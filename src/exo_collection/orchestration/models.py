"""Serializable inputs and outputs for one simulated Trial worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class SubjectExperimentMetadata(OrchestrationModel):
    """Optional subject measurements recorded at collection time."""

    height_cm: float | None = Field(default=None, ge=30, le=250)
    weight_kg: float | None = Field(default=None, ge=1, le=500)
    leg_length_cm: float | None = Field(default=None, ge=10, le=200)
    sex: Literal["female", "male", "other"] | None = None
    age_years: int | None = Field(default=None, ge=0, le=120)


class UltrasoundProbeMetadata(OrchestrationModel):
    """Optional ultrasound placement and fixation record.

    Strap pressure remains free text because the current hardware/protocol does
    not define a trustworthy pressure unit.  This avoids inventing a unit while
    still preserving the operator's experimental record.
    """

    muscle: str | None = Field(default=None, max_length=200)
    laterality: Literal["left", "right"] | None = None
    longitudinal_position: Literal["proximal", "middle", "distal"] | None = None
    channel_mapping: tuple[
        str | None,
        str | None,
        str | None,
        str | None,
    ] = (None, None, None, None)
    fixation_method: str | None = Field(default=None, max_length=500)
    strap_pressure: str | None = Field(default=None, max_length=200)
    probe_reapplied: bool | None = None

    @field_validator(
        "muscle",
        "fixation_method",
        "strap_pressure",
        mode="before",
    )
    @classmethod
    def optional_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("channel_mapping", mode="before")
    @classmethod
    def normalize_channel_mapping(cls, value: Any) -> Any:
        if value is None:
            return (None, None, None, None)
        if isinstance(value, (list, tuple)):
            return tuple(
                (item.strip() or None) if isinstance(item, str) else item
                for item in value
            )
        return value


class MeasuredConditionMetadata(OrchestrationModel):
    """Optional measured condition values, distinct from protocol targets."""

    treadmill_speed_mps: float | None = Field(default=None, ge=0, le=15)
    assist_level: float | None = Field(default=None, ge=0, le=100)
    load_kg: float | None = Field(default=None, ge=0, le=500)
    slope_deg: float | None = Field(default=None, ge=-45, le=45)


class TrialExperimentMetadata(OrchestrationModel):
    """Structured, optional experiment notes attached to one Trial request."""

    subject: SubjectExperimentMetadata = Field(
        default_factory=SubjectExperimentMetadata
    )
    ultrasound_probe: UltrasoundProbeMetadata = Field(
        default_factory=UltrasoundProbeMetadata
    )
    measured_condition: MeasuredConditionMetadata = Field(
        default_factory=MeasuredConditionMetadata
    )
    trial_notes: str | None = Field(default=None, max_length=4000)

    @field_validator("trial_notes", mode="before")
    @classmethod
    def optional_notes(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value


class TrialRunRequest(OrchestrationModel):
    data_root: Path
    device_profile_key: Literal["simulated", "hardware"] = "simulated"
    device_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Interactive collection has no predetermined duration.  CLI/smoke callers
    # can still provide a finite duration.  It is measured from a qualified
    # sync trigger when one arrives in time, otherwise from recording-gate
    # start so an optional missing pulse can never hang the Worker.
    duration_s: float | None = Field(default=None, gt=0, le=86_400)
    sync_wait_timeout_s: float | None = Field(default=None, gt=0, le=86_400)
    project_code: Literal["F", "T"] = "T"
    project_uuid: UUID = Field(default_factory=uuid4)
    subject_uuid: UUID = Field(default_factory=uuid4)
    session_uuid: UUID = Field(default_factory=uuid4)
    trial_uuid: UUID = Field(default_factory=uuid4)
    project_name: str = "测试"
    principal_investigator: str = "Not specified"
    subject_code: str = "001"
    subject_group: str | None = None
    # The current experiment workflow has no operator input.  The persisted
    # Session field remains for backward/schema compatibility and audit tools.
    operator: str = "not_recorded"
    condition_code: str = "WALK_LEVEL"
    condition_name: str = "Level walking"
    condition_level: int | str | None = 2
    condition_parameters: dict[str, JsonValue] = Field(
        default_factory=lambda: {"speed_mps": 0.8, "assist_level": 3, "simulated": True}
    )
    repeat_index: int = Field(default=1, ge=1)
    protocol_version: str = "1.0.0"
    config_version: str = "1.0.0"
    experiment_metadata: TrialExperimentMetadata = Field(
        default_factory=TrialExperimentMetadata
    )
    simulation: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Subset of modalities enabled for this trial.  When None or empty the
    # worker preserves backward behaviour and connects everything in the
    # profile.  The collector UI sets this from the currently-previewed set.
    enabled_modalities: frozenset[str] | None = None

    @field_validator("device_overrides")
    @classmethod
    def validate_device_override_modalities(
        cls, value: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        allowed = {"ultrasound", "imu", "encoder", "sync_pulse"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "unknown device override modalities: " + ", ".join(sorted(unknown))
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def stable_hierarchy_ids(cls, value: Any) -> Any:
        """Derive stable Project/Subject UUIDs when callers omit them.

        F/T and the human subject code remain readable labels only; UUIDs are
        still the authoritative relationship keys in the Manifest and Catalog.
        """

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        project_code = str(normalized.get("project_code", "T")).strip().upper()
        if "project_uuid" not in normalized:
            normalized["project_uuid"] = uuid5(
                NAMESPACE_URL,
                f"exo-collection:project:{project_code}",
            )
        if "subject_uuid" not in normalized:
            subject_code = str(normalized.get("subject_code", "001")).strip()
            normalized["subject_uuid"] = uuid5(
                UUID(str(normalized["project_uuid"])),
                f"subject:{subject_code}",
            )
        return normalized

    @field_validator(
        "project_name",
        "principal_investigator",
        "subject_code",
        "operator",
        "condition_code",
        "condition_name",
        "protocol_version",
        "config_version",
    )
    @classmethod
    def non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


class TrialRunResult(OrchestrationModel):
    trial_uuid: UUID
    state: str
    trial_directory: Path
    manifest_path: Path
    duration_s: float = Field(ge=0)
    modality_counts: dict[str, int] = Field(default_factory=dict)
    pulse_event_count: int = Field(default=0, ge=0)
    trigger_count: int = Field(default=0, ge=0)
    first_trigger_host_monotonic_ns: int | None = Field(default=None, ge=0)
    quality_grade: str
