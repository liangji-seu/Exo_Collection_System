"""Core UUID-linked domain models for the collection system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from .states import TrialState


def utc_now() -> datetime:
    """Return an aware UTC timestamp suitable for audit fields."""

    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalize aware values to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


UTCDateTime = Annotated[datetime, AfterValidator(ensure_utc)]
NonEmptyStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class FrozenDict(dict[str, Any]):
    """JSON-compatible mapping that rejects in-place changes."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("frozen condition data cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __deepcopy__(self, _memo: dict[int, Any]) -> FrozenDict:
        return self


class FrozenList(list[Any]):
    """JSON-compatible sequence that rejects in-place changes."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("frozen condition data cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def __deepcopy__(self, _memo: dict[int, Any]) -> FrozenList:
        return self


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_freeze_json(item) for item in value)
    return value


def normalize_relative_path(value: str) -> str:
    """Validate and canonicalize a portable Trial-relative path."""

    text = value.strip().replace("\\", "/")
    if not text:
        raise ValueError("relative_path must not be empty")
    windows_path = PureWindowsPath(text)
    path = PurePosixPath(text)
    if windows_path.drive or windows_path.root or path.is_absolute():
        raise ValueError("relative_path must not be absolute")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative_path may not contain '.' or '..' components")
    windows_reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}"
        for prefix in ("COM", "LPT")
        for number in range(1, 10)
    }
    for part in path.parts:
        if (
            any(ord(character) < 32 for character in part)
            or any(character in '<>:"|?*' for character in part)
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].upper() in windows_reserved
        ):
            raise ValueError(
                "relative_path contains a component that is unsafe on Windows"
            )
    return path.as_posix()


class DomainModel(BaseModel):
    """Strict base model shared by persisted domain entities."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        validate_default=True,
    )


class QualityGrade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    INVALID = "INVALID"


class UploadState(StrEnum):
    NOT_SELECTED = "NOT_SELECTED"
    QUEUED = "QUEUED"
    TRANSFERRING = "TRANSFERRING"
    TRANSFERRED = "TRANSFERRED"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


class ArtifactKind(StrEnum):
    RAW = "raw"
    DERIVED = "derived"
    REPORT = "report"
    EXTERNAL = "external"
    LOG = "log"


class Project(DomainModel):
    project_uuid: UUID = Field(default_factory=uuid4)
    project_code: NonEmptyStr | None = None
    project_name: NonEmptyStr
    principal_investigator: NonEmptyStr
    protocol_version: NonEmptyStr
    data_root: NonEmptyStr
    condition_definition_version: NonEmptyStr
    default_device_config: dict[str, JsonValue] = Field(default_factory=dict)
    created_at_utc: UTCDateTime = Field(default_factory=utc_now)
    updated_at_utc: UTCDateTime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_audit_order(self) -> Project:
        if self.updated_at_utc < self.created_at_utc:
            raise ValueError("updated_at_utc precedes created_at_utc")
        return self


class Subject(DomainModel):
    subject_uuid: UUID = Field(default_factory=uuid4)
    project_uuid: UUID
    subject_code: NonEmptyStr
    group: str | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    created_at_utc: UTCDateTime = Field(default_factory=utc_now)
    updated_at_utc: UTCDateTime = Field(default_factory=utc_now)

    @field_validator("group")
    @classmethod
    def normalize_group(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_audit_order(self) -> Subject:
        if self.updated_at_utc < self.created_at_utc:
            raise ValueError("updated_at_utc precedes created_at_utc")
        return self


class DeviceReference(DomainModel):
    device_id: NonEmptyStr
    modality: NonEmptyStr
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    firmware_version: str | None = None
    driver_version: str | None = None
    required: bool = True
    clock_domain: NonEmptyStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CalibrationReference(DomainModel):
    calibration_uuid: UUID = Field(default_factory=uuid4)
    device_id: NonEmptyStr
    calibration_version: NonEmptyStr
    performed_at_utc: UTCDateTime
    operator: str | None = None
    artifact_uuid: UUID | None = None
    notes: str | None = None


class Session(DomainModel):
    session_uuid: UUID = Field(default_factory=uuid4)
    project_uuid: UUID
    subject_uuid: UUID
    operator: NonEmptyStr
    software_version: NonEmptyStr
    devices: list[DeviceReference] = Field(default_factory=list)
    calibrations: list[CalibrationReference] = Field(default_factory=list)
    started_at_utc: UTCDateTime = Field(default_factory=utc_now)
    ended_at_utc: UTCDateTime | None = None
    created_at_utc: UTCDateTime = Field(default_factory=utc_now)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_session(self) -> Session:
        if self.ended_at_utc is not None and self.ended_at_utc < self.started_at_utc:
            raise ValueError("ended_at_utc precedes started_at_utc")
        device_ids = [device.device_id for device in self.devices]
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("Session device_id values must be unique")
        return self


class Condition(DomainModel):
    """Frozen snapshot of the condition selected before recording starts.

    Condition and signal quality are intentionally separate concepts.  A Trial
    embeds this snapshot; quality is recorded in ``Trial.quality_grade`` and the
    Manifest quality summary.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    condition_code: NonEmptyStr
    condition_name: NonEmptyStr
    condition_level: int | str | None = None
    levels: dict[str, int | str] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    repeat_index: int = Field(ge=1)
    protocol_version: NonEmptyStr
    grading_rule_version: str | None = None
    selected_at_utc: UTCDateTime = Field(default_factory=utc_now)

    @field_validator("levels", "parameters", mode="after")
    @classmethod
    def freeze_nested_condition_data(cls, value: dict[str, Any]) -> FrozenDict:
        return _freeze_json(value)


class Artifact(DomainModel):
    artifact_uuid: UUID = Field(default_factory=uuid4)
    trial_uuid: UUID
    modality: NonEmptyStr
    kind: ArtifactKind
    media_type: NonEmptyStr
    relative_path: NonEmptyStr
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: Sha256 | None = None
    created_at_utc: UTCDateTime = Field(default_factory=utc_now)
    finalized_at_utc: UTCDateTime | None = None
    source_artifact_uuids: list[UUID] = Field(default_factory=list)
    immutable: bool = True
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str | None) -> str | None:
        return value.lower() if value is not None else None

    @model_validator(mode="after")
    def validate_artifact(self) -> Artifact:
        if self.kind is ArtifactKind.RAW and not self.immutable:
            raise ValueError("raw Artifacts must be immutable")
        if (
            self.finalized_at_utc is not None
            and self.finalized_at_utc < self.created_at_utc
        ):
            raise ValueError("finalized_at_utc precedes created_at_utc")
        if len(self.source_artifact_uuids) != len(set(self.source_artifact_uuids)):
            raise ValueError("source_artifact_uuids must be unique")
        if self.artifact_uuid in self.source_artifact_uuids:
            raise ValueError("an Artifact may not derive from itself")
        return self


class Trial(DomainModel):
    trial_uuid: UUID = Field(default_factory=uuid4)
    project_uuid: UUID
    subject_uuid: UUID
    session_uuid: UUID
    condition: Condition
    state: TrialState = TrialState.IDLE
    modalities: list[str] = Field(default_factory=list)
    devices: list[DeviceReference] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    software_version: NonEmptyStr
    config_version: NonEmptyStr
    git_commit: str | None = None
    started_at_utc: UTCDateTime | None = None
    stopped_at_utc: UTCDateTime | None = None
    finalized_at_utc: UTCDateTime | None = None
    start_host_monotonic_ns: int | None = Field(default=None, ge=0)
    stop_host_monotonic_ns: int | None = Field(default=None, ge=0)
    quality_grade: QualityGrade | None = None
    abnormal_stop: bool = False
    upload_state: UploadState = UploadState.NOT_SELECTED
    created_at_utc: UTCDateTime = Field(default_factory=utc_now)

    @field_validator("modalities")
    @classmethod
    def normalize_modalities(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("modality names must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("modalities must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_trial(self) -> Trial:
        times = [
            value
            for value in (
                self.started_at_utc,
                self.stopped_at_utc,
                self.finalized_at_utc,
            )
            if value is not None
        ]
        if times != sorted(times):
            raise ValueError("Trial UTC timestamps are out of order")
        if (
            self.start_host_monotonic_ns is not None
            and self.stop_host_monotonic_ns is not None
            and self.stop_host_monotonic_ns < self.start_host_monotonic_ns
        ):
            raise ValueError("stop_host_monotonic_ns precedes start")
        artifact_ids = [artifact.artifact_uuid for artifact in self.artifacts]
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Artifact UUID values must be unique within a Trial")
        if len(paths) != len(set(paths)):
            raise ValueError("Artifact relative paths must be unique within a Trial")
        if any(artifact.trial_uuid != self.trial_uuid for artifact in self.artifacts):
            raise ValueError("every Artifact must reference this Trial UUID")
        device_ids = [device.device_id for device in self.devices]
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("Trial device_id values must be unique")
        return self
