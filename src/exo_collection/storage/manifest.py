"""Versioned, strictly validated Trial Manifest contract.

``manifest.json`` is the entry point to a published Trial package.  This module
contains no device-specific assumptions and is shared by Collector, recovery,
Catalog rebuilding, and Data Studio.
"""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from exo_collection.domain.models import (
    Artifact,
    ArtifactKind,
    Condition,
    NonEmptyStr,
    QualityGrade,
    Sha256,
    UTCDateTime,
    UploadState,
    normalize_relative_path,
    utc_now,
)
from exo_collection.domain.states import TrialState


MANIFEST_SCHEMA_VERSION = "1.0.0"
SemVer = Annotated[
    str,
    StringConstraints(
        pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
        r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
        r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
    ),
]


class ManifestModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        use_enum_values=False,
    )


class ManifestArtifact(Artifact):
    """An Artifact after Writer closure and integrity calculation."""

    size_bytes: int = Field(ge=0)
    sha256: Sha256
    finalized_at_utc: UTCDateTime

    @model_validator(mode="after")
    def reject_temporary_artifact(self) -> ManifestArtifact:
        components = self.relative_path.replace("\\", "/").split("/")
        if any(
            component.endswith(".partial") or component.endswith(".recording")
            for component in components
        ):
            raise ValueError("Manifest Artifacts may not refer to temporary paths")
        return self


class DeviceProvenance(ManifestModel):
    device_id: NonEmptyStr
    modality: NonEmptyStr
    adapter_type: NonEmptyStr
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    firmware_version: str | None = None
    driver_version: str | None = None
    calibration_version: str | None = None
    calibration_artifact_uuid: UUID | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ModalityManifest(ManifestModel):
    modality: NonEmptyStr
    required: bool = True
    adapter_type: NonEmptyStr
    writer_type: NonEmptyStr
    clock_domain: NonEmptyStr
    device_ids: list[str] = Field(default_factory=list)
    artifact_uuids: list[UUID] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    units: list[str] = Field(default_factory=list)
    sample_count: int | None = Field(default=None, ge=0)
    frame_count: int | None = Field(default=None, ge=0)
    first_sample_index: int | None = Field(default=None, ge=0)
    last_sample_index: int | None = Field(default=None, ge=0)
    sequence_gap_count: int = Field(default=0, ge=0)
    nominal_sample_rate_hz: float | None = Field(default=None, gt=0)
    actual_sample_rate_hz: float | None = Field(default=None, ge=0)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_modality(self) -> ModalityManifest:
        if len(self.device_ids) != len(set(self.device_ids)):
            raise ValueError("device_ids must be unique within a modality")
        if len(self.artifact_uuids) != len(set(self.artifact_uuids)):
            raise ValueError("artifact_uuids must be unique within a modality")
        if len(self.channels) != len(set(self.channels)):
            raise ValueError("channel names must be unique")
        if self.units and len(self.units) != len(self.channels):
            raise ValueError("units must be empty or match the channel count")
        if (
            self.first_sample_index is not None
            and self.last_sample_index is not None
            and self.last_sample_index < self.first_sample_index
        ):
            raise ValueError("last_sample_index precedes first_sample_index")
        return self


class TrialTiming(ManifestModel):
    """Both audit time and the non-jumping common acquisition time."""

    started_at_utc: UTCDateTime
    stopped_at_utc: UTCDateTime | None = None
    finalized_at_utc: UTCDateTime | None = None
    start_host_monotonic_ns: int = Field(ge=0)
    stop_host_monotonic_ns: int | None = Field(default=None, ge=0)
    finalize_host_monotonic_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_timing_order(self) -> TrialTiming:
        utc_values = [
            item
            for item in (
                self.started_at_utc,
                self.stopped_at_utc,
                self.finalized_at_utc,
            )
            if item is not None
        ]
        if utc_values != sorted(utc_values):
            raise ValueError("Trial UTC timestamps are out of order")
        monotonic_values = [
            item
            for item in (
                self.start_host_monotonic_ns,
                self.stop_host_monotonic_ns,
                self.finalize_host_monotonic_ns,
            )
            if item is not None
        ]
        if monotonic_values != sorted(monotonic_values):
            raise ValueError("Trial host monotonic timestamps are out of order")
        return self


class SoftwareProvenance(ManifestModel):
    application: NonEmptyStr
    application_version: NonEmptyStr
    core_version: NonEmptyStr
    git_commit: NonEmptyStr
    python_version: str | None = None
    build_id: str | None = None


class ConfigurationSnapshot(ManifestModel):
    config_version: NonEmptyStr
    protocol_version: NonEmptyStr
    condition_definition_version: NonEmptyStr
    content_sha256: Sha256
    snapshot_relative_path: str | None = None
    source_files: list[str] = Field(default_factory=list)

    @field_validator("snapshot_relative_path")
    @classmethod
    def validate_snapshot_path(cls, value: str | None) -> str | None:
        return normalize_relative_path(value) if value is not None else None

    @field_validator("source_files")
    @classmethod
    def validate_source_files(cls, values: list[str]) -> list[str]:
        normalized = [normalize_relative_path(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("source_files must be unique")
        return normalized


class ClockDomainKind(StrEnum):
    HOST_MONOTONIC = "host_monotonic"
    DEVICE_TICK = "device_tick"
    DEVICE_TIMESTAMP = "device_timestamp"
    EXTERNAL = "external"


class ClockDomainManifest(ManifestModel):
    clock_domain: NonEmptyStr
    kind: ClockDomainKind
    unit: NonEmptyStr
    device_id: str | None = None
    nominal_rate_hz: float | None = Field(default=None, gt=0)
    description: str | None = None


class ResidualStatistics(ManifestModel):
    count: int = Field(ge=0)
    mean_ns: float | None = None
    rms_ns: float | None = Field(default=None, ge=0)
    standard_deviation_ns: float | None = Field(default=None, ge=0)
    p95_absolute_ns: float | None = Field(default=None, ge=0)
    max_absolute_ns: float | None = Field(default=None, ge=0)


class ClockMapping(ManifestModel):
    """Persisted ``t_target_ns = a * t_source + b`` clock model."""

    mapping_uuid: UUID = Field(default_factory=uuid4)
    source_clock_domain: NonEmptyStr
    target_clock_domain: NonEmptyStr = "host_monotonic"
    scale_a: float
    offset_b_ns: float
    valid_source_start: int | float | None = None
    valid_source_end: int | float | None = None
    anchor_count: int = Field(ge=1)
    residuals: ResidualStatistics
    algorithm_version: NonEmptyStr
    created_at_utc: UTCDateTime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_mapping(self) -> ClockMapping:
        if self.scale_a <= 0:
            raise ValueError("scale_a must be positive")
        if (
            self.valid_source_start is not None
            and self.valid_source_end is not None
            and self.valid_source_end < self.valid_source_start
        ):
            raise ValueError("clock mapping valid interval is reversed")
        if self.source_clock_domain == self.target_clock_domain:
            raise ValueError("clock mapping source and target must differ")
        return self


class AlignmentQuality(StrEnum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    UNAVAILABLE = "UNAVAILABLE"


class AlignmentRecord(ManifestModel):
    alignment_uuid: UUID = Field(default_factory=uuid4)
    source_artifact_uuid: UUID
    source_clock_domain: NonEmptyStr
    pulse_ids: list[str] = Field(default_factory=list)
    mapping_uuid: UUID | None = None
    offset_only: bool = False
    quality: AlignmentQuality = AlignmentQuality.UNAVAILABLE
    match_error_rms_ns: float | None = Field(default=None, ge=0)
    algorithm_version: NonEmptyStr
    derived_artifact_uuid: UUID | None = None
    created_at_utc: UTCDateTime = Field(default_factory=utc_now)

    @field_validator("pulse_ids")
    @classmethod
    def validate_pulse_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("pulse_id must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("pulse_ids must be unique")
        return normalized


class ClockAndAlignment(ManifestModel):
    common_clock_domain: Literal["host_monotonic"] = "host_monotonic"
    host_clock_api: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"
    clock_domains: list[ClockDomainManifest] = Field(default_factory=list)
    mappings: list[ClockMapping] = Field(default_factory=list)
    raw_sync_pulse_artifact_uuids: list[UUID] = Field(default_factory=list)
    sync_event_artifact_uuids: list[UUID] = Field(default_factory=list)
    alignments: list[AlignmentRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_clock_graph(self) -> ClockAndAlignment:
        domains = [item.clock_domain for item in self.clock_domains]
        if len(domains) != len(set(domains)):
            raise ValueError("clock_domain values must be unique")
        known_domains = set(domains) | {self.common_clock_domain}
        mapping_ids = [mapping.mapping_uuid for mapping in self.mappings]
        if len(mapping_ids) != len(set(mapping_ids)):
            raise ValueError("clock mapping UUID values must be unique")
        for mapping in self.mappings:
            if mapping.source_clock_domain not in known_domains:
                raise ValueError(
                    f"unknown source clock domain: {mapping.source_clock_domain}"
                )
            if mapping.target_clock_domain not in known_domains:
                raise ValueError(
                    f"unknown target clock domain: {mapping.target_clock_domain}"
                )
        known_mapping_ids = set(mapping_ids)
        alignment_ids = [item.alignment_uuid for item in self.alignments]
        if len(alignment_ids) != len(set(alignment_ids)):
            raise ValueError("alignment UUID values must be unique")
        for alignment in self.alignments:
            if (
                alignment.mapping_uuid is not None
                and alignment.mapping_uuid not in known_mapping_ids
            ):
                raise ValueError("Alignment refers to an unknown clock mapping")
        return self


class QualityIssueSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class QualityIssue(ManifestModel):
    code: NonEmptyStr
    severity: QualityIssueSeverity
    message: NonEmptyStr
    modality: str | None = None
    artifact_uuid: UUID | None = None
    metric: str | None = None
    observed_value: JsonValue | None = None
    threshold: JsonValue | None = None


class QualitySummary(ManifestModel):
    """Computed result plus a non-destructive optional human review."""

    computed_grade: QualityGrade | None = None
    required_artifacts_complete: bool = False
    integrity_checks_passed: bool = False
    algorithm_version: str | None = None
    assessed_at_utc: UTCDateTime | None = None
    issues: list[QualityIssue] = Field(default_factory=list)
    metric_count: int = Field(default=0, ge=0)
    report_artifact_uuid: UUID | None = None
    reviewed_grade: QualityGrade | None = None
    reviewed_by: str | None = None
    reviewed_at_utc: UTCDateTime | None = None
    review_reason: str | None = None

    @model_validator(mode="after")
    def validate_review(self) -> QualitySummary:
        review_values = (
            self.reviewed_by,
            self.reviewed_at_utc,
            self.review_reason,
        )
        if self.reviewed_grade is not None and not all(review_values):
            raise ValueError(
                "a reviewed grade requires reviewer, UTC time, and reason"
            )
        if self.reviewed_grade is None and any(review_values):
            raise ValueError("review metadata requires reviewed_grade")
        if self.assessed_at_utc is not None and self.algorithm_version is None:
            raise ValueError("an assessed quality result needs algorithm_version")
        return self


class AbnormalTermination(ManifestModel):
    occurred: bool = False
    reason: str | None = None
    error_code: str | None = None
    last_state: TrialState | None = None
    occurred_at_utc: UTCDateTime | None = None
    related_event_uuids: list[UUID] = Field(default_factory=list)
    recovery_notes: str | None = None

    @model_validator(mode="after")
    def validate_abnormal_details(self) -> AbnormalTermination:
        details = (
            self.reason,
            self.error_code,
            self.last_state,
            self.occurred_at_utc,
            self.related_event_uuids,
            self.recovery_notes,
        )
        if not self.occurred and any(details):
            raise ValueError("abnormal termination details require occurred=true")
        if self.occurred and not self.reason:
            raise ValueError("abnormal termination requires a reason")
        return self


class ExternalArtifactReference(ManifestModel):
    artifact_uuid: UUID
    source_system: NonEmptyStr
    external_clock_domain: str | None = None
    original_filename: str | None = None
    imported_at_utc: UTCDateTime
    alignment_uuid: UUID | None = None


class UploadRecordReference(ManifestModel):
    """Reference only: credentials and passwords never belong in a Manifest."""

    transfer_batch_uuid: UUID
    status: UploadState
    catalog_record_id: str | None = None
    remote_profile_name: str | None = None
    remote_relative_path: str | None = None
    created_at_utc: UTCDateTime
    verified_at_utc: UTCDateTime | None = None

    @field_validator("remote_relative_path")
    @classmethod
    def validate_remote_path(cls, value: str | None) -> str | None:
        return normalize_relative_path(value) if value is not None else None

    @model_validator(mode="after")
    def validate_upload_record(self) -> UploadRecordReference:
        if self.status is UploadState.VERIFIED and self.verified_at_utc is None:
            raise ValueError("VERIFIED upload reference needs verified_at_utc")
        if self.status is not UploadState.VERIFIED and self.verified_at_utc is not None:
            raise ValueError("verified_at_utc is only valid for VERIFIED uploads")
        if (
            self.verified_at_utc is not None
            and self.verified_at_utc < self.created_at_utc
        ):
            raise ValueError("verified_at_utc precedes upload creation")
        return self


PUBLISHED_MANIFEST_STATES = frozenset(
    {TrialState.FINALIZED, TrialState.ABORTED, TrialState.RECOVERABLE}
)


class TrialManifest(ManifestModel):
    """Schema for a single UUID-linked Trial package."""

    schema_version: SemVer = MANIFEST_SCHEMA_VERSION
    manifest_uuid: UUID = Field(default_factory=uuid4)
    project_uuid: UUID
    subject_uuid: UUID
    session_uuid: UUID
    trial_uuid: UUID
    state: TrialState
    condition: Condition
    timing: TrialTiming
    software: SoftwareProvenance
    configuration: ConfigurationSnapshot
    devices: list[DeviceProvenance] = Field(default_factory=list)
    modalities: list[ModalityManifest] = Field(default_factory=list)
    artifacts: list[ManifestArtifact] = Field(default_factory=list)
    clock_and_alignment: ClockAndAlignment = Field(
        default_factory=ClockAndAlignment
    )
    quality: QualitySummary = Field(default_factory=QualitySummary)
    abnormal_termination: AbnormalTermination = Field(
        default_factory=AbnormalTermination
    )
    external_artifacts: list[ExternalArtifactReference] = Field(default_factory=list)
    upload_records: list[UploadRecordReference] = Field(default_factory=list)
    created_at_utc: UTCDateTime = Field(default_factory=utc_now)

    @field_validator("state")
    @classmethod
    def validate_published_state(cls, value: TrialState) -> TrialState:
        if value not in PUBLISHED_MANIFEST_STATES:
            allowed = ", ".join(sorted(state.value for state in PUBLISHED_MANIFEST_STATES))
            raise ValueError(f"Manifest state must be one of: {allowed}")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> TrialManifest:
        if self.state is TrialState.FINALIZED:
            if (
                self.timing.stopped_at_utc is None
                or self.timing.finalized_at_utc is None
                or self.timing.stop_host_monotonic_ns is None
                or self.timing.finalize_host_monotonic_ns is None
            ):
                raise ValueError("FINALIZED Manifest needs stop and finalization times")
        if self.state is TrialState.ABORTED and self.timing.stopped_at_utc is None:
            raise ValueError("ABORTED Manifest needs stopped_at_utc")
        if self.state in {TrialState.ABORTED, TrialState.RECOVERABLE}:
            if not self.abnormal_termination.occurred:
                raise ValueError(
                    "ABORTED/RECOVERABLE Manifest requires abnormal termination details"
                )

        artifact_ids = [artifact.artifact_uuid for artifact in self.artifacts]
        artifact_paths = [artifact.relative_path for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Artifact UUID values must be unique")
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("Artifact relative paths must be unique")
        if any(artifact.trial_uuid != self.trial_uuid for artifact in self.artifacts):
            raise ValueError("every Artifact must reference the Manifest Trial UUID")
        known_artifacts = {artifact.artifact_uuid: artifact for artifact in self.artifacts}
        for artifact in self.artifacts:
            unknown_sources = set(artifact.source_artifact_uuids) - set(known_artifacts)
            if unknown_sources:
                raise ValueError("Artifact has unknown source_artifact_uuids")

        modality_names = [modality.modality for modality in self.modalities]
        if len(modality_names) != len(set(modality_names)):
            raise ValueError("modality names must be unique")
        for modality in self.modalities:
            for artifact_uuid in modality.artifact_uuids:
                artifact = known_artifacts.get(artifact_uuid)
                if artifact is None:
                    raise ValueError("Modality refers to an unknown Artifact")
                if artifact.modality != modality.modality:
                    raise ValueError("Modality refers to an Artifact of another modality")
            if (
                self.state is TrialState.FINALIZED
                and modality.required
                and not modality.artifact_uuids
            ):
                raise ValueError("required modality has no Artifact")

        device_ids = [device.device_id for device in self.devices]
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("device_id values must be unique")
        known_devices = set(device_ids)
        for modality in self.modalities:
            if set(modality.device_ids) - known_devices:
                raise ValueError("Modality refers to an unknown device_id")

        clock_artifact_ids = set(
            self.clock_and_alignment.raw_sync_pulse_artifact_uuids
            + self.clock_and_alignment.sync_event_artifact_uuids
        )
        if clock_artifact_ids - set(known_artifacts):
            raise ValueError("clock section refers to an unknown Artifact")
        alignment_ids = {
            alignment.alignment_uuid
            for alignment in self.clock_and_alignment.alignments
        }
        for alignment in self.clock_and_alignment.alignments:
            if alignment.source_artifact_uuid not in known_artifacts:
                raise ValueError("Alignment source Artifact is unknown")
            if (
                alignment.derived_artifact_uuid is not None
                and alignment.derived_artifact_uuid not in known_artifacts
            ):
                raise ValueError("Alignment derived Artifact is unknown")

        external_ids: list[UUID] = []
        for reference in self.external_artifacts:
            external_ids.append(reference.artifact_uuid)
            artifact = known_artifacts.get(reference.artifact_uuid)
            if artifact is None or artifact.kind is not ArtifactKind.EXTERNAL:
                raise ValueError("external reference must target an external Artifact")
            if (
                reference.alignment_uuid is not None
                and reference.alignment_uuid not in alignment_ids
            ):
                raise ValueError("external reference has unknown alignment_uuid")
        if len(external_ids) != len(set(external_ids)):
            raise ValueError("external Artifact references must be unique")

        if self.quality.report_artifact_uuid is not None:
            report = known_artifacts.get(self.quality.report_artifact_uuid)
            if report is None or report.kind is not ArtifactKind.REPORT:
                raise ValueError("quality report must refer to a report Artifact")
        for issue in self.quality.issues:
            if issue.artifact_uuid is not None and issue.artifact_uuid not in known_artifacts:
                raise ValueError("quality issue refers to an unknown Artifact")

        batch_ids = [record.transfer_batch_uuid for record in self.upload_records]
        if len(batch_ids) != len(set(batch_ids)):
            raise ValueError("transfer batch references must be unique")
        return self


# Short compatibility name for callers that refer to the file rather than Trial.
Manifest = TrialManifest


def manifest_json_schema() -> dict[str, Any]:
    """Return the versioned JSON Schema used by external tooling."""

    schema = TrialManifest.model_json_schema()
    schema["$id"] = (
        "https://exo-collection.local/schemas/manifest/"
        f"{MANIFEST_SCHEMA_VERSION}.json"
    )
    schema["title"] = f"Exo Collection Trial Manifest {MANIFEST_SCHEMA_VERSION}"
    return schema


def _reject_partial_path(path: Path) -> None:
    if any(part.endswith(".partial") for part in path.parts):
        raise ValueError("refusing to read or publish a .partial Manifest path")


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_manifest(
    path: str | os.PathLike[str],
    manifest: TrialManifest | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically publish a UTF-8 Manifest.

    Existing Manifests are protected by default to preserve a finalized Trial's
    immutable record.  Schema migration tools must opt in to replacement
    explicitly and should retain their own audit trail.
    """

    target = Path(path)
    _reject_partial_path(target)
    validated = (
        manifest
        if isinstance(manifest, TrialManifest)
        else TrialManifest.model_validate(manifest)
    )
    if target.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {target}")
    _atomic_json_write(target, validated.model_dump(mode="json"))
    return target


def load_manifest(path: str | os.PathLike[str]) -> TrialManifest:
    """Load a complete Manifest; temporary ``.partial`` inputs are rejected."""

    source = Path(path)
    _reject_partial_path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    return TrialManifest.model_validate_json(source.read_text(encoding="utf-8"))


def export_manifest_json_schema(
    path: str | os.PathLike[str], *, overwrite: bool = True
) -> Path:
    """Atomically export the public Manifest JSON Schema."""

    target = Path(path)
    _reject_partial_path(target)
    if target.exists() and not overwrite:
        raise FileExistsError(f"schema already exists: {target}")
    _atomic_json_write(target, manifest_json_schema())
    return target

