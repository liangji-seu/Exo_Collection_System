"""Manifest/Catalog-driven local dataset management without raw-data mutation.

The module deliberately has no Qt dependency.  It provides immutable records
for a future Data Studio UI and treats human quality reviews and upload audits
as external, verifiable sidecars.  Artifact payloads are never opened for
filtering, coverage calculation, state summaries, or inventory export.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import StrEnum
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Sequence
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from exo_collection.domain.states import TrialState
from exo_collection.external import ExternalAnnexManifest
from exo_collection.protocols.models import (
    ConditionDefinition,
    ProtocolDefinition,
    load_default_protocol,
)
from exo_collection.storage.activity import read_activity
from exo_collection.storage.checksum import sha256_file
from exo_collection.storage.layout import (
    iter_aborted_directories,
    iter_finalized_manifest_paths,
    iter_recording_directories,
    name_has_storage_suffix,
    path_has_unpublished_component,
)
from exo_collection.storage.manifest import TrialManifest, load_manifest

from .quality_reviews import (
    QUALITY_REVIEW_DIRECTORY,
    QualityReviewError,
    list_quality_reviews,
)
from .service import DataStudioSnapshot, load_catalog_snapshot


UPLOAD_AUDIT_DIRECTORY = ".upload-audit"
ANNEX_DIRECTORY_NAME = "external_annexes"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManagementError(RuntimeError):
    """A local management operation cannot produce trustworthy results."""


class ManagementBusyError(ManagementError):
    """The Collector owns the dataset root, so full management is deferred."""


class QualityReviewStatus(StrEnum):
    PENDING = "PENDING"
    REVIEWED = "REVIEWED"
    INVALID_SIDECAR = "INVALID_SIDECAR"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class UploadAuditStatus(StrEnum):
    PENDING = "PENDING"
    FAILED = "FAILED"
    VERIFIED = "VERIFIED"
    INVALID_SIDECAR = "INVALID_SIDECAR"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PackageState(StrEnum):
    PENDING_RECOVERY = "PENDING_RECOVERY"
    ABORTED = "ABORTED"
    ABORTED_UNVERIFIED = "ABORTED_UNVERIFIED"


class AnnexValidationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class TrialManagementRecord:
    data_root: Path
    manifest_path: Path
    manifest_relative_path: str
    project_uuid: str
    project_code: str | None
    project_name: str | None
    subject_uuid: str
    subject_code: str | None
    session_uuid: str
    trial_uuid: str
    state: str
    condition_code: str
    condition_name: str
    repeat_index: int
    started_at_utc: datetime
    duration_s: float
    computed_quality_grade: str
    effective_quality_grade: str
    quality_review_status: QualityReviewStatus
    quality_reviewed_at_utc: datetime | None
    upload_status: UploadAuditStatus
    upload_verified_at_utc_ns: int | None
    artifact_count: int
    artifact_total_bytes: int
    sidecar_errors: tuple[str, ...] = ()

    @property
    def started_date(self) -> date:
        return self.started_at_utc.astimezone(timezone.utc).date()

    @property
    def pending_quality_review(self) -> bool:
        return (
            self.state == TrialState.FINALIZED.value
            and self.quality_review_status
            in {QualityReviewStatus.PENDING, QualityReviewStatus.INVALID_SIDECAR}
        )

    @property
    def pending_upload(self) -> bool:
        return (
            self.state == TrialState.FINALIZED.value
            and self.upload_status is not UploadAuditStatus.VERIFIED
        )


@dataclass(frozen=True, slots=True)
class ManagementIndex:
    data_root: Path
    records: tuple[TrialManagementRecord, ...]
    catalog_scan_failures: tuple[tuple[str, str], ...]
    manifest_failures: tuple[tuple[str, str], ...]


class TrialFilter(BaseModel):
    """Composable, strict filter accepted by the non-Qt management backend."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    projects: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    start_date: date | None = None
    end_date: date | None = None
    qualities: tuple[str, ...] = ()
    text: str | None = Field(default=None, max_length=500)

    @field_validator(
        "projects", "subjects", "sessions", "conditions", "qualities", mode="before"
    )
    @classmethod
    def normalize_terms(cls, value: Any) -> Any:
        if value is None:
            return ()
        if isinstance(value, str):
            value = (value,)
        if isinstance(value, (list, tuple, set, frozenset)):
            normalized = tuple(str(item).strip() for item in value)
            if any(not item for item in normalized):
                raise ValueError("filter terms must not be blank")
            return tuple(dict.fromkeys(normalized))
        return value

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = " ".join(value.strip().split())
            return normalized or None
        return value

    @model_validator(mode="after")
    def validate_dates(self) -> TrialFilter:
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date < self.start_date
        ):
            raise ValueError("end_date must not precede start_date")
        return self


class ConditionCompletionStatus(StrEnum):
    COMPLETED = "COMPLETED"
    ATTEMPTED_NO_VALID_TRIAL = "ATTEMPTED_NO_VALID_TRIAL"
    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class ConditionCoverage:
    condition_code: str
    condition_name: str
    status: ConditionCompletionStatus
    trial_count: int
    finalized_trial_count: int
    valid_trial_count: int
    repeat_indices: tuple[int, ...]
    valid_repeat_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SubjectCoverage:
    project_uuid: str
    project_code: str | None
    subject_uuid: str
    subject_code: str | None
    total_trial_count: int
    finalized_trial_count: int
    valid_trial_count: int
    completed_condition_codes: tuple[str, ...]
    missing_condition_codes: tuple[str, ...]
    attempted_without_valid_condition_codes: tuple[str, ...]
    never_attempted_condition_codes: tuple[str, ...]
    coverage_fraction: float
    conditions: tuple[ConditionCoverage, ...]


@dataclass(frozen=True, slots=True)
class PackageStatusRecord:
    path: Path
    trial_uuid: str | None
    state: PackageState
    evidence_verified: bool
    message: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetStateSummary:
    finalized_trial_uuids: tuple[str, ...]
    published_nonfinalized_trial_uuids: tuple[str, ...]
    pending_recovery: tuple[PackageStatusRecord, ...]
    aborted: tuple[PackageStatusRecord, ...]
    pending_quality_trial_uuids: tuple[str, ...]
    pending_upload_trial_uuids: tuple[str, ...]
    reviewed_trial_uuids: tuple[str, ...]
    verified_upload_trial_uuids: tuple[str, ...]
    sidecar_error_trial_uuids: tuple[str, ...]

    @property
    def finalized_count(self) -> int:
        return len(self.finalized_trial_uuids)

    @property
    def pending_recovery_count(self) -> int:
        return len(self.pending_recovery)

    @property
    def aborted_count(self) -> int:
        return len(self.aborted)

    @property
    def pending_quality_count(self) -> int:
        return len(self.pending_quality_trial_uuids)

    @property
    def pending_upload_count(self) -> int:
        return len(self.pending_upload_trial_uuids)


@dataclass(frozen=True, slots=True)
class InventoryExportResult:
    csv_path: Path
    json_path: Path
    record_count: int


@dataclass(frozen=True, slots=True)
class AnnexArtifactSummary:
    artifact_uuid: str
    role: str
    relative_path: str
    media_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ExternalAnnexSummary:
    """One published external-modality annex, including failed validation."""

    annex_directory: Path
    annex_manifest_path: Path
    validation_status: AnnexValidationStatus
    annex_uuid: str | None
    trial_uuid: str | None
    modality: str | None
    modality_label: str | None
    source_system: str | None
    imported_at_utc: datetime | None
    mapping_quality: str | None
    mapping_offset_only: bool | None
    mapping_anchor_count: int | None
    file_count: int
    total_bytes: int
    files: tuple[AnnexArtifactSummary, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnexScanResult:
    data_root: Path
    annexes: tuple[ExternalAnnexSummary, ...]
    scan_failures: tuple[tuple[str, str], ...] = ()

    def by_trial_uuid(self) -> dict[str, tuple[ExternalAnnexSummary, ...]]:
        grouped: dict[str, list[ExternalAnnexSummary]] = {}
        for annex in self.annexes:
            if annex.trial_uuid is not None:
                grouped.setdefault(annex.trial_uuid, []).append(annex)
        return {
            trial_uuid: tuple(
                sorted(
                    items,
                    key=lambda item: (
                        item.imported_at_utc or datetime.min.replace(tzinfo=timezone.utc),
                        item.annex_uuid or "",
                    ),
                )
            )
            for trial_uuid, items in sorted(grouped.items())
        }


@dataclass(frozen=True, slots=True)
class ManagementRefreshResult:
    index: ManagementIndex
    annex_scan: AnnexScanResult


@dataclass(frozen=True, slots=True)
class ManagementSummaryResult:
    index: ManagementIndex
    subject_coverage: tuple[SubjectCoverage, ...]
    dataset_states: DatasetStateSummary


def _require_idle(root: Path) -> None:
    if read_activity(root) is not None:
        raise ManagementBusyError(
            "Collector 正在采集；筛选重建、覆盖率、状态扫描和导出已暂停。"
        )


def _duration_s(manifest: TrialManifest) -> float:
    timing = manifest.timing
    if timing.stop_host_monotonic_ns is not None:
        return max(
            0.0,
            (timing.stop_host_monotonic_ns - timing.start_host_monotonic_ns) / 1e9,
        )
    if timing.stopped_at_utc is not None:
        return max(
            0.0,
            (timing.stopped_at_utc - timing.started_at_utc).total_seconds(),
        )
    return 0.0


def _catalog_manifest_paths(snapshot: DataStudioSnapshot) -> tuple[Path, ...]:
    paths: list[Path] = []
    for project in snapshot.tree:
        for subject in project.get("children", []):
            for session in subject.get("children", []):
                for trial in session.get("children", []):
                    value = trial.get("manifest_path")
                    if value:
                        paths.append(Path(str(value)).expanduser().resolve())
    return tuple(dict.fromkeys(paths))


def _quality_sidecar(
    root: Path,
    manifest_path: Path,
    manifest: TrialManifest,
) -> tuple[QualityReviewStatus, str, datetime | None, tuple[str, ...]]:
    computed = manifest.quality.computed_grade
    effective = (
        manifest.quality.reviewed_grade
        or computed
    )
    effective_value = effective.value if effective is not None else "UNASSESSED"
    if manifest.state is not TrialState.FINALIZED:
        return QualityReviewStatus.NOT_APPLICABLE, effective_value, None, ()
    review_directory = root / QUALITY_REVIEW_DIRECTORY / str(manifest.trial_uuid)
    if review_directory.is_symlink() or (
        review_directory.is_dir()
        and any(path.is_symlink() for path in review_directory.glob("*.json"))
    ):
        return (
            QualityReviewStatus.INVALID_SIDECAR,
            effective_value,
            manifest.quality.reviewed_at_utc,
            ("quality review sidecar contains a symbolic link",),
        )
    try:
        reviews = list_quality_reviews(root, manifest_path)
    except QualityReviewError as exc:
        return (
            QualityReviewStatus.INVALID_SIDECAR,
            effective_value,
            manifest.quality.reviewed_at_utc,
            (str(exc),),
        )
    if reviews:
        latest = reviews[-1].record
        return (
            QualityReviewStatus.REVIEWED,
            latest.reviewed_grade.value,
            latest.reviewed_at_utc,
            (),
        )
    if manifest.quality.reviewed_grade is not None:
        return (
            QualityReviewStatus.REVIEWED,
            manifest.quality.reviewed_grade.value,
            manifest.quality.reviewed_at_utc,
            (),
        )
    return QualityReviewStatus.PENDING, effective_value, None, ()


def _safe_audit_relative(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("upload audit relative_path must be a string")
    relative = PurePosixPath(value)
    if (
        "\\" in value
        or any(ord(character) < 32 for character in value)
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(":" in part for part in relative.parts)
    ):
        raise ValueError("upload audit contains an unsafe relative_path")
    return relative.as_posix()


def _expected_upload_files(
    manifest_path: Path,
    manifest: TrialManifest,
) -> dict[str, tuple[int, str]]:
    expected = {
        artifact.relative_path: (artifact.size_bytes, artifact.sha256)
        for artifact in manifest.artifacts
    }
    expected["manifest.json"] = (
        manifest_path.stat().st_size,
        sha256_file(manifest_path),
    )
    checksum_path = manifest_path.parent / "checksums.sha256"
    if not checksum_path.is_file():
        raise ValueError("finalized Trial has no checksums.sha256 for upload audit")
    expected["checksums.sha256"] = (
        checksum_path.stat().st_size,
        sha256_file(checksum_path),
    )
    return expected


def _validate_upload_audit(
    path: Path,
    *,
    manifest: TrialManifest,
    expected_files: dict[str, tuple[int, str]],
) -> tuple[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0.0":
        raise ValueError("unsupported upload audit schema")
    transfer_uuid = UUID(str(payload.get("transfer_batch_uuid")))
    if path.stem != str(transfer_uuid):
        raise ValueError("upload audit filename does not match transfer UUID")
    if UUID(str(payload.get("trial_uuid"))) != manifest.trial_uuid:
        raise ValueError("upload audit Trial UUID mismatch")
    status = payload.get("status")
    if status not in {"FAILED", "VERIFIED"}:
        raise ValueError("upload audit status is not terminal")
    started = payload.get("started_at_utc_ns")
    completed = payload.get("completed_at_utc_ns")
    if (
        not isinstance(started, int)
        or not isinstance(completed, int)
        or started < 0
        or completed < started
    ):
        raise ValueError("upload audit timestamps are invalid")
    remote = payload.get("remote")
    if not isinstance(remote, dict):
        raise ValueError("upload audit remote endpoint is missing")
    for field in ("host", "username", "trial_directory"):
        value = remote.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"upload audit remote {field} is invalid")
    port = remote.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("upload audit remote port is invalid")
    if remote.get("authentication_method") not in {"PASSWORD", "PRIVATE_KEY"}:
        raise ValueError("upload audit authentication method is invalid")
    error = payload.get("error")
    if status == "VERIFIED" and error is not None:
        raise ValueError("verified upload audit must not contain an error")
    if status == "FAILED":
        if not isinstance(error, dict):
            raise ValueError("failed upload audit has no structured error")
        if not isinstance(error.get("code"), str) or not error["code"].strip():
            raise ValueError("failed upload audit error code is invalid")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("upload audit files must be a list")
    actual: dict[str, tuple[int, str, str | None]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("upload audit file item is invalid")
        relative = _safe_audit_relative(item.get("relative_path"))
        if relative in actual:
            raise ValueError("upload audit contains duplicate file paths")
        size = item.get("size_bytes")
        local_sha = item.get("local_sha256")
        remote_sha = item.get("remote_sha256")
        if not isinstance(size, int) or size < 0:
            raise ValueError("upload audit file size is invalid")
        if not isinstance(local_sha, str) or not _SHA256_PATTERN.fullmatch(local_sha):
            raise ValueError("upload audit local SHA-256 is invalid")
        if remote_sha is not None and (
            not isinstance(remote_sha, str)
            or not _SHA256_PATTERN.fullmatch(remote_sha)
        ):
            raise ValueError("upload audit remote SHA-256 is invalid")
        actual[relative] = (size, local_sha, remote_sha)
    if set(actual) != set(expected_files):
        raise ValueError("upload audit does not exactly cover the finalized package")
    for relative, (expected_size, expected_sha) in expected_files.items():
        size, local_sha, remote_sha = actual[relative]
        if size != expected_size or local_sha != expected_sha:
            raise ValueError(f"upload audit local evidence mismatch: {relative}")
        if status == "VERIFIED" and remote_sha != local_sha:
            raise ValueError(f"upload audit remote evidence mismatch: {relative}")
    return str(status), completed


def _upload_sidecar(
    root: Path,
    manifest_path: Path,
    manifest: TrialManifest,
) -> tuple[UploadAuditStatus, int | None, tuple[str, ...]]:
    if manifest.state is not TrialState.FINALIZED:
        return UploadAuditStatus.NOT_APPLICABLE, None, ()
    if any(record.status.value == "VERIFIED" for record in manifest.upload_records):
        verified = [
            record.verified_at_utc
            for record in manifest.upload_records
            if record.status.value == "VERIFIED" and record.verified_at_utc is not None
        ]
        latest = max(verified) if verified else None
        return (
            UploadAuditStatus.VERIFIED,
            int(latest.timestamp() * 1e9) if latest is not None else None,
            (),
        )
    directory = root / UPLOAD_AUDIT_DIRECTORY / str(manifest.trial_uuid)
    if not directory.is_dir():
        return UploadAuditStatus.PENDING, None, ()
    if directory.is_symlink():
        return (
            UploadAuditStatus.INVALID_SIDECAR,
            None,
            ("upload audit directory is a symbolic link",),
        )
    try:
        expected = _expected_upload_files(manifest_path, manifest)
    except (OSError, ValueError) as exc:
        return UploadAuditStatus.INVALID_SIDECAR, None, (str(exc),)
    valid: list[tuple[str, int]] = []
    errors: list[str] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith(".") or name_has_storage_suffix(path.name):
            continue
        try:
            if path.is_symlink():
                raise ValueError("upload audit file is a symbolic link")
            valid.append(
                _validate_upload_audit(
                    path,
                    manifest=manifest,
                    expected_files=expected,
                )
            )
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    verified_times = [completed for status, completed in valid if status == "VERIFIED"]
    if verified_times:
        return UploadAuditStatus.VERIFIED, max(verified_times), tuple(errors)
    failed_times = [completed for status, completed in valid if status == "FAILED"]
    if failed_times:
        return UploadAuditStatus.FAILED, None, tuple(errors)
    if errors:
        return UploadAuditStatus.INVALID_SIDECAR, None, tuple(errors)
    return UploadAuditStatus.PENDING, None, ()


def _record_from_manifest(
    root: Path,
    manifest_path: Path,
    manifest: TrialManifest,
) -> TrialManagementRecord:
    try:
        relative = manifest_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ManagementError("Catalog Manifest 路径逃逸当前数据根目录") from exc
    review_status, effective_quality, reviewed_at, quality_errors = _quality_sidecar(
        root, manifest_path, manifest
    )
    upload_status, verified_ns, upload_errors = _upload_sidecar(
        root, manifest_path, manifest
    )
    computed = manifest.quality.computed_grade
    return TrialManagementRecord(
        data_root=root,
        manifest_path=manifest_path,
        manifest_relative_path=relative,
        project_uuid=str(manifest.project_uuid),
        project_code=manifest.project_code,
        project_name=manifest.project_name,
        subject_uuid=str(manifest.subject_uuid),
        subject_code=manifest.subject_code,
        session_uuid=str(manifest.session_uuid),
        trial_uuid=str(manifest.trial_uuid),
        state=manifest.state.value,
        condition_code=manifest.condition.condition_code,
        condition_name=manifest.condition.condition_name,
        repeat_index=manifest.condition.repeat_index,
        started_at_utc=manifest.timing.started_at_utc,
        duration_s=_duration_s(manifest),
        computed_quality_grade=(
            computed.value if computed is not None else "UNASSESSED"
        ),
        effective_quality_grade=effective_quality,
        quality_review_status=review_status,
        quality_reviewed_at_utc=reviewed_at,
        upload_status=upload_status,
        upload_verified_at_utc_ns=verified_ns,
        artifact_count=len(manifest.artifacts),
        artifact_total_bytes=sum(item.size_bytes for item in manifest.artifacts),
        sidecar_errors=(*quality_errors, *upload_errors),
    )


def build_management_index(snapshot: DataStudioSnapshot) -> ManagementIndex:
    """Build immutable Manifest rows from an already refreshed Catalog snapshot."""

    if not isinstance(snapshot, DataStudioSnapshot):
        raise TypeError("snapshot must be a DataStudioSnapshot")
    root = snapshot.data_root.expanduser().resolve()
    _require_idle(root)
    if snapshot.lightweight_mode:
        raise ManagementBusyError("Collector 在 Catalog 刷新期间开始采集。")
    records: list[TrialManagementRecord] = []
    failures: list[tuple[str, str]] = []
    seen_trials: set[str] = set()
    for manifest_path in _catalog_manifest_paths(snapshot):
        try:
            if path_has_unpublished_component(manifest_path):
                raise ValueError("Catalog points to an unpublished Trial package")
            manifest = load_manifest(manifest_path)
            trial_uuid = str(manifest.trial_uuid)
            if trial_uuid in seen_trials:
                raise ValueError("Catalog contains a duplicate Trial UUID")
            seen_trials.add(trial_uuid)
            records.append(_record_from_manifest(root, manifest_path, manifest))
        except Exception as exc:
            failures.append(
                (str(manifest_path), f"{type(exc).__name__}: {exc}")
            )
    _require_idle(root)
    records.sort(key=lambda item: (item.started_at_utc, item.trial_uuid))
    return ManagementIndex(
        data_root=root,
        records=tuple(records),
        catalog_scan_failures=tuple(sorted(snapshot.scan_report.failures.items())),
        manifest_failures=tuple(failures),
    )


def load_management_index(data_root: str | Path) -> ManagementIndex:
    """Refresh Catalog and build immutable Manifest-backed management rows."""

    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _require_idle(root)
    return build_management_index(load_catalog_snapshot(root))


def _matches_exact(terms: tuple[str, ...], candidates: Iterable[str | None]) -> bool:
    if not terms:
        return True
    expected = {term.casefold() for term in terms}
    values = {value.casefold() for value in candidates if value}
    return bool(expected & values)


def filter_trial_records(
    records: Iterable[TrialManagementRecord],
    criteria: TrialFilter | dict[str, Any] | None = None,
) -> tuple[TrialManagementRecord, ...]:
    """Apply every configured filter with AND semantics and stable ordering."""

    selected = (
        criteria
        if isinstance(criteria, TrialFilter)
        else TrialFilter.model_validate(criteria or {})
    )
    text = selected.text.casefold() if selected.text else None
    result: list[TrialManagementRecord] = []
    for record in records:
        if not _matches_exact(
            selected.projects,
            (record.project_uuid, record.project_code, record.project_name),
        ):
            continue
        if not _matches_exact(
            selected.subjects,
            (record.subject_uuid, record.subject_code),
        ):
            continue
        if not _matches_exact(selected.sessions, (record.session_uuid,)):
            continue
        if not _matches_exact(
            selected.conditions,
            (record.condition_code, record.condition_name),
        ):
            continue
        if not _matches_exact(
            selected.qualities,
            (record.effective_quality_grade,),
        ):
            continue
        if selected.start_date is not None and record.started_date < selected.start_date:
            continue
        if selected.end_date is not None and record.started_date > selected.end_date:
            continue
        if text is not None:
            haystack = " ".join(
                str(value)
                for value in (
                    record.project_uuid,
                    record.project_code,
                    record.project_name,
                    record.subject_uuid,
                    record.subject_code,
                    record.session_uuid,
                    record.trial_uuid,
                    record.state,
                    record.condition_code,
                    record.condition_name,
                    record.effective_quality_grade,
                    record.manifest_relative_path,
                )
                if value is not None
            ).casefold()
            if text not in haystack:
                continue
        result.append(record)
    return tuple(sorted(result, key=lambda item: (item.started_at_utc, item.trial_uuid)))


def compute_subject_coverage(
    records: Iterable[TrialManagementRecord],
    *,
    protocol: ProtocolDefinition | None = None,
    valid_quality_grades: Iterable[str] = ("A", "B"),
) -> tuple[SubjectCoverage, ...]:
    """Summarize expected-condition coverage per Subject.

    A condition is complete only when it has at least one FINALIZED Trial whose
    effective (possibly human-reviewed) quality is in ``valid_quality_grades``.
    Attempts with no valid Trial remain distinguishable from never attempted.
    """

    definition = protocol or load_default_protocol()
    expected: tuple[ConditionDefinition, ...] = tuple(definition.conditions)
    valid_grades = {str(value).strip().upper() for value in valid_quality_grades}
    if not valid_grades or any(not value for value in valid_grades):
        raise ValueError("valid_quality_grades must not be empty")
    grouped: dict[tuple[str, str], list[TrialManagementRecord]] = {}
    for record in records:
        grouped.setdefault((record.project_uuid, record.subject_uuid), []).append(record)
    summaries: list[SubjectCoverage] = []
    for (project_uuid, subject_uuid), subject_records in grouped.items():
        first = subject_records[0]
        condition_rows: list[ConditionCoverage] = []
        for condition in expected:
            attempts = [
                record
                for record in subject_records
                if record.condition_code == condition.condition_code
            ]
            finalized_attempts = [
                record
                for record in attempts
                if record.state == TrialState.FINALIZED.value
            ]
            valid = [
                record
                for record in finalized_attempts
                if record.effective_quality_grade.upper() in valid_grades
            ]
            status = (
                ConditionCompletionStatus.COMPLETED
                if valid
                else ConditionCompletionStatus.ATTEMPTED_NO_VALID_TRIAL
                if attempts
                else ConditionCompletionStatus.MISSING
            )
            condition_rows.append(
                ConditionCoverage(
                    condition_code=condition.condition_code,
                    condition_name=condition.condition_name,
                    status=status,
                    trial_count=len(attempts),
                    finalized_trial_count=len(finalized_attempts),
                    valid_trial_count=len(valid),
                    repeat_indices=tuple(sorted({item.repeat_index for item in attempts})),
                    valid_repeat_indices=tuple(
                        sorted({item.repeat_index for item in valid})
                    ),
                )
            )
        completed = tuple(
            item.condition_code
            for item in condition_rows
            if item.status is ConditionCompletionStatus.COMPLETED
        )
        missing = tuple(
            item.condition_code
            for item in condition_rows
            if item.status is not ConditionCompletionStatus.COMPLETED
        )
        attempted_without_valid = tuple(
            item.condition_code
            for item in condition_rows
            if item.status is ConditionCompletionStatus.ATTEMPTED_NO_VALID_TRIAL
        )
        never_attempted = tuple(
            item.condition_code
            for item in condition_rows
            if item.status is ConditionCompletionStatus.MISSING
        )
        finalized = [
            item for item in subject_records if item.state == TrialState.FINALIZED.value
        ]
        valid_trials = [
            item
            for item in finalized
            if item.effective_quality_grade.upper() in valid_grades
        ]
        summaries.append(
            SubjectCoverage(
                project_uuid=project_uuid,
                project_code=first.project_code,
                subject_uuid=subject_uuid,
                subject_code=first.subject_code,
                total_trial_count=len(subject_records),
                finalized_trial_count=len(finalized),
                valid_trial_count=len(valid_trials),
                completed_condition_codes=completed,
                missing_condition_codes=missing,
                attempted_without_valid_condition_codes=attempted_without_valid,
                never_attempted_condition_codes=never_attempted,
                coverage_fraction=(len(completed) / len(expected) if expected else 1.0),
                conditions=tuple(condition_rows),
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda item: (
                item.project_code or item.project_uuid,
                item.subject_code or item.subject_uuid,
            ),
        )
    )


def _directory_uuid(path: Path, suffix: str) -> str | None:
    try:
        return str(UUID(path.name[: -len(suffix)]))
    except (ValueError, AttributeError):
        return None


def _verify_aborted_package(path: Path, data_root: Path) -> PackageStatusRecord:
    trial_uuid = _directory_uuid(path, ".aborted")
    if trial_uuid is None:
        return PackageStatusRecord(
            path,
            None,
            PackageState.ABORTED_UNVERIFIED,
            False,
            "aborted directory name is not a Trial UUID",
        )
    if path.is_symlink():
        return PackageStatusRecord(
            path,
            trial_uuid,
            PackageState.ABORTED_UNVERIFIED,
            False,
            "aborted directory is a symbolic link",
        )
    audits = sorted((path / "reports").glob("recovery-abort-*.json"))
    errors: list[str] = []
    for audit in audits:
        try:
            if audit.is_symlink():
                raise ValueError("abort audit is a symbolic link")
            document = json.loads(audit.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("audit root is not an object")
            if document.get("schema_version") != "1.0.0":
                raise ValueError("unsupported abort audit schema")
            operation_uuid = str(UUID(str(document.get("operation_uuid"))))
            if audit.name != f"recovery-abort-{operation_uuid}.json":
                raise ValueError("abort audit filename does not match operation UUID")
            if document.get("action") != "ABORT_PRESERVING_DATA":
                raise ValueError("abort action mismatch")
            if document.get("state") != TrialState.ABORTED.value:
                raise ValueError("abort state mismatch")
            if document.get("confirmed") is not True:
                raise ValueError("abort decision was not explicitly confirmed")
            reason = document.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("abort reason is missing")
            decided = datetime.fromisoformat(
                str(document.get("decided_at_utc")).replace("Z", "+00:00")
            )
            if decided.tzinfo is None or decided.utcoffset() is None:
                raise ValueError("abort decision timestamp is not timezone-aware")
            if str(UUID(str(document.get("trial_uuid")))) != trial_uuid:
                raise ValueError("abort Trial UUID mismatch")
            destination = Path(str(document.get("destination_directory"))).resolve()
            if destination != path.resolve():
                raise ValueError("abort destination mismatch")
            evidence = document.get("original_evidence")
            if not isinstance(evidence, list):
                raise ValueError("abort evidence is missing")
            relative_paths: set[str] = set()
            evidence_contract: dict[str, tuple[int, str]] = {}
            for item in evidence:
                if not isinstance(item, dict):
                    raise ValueError("abort evidence item is invalid")
                relative = _safe_audit_relative(item.get("relative_path"))
                digest = item.get("sha256")
                size = item.get("size_bytes")
                if relative in relative_paths:
                    raise ValueError("abort evidence paths are duplicated")
                if not isinstance(size, int) or size < 0:
                    raise ValueError("abort evidence size is invalid")
                if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
                    raise ValueError("abort evidence SHA-256 is invalid")
                relative_paths.add(relative)
                evidence_contract[relative] = (size, digest)
            descendants = tuple(path.rglob("*"))
            if any(candidate.is_symlink() for candidate in descendants):
                raise ValueError("aborted package contains a symbolic link")
            actual_files = {
                candidate.relative_to(path).as_posix()
                for candidate in descendants
                if candidate.is_file()
                and not (
                    candidate.parent == path / "reports"
                    and candidate.name.startswith("recovery-abort-")
                    and candidate.suffix == ".json"
                )
            }
            if actual_files != set(evidence_contract):
                raise ValueError("abort evidence does not exactly cover retained files")
            for relative, (size, digest) in evidence_contract.items():
                candidate = _package_regular_file(
                    path,
                    relative,
                    forbid_temporary=False,
                )
                if candidate.stat().st_size != size:
                    raise ValueError(f"abort evidence size mismatch: {relative}")
                if _sha256_file_idle(candidate, data_root) != digest:
                    raise ValueError(f"abort evidence SHA-256 mismatch: {relative}")
            return PackageStatusRecord(
                path, trial_uuid, PackageState.ABORTED, True, None
            )
        except Exception as exc:
            errors.append(f"{audit.name}: {type(exc).__name__}: {exc}")
    return PackageStatusRecord(
        path,
        trial_uuid,
        PackageState.ABORTED_UNVERIFIED,
        False,
        "; ".join(errors) if errors else "abort audit sidecar is missing",
    )


def summarize_dataset_states(
    data_root: str | Path,
    records: Iterable[TrialManagementRecord] | None = None,
) -> DatasetStateSummary:
    root = Path(data_root).expanduser().resolve()
    _require_idle(root)
    rows = tuple(records) if records is not None else load_management_index(root).records
    if any(item.data_root.resolve() != root for item in rows):
        raise ManagementError("状态汇总记录来自不同的数据根目录")
    pending_recovery = tuple(
        PackageStatusRecord(
            path=path,
            trial_uuid=_directory_uuid(path, ".recording"),
            state=PackageState.PENDING_RECOVERY,
            evidence_verified=False,
            message="package requires recovery inspection",
        )
        for path in iter_recording_directories(root)
    )
    aborted = tuple(
        _verify_aborted_package(path, root) for path in iter_aborted_directories(root)
    )
    finalized = tuple(
        sorted(item.trial_uuid for item in rows if item.state == TrialState.FINALIZED.value)
    )
    published_nonfinalized = tuple(
        sorted(item.trial_uuid for item in rows if item.state != TrialState.FINALIZED.value)
    )
    pending_quality = tuple(
        sorted(item.trial_uuid for item in rows if item.pending_quality_review)
    )
    pending_upload = tuple(sorted(item.trial_uuid for item in rows if item.pending_upload))
    reviewed = tuple(
        sorted(
            item.trial_uuid
            for item in rows
            if item.quality_review_status is QualityReviewStatus.REVIEWED
        )
    )
    uploaded = tuple(
        sorted(
            item.trial_uuid
            for item in rows
            if item.upload_status is UploadAuditStatus.VERIFIED
        )
    )
    sidecar_errors = tuple(sorted(item.trial_uuid for item in rows if item.sidecar_errors))
    _require_idle(root)
    return DatasetStateSummary(
        finalized_trial_uuids=finalized,
        published_nonfinalized_trial_uuids=published_nonfinalized,
        pending_recovery=pending_recovery,
        aborted=aborted,
        pending_quality_trial_uuids=pending_quality,
        pending_upload_trial_uuids=pending_upload,
        reviewed_trial_uuids=reviewed,
        verified_upload_trial_uuids=uploaded,
        sidecar_error_trial_uuids=sidecar_errors,
    )


def _temporary_annex_component(path: Path) -> bool:
    return path_has_unpublished_component(path)


def _uuid_text(value: str) -> str | None:
    try:
        return str(UUID(value))
    except (ValueError, AttributeError):
        return None


def _annex_file_summary(
    manifest: ExternalAnnexManifest | None,
) -> tuple[AnnexArtifactSummary, ...]:
    if manifest is None:
        return ()
    return tuple(
        AnnexArtifactSummary(
            artifact_uuid=str(item.artifact_uuid),
            role=item.role,
            relative_path=item.relative_path,
            media_type=item.media_type,
            size_bytes=item.size_bytes,
            sha256=item.sha256,
        )
        for item in manifest.files
    )


def _annex_summary(
    directory: Path,
    *,
    manifest: ExternalAnnexManifest | None,
    status: AnnexValidationStatus,
    errors: tuple[str, ...] = (),
) -> ExternalAnnexSummary:
    files = _annex_file_summary(manifest)
    inferred_trial = _uuid_text(directory.parent.name)
    inferred_annex = _uuid_text(directory.name)
    return ExternalAnnexSummary(
        annex_directory=directory,
        annex_manifest_path=directory / "annex_manifest.json",
        validation_status=status,
        annex_uuid=inferred_annex or (
            str(manifest.annex_uuid) if manifest is not None else None
        ),
        trial_uuid=inferred_trial or (
            str(manifest.trial_uuid) if manifest is not None else None
        ),
        modality=manifest.modality.value if manifest is not None else None,
        modality_label=(
            manifest.other_modality_label or manifest.modality.value
            if manifest is not None
            else None
        ),
        source_system=manifest.source_system if manifest is not None else None,
        imported_at_utc=manifest.imported_at_utc if manifest is not None else None,
        mapping_quality=manifest.mapping.quality if manifest is not None else None,
        mapping_offset_only=(
            manifest.mapping.offset_only if manifest is not None else None
        ),
        mapping_anchor_count=(
            manifest.mapping.anchor_count if manifest is not None else None
        ),
        file_count=len(files),
        total_bytes=sum(item.size_bytes for item in files),
        files=files,
        errors=errors,
    )


def _sha256_file_idle(path: Path, data_root: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            _require_idle(data_root)
            digest.update(chunk)
    return digest.hexdigest()


def _package_regular_file(
    directory: Path,
    relative_path: str,
    *,
    forbid_temporary: bool,
) -> Path:
    relative = _safe_audit_relative(relative_path)
    if forbid_temporary and _temporary_annex_component(Path(relative)):
        raise ValueError("annex references a building/partial path")
    lexical = directory.joinpath(*PurePosixPath(relative).parts)
    if lexical.is_symlink():
        raise ValueError(f"annex file is a symbolic link: {relative}")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(directory.resolve())
    except ValueError:
        raise ValueError(f"annex file escapes its package: {relative}") from None
    if not resolved.is_file():
        raise ValueError(f"annex file is missing: {relative}")
    return resolved


def _annex_regular_file(directory: Path, relative_path: str) -> Path:
    return _package_regular_file(
        directory,
        relative_path,
        forbid_temporary=True,
    )


def _annex_checksum_contract(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("annex checksums.sha256 is missing or symbolic")
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, relative_value = line.partition("  ")
        if not separator or not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError("annex checksum line is invalid")
        relative = _safe_audit_relative(relative_value)
        if relative in entries:
            raise ValueError("annex checksum paths are duplicated")
        entries[relative] = digest
    if not entries:
        raise ValueError("annex checksum manifest is empty")
    return entries


def _validate_annex_directory(
    data_root: Path,
    directory: Path,
) -> ExternalAnnexSummary:
    manifest: ExternalAnnexManifest | None = None
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("annex package is not a regular directory")
        if _temporary_annex_component(directory.relative_to(data_root)):
            raise ValueError("temporary annex packages must not be scanned")
        expected_annex_uuid = _uuid_text(directory.name)
        expected_trial_uuid = _uuid_text(directory.parent.name)
        if expected_annex_uuid is None or expected_trial_uuid is None:
            raise ValueError("annex hierarchy does not use Trial/Annex UUIDs")
        manifest_path = directory / "annex_manifest.json"
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or manifest_path.stat().st_size > 2 * 1024 * 1024
        ):
            raise ValueError("annex_manifest.json is missing, symbolic, or oversized")
        manifest = ExternalAnnexManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if str(manifest.annex_uuid) != expected_annex_uuid:
            raise ValueError("annex UUID does not match its directory")
        if str(manifest.trial_uuid) != expected_trial_uuid:
            raise ValueError("annex Trial UUID does not match its parent directory")

        base_relative = _safe_audit_relative(manifest.base_manifest_relative_path)
        if _temporary_annex_component(Path(base_relative)):
            raise ValueError("annex binds to a temporary Trial Manifest")
        base_lexical = data_root.joinpath(*PurePosixPath(base_relative).parts)
        if base_lexical.is_symlink():
            raise ValueError("base Trial Manifest is a symbolic link")
        base_path = base_lexical.resolve(strict=True)
        try:
            base_path.relative_to(data_root)
        except ValueError:
            raise ValueError("base Trial Manifest escapes the dataset") from None
        if base_path.name != "manifest.json" or not base_path.is_file():
            raise ValueError("annex base Manifest path is invalid")
        base = load_manifest(base_path)
        if base.state is not TrialState.FINALIZED:
            raise ValueError("annex base Trial is not FINALIZED")
        if base.trial_uuid != manifest.trial_uuid:
            raise ValueError("annex and base Manifest Trial UUIDs differ")
        if base.manifest_uuid != manifest.base_manifest_uuid:
            raise ValueError("annex base Manifest UUID mismatch")
        if base.schema_version != manifest.base_manifest_schema_version:
            raise ValueError("annex base Manifest schema mismatch")
        if _sha256_file_idle(base_path, data_root) != manifest.base_manifest_sha256:
            raise ValueError("annex base Manifest SHA-256 mismatch")

        described: dict[str, tuple[int, str]] = {
            item.relative_path: (item.size_bytes, item.sha256)
            for item in manifest.files
        }
        described[manifest.mapping.relative_path] = (
            manifest.mapping.size_bytes,
            manifest.mapping.sha256,
        )
        described["annex_manifest.json"] = (
            manifest_path.stat().st_size,
            _sha256_file_idle(manifest_path, data_root),
        )
        if len(described) != len(manifest.files) + 2:
            raise ValueError("annex has duplicate described paths")
        checksums = _annex_checksum_contract(directory / manifest.checksum_manifest)
        if set(checksums) != set(described):
            raise ValueError("annex checksums do not exactly cover described files")

        actual_files: set[str] = set()
        for candidate in directory.rglob("*"):
            relative = candidate.relative_to(directory)
            if _temporary_annex_component(relative):
                raise ValueError("published annex contains building/partial content")
            if candidate.is_symlink():
                raise ValueError("published annex contains a symbolic link")
            if candidate.is_file():
                actual_files.add(relative.as_posix())
        if actual_files != set(described) | {manifest.checksum_manifest}:
            raise ValueError("annex contains files outside its checksum contract")

        for relative, (size, expected_sha) in described.items():
            artifact = _annex_regular_file(directory, relative)
            if artifact.stat().st_size != size:
                raise ValueError(f"annex file size mismatch: {relative}")
            actual_sha = _sha256_file_idle(artifact, data_root)
            if actual_sha != expected_sha or checksums[relative] != expected_sha:
                raise ValueError(f"annex file SHA-256 mismatch: {relative}")
        return _annex_summary(
            directory,
            manifest=manifest,
            status=AnnexValidationStatus.VERIFIED,
        )
    except Exception as exc:
        return _annex_summary(
            directory,
            manifest=manifest,
            status=AnnexValidationStatus.INVALID,
            errors=(f"{type(exc).__name__}: {exc}",),
        )


def scan_external_annexes(
    data_root: str | Path,
    *,
    trial_uuid: str | UUID | None = None,
) -> AnnexScanResult:
    """Discover and fully verify published annexes without touching Trial raw data.

    Private ``.building`` and ``.partial`` paths are deliberately invisible to
    consumers.  Hashing stops promptly if Collector activity starts.
    """

    root = Path(data_root).expanduser().resolve()
    _require_idle(root)
    selected_trial = str(UUID(str(trial_uuid))) if trial_uuid is not None else None
    annex_root = root / ANNEX_DIRECTORY_NAME
    if not annex_root.exists():
        return AnnexScanResult(root, ())
    if annex_root.is_symlink() or not annex_root.is_dir():
        raise ManagementError("external_annexes 必须是数据根目录内的普通目录")
    annex_root = annex_root.resolve()
    try:
        annex_root.relative_to(root)
    except ValueError:
        raise ManagementError("external_annexes 路径逃逸数据根目录") from None

    summaries: list[ExternalAnnexSummary] = []
    failures: list[tuple[str, str]] = []
    for trial_directory in sorted(annex_root.iterdir(), key=lambda item: item.name):
        if _temporary_annex_component(trial_directory.relative_to(annex_root)):
            continue
        if selected_trial is not None and trial_directory.name != selected_trial:
            continue
        if trial_directory.is_symlink() or not trial_directory.is_dir():
            failures.append(
                (str(trial_directory), "Trial annex parent is not a regular directory")
            )
            continue
        for annex_directory in sorted(trial_directory.iterdir(), key=lambda item: item.name):
            if _temporary_annex_component(annex_directory.relative_to(annex_root)):
                continue
            if annex_directory.is_symlink() or not annex_directory.is_dir():
                failures.append((str(annex_directory), "annex entry is not a regular directory"))
                continue
            summaries.append(_validate_annex_directory(root, annex_directory))
    _require_idle(root)
    summaries.sort(
        key=lambda item: (
            item.trial_uuid or "",
            item.imported_at_utc or datetime.min.replace(tzinfo=timezone.utc),
            item.annex_uuid or "",
        )
    )
    return AnnexScanResult(root, tuple(summaries), tuple(failures))


def load_management_refresh(snapshot: DataStudioSnapshot) -> ManagementRefreshResult:
    """Build filter rows and verify annexes in one spawned-worker payload."""

    index = build_management_index(snapshot)
    annex_scan = scan_external_annexes(index.data_root)
    return ManagementRefreshResult(index=index, annex_scan=annex_scan)


def load_management_summary(data_root: str | Path) -> ManagementSummaryResult:
    """Build a current coverage/state summary for the management dialog."""

    index = load_management_index(data_root)
    return ManagementSummaryResult(
        index=index,
        subject_coverage=compute_subject_coverage(index.records),
        dataset_states=summarize_dataset_states(index.data_root, index.records),
    )


_EXPORT_FIELDS = (
    "project_uuid",
    "project_code",
    "project_name",
    "subject_uuid",
    "subject_code",
    "session_uuid",
    "trial_uuid",
    "manifest_path",
    "manifest_relative_path",
    "state",
    "computed_quality_grade",
    "effective_quality_grade",
    "condition_code",
    "condition_name",
    "repeat_index",
    "started_at_utc",
    "date_utc",
    "duration_s",
    "artifact_count",
    "artifact_total_bytes",
    "quality_review_status",
    "upload_status",
)


def _export_row(record: TrialManagementRecord) -> dict[str, Any]:
    return {
        "project_uuid": record.project_uuid,
        "project_code": record.project_code,
        "project_name": record.project_name,
        "subject_uuid": record.subject_uuid,
        "subject_code": record.subject_code,
        "session_uuid": record.session_uuid,
        "trial_uuid": record.trial_uuid,
        "manifest_path": str(record.manifest_path),
        "manifest_relative_path": record.manifest_relative_path,
        "state": record.state,
        "computed_quality_grade": record.computed_quality_grade,
        "effective_quality_grade": record.effective_quality_grade,
        "condition_code": record.condition_code,
        "condition_name": record.condition_name,
        "repeat_index": record.repeat_index,
        "started_at_utc": record.started_at_utc.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "date_utc": record.started_date.isoformat(),
        "duration_s": record.duration_s,
        "artifact_count": record.artifact_count,
        "artifact_total_bytes": record.artifact_total_bytes,
        "quality_review_status": record.quality_review_status.value,
        "upload_status": record.upload_status.value,
    }


def _reject_trial_package_destination(
    path: Path,
    records: Sequence[TrialManagementRecord],
) -> None:
    if path_has_unpublished_component(path):
        raise ManagementError("导出目标不能位于 recording/partial/aborted 包中")
    for record in records:
        try:
            path.relative_to(record.manifest_path.parent)
        except ValueError:
            continue
        raise ManagementError("导出目标不能写入不可变 Trial 包")


def _write_temporary(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _publish_temporary(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        os.replace(temporary, destination)
        return
    # A same-directory hard link creates the destination atomically and fails
    # when it already exists on both Windows and POSIX.  Unlinking the private
    # temporary name afterwards leaves the published bytes untouched.
    os.link(temporary, destination)
    temporary.unlink()


def export_manifest_inventory(
    records: Iterable[TrialManagementRecord],
    destination_stem: str | Path,
    *,
    overwrite: bool = False,
) -> InventoryExportResult:
    """Atomically export selected/filtered rows as sibling CSV and JSON files."""

    rows = tuple(sorted(records, key=lambda item: (item.started_at_utc, item.trial_uuid)))
    stem = Path(destination_stem).expanduser().resolve()
    if stem.suffix.casefold() in {".csv", ".json"}:
        stem = stem.with_suffix("")
    csv_path = stem.with_suffix(".csv")
    json_path = stem.with_suffix(".json")
    for path in (csv_path, json_path):
        _reject_trial_package_destination(path, rows)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    exported = [_export_row(record) for record in rows]
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_stream,
        fieldnames=_EXPORT_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(exported)
    csv_bytes = csv_stream.getvalue().encode("utf-8-sig")
    json_bytes = (
        json.dumps(
            {
                "schema_version": "1.0.0",
                "record_count": len(exported),
                "records": exported,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary_csv: Path | None = None
    temporary_json: Path | None = None
    try:
        temporary_csv = _write_temporary(csv_path, csv_bytes)
        temporary_json = _write_temporary(json_path, json_bytes)
        _publish_temporary(temporary_csv, csv_path, overwrite=overwrite)
        temporary_csv = None
        _publish_temporary(temporary_json, json_path, overwrite=overwrite)
        temporary_json = None
    finally:
        if temporary_csv is not None:
            temporary_csv.unlink(missing_ok=True)
        if temporary_json is not None:
            temporary_json.unlink(missing_ok=True)
    return InventoryExportResult(csv_path, json_path, len(exported))


def export_manifest_inventory_checked(
    data_root: str | Path,
    records: Iterable[TrialManagementRecord],
    destination_stem: str | Path,
    *,
    overwrite: bool = False,
) -> InventoryExportResult:
    """Worker entry that rechecks Collector activity around inventory export."""

    root = Path(data_root).expanduser().resolve()
    rows = tuple(records)
    if any(item.data_root.resolve() != root for item in rows):
        raise ManagementError("导出记录来自不同的数据根目录")
    _require_idle(root)
    stem = Path(destination_stem).expanduser().resolve()
    if stem.suffix.casefold() in {".csv", ".json"}:
        stem = stem.with_suffix("")
    targets = (stem.with_suffix(".csv"), stem.with_suffix(".json"))
    immutable_roots = tuple(
        manifest_path.parent.resolve()
        for manifest_path in iter_finalized_manifest_paths(root)
    )
    for target in targets:
        for trial_root in immutable_roots:
            try:
                target.relative_to(trial_root)
            except ValueError:
                continue
            raise ManagementError("导出目标不能写入任何不可变 Trial 包")
        annex_root = (root / ANNEX_DIRECTORY_NAME).resolve()
        try:
            annex_relative = target.relative_to(annex_root)
        except ValueError:
            pass
        else:
            if len(annex_relative.parts) >= 3:
                raise ManagementError("导出目标不能写入不可变 external annex 包")
    return export_manifest_inventory(rows, stem, overwrite=overwrite)


__all__ = [
    "AnnexArtifactSummary",
    "AnnexScanResult",
    "AnnexValidationStatus",
    "ConditionCompletionStatus",
    "ConditionCoverage",
    "DatasetStateSummary",
    "ExternalAnnexSummary",
    "InventoryExportResult",
    "ManagementBusyError",
    "ManagementError",
    "ManagementIndex",
    "ManagementRefreshResult",
    "ManagementSummaryResult",
    "PackageState",
    "PackageStatusRecord",
    "QualityReviewStatus",
    "SubjectCoverage",
    "TrialFilter",
    "TrialManagementRecord",
    "UploadAuditStatus",
    "build_management_index",
    "compute_subject_coverage",
    "export_manifest_inventory",
    "export_manifest_inventory_checked",
    "filter_trial_records",
    "load_management_index",
    "load_management_refresh",
    "load_management_summary",
    "scan_external_annexes",
    "summarize_dataset_states",
]
