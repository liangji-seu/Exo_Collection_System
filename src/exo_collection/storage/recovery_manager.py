"""Auditable recovery workflow for interrupted Trial packages.

Discovery and inspection are read-only.  Every mutating decision first owns
the same dataset-root lease used by Collector, so recovery can never repair or
rename files that an active writer may still have open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import json
import os
from pathlib import Path
import stat
import string
from typing import Any
from uuid import UUID, uuid4

import h5py

from exo_collection.domain.states import TrialState

from .activity import AcquisitionLock, read_activity
from .checksum import sha256_file
from .layout import (
    iter_recording_directories,
    name_has_storage_suffix,
    safe_relative_path,
)
from .manifest import TrialManifest, load_manifest
from .recovery import UltrasoundRecoveryResult, recover_ultrasound_file, scan_ultrasound_file


class RecoveryAction(StrEnum):
    """Operator decisions that the current evidence permits."""

    REPAIR_SAFE_TAIL = "REPAIR_SAFE_TAIL"
    FINALIZE_PREPARED = "FINALIZE_PREPARED"
    ABORT_PRESERVING_DATA = "ABORT_PRESERVING_DATA"


class UnsafeRecoveryDecisionError(RuntimeError):
    """Raised when available evidence cannot prove a requested decision safe."""


class RecoveryConfirmationRequiredError(ValueError):
    """Raised when a destructive-looking lifecycle decision lacks confirmation."""


@dataclass(frozen=True, slots=True)
class Hdf5RecoveryStatus:
    path: Path
    readable: bool
    closed_cleanly: bool
    sample_count: int | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedPublicationStatus:
    """Proof that a crash happened after package preparation but before rename."""

    manifest_path: Path
    checksum_path: Path
    manifest_state: TrialState | None
    manifest_trial_uuid: UUID | None
    checksum_results: tuple[tuple[str, bool], ...]
    artifact_integrity_verified: bool
    complete_file_coverage: bool
    ready: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrialRecoveryReport:
    recording_directory: Path
    trial_uuid: UUID | None
    state: TrialState
    ultrasound: UltrasoundRecoveryResult | None
    hdf5_files: tuple[Hdf5RecoveryStatus, ...]
    partial_files: tuple[Path, ...]
    recoverable: bool
    inspected_at_utc: datetime
    repair_log_path: Path | None = None
    active_collection: bool = False
    active_trial_uuid: str | None = None
    prepared_publication: PreparedPublicationStatus | None = None
    allowed_actions: tuple[RecoveryAction, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def can_finalize(self) -> bool:
        return RecoveryAction.FINALIZE_PREPARED in self.allowed_actions

    @property
    def can_repair(self) -> bool:
        return RecoveryAction.REPAIR_SAFE_TAIL in self.allowed_actions

    @property
    def can_abort(self) -> bool:
        return RecoveryAction.ABORT_PRESERVING_DATA in self.allowed_actions


@dataclass(frozen=True, slots=True)
class RecoveryDecisionResult:
    action: RecoveryAction
    trial_uuid: UUID | None
    source_directory: Path
    destination_directory: Path
    audit_path: Path
    decided_at_utc: datetime


def _trial_uuid(path: Path) -> UUID | None:
    if not name_has_storage_suffix(path.name, (".recording",)):
        return None
    name = path.name[: -len(".recording")]
    try:
        return UUID(name)
    except ValueError:
        return None


def _validate_recording_directory(recording_directory: str | Path) -> tuple[Path, Path]:
    directory = Path(recording_directory).expanduser().resolve()
    if not directory.is_dir() or not name_has_storage_suffix(
        directory.name, (".recording",)
    ):
        raise ValueError("expected an existing .recording Trial directory")
    if directory.parent.name != "trials" or len(directory.parents) < 5:
        raise ValueError("recording directory is outside the canonical dataset layout")
    # Canonical layout is root/project-or-partition/subject/session/trials/Trial.
    dataset_root = directory.parents[4]
    return directory, dataset_root


def _inspect_hdf5(path: Path) -> Hdf5RecoveryStatus:
    try:
        with h5py.File(path, "r") as file:
            data_count = int(file["samples/data"].shape[0])
            index_count = int(file["samples/sample_index"].shape[0])
            device_count = int(file["samples/device_time"].shape[0])
            host_count = int(file["samples/host_monotonic_ns"].shape[0])
            optional_counts = [
                int(file[f"samples/{name}"].shape[0])
                for name in ("host_utc_ns", "source_sequence")
                if f"samples/{name}" in file
            ]
            if len({data_count, index_count, device_count, host_count, *optional_counts}) != 1:
                raise ValueError("sample datasets have inconsistent lengths")
            clean = bool(file.attrs.get("closed_cleanly", False))
            declared = int(file.attrs.get("sample_count", data_count))
            if clean and declared != data_count:
                raise ValueError("declared sample_count differs from dataset length")
        return Hdf5RecoveryStatus(path, True, clean, data_count)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return Hdf5RecoveryStatus(path, False, False, None, f"{type(exc).__name__}: {exc}")


def _path_inside_trial(root: Path, relative: str) -> Path:
    safe = safe_relative_path(relative)
    candidate = root.joinpath(*safe.parts)
    current = root
    for component in safe.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"symbolic links are forbidden in a Trial package: {relative}")
    resolved = candidate.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"path escapes Trial directory: {relative}")
    return candidate


def _assert_owned_regular_file(path: Path, package_root: Path) -> None:
    """Reject link/reparse aliases before recovery mutates or publishes them.

    A symbolic link, Windows junction/reparse point, or hard link can make an
    apparently local recovery file alias bytes outside the Trial package.
    Recovery is allowed to truncate an incomplete ultrasound tail, so this
    ownership proof is deliberately stricter than ordinary read-only scans.
    """

    root = package_root.resolve()
    try:
        relative = path.relative_to(package_root)
    except ValueError as exc:
        raise UnsafeRecoveryDecisionError(
            f"recovery file is outside the Trial package: {path}"
        ) from exc

    current = package_root
    for component in relative.parts:
        current = current / component
        try:
            information = current.lstat()
        except OSError as exc:
            raise UnsafeRecoveryDecisionError(
                f"recovery file cannot be inspected safely: {current}"
            ) from exc
        file_attributes = int(getattr(information, "st_file_attributes", 0))
        if stat.S_ISLNK(information.st_mode) or file_attributes & 0x400:
            raise UnsafeRecoveryDecisionError(
                f"links/reparse points are forbidden in a recoverable package: {current}"
            )

    try:
        resolved = path.resolve(strict=True)
        information = path.stat()
    except OSError as exc:
        raise UnsafeRecoveryDecisionError(
            f"recovery file cannot be inspected safely: {path}"
        ) from exc
    if resolved != root and root not in resolved.parents:
        raise UnsafeRecoveryDecisionError(
            f"recovery file escapes the Trial package: {path}"
        )
    if not stat.S_ISREG(information.st_mode):
        raise UnsafeRecoveryDecisionError(
            f"recovery target is not a regular file: {path}"
        )
    if information.st_nlink != 1:
        raise UnsafeRecoveryDecisionError(
            f"hard-linked files are forbidden in a recoverable package: {path}"
        )


def _read_checksum_contract(path: Path, root: Path) -> dict[str, tuple[str, bool]]:
    entries: dict[str, tuple[str, bool]] = {}
    hexdigits = set(string.hexdigits)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, separator, relative = line.partition("  ")
        if (
            not separator
            or len(expected) != 64
            or any(character not in hexdigits for character in expected)
        ):
            raise ValueError(f"invalid checksum line: {line!r}")
        normalized = safe_relative_path(relative).as_posix()
        if normalized in entries:
            raise ValueError(f"duplicate checksum path: {normalized}")
        candidate = _path_inside_trial(root, normalized)
        verified = candidate.is_file() and sha256_file(candidate) == expected.lower()
        entries[normalized] = (expected.lower(), verified)
    return entries


def _inspect_prepared_publication(
    directory: Path,
    trial_uuid: UUID | None,
    partial_files: tuple[Path, ...],
    ultrasound: UltrasoundRecoveryResult | None,
    hdf5_files: tuple[Hdf5RecoveryStatus, ...],
) -> PreparedPublicationStatus:
    manifest_path = directory / "manifest.json"
    checksum_path = directory / "checksums.sha256"
    reasons: list[str] = []
    manifest: TrialManifest | None = None
    checksum_results: tuple[tuple[str, bool], ...] = ()
    artifact_integrity = False
    complete_coverage = False

    if partial_files:
        reasons.append("package still contains .partial files")
    if not manifest_path.is_file():
        reasons.append("manifest.json is absent")
    else:
        try:
            manifest = load_manifest(manifest_path)
        except Exception as exc:
            reasons.append(f"Manifest validation failed: {type(exc).__name__}: {exc}")
    if not checksum_path.is_file():
        reasons.append("checksums.sha256 is absent")

    if manifest is not None:
        if manifest.state is not TrialState.FINALIZED:
            reasons.append("prepared Manifest state is not FINALIZED")
        if trial_uuid is None or manifest.trial_uuid != trial_uuid:
            reasons.append("Manifest Trial UUID does not match directory name")

        artifact_failures: list[str] = []
        for artifact in manifest.artifacts:
            try:
                candidate = _path_inside_trial(directory, artifact.relative_path)
                if not candidate.is_file():
                    artifact_failures.append(f"missing Artifact: {artifact.relative_path}")
                elif candidate.stat().st_size != artifact.size_bytes:
                    artifact_failures.append(f"Artifact size mismatch: {artifact.relative_path}")
                elif sha256_file(candidate) != artifact.sha256:
                    artifact_failures.append(f"Artifact SHA-256 mismatch: {artifact.relative_path}")
            except (OSError, ValueError) as exc:
                artifact_failures.append(f"invalid Artifact path {artifact.relative_path}: {exc}")
        artifact_integrity = not artifact_failures
        reasons.extend(artifact_failures)

    if manifest is not None and checksum_path.is_file():
        try:
            package_files = [path for path in directory.rglob("*") if path.is_file()]
            for package_file in package_files:
                _assert_owned_regular_file(package_file, directory)
            entries = _read_checksum_contract(checksum_path, directory)
            checksum_results = tuple(
                sorted((relative, verified) for relative, (_digest, verified) in entries.items())
            )
            expected_paths = {"manifest.json"} | {
                artifact.relative_path for artifact in manifest.artifacts
            }
            actual_paths = {
                path.relative_to(directory).as_posix()
                for path in directory.rglob("*")
                if path.is_file() and path != checksum_path
            }
            complete_coverage = set(entries) == expected_paths and actual_paths == expected_paths
            if set(entries) != expected_paths:
                reasons.append("checksum entries do not exactly cover Manifest Artifacts")
            if actual_paths != expected_paths:
                reasons.append("package contains files not covered by Manifest/checksums")
            if not entries or not all(verified for _digest, verified in entries.values()):
                reasons.append("one or more checksum entries failed verification")
        except (
            OSError,
            UnicodeError,
            UnsafeRecoveryDecisionError,
            ValueError,
        ) as exc:
            reasons.append(f"checksum validation failed: {type(exc).__name__}: {exc}")

    if ultrasound is not None and not ultrasound.is_clean:
        reasons.append("ultrasound binary is not structurally complete")
    if any(not item.readable or not item.closed_cleanly for item in hdf5_files):
        reasons.append("one or more HDF5 files are unreadable or not cleanly closed")

    ready = (
        manifest is not None
        and manifest.state is TrialState.FINALIZED
        and trial_uuid is not None
        and manifest.trial_uuid == trial_uuid
        and not partial_files
        and artifact_integrity
        and complete_coverage
        and bool(checksum_results)
        and all(verified for _relative, verified in checksum_results)
        and (ultrasound is None or ultrasound.is_clean)
        and all(item.readable and item.closed_cleanly for item in hdf5_files)
        and not reasons
    )
    return PreparedPublicationStatus(
        manifest_path=manifest_path,
        checksum_path=checksum_path,
        manifest_state=manifest.state if manifest is not None else None,
        manifest_trial_uuid=manifest.trial_uuid if manifest is not None else None,
        checksum_results=checksum_results,
        artifact_integrity_verified=artifact_integrity,
        complete_file_coverage=complete_coverage,
        ready=ready,
        reasons=tuple(reasons),
    )


def _inspect_recording_directory(directory: Path, dataset_root: Path, *, guard_activity: bool) -> TrialRecoveryReport:
    trial_uuid = _trial_uuid(directory)
    inspected_at = datetime.now(timezone.utc)
    activity = read_activity(dataset_root) if guard_activity else None
    if activity is not None:
        return TrialRecoveryReport(
            recording_directory=directory,
            trial_uuid=trial_uuid,
            state=TrialState.RECOVERABLE,
            ultrasound=None,
            hdf5_files=(),
            partial_files=(),
            recoverable=False,
            inspected_at_utc=inspected_at,
            active_collection=True,
            active_trial_uuid=activity.trial_uuid,
            issues=(
                "Collector owns the dataset root; large .recording/.partial files were not opened.",
            ),
        )

    raw_directory = directory / "raw"
    raw_files = (
        tuple(path for path in raw_directory.iterdir() if path.is_file())
        if raw_directory.is_dir()
        else ()
    )
    ultrasound_path = next(
        (
            path
            for expected in ("ultrasound.bin.partial", "ultrasound.bin")
            for path in raw_files
            if path.name.casefold() == expected
        ),
        None,
    )
    ultrasound = scan_ultrasound_file(ultrasound_path) if ultrasound_path is not None else None
    hdf5_paths = sorted(
        path
        for path in raw_files
        if path.name.casefold().endswith((".h5.partial", ".h5"))
    )
    hdf5_statuses = tuple(_inspect_hdf5(path) for path in hdf5_paths)
    partials = tuple(
        sorted(
            path
            for path in directory.rglob("*")
            if name_has_storage_suffix(path.name)
        )
    )
    prepared = _inspect_prepared_publication(
        directory, trial_uuid, partials, ultrasound, hdf5_statuses
    )
    has_valid_source = (
        ultrasound is not None and ultrasound.complete_block_count > 0
    ) or any(status.readable and (status.sample_count or 0) > 0 for status in hdf5_statuses)
    no_unsafe_ultrasound_corruption = ultrasound is None or not ultrasound.intermediate_corruption
    recoverable = prepared.ready or (has_valid_source and no_unsafe_ultrasound_corruption)
    actions: list[RecoveryAction] = [RecoveryAction.ABORT_PRESERVING_DATA]
    if (
        ultrasound is not None
        and ultrasound.tail_recoverable
        and not ultrasound.intermediate_corruption
        and not prepared.ready
    ):
        actions.insert(0, RecoveryAction.REPAIR_SAFE_TAIL)
    if prepared.ready:
        actions.insert(0, RecoveryAction.FINALIZE_PREPARED)
    issues = list(prepared.reasons)
    if ultrasound is not None and ultrasound.intermediate_corruption:
        issues.append("ultrasound has possible middle-file corruption; automatic truncation is forbidden")
    return TrialRecoveryReport(
        recording_directory=directory,
        trial_uuid=trial_uuid,
        state=TrialState.RECOVERABLE,
        ultrasound=ultrasound,
        hdf5_files=hdf5_statuses,
        partial_files=partials,
        recoverable=recoverable,
        inspected_at_utc=inspected_at,
        prepared_publication=prepared,
        allowed_actions=tuple(actions),
        issues=tuple(dict.fromkeys(issues)),
    )


def inspect_recording_directory(recording_directory: str | Path) -> TrialRecoveryReport:
    """Inspect an interrupted Trial without mutating or publishing any file.

    If Collector currently owns the dataset-root lease, inspection deliberately
    stops before opening ultrasound/HDF5 payloads and returns an occupied report.
    """

    directory, dataset_root = _validate_recording_directory(recording_directory)
    if read_activity(dataset_root) is not None:
        return _inspect_recording_directory(directory, dataset_root, guard_activity=True)
    # Close the check/use race: while payloads are inspected, recovery owns the
    # same exclusive lease as Collector.  This writes only the short-lived root
    # lock; the Trial package itself remains byte-for-byte read-only.
    try:
        with AcquisitionLock(dataset_root, _trial_uuid(directory)):
            return _inspect_recording_directory(
                directory, dataset_root, guard_activity=False
            )
    except FileExistsError:
        # Collector won the race after the first lightweight activity check.
        return TrialRecoveryReport(
            recording_directory=directory,
            trial_uuid=_trial_uuid(directory),
            state=TrialState.RECOVERABLE,
            ultrasound=None,
            hdf5_files=(),
            partial_files=(),
            recoverable=False,
            inspected_at_utc=datetime.now(timezone.utc),
            active_collection=True,
            active_trial_uuid=(
                activity.trial_uuid
                if (activity := read_activity(dataset_root)) is not None
                else None
            ),
            issues=(
                "Collector/recovery lease appeared during inspection; payload files were not opened.",
            ),
        )


def discover_recoverable_trials(dataset_root: str | Path) -> tuple[TrialRecoveryReport, ...]:
    """Startup/manual discovery entry point for every ``.recording`` package."""

    return tuple(inspect_recording_directory(path) for path in iter_recording_directories(dataset_root))


def _encode_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, TrialState):
        return value.value
    raise TypeError(type(value).__name__)


def _publish_append_only_document(path: Path, document: dict[str, Any]) -> Path:
    """Create one immutable audit document; never replace an existing record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document,
        default=_encode_json,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def repair_recording_directory(
    recording_directory: str | Path,
    *,
    truncate_ultrasound_tail: bool = True,
) -> TrialRecoveryReport:
    """Explicitly repair only an objectively incomplete ultrasound tail.

    HDF5 and every complete ultrasound block remain untouched.  The Trial stays
    ``.recording``/``RECOVERABLE``; repair alone can never publish it.
    """

    directory, dataset_root = _validate_recording_directory(recording_directory)
    operation_uuid = uuid4()
    report_directory = directory / "reports"
    intent_path = report_directory / f"recovery-{operation_uuid}.intent.json"
    report_path = report_directory / f"recovery-{operation_uuid}.result.json"

    with AcquisitionLock(dataset_root, _trial_uuid(directory)):
        before = _inspect_recording_directory(directory, dataset_root, guard_activity=False)
        if not before.can_repair or before.ultrasound is None:
            raise UnsafeRecoveryDecisionError(
                "safe tail repair is unavailable; the Trial remains RECOVERABLE"
            )
        _assert_owned_regular_file(before.ultrasound.data_path, directory)
        report_directory.mkdir(parents=True, exist_ok=True)
        intent = {
            "schema_version": "1.0.0",
            "operation_uuid": str(operation_uuid),
            "action": RecoveryAction.REPAIR_SAFE_TAIL.value,
            "status": "INTENT_RECORDED",
            "decided_at_utc": datetime.now(timezone.utc),
            "before": asdict(before),
            "before_ultrasound_sha256": sha256_file(before.ultrasound.data_path),
            "raw_data_policy": (
                "Only an objectively incomplete ultrasound tail may be truncated; "
                "HDF5 and complete blocks are not rewritten."
            ),
        }
        _publish_append_only_document(intent_path, intent)

        try:
            recover_ultrasound_file(
                before.ultrasound.data_path,
                truncate=truncate_ultrasound_tail,
                rebuild_idx=True,
            )
            after = _inspect_recording_directory(directory, dataset_root, guard_activity=False)
            _publish_append_only_document(
                report_path,
                {**intent, "status": "COMPLETED", "after": asdict(after)},
            )
        except BaseException as exc:
            failed_path = report_directory / f"recovery-{operation_uuid}.failed.json"
            try:
                _publish_append_only_document(
                    failed_path,
                    {
                        **intent,
                        "status": "FAILED",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            except BaseException:
                pass
            raise
        return TrialRecoveryReport(
            recording_directory=after.recording_directory,
            trial_uuid=after.trial_uuid,
            state=after.state,
            ultrasound=after.ultrasound,
            hdf5_files=after.hdf5_files,
            partial_files=after.partial_files,
            recoverable=after.recoverable,
            inspected_at_utc=after.inspected_at_utc,
            repair_log_path=report_path,
            prepared_publication=after.prepared_publication,
            allowed_actions=after.allowed_actions,
            issues=after.issues,
        )


def _evidence_snapshot(directory: Path) -> tuple[dict[str, Any], ...]:
    evidence: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise UnsafeRecoveryDecisionError(
                f"symbolic links are forbidden in a recoverable package: {path}"
            )
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        stat = path.stat()
        evidence.append(
            {
                "relative_path": relative,
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
                "modified_utc_ns": stat.st_mtime_ns,
                "raw_artifact": relative.startswith("raw/"),
            }
        )
    return tuple(evidence)


def finalize_prepared_recording(
    recording_directory: str | Path,
    *,
    confirmed: bool = False,
    confirmed_by: str | None = None,
) -> RecoveryDecisionResult:
    """Atomically publish only a package already proven complete.

    This operation does not rewrite Manifest, checksums, or Artifact bytes.  A
    package lacking any part of the proof is refused rather than made to look
    ``FINALIZED``.
    """

    if not confirmed:
        raise RecoveryConfirmationRequiredError("explicit FINALIZED confirmation is required")
    directory, dataset_root = _validate_recording_directory(recording_directory)
    destination = directory.with_name(directory.name[: -len(".recording")])
    operation_uuid = uuid4()
    decided_at = datetime.now(timezone.utc)

    with AcquisitionLock(dataset_root, _trial_uuid(directory)):
        report = _inspect_recording_directory(directory, dataset_root, guard_activity=False)
        if not report.can_finalize or report.prepared_publication is None:
            reasons = "; ".join(report.issues) or "complete publication proof is unavailable"
            raise UnsafeRecoveryDecisionError(
                f"refusing to publish RECOVERABLE Trial as FINALIZED: {reasons}"
            )
        if destination.exists():
            raise FileExistsError(f"recovery destination already exists: {destination}")
        audit_path = dataset_root / "recovery-audit" / f"finalize-{operation_uuid}.json"
        _publish_append_only_document(
            audit_path,
            {
                "schema_version": "1.0.0",
                "operation_uuid": str(operation_uuid),
                "action": RecoveryAction.FINALIZE_PREPARED.value,
                "confirmed": True,
                "confirmed_by": (confirmed_by.strip() if confirmed_by and confirmed_by.strip() else None),
                "decided_at_utc": decided_at,
                "trial_uuid": report.trial_uuid,
                "source_directory": str(directory),
                "destination_directory": str(destination),
                "proof": asdict(report.prepared_publication),
                "raw_data_policy": "No Trial-package file is changed; publication is one atomic directory rename.",
            },
        )
        os.replace(directory, destination)

    return RecoveryDecisionResult(
        action=RecoveryAction.FINALIZE_PREPARED,
        trial_uuid=report.trial_uuid,
        source_directory=directory,
        destination_directory=destination,
        audit_path=audit_path,
        decided_at_utc=decided_at,
    )


def abort_recording_preserving_data(
    recording_directory: str | Path,
    *,
    reason: str,
    confirmed: bool = False,
    confirmed_by: str | None = None,
) -> RecoveryDecisionResult:
    """Record an append-only decision and atomically retain the package as aborted.

    No existing file is modified or deleted.  Every pre-decision file is hashed
    into the audit record before the directory is renamed to ``.aborted``.
    """

    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ValueError("an ABORTED decision requires a non-empty reason")
    if not confirmed:
        raise RecoveryConfirmationRequiredError("explicit ABORTED confirmation is required")
    directory, dataset_root = _validate_recording_directory(recording_directory)
    destination = directory.with_name(
        f"{directory.name[: -len('.recording')]}.aborted"
    )
    operation_uuid = uuid4()
    decided_at = datetime.now(timezone.utc)

    with AcquisitionLock(dataset_root, _trial_uuid(directory)):
        report = _inspect_recording_directory(directory, dataset_root, guard_activity=False)
        if destination.exists():
            raise FileExistsError(f"recovery destination already exists: {destination}")
        evidence = _evidence_snapshot(directory)
        relative_audit = Path("reports") / f"recovery-abort-{operation_uuid}.json"
        audit_path = directory / relative_audit
        _publish_append_only_document(
            audit_path,
            {
                "schema_version": "1.0.0",
                "operation_uuid": str(operation_uuid),
                "action": RecoveryAction.ABORT_PRESERVING_DATA.value,
                "state": TrialState.ABORTED.value,
                "confirmed": True,
                "confirmed_by": (confirmed_by.strip() if confirmed_by and confirmed_by.strip() else None),
                "reason": normalized_reason,
                "decided_at_utc": decided_at,
                "trial_uuid": report.trial_uuid,
                "source_directory": str(directory),
                "destination_directory": str(destination),
                "original_evidence": evidence,
                "raw_data_policy": "All existing bytes are retained; no file is modified or deleted.",
            },
        )
        os.replace(directory, destination)

    return RecoveryDecisionResult(
        action=RecoveryAction.ABORT_PRESERVING_DATA,
        trial_uuid=report.trial_uuid,
        source_directory=directory,
        destination_directory=destination,
        audit_path=destination / relative_audit,
        decided_at_utc=decided_at,
    )


__all__ = [
    "Hdf5RecoveryStatus",
    "PreparedPublicationStatus",
    "RecoveryAction",
    "RecoveryConfirmationRequiredError",
    "RecoveryDecisionResult",
    "TrialRecoveryReport",
    "UnsafeRecoveryDecisionError",
    "abort_recording_preserving_data",
    "discover_recoverable_trials",
    "finalize_prepared_recording",
    "inspect_recording_directory",
    "repair_recording_directory",
]
