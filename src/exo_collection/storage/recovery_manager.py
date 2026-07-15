"""Read-only discovery and explicit repair planning for interrupted Trials."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import h5py

from exo_collection.domain.states import TrialState

from .layout import iter_recording_directories
from .recovery import UltrasoundRecoveryResult, recover_ultrasound_file, scan_ultrasound_file


@dataclass(frozen=True, slots=True)
class Hdf5RecoveryStatus:
    path: Path
    readable: bool
    closed_cleanly: bool
    sample_count: int | None
    error: str | None = None


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


def _trial_uuid(path: Path) -> UUID | None:
    name = path.name[: -len(".recording")]
    try:
        return UUID(name)
    except ValueError:
        return None


def _inspect_hdf5(path: Path) -> Hdf5RecoveryStatus:
    try:
        with h5py.File(path, "r") as file:
            data_count = int(file["samples/data"].shape[0])
            index_count = int(file["samples/sample_index"].shape[0])
            device_count = int(file["samples/device_time"].shape[0])
            host_count = int(file["samples/host_monotonic_ns"].shape[0])
            if len({data_count, index_count, device_count, host_count}) != 1:
                raise ValueError("sample datasets have inconsistent lengths")
            clean = bool(file.attrs.get("closed_cleanly", False))
            declared = int(file.attrs.get("sample_count", data_count))
            if clean and declared != data_count:
                raise ValueError("declared sample_count differs from dataset length")
        return Hdf5RecoveryStatus(path, True, clean, data_count)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return Hdf5RecoveryStatus(path, False, False, None, f"{type(exc).__name__}: {exc}")


def inspect_recording_directory(recording_directory: str | Path) -> TrialRecoveryReport:
    """Inspect an interrupted Trial without opening it in Data Studio or mutating it."""

    directory = Path(recording_directory).expanduser().resolve()
    if not directory.is_dir() or not directory.name.endswith(".recording"):
        raise ValueError("expected an existing .recording Trial directory")
    ultrasound_path = next(
        (
            path
            for path in (directory / "raw/ultrasound.bin.partial", directory / "raw/ultrasound.bin")
            if path.is_file()
        ),
        None,
    )
    ultrasound = scan_ultrasound_file(ultrasound_path) if ultrasound_path is not None else None
    hdf5_paths = sorted((directory / "raw").glob("*.h5.partial")) + sorted(
        (directory / "raw").glob("*.h5")
    )
    hdf5_statuses = tuple(_inspect_hdf5(path) for path in hdf5_paths)
    partials = tuple(sorted(path for path in directory.rglob("*.partial") if path.is_file()))
    has_valid_source = (
        ultrasound is not None and ultrasound.complete_block_count > 0
    ) or any(status.readable and (status.sample_count or 0) > 0 for status in hdf5_statuses)
    no_unsafe_ultrasound_corruption = ultrasound is None or not ultrasound.intermediate_corruption
    recoverable = has_valid_source and no_unsafe_ultrasound_corruption
    return TrialRecoveryReport(
        recording_directory=directory,
        trial_uuid=_trial_uuid(directory),
        state=TrialState.RECOVERABLE,
        ultrasound=ultrasound,
        hdf5_files=hdf5_statuses,
        partial_files=partials,
        recoverable=recoverable,
        inspected_at_utc=datetime.now(timezone.utc),
    )


def discover_recoverable_trials(dataset_root: str | Path) -> tuple[TrialRecoveryReport, ...]:
    return tuple(inspect_recording_directory(path) for path in iter_recording_directories(dataset_root))


def repair_recording_directory(
    recording_directory: str | Path,
    *,
    truncate_ultrasound_tail: bool = True,
) -> TrialRecoveryReport:
    """Explicitly repair a safe ultrasound tail and write an audit report.

    HDF5 is inspected but never rewritten. The Trial remains `.recording` and
    `RECOVERABLE`; an operator must later decide whether to finalize or abort it.
    """

    before = inspect_recording_directory(recording_directory)
    if before.ultrasound is not None:
        recover_ultrasound_file(
            before.ultrasound.data_path,
            truncate=truncate_ultrasound_tail,
            rebuild_idx=True,
        )
    after = inspect_recording_directory(recording_directory)
    report_path = after.recording_directory / "reports/recovery_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial = report_path.with_name(report_path.name + ".partial")

    def encode(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat().replace("+00:00", "Z")
        if isinstance(value, TrialState):
            return value.value
        raise TypeError(type(value).__name__)

    document = {
        "schema_version": "1.0.0",
        "action": "explicit_recovery",
        "before": asdict(before),
        "after": asdict(after),
        "raw_data_policy": "Only an objectively incomplete ultrasound tail may be truncated; HDF5 is not rewritten.",
    }
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, default=encode, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, report_path)
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
    )


__all__ = [
    "Hdf5RecoveryStatus",
    "TrialRecoveryReport",
    "discover_recoverable_trials",
    "inspect_recording_directory",
    "repair_recording_directory",
]

