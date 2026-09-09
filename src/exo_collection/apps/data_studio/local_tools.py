"""Read-only local analysis tools used by Exo Data Studio.

Every entry point in this module accepts only an atomically finalized Trial.
The guards are deliberately repeated at the service boundary so a future UI
change cannot make playback or integrity verification inspect an active
``.recording`` directory or a ``.partial`` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from numpy.typing import NDArray

from exo_collection.adapters.ultrasound.raw_ethernet import (
    decode_raw_ethernet_flags,
)
from exo_collection.domain.states import TrialState
from exo_collection.domain.prompt_labels import (
    PromptLabelSource,
    load_prompt_label_events,
)
from exo_collection.readers.binary_block import BlockBinaryReader, scan_binary_file
from exo_collection.storage.activity import read_activity
from exo_collection.storage.layout import path_has_unpublished_component
from exo_collection.storage.manifest import TrialManifest, load_manifest
from exo_collection.timing.clock_model import fit_affine_clock
from exo_collection.writers.binary_block import companion_paths

from .service import load_catalog_snapshot

_log = logging.getLogger(__name__)


class DataStudioToolError(RuntimeError):
    """A local tool cannot safely operate on the requested data."""


class AcquisitionBecameActiveError(DataStudioToolError):
    """Collector became active while a disk-heavy local tool was running."""


@dataclass(frozen=True, slots=True)
class SignalPlayback:
    """A bounded, downsampled signal suitable for plotting in the GUI."""

    time_s: NDArray[np.float64]
    values: NDArray[np.generic]
    channels: tuple[str, ...]
    units: tuple[str, ...]
    # Physical sensor labels in the leading sample-shape order.  Empty means
    # that the source file did not publish enough information to identify
    # individual devices; the UI must display this honestly rather than
    # inventing device IDs.
    sensor_labels: tuple[str, ...] = ()
    # ``True`` means that the corresponding sample is the first sample after
    # a discontinuity in the source device clock.  Plotters must not connect
    # that sample to the preceding one.
    break_before: NDArray[np.bool_] | None = None


@dataclass(frozen=True, slots=True)
class UltrasoundPlayback:
    """A bounded A-line waterfall plus the latest displayed frame."""

    time_s: NDArray[np.float64]
    # Shape is (channel, frame, depth).
    waterfall: NDArray[np.generic]
    latest_frame: NDArray[np.generic]
    channels: tuple[str, ...]
    source_frame_count: int
    source_packet_count: int = 0
    source_trailer_packet_count: int = 0
    alignment_semantics: str = "device_synchronized_frames"
    device_synchronized: bool = True


@dataclass(frozen=True, slots=True)
class PromptLabelPlaybackEvent:
    time_s: float
    source: PromptLabelSource
    label: str
    key: str


@dataclass(frozen=True, slots=True)
class MocapPlayback:
    """Bounded, plot-ready optical marker trajectory for the 3D view."""

    time_s: NDArray[np.float64]
    # Shape is (frame, marker, 3) in millimetres.
    positions: NDArray[np.generic]
    marker_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrialPlayback:
    manifest_path: Path
    trial_uuid: str
    condition_code: str
    formal_t0_host_monotonic_ns: int
    ultrasound: UltrasoundPlayback | None
    imu: SignalPlayback | None
    encoder: SignalPlayback | None
    sync: SignalPlayback | None
    sync_trigger_times_s: NDArray[np.float64]
    # When the IMU recording is decoupled (each physical unit on its own
    # independent clock), ``imu`` is ``None`` and every physical sensor is
    # exposed as its own time-aligned series here.  Coupled / single-device
    # recordings keep the shared-axis ``imu`` and leave this empty.
    imu_sensors: tuple[SignalPlayback, ...] = ()
    emg: SignalPlayback | None = None
    mocap: MocapPlayback | None = None
    moment: SignalPlayback | None = None
    prompt_labels: tuple[PromptLabelPlaybackEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class FullStatistics:
    projects: int
    subjects: int
    sessions: int
    trials: int
    finalized_trials: int
    total_duration_s: float
    artifact_count: int
    artifact_bytes: int
    by_condition: dict[str, dict[str, float | int]]
    by_quality: dict[str, int]
    by_modality: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class ChecksumItem:
    relative_path: str
    expected_sha256: str
    actual_sha256: str | None
    size_bytes: int | None
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class ChecksumReport:
    manifest_path: Path
    trial_uuid: str
    items: tuple[ChecksumItem, ...]

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item.passed for item in self.items)


@dataclass(frozen=True, slots=True)
class QualityAudit:
    manifest_path: Path
    trial_uuid: str
    computed_grade: str
    reviewed_grade: str | None
    reviewed_by: str | None
    reviewed_at_utc: str | None
    review_reason: str | None
    review_count: int
    required_artifacts_complete: bool
    integrity_checks_passed: bool
    algorithm_version: str | None
    issues: tuple[dict[str, Any], ...]
    devices: tuple[dict[str, str], ...]
    sync_checks: tuple[dict[str, str], ...]
    warnings_text: str
    soft_metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Hdf5ChannelStat:
    channel: str
    unit: str
    min: float
    max: float
    mean: float
    std: float
    nan_count: int


@dataclass(frozen=True, slots=True)
class Hdf5Inspection:
    relative_path: str
    size_bytes: int
    closed_cleanly: bool
    sample_count: int
    dtype: str
    sample_shape: tuple[int, ...]
    nominal_rate_hz: float | None
    channels: tuple[str, ...]
    units: tuple[str, ...]
    device_metadata: dict[str, Any]
    trial_metadata: dict[str, Any]
    clock_model: dict[str, Any]
    root_attrs: tuple[tuple[str, str], ...]
    structure: tuple[str, ...]
    stats: tuple[Hdf5ChannelStat, ...]
    preview_columns: tuple[str, ...]
    preview_rows: tuple[tuple[Any, ...], ...]
    discontinuity_count: int
    event_count: int


@dataclass(frozen=True, slots=True)
class UltrasoundInspection:
    relative_path: str
    size_bytes: int
    meta_path: str
    index_path: str
    metadata: dict[str, Any]
    block_count: int


@dataclass(frozen=True, slots=True)
class JsonlInspection:
    relative_path: str
    size_bytes: int
    event_count: int
    preview: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    modality: str
    relative_path: str
    size_bytes: int
    kind: str
    message: str = ""
    hdf5: Hdf5Inspection | None = None
    ultrasound: UltrasoundInspection | None = None
    jsonl: JsonlInspection | None = None


@dataclass(frozen=True, slots=True)
class TrialInspection:
    manifest_path: Path
    trial_uuid: str
    condition_code: str
    artifact_count: int
    artifacts: tuple[ArtifactInspection, ...]


def _has_active_component(path: Path) -> bool:
    return path_has_unpublished_component(path)


def _require_idle(data_root: Path) -> None:
    if read_activity(data_root) is not None:
        raise AcquisitionBecameActiveError(
            "Collector 已开始采集，后台工具已停止以保护原始采集。"
        )


def _require_trial_under_data_root(manifest_path: Path, data_root: Path) -> None:
    try:
        manifest_path.relative_to(data_root)
    except ValueError as exc:
        raise DataStudioToolError("Trial Manifest 不在当前数据根目录中") from exc


def _load_finalized_trial(
    manifest_path: str | Path,
) -> tuple[Path, Path, TrialManifest]:
    supplied = Path(manifest_path).expanduser()
    if _has_active_component(supplied):
        raise DataStudioToolError("拒绝读取 .recording/.partial 路径")
    path = supplied.resolve()
    if path.name != "manifest.json" or not path.is_file():
        raise DataStudioToolError("请选择包含 manifest.json 的 Trial")
    if _has_active_component(path):
        raise DataStudioToolError("拒绝读取 .recording/.partial 路径")
    manifest = load_manifest(path)
    if manifest.state is not TrialState.FINALIZED:
        raise DataStudioToolError(
            f"只能处理 FINALIZED Trial，当前状态为 {manifest.state.value}"
        )
    trial_root = path.parent.resolve()
    if trial_root.name == ".exo":
        trial_root = trial_root.parent.resolve()
    return path, trial_root, manifest


def _artifact_path(trial_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or _has_active_component(relative):
        raise DataStudioToolError(f"非法 Artifact 路径：{relative_path}")
    candidate = (trial_root / relative).resolve()
    try:
        candidate.relative_to(trial_root)
    except ValueError as exc:
        raise DataStudioToolError(
            f"Artifact 路径逃逸 Trial 目录：{relative_path}"
        ) from exc
    if _has_active_component(candidate):
        raise DataStudioToolError(f"拒绝读取临时 Artifact：{relative_path}")
    return candidate


def _artifact_for(
    manifest: TrialManifest,
    *,
    modality: str,
    suffix: str,
) -> str | None:
    matches = [
        artifact.relative_path
        for artifact in manifest.artifacts
        if artifact.modality == modality
        and artifact.relative_path.casefold().endswith(suffix.casefold())
    ]
    return matches[0] if matches else None


def _artifact_named(manifest: TrialManifest, relative_path: str) -> str | None:
    return next(
        (
            artifact.relative_path
            for artifact in manifest.artifacts
            if artifact.relative_path == relative_path
        ),
        None,
    )


def _internal_relative(manifest_path: Path, filename: str) -> str:
    """Return the metadata path for both current and legacy Trial layouts."""

    return f".exo/{filename}" if manifest_path.parent.name == ".exo" else filename


def _internal_artifact_named(
    manifest: TrialManifest, manifest_path: Path, filename: str
) -> str | None:
    preferred = _internal_relative(manifest_path, filename)
    candidates = (preferred, f"reports/{filename}", filename, f".exo/{filename}")
    return next(
        (
            listed
            for candidate in dict.fromkeys(candidates)
            if (listed := _artifact_named(manifest, candidate)) is not None
        ),
        None,
    )


def _even_indices(count: int, limit: int) -> NDArray[np.int64] | slice:
    if count <= limit:
        return slice(None)
    return np.unique(np.linspace(0, count - 1, limit, dtype=np.int64))


def _decode_strings(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8", errors="replace"))
        else:
            result.append(str(value))
    return tuple(result)


def _flatten_channel_labels(
    base: tuple[str, ...], units: tuple[str, ...], column_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(base) == column_count:
        return base, units
    if base and column_count % len(base) == 0:
        groups = column_count // len(base)
        labels = tuple(
            f"{group + 1}:{channel}"
            for group in range(groups)
            for channel in base
        )
        expanded_units = tuple(
            units[index] if index < len(units) else ""
            for _group in range(groups)
            for index in range(len(base))
        )
        return labels, expanded_units
    return (
        tuple(f"ch_{index + 1}" for index in range(column_count)),
        ("",) * column_count,
    )


def _unwrap_device_clock(values: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Unwrap a uint16 PacketCounter (or uint32 SampleTimeFine) sequence.

    Returns ``values`` unchanged when already monotonic.  Detects wrap points
    (negative jumps) and accumulates the wrap modulus so the result is strictly
    non-decreasing.  The modulus is inferred from the value range: counter
    values are < 65536 (mod 65536); larger values are treated as SampleTimeFine
    (mod 2^32), which in practice never wraps within a trial.
    """
    if values.size < 2:
        return values if values.size == 1 else None
    diffs = np.diff(values)
    if not np.any(diffs < 0):
        return values
    mod = 65536.0 if float(np.max(values)) < 65536.0 else 4294967296.0
    correction = np.zeros(values.shape, dtype=np.float64)
    acc = 0.0
    for index in range(1, values.size):
        if diffs[index - 1] < 0:
            acc += mod
        correction[index] = acc
    return values + correction


def _decode_device_metadata(
    raw_device: Any, path: Path | None = None
) -> dict[str, Any]:
    """Parse the JSON object stored under ``metadata/device``."""
    if isinstance(raw_device, bytes):
        raw_device = raw_device.decode("utf-8", errors="replace")
    try:
        result = json.loads(str(raw_device))
    except (TypeError, ValueError, json.JSONDecodeError):
        if path is not None:
            _log.warning("HDF5 device metadata is not valid JSON: %s", path)
        return {}
    return result if isinstance(result, dict) else {}


def _sensor_labels_from_metadata(device_metadata: dict[str, Any]) -> tuple[str, ...]:
    candidates = (
        device_metadata.get("preview_labels")
        or device_metadata.get("device_ids")
        or ()
    )
    if isinstance(candidates, (list, tuple)):
        return tuple(str(value) for value in candidates)
    return ()


def _is_decoupled_imu(
    device_metadata: dict[str, Any],
    device_time: NDArray[np.float64] | None,
    n_devices: int,
) -> bool:
    """Detect independent per-unit streams that must not share one time axis.

    The wired MTw adapter stamps each packet with that unit's own SampleTimeFine
    and records ``alignment_mode: "none_independent_streams"``.  As a fallback
    for older files without that marker, interleaved per-unit clocks also show
    up as many negative diffs in the raw device_time array, whereas a single
    shared clock is monotonic except for at most a couple of wrap points.
    """
    if n_devices <= 1:
        return False
    if device_metadata.get("alignment_mode") == "none_independent_streams":
        return True
    if (
        device_time is not None
        and device_time.size >= 2
        and np.all(np.isfinite(device_time))
    ):
        return int(np.count_nonzero(np.diff(device_time) < 0)) > 2
    return False


def _read_hdf5_signal(
    path: Path,
    *,
    formal_t0_ns: int,
    max_points: int,
) -> tuple[SignalPlayback, NDArray[np.float64]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("closed_cleanly", False)):
            raise DataStudioToolError(f"HDF5 未正常关闭：{path.name}")
        if "samples/data" not in handle or "samples/host_monotonic_ns" not in handle:
            raise DataStudioToolError(f"HDF5 结构不完整：{path.name}")
        count = int(handle["samples/data"].shape[0])
        selector = _even_indices(count, max_points)
        selected_rows = (
            np.arange(*selector.indices(count), dtype=np.int64)
            if isinstance(selector, slice)
            else np.asarray(selector, dtype=np.int64)
        )
        data = np.asarray(handle["samples/data"][selector])
        break_before = np.zeros(selected_rows.size, dtype=np.bool_)
        host_ns = np.asarray(
            handle["samples/host_monotonic_ns"][selector], dtype=np.float64
        )
        if "samples/device_time" in handle:
            device_time = np.asarray(
                handle["samples/device_time"][selector], dtype=np.float64
            )
            if device_time.size >= 2 and np.all(np.isfinite(device_time)):
                # Independent-streams recordings interleave several per-unit
                # clocks into one array, so raw device_time shows many negative
                # diffs.  A single shared clock is monotonic except for at most
                # a couple of wrap points.  Only reconstruct a uniform axis from
                # a genuine shared clock; decoupled data keeps the real per-row
                # host arrival time instead.
                decoupled = int(np.count_nonzero(np.diff(device_time) < 0)) > 2
                if not decoupled:
                    source = _unwrap_device_clock(device_time)
                    if source is not None and np.all(np.diff(source) > 0):
                        if source.size >= 2:
                            selected_row_steps = np.diff(selected_rows).astype(
                                np.float64
                            )
                            clock_steps = np.diff(source)
                            clock_step_per_row = clock_steps / selected_row_steps
                            nominal_clock_step = float(np.median(clock_step_per_row))
                            if (
                                np.isfinite(nominal_clock_step)
                                and nominal_clock_step > 0
                            ):
                                # Account for the rows intentionally omitted by
                                # GUI downsampling.  A further half device tick
                                # is enough to distinguish an actual missing
                                # packet from normal floating-point clock jitter.
                                expected_clock_steps = (
                                    nominal_clock_step * selected_row_steps
                                )
                                tolerance = 0.5 * nominal_clock_step
                                break_before[1:] = (
                                    clock_steps > expected_clock_steps + tolerance
                                )
                        try:
                            host_ns = fit_affine_clock(source, host_ns).map(source)
                        except ValueError:
                            pass  # fall back to raw host_ns on a degenerate fit
        channels = (
            _decode_strings(handle["metadata/channels"][:])
            if "metadata/channels" in handle
            else ()
        )
        units = (
            _decode_strings(handle["metadata/units"][:])
            if "metadata/units" in handle
            else ()
        )
        sensor_labels: tuple[str, ...] = ()
        if "metadata/device" in handle:
            device_metadata = _decode_device_metadata(
                handle["metadata/device"][()], path
            )
            sensor_labels = _sensor_labels_from_metadata(device_metadata)
        trigger_times: list[float] = []
        if "events/records" in handle:
            records = handle["events/records"]
            event_selector = _even_indices(int(records.shape[0]), 2000)
            for raw in records[event_selector]:
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if (
                    payload.get("event_type") == "sync_pulse"
                    and payload.get("edge_type") == "rising"
                ):
                    timestamp = payload.get("host_monotonic_ns")
                    if isinstance(timestamp, (int, float)):
                        trigger_times.append((float(timestamp) - formal_t0_ns) / 1e9)

    if count == 0:
        values = np.empty((0, max(1, len(channels))), dtype=np.float64)
        time_s = np.empty((0,), dtype=np.float64)
    else:
        values = data.reshape((data.shape[0], -1))
        time_s = (host_ns - float(formal_t0_ns)) / 1e9
    labels, expanded_units = _flatten_channel_labels(
        channels, units, int(values.shape[1])
    )
    return (
        SignalPlayback(
            time_s=np.asarray(time_s, dtype=np.float64),
            values=values,
            channels=labels,
            units=expanded_units,
            break_before=break_before,
            sensor_labels=sensor_labels,
        ),
        np.asarray(trigger_times, dtype=np.float64),
    )


def _read_hdf5_imu_sensors(
    path: Path,
    *,
    formal_t0_ns: int,
    max_points: int,
) -> tuple[SignalPlayback, ...]:
    """Read an IMU HDF5, splitting decoupled devices into per-sensor series.

    The wired MTw adapter records each unit independently: every parsed packet
    is its own one-sample batch with only that unit's row populated, so the raw
    ``samples/data`` interleaves several independent per-unit clocks.  A single
    shared time axis cannot represent that faithfully, so when the recording is
    decoupled the reader returns one ``SignalPlayback`` per physical sensor,
    each carrying that sensor's own host-arrival timestamps.  Coupled (Awinda)
    or single-device recordings return a one-element tuple and delegate to the
    shared-clock reconstruction in :func:`_read_hdf5_signal`.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("closed_cleanly", False)):
            raise DataStudioToolError(f"HDF5 未正常关闭：{path.name}")
        if "samples/data" not in handle or "samples/host_monotonic_ns" not in handle:
            raise DataStudioToolError(f"HDF5 结构不完整：{path.name}")
        data_ds = handle["samples/data"]
        n_devices = int(data_ds.shape[1]) if data_ds.ndim == 3 else 1
        channels = (
            _decode_strings(handle["metadata/channels"][:])
            if "metadata/channels" in handle
            else ()
        )
        units = (
            _decode_strings(handle["metadata/units"][:])
            if "metadata/units" in handle
            else ()
        )
        device_metadata = (
            _decode_device_metadata(handle["metadata/device"][()], path)
            if "metadata/device" in handle
            else {}
        )
        sensor_labels = _sensor_labels_from_metadata(device_metadata)
        device_time = (
            np.asarray(handle["samples/device_time"], dtype=np.float64)
            if "samples/device_time" in handle
            else None
        )

        decoupled = (
            _is_decoupled_imu(device_metadata, device_time, n_devices)
            and data_ds.ndim == 3
            and len(sensor_labels) == n_devices
            and len(channels) == int(data_ds.shape[2])
        )
        if not decoupled:
            series, _ = _read_hdf5_signal(
                path, formal_t0_ns=formal_t0_ns, max_points=max_points
            )
            return (series,)

        data = np.asarray(data_ds)
        host_ns = np.asarray(
            handle["samples/host_monotonic_ns"], dtype=np.float64
        )
        series_list: list[SignalPlayback] = []
        for slot in range(n_devices):
            row_index = np.flatnonzero(np.isfinite(data[:, slot, 0]))
            if row_index.size == 0:
                continue
            chosen = row_index[_even_indices(int(row_index.size), max_points)]
            slot_values = data[chosen, slot, :].reshape(-1, len(channels))
            slot_time_s = (host_ns[chosen] - float(formal_t0_ns)) / 1e9
            series_list.append(
                SignalPlayback(
                    time_s=np.asarray(slot_time_s, dtype=np.float64),
                    values=slot_values,
                    channels=channels,
                    units=units,
                    break_before=None,
                    sensor_labels=(sensor_labels[slot],),
                )
            )
        return tuple(series_list)


def _read_hdf5_mocap(
    path: Path,
    *,
    formal_t0_ns: int,
    max_points: int,
) -> MocapPlayback:
    """Read optical-marker trajectories (``samples/data`` = frame × marker × 3)."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("closed_cleanly", False)):
            raise DataStudioToolError(f"HDF5 未正常关闭：{path.name}")
        if "samples/data" not in handle or "samples/host_monotonic_ns" not in handle:
            raise DataStudioToolError(f"HDF5 结构不完整：{path.name}")
        data = np.asarray(handle["samples/data"], dtype=np.float64)
        if data.ndim != 3 or data.shape[2] != 3:
            raise DataStudioToolError(
                f"动捕 HDF5 形状应为 (帧, marker, 3)，实际为 {data.shape}"
            )
        count = int(data.shape[0])
        selector = _even_indices(count, max_points)
        positions = data[selector]
        host_ns = np.asarray(
            handle["samples/host_monotonic_ns"][selector], dtype=np.float64
        )
        marker_names = (
            _decode_strings(handle["metadata/channels"][:])
            if "metadata/channels" in handle
            else ()
        )
        if len(marker_names) != positions.shape[1]:
            device_metadata = (
                _decode_device_metadata(handle["metadata/device"][()], path)
                if "metadata/device" in handle
                else {}
            )
            raw_names = device_metadata.get("marker_names")
            if isinstance(raw_names, (list, tuple)):
                marker_names = tuple(str(name) for name in raw_names)
        if len(marker_names) != positions.shape[1]:
            marker_names = tuple(
                f"marker_{index + 1:02d}" for index in range(positions.shape[1])
            )
    time_s = (host_ns - float(formal_t0_ns)) / 1e9
    return MocapPlayback(
        time_s=np.asarray(time_s, dtype=np.float64),
        positions=positions,
        marker_names=tuple(marker_names),
    )


def _read_moment_csv(path: Path) -> SignalPlayback | None:
    """Best-effort read of a ``ground_truth.csv`` moment truth sidecar.

    The first column is assumed to be ``time_s`` in seconds; every remaining
    numeric column becomes a channel.  Returns ``None`` when the file is absent
    or unparseable so a trial without a truth sidecar still opens.
    """

    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    rows = [line for line in text.splitlines() if line.strip()]
    if len(rows) < 2:
        return None
    import csv as _csv

    try:
        records = list(_csv.reader(rows))
    except _csv.Error:
        return None
    header = records[0]
    if not header or header[0].strip().casefold() not in {"time_s", "time", "t"}:
        return None
    raw_channels = tuple(str(name).strip() for name in header[1:])
    try:
        matrix = np.asarray(
            [[float(cell) for cell in row] for row in records[1:]],
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return None
    time_s = matrix[:, 0]
    values = matrix[:, 1:]
    # ground_truth.csv 的表头为 time_s + 12 个 imu_* 特征列 + 6 个关节力矩列
    # （hip_flexion_r/l、knee_angle_r/l、ankle_angle_r/l）。力矩真值面板只展示两个
    # 髋关节力矩（IMU 特征另有独立面板，来自 imu.h5；膝/踝力矩暂不展示），故剔除
    # imu_ 特征列并只保留 hip 列。
    keep = [
        index
        for index, name in enumerate(raw_channels)
        if not name.casefold().startswith("imu_") and "hip" in name.casefold()
    ]
    if not keep:
        return None
    channels = tuple(raw_channels[index] for index in keep)
    values = values[:, keep]
    return SignalPlayback(
        time_s=np.asarray(time_s, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        channels=channels[: values.shape[1]],
        units=("N·m",) * values.shape[1],
    )


def _is_raw_ethernet_ultrasound(metadata: dict[str, Any]) -> bool:
    """Return whether binary metadata declares the packet-per-channel format."""

    protocol = str(metadata.get("protocol", "")).strip().casefold()
    transport = str(metadata.get("transport", "")).strip().casefold()
    return protocol == "raw_ethernet_uint8" or transport in {
        "raw_ethernet",
        "raw_ethernet_scapy_npcap",
    }


def _raw_ultrasound_channel_labels(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw_channels = metadata.get("channels")
    if isinstance(raw_channels, list) and len(raw_channels) == 4:
        labels: list[str] = []
        for index, value in enumerate(raw_channels):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                labels.append(f"ch_{int(value)}")
            else:
                label = str(value).strip()
                labels.append(label or f"ch_{index + 1}")
        return tuple(labels)
    return ("ch_1", "ch_2", "ch_3", "ch_4")


def _raw_ultrasound_adc_view(
    packet: NDArray[np.generic],
    *,
    channel: int,
    metadata: dict[str, Any],
) -> NDArray[np.generic]:
    """Return derived ADC samples while leaving stored raw bytes untouched."""

    preservation = str(metadata.get("raw_preservation", "")).strip().casefold()
    if preservation != "complete captured frame":
        return packet
    if (
        packet.ndim != 1
        or packet.size < 3
        or int(packet[0]) != 0x00
        or int(packet[1]) != channel + 1
        or int(packet[-1]) != 0xFF
    ):
        raise DataStudioToolError(
            "Raw Ethernet ultrasound frame signature does not match its block flags"
        )
    return packet[2:-1]


def _read_raw_ethernet_ultrasound(
    reader: BlockBinaryReader,
    *,
    formal_t0_ns: int,
    max_frames: int,
    max_depth_points: int,
    idle_check: Callable[[], None],
) -> UltrasoundPlayback:
    """Load independent channel packets without claiming device synchrony.

    The raw Ethernet device emits one channel per packet.  Offline display
    pairs the kth packet received for each channel, solely to make a bounded
    four-channel playback view.  The authoritative binary order and packet
    CRC are still read sequentially before this derived grouping is formed.
    """

    channel_counts = [0, 0, 0, 0]
    source_packet_count = 0
    trailer_packet_count = 0
    stored_depth_count = int(reader.metadata.get("sample_shape", [0])[-1])
    complete_wire_frame = (
        str(reader.metadata.get("raw_preservation", "")).strip().casefold()
        == "complete captured frame"
    )
    depth_count = stored_depth_count - 3 if complete_wire_frame else stored_depth_count
    if depth_count <= 0:
        raise DataStudioToolError("Raw Ethernet ultrasound depth is invalid")
    depth_selector = _even_indices(depth_count, max_depth_points)

    # First pass validates every packet/CRC in authoritative storage order and
    # establishes how many complete four-channel ordinals exist.  Keeping only
    # counters here avoids loading an arbitrarily long Trial into RAM.
    for ordinal in range(reader.block_count):
        idle_check()
        record = reader.read_block(ordinal=ordinal)
        if record.header.sample_count != 1 or len(record.data) != 1:
            raise DataStudioToolError(
                "Raw Ethernet ultrasound blocks must contain exactly one packet"
            )
        decoded = decode_raw_ethernet_flags(record.header.flags)
        packet = np.asarray(record.data[0])
        if packet.ndim != 1:
            raise DataStudioToolError(
                "Raw Ethernet ultrasound packets must be one-dimensional A-lines"
            )
        _raw_ultrasound_adc_view(
            packet,
            channel=decoded.channel,
            metadata=reader.metadata,
        )
        channel_counts[decoded.channel] += 1
        source_packet_count += 1
        trailer_packet_count += int(decoded.has_trailer)

    channels = _raw_ultrasound_channel_labels(reader.metadata)
    complete_count = min(channel_counts)
    retained_depth = min(depth_count, max_depth_points)
    if complete_count == 0:
        return UltrasoundPlayback(
            time_s=np.empty((0,), dtype=np.float64),
            waterfall=np.empty((4, 0, retained_depth), dtype=reader.dtype),
            latest_frame=np.empty((0, retained_depth), dtype=reader.dtype),
            channels=channels,
            source_frame_count=0,
            source_packet_count=source_packet_count,
            source_trailer_packet_count=trailer_packet_count,
            alignment_semantics="independent_channel_arrival_ordinal_for_playback_only",
            device_synchronized=False,
        )

    keep = _even_indices(complete_count, max_frames)
    complete_ordinals = np.arange(complete_count, dtype=np.int64)[keep]
    retained_ordinals = {int(value) for value in complete_ordinals}
    channel_ordinals = [0, 0, 0, 0]
    channel_packets: list[list[NDArray[np.generic]]] = [[], [], [], []]
    channel_arrival_ns: list[list[int]] = [[], [], [], []]

    # Second pass still follows exact binary packet order, but retains only
    # the bounded ordinal set selected for playback.
    for ordinal in range(reader.block_count):
        idle_check()
        record = reader.read_block(ordinal=ordinal)
        decoded = decode_raw_ethernet_flags(record.header.flags)
        channel_ordinal = channel_ordinals[decoded.channel]
        channel_ordinals[decoded.channel] += 1
        if channel_ordinal not in retained_ordinals:
            continue
        packet = _raw_ultrasound_adc_view(
            np.asarray(record.data[0]),
            channel=decoded.channel,
            metadata=reader.metadata,
        )
        channel_packets[decoded.channel].append(packet[depth_selector].copy())
        channel_arrival_ns[decoded.channel].append(
            int(record.header.host_monotonic_ns)
        )

    frames = np.stack(
        [
            np.stack(
                [channel_packets[channel][position] for channel in range(4)],
                axis=0,
            )
            for position in range(len(complete_ordinals))
        ],
        axis=0,
    )
    times = np.asarray(
        [
            (
                max(
                    channel_arrival_ns[channel][position]
                    for channel in range(4)
                )
                - formal_t0_ns
            )
            / 1e9
            for position in range(len(complete_ordinals))
        ],
        dtype=np.float64,
    )
    return UltrasoundPlayback(
        time_s=times,
        waterfall=np.transpose(frames, (1, 0, 2)),
        latest_frame=np.asarray(frames[-1]),
        channels=channels,
        source_frame_count=complete_count,
        source_packet_count=source_packet_count,
        source_trailer_packet_count=trailer_packet_count,
        alignment_semantics="independent_channel_arrival_ordinal_for_playback_only",
        device_synchronized=False,
    )


def _read_ultrasound(
    path: Path,
    *,
    meta_path: Path,
    index_path: Path,
    formal_t0_ns: int,
    max_frames: int,
    max_depth_points: int,
    idle_check: Callable[[], None],
) -> UltrasoundPlayback:
    with BlockBinaryReader(
        path,
        meta_path=meta_path,
        index_path=index_path,
        validate_crc=True,
        auto_rebuild_index=False,
    ) as reader:
        block_count = reader.block_count
        if _is_raw_ethernet_ultrasound(reader.metadata):
            return _read_raw_ethernet_ultrasound(
                reader,
                formal_t0_ns=formal_t0_ns,
                max_frames=max_frames,
                max_depth_points=max_depth_points,
                idle_check=idle_check,
            )
        if block_count == 0:
            return UltrasoundPlayback(
                time_s=np.empty((0,), dtype=np.float64),
                waterfall=np.empty((0, 0, 0), dtype=np.float32),
                latest_frame=np.empty((0, 0), dtype=np.float32),
                channels=(),
                source_frame_count=0,
            )
        selected_blocks = np.unique(
            np.linspace(
                0,
                block_count - 1,
                min(block_count, max_frames),
                dtype=np.int64,
            )
        )
        arrays: list[NDArray[np.generic]] = []
        times: list[float] = []
        source_frame_count = 0
        rate = float(reader.metadata.get("nominal_frame_rate_hz") or 0.0)
        for ordinal in selected_blocks:
            idle_check()
            record = reader.read_block(ordinal=int(ordinal))
            source_frame_count += int(record.header.sample_count)
            per_block_limit = max(1, math.ceil(max_frames / len(selected_blocks)))
            local_selector = _even_indices(len(record.data), per_block_limit)
            # Keep the positions in the uncompressed source block.  Enumerating
            # the downsampled array would turn source offsets [0, 99] into
            # [0, 1] and silently compress its playback time axis.
            source_offsets = np.arange(len(record.data), dtype=np.int64)[
                local_selector
            ]
            selected = np.asarray(record.data[source_offsets])
            if selected.ndim == 2:
                selected = selected[:, np.newaxis, :]
            elif selected.ndim > 3:
                selected = selected.reshape(
                    selected.shape[0], -1, selected.shape[-1]
                )
            depth_selector = _even_indices(selected.shape[-1], max_depth_points)
            selected = selected[..., depth_selector]
            arrays.append(selected)
            for source_offset in source_offsets:
                offset_s = float(source_offset) / rate if rate > 0 else 0.0
                times.append(
                    (record.header.host_monotonic_ns - formal_t0_ns) / 1e9
                    + offset_s
                )
        frames = np.concatenate(arrays, axis=0)
        if frames.shape[0] > max_frames:
            keep = _even_indices(frames.shape[0], max_frames)
            frames = frames[keep]
            times = list(np.asarray(times, dtype=np.float64)[keep])
        channel_count = int(frames.shape[1])
        raw_channels = reader.metadata.get("channels")
        if isinstance(raw_channels, list) and len(raw_channels) == channel_count:
            channels = tuple(str(value) for value in raw_channels)
        else:
            channels = tuple(f"ch_{index + 1}" for index in range(channel_count))
        return UltrasoundPlayback(
            time_s=np.asarray(times, dtype=np.float64),
            waterfall=np.transpose(frames, (1, 0, 2)),
            latest_frame=np.asarray(frames[-1]),
            channels=channels,
            source_frame_count=source_frame_count,
        )


def _emg_channel_labels(
    metadata: dict[str, Any], channel_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_channels = metadata.get("channel_names")
    if isinstance(raw_channels, (list, tuple)) and len(raw_channels) == channel_count:
        channels = tuple(str(value) for value in raw_channels)
    else:
        channels = tuple(f"ch_{index + 1}" for index in range(channel_count))
    raw_units = metadata.get("units")
    unit_values = (
        [str(value) for value in raw_units]
        if isinstance(raw_units, (list, tuple))
        else []
    )
    if len(unit_values) < channel_count:
        fallback = unit_values[-1] if unit_values else ""
        unit_values.extend([fallback] * (channel_count - len(unit_values)))
    return channels, tuple(unit_values[:channel_count])


def _read_emg(
    path: Path,
    *,
    meta_path: Path,
    index_path: Path,
    formal_t0_ns: int,
    max_points: int,
    idle_check: Callable[[], None],
) -> SignalPlayback:
    """Load a bounded, time-aligned EMG view from its block-binary artifact.

    EMG shares the block-binary layout with ultrasound, but every block stores a
    ``(sample_count, n_channels)`` batch of µV samples instead of A-lines.  The
    Noraxon SDK stamps each block's ``host_monotonic_ns`` at *arrival* time,
    which corresponds to the block's *last* sample, so the per-sample clock is
    reconstructed with ``fit_affine_clock(last_sample_index, host_ns)`` rather
    than read from the file.
    """

    with BlockBinaryReader(
        path,
        meta_path=meta_path,
        index_path=index_path,
        validate_crc=True,
        auto_rebuild_index=False,
    ) as reader:
        block_count = reader.block_count
        shape = reader.sample_shape
        channel_count = int(shape[0]) if shape else 0
        channels, units = _emg_channel_labels(reader.metadata, channel_count)
        if block_count == 0:
            return SignalPlayback(
                time_s=np.empty((0,), dtype=np.float64),
                values=np.empty((0, channel_count), dtype=np.float32),
                channels=channels,
                units=units,
            )

        # Anchor the sample clock to host arrival time with a header-only scan
        # (skips payload), then fit once so the whole axis is smooth/monotonic
        # instead of inheriting per-block arrival jitter.
        scan = scan_binary_file(path, validate_crc=False)
        if len(scan.headers) != block_count:
            raise DataStudioToolError("EMG 索引与数据文件块数不一致")
        anchor_selector = _even_indices(len(scan.headers), 2000)
        anchor_ordinals = np.arange(len(scan.headers), dtype=np.int64)[
            anchor_selector
        ]
        source = np.asarray(
            [
                scan.headers[int(ordinal)].first_sample_index
                + scan.headers[int(ordinal)].sample_count
                - 1
                for ordinal in anchor_ordinals
            ],
            dtype=np.float64,
        )
        target = np.asarray(
            [scan.headers[int(ordinal)].host_monotonic_ns for ordinal in anchor_ordinals],
            dtype=np.float64,
        )
        model = fit_affine_clock(source, target)

        # A source discontinuity only exists where a block's first sample does
        # not continue the previous block's sample run.
        gap_before_block = np.zeros(len(scan.headers), dtype=np.bool_)
        for block_ordinal in range(1, len(scan.headers)):
            previous = scan.headers[block_ordinal - 1]
            current = scan.headers[block_ordinal]
            gap_before_block[block_ordinal] = (
                current.first_sample_index
                != previous.first_sample_index + previous.sample_count
            )

        # EMG is one continuous sample stream across blocks, so downsample
        # *globally* (uniform source positions) rather than per block.  Uneven
        # block sizes would otherwise make the retained sample spacing ragged,
        # which the sweep plotter renders as spurious "丢包" gaps.
        total_samples = sum(int(header.sample_count) for header in scan.headers)
        keep = np.arange(total_samples, dtype=np.int64)[
            _even_indices(total_samples, max_points)
        ]
        values = np.empty((keep.size, channel_count), dtype=np.float32)
        retained_indices = np.empty(keep.size, dtype=np.int64)
        break_before = np.zeros(keep.size, dtype=np.bool_)
        cursor = 0
        stream_start = 0  # global position of the current block's first row
        for ordinal in range(block_count):
            block_size = int(scan.headers[ordinal].sample_count)
            stream_end = stream_start + block_size
            if cursor >= keep.size:
                break
            if keep[cursor] >= stream_end:
                stream_start = stream_end
                continue
            idle_check()
            begin = cursor
            while cursor < keep.size and keep[cursor] < stream_end:
                cursor += 1
            local_offsets = keep[begin:cursor] - stream_start
            record = reader.read_block(ordinal=ordinal)
            values[begin:cursor] = record.data[local_offsets]
            retained_indices[begin:cursor] = (
                scan.headers[ordinal].first_sample_index + local_offsets
            )
            if gap_before_block[ordinal] and local_offsets[0] == 0:
                break_before[begin] = True
            stream_start = stream_end

        time_s = (
            model.map(retained_indices.astype(np.float64)) - float(formal_t0_ns)
        ) / 1e9
        return SignalPlayback(
            time_s=np.asarray(time_s, dtype=np.float64),
            values=values,
            channels=channels,
            units=units,
            break_before=break_before,
        )


def load_trial_playback(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    max_signal_points: int = 4000,
    max_ultrasound_frames: int = 4000,
    max_ultrasound_depth_points: int = 1000,
) -> TrialPlayback:
    """Load a bounded, plot-ready view of one finalized Trial."""

    _log.info("=== load_trial_playback 开始 ===")
    _log.info("manifest_path=%s, data_root=%s", manifest_path, data_root)

    if min(max_signal_points, max_ultrasound_frames, max_ultrasound_depth_points) <= 0:
        raise ValueError("playback limits must be positive")
    path, trial_root, manifest = _load_finalized_trial(manifest_path)
    _log.info("Manifest 已加载: trial_uuid=%s, trial_root=%s", manifest.trial_uuid, trial_root)
    dataset_root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else trial_root
    )
    _require_trial_under_data_root(path, dataset_root)

    def idle_check() -> None:
        _require_idle(dataset_root)

    idle_check()
    formal_t0_ns = manifest.timing.start_host_monotonic_ns
    _log.info("formal_t0_ns=%d", formal_t0_ns)

    ultrasound: UltrasoundPlayback | None = None
    ultrasound_relative = _artifact_for(
        manifest, modality="ultrasound", suffix=".bin"
    )
    _log.info("超声 artifact: %s", ultrasound_relative)
    if ultrasound_relative is not None:
        _log.info("正在加载超声数据…")
        relative_meta, relative_index = companion_paths(ultrasound_relative)
        published_paths = {artifact.relative_path for artifact in manifest.artifacts}
        companion_relatives = (
            relative_meta.as_posix(),
            relative_index.as_posix(),
        )
        missing_companions = set(companion_relatives) - published_paths
        if missing_companions:
            raise DataStudioToolError(
                "超声回放缺少 Manifest 所列 companion Artifact："
                + ", ".join(sorted(missing_companions))
            )
        _log.info("超声 .bin: %s, .meta: %s, .idx: %s",
                  ultrasound_relative, companion_relatives[0], companion_relatives[1])
        ultrasound = _read_ultrasound(
            _artifact_path(trial_root, ultrasound_relative),
            meta_path=_artifact_path(trial_root, companion_relatives[0]),
            index_path=_artifact_path(trial_root, companion_relatives[1]),
            formal_t0_ns=formal_t0_ns,
            max_frames=max_ultrasound_frames,
            max_depth_points=max_ultrasound_depth_points,
            idle_check=idle_check,
        )
        _log.info("超声加载完成: waterfall shape=%s, frames=%d",
                  ultrasound.waterfall.shape, ultrasound.source_frame_count)

    signals: dict[str, SignalPlayback | None] = {
        "imu": None,
        "encoder": None,
        "sync_pulse": None,
        "emg": None,
    }
    sync_trigger_times = np.empty((0,), dtype=np.float64)
    imu_sensors: tuple[SignalPlayback, ...] = ()
    for modality in ("imu", "encoder", "sync_pulse"):
        idle_check()
        relative = _artifact_for(manifest, modality=modality, suffix=".h5")
        _log.info("HDF5 artifact [%s]: %s", modality, relative)
        if relative is None:
            continue
        if modality == "imu":
            sensor_series = _read_hdf5_imu_sensors(
                _artifact_path(trial_root, relative),
                formal_t0_ns=formal_t0_ns,
                max_points=max_signal_points,
            )
            if len(sensor_series) > 1:
                imu_sensors = sensor_series
                signals["imu"] = None
                _log.info(
                    "[imu] 加载完成: %d 颗解耦传感器序列", len(sensor_series)
                )
            else:
                signals["imu"] = sensor_series[0] if sensor_series else None
                if signals["imu"] is not None:
                    _log.info(
                        "[imu] 加载完成: time_s=%d points, values shape=%s",
                        signals["imu"].time_s.size,
                        signals["imu"].values.shape,
                    )
            continue
        series, trigger_times = _read_hdf5_signal(
            _artifact_path(trial_root, relative),
            formal_t0_ns=formal_t0_ns,
            max_points=max_signal_points,
        )
        signals[modality] = series
        _log.info("[%s] 加载完成: time_s=%d points, values shape=%s",
                  modality, series.time_s.size, series.values.shape)
        if modality == "sync_pulse":
            sync_trigger_times = trigger_times

    # EMG is block-binary (.bin + .meta.json + .idx), not HDF5.
    emg_relative = _artifact_for(manifest, modality="emg", suffix=".bin")
    _log.info("EMG artifact: %s", emg_relative)
    if emg_relative is not None:
        idle_check()
        relative_meta, relative_index = companion_paths(emg_relative)
        published_paths = {artifact.relative_path for artifact in manifest.artifacts}
        companion_relatives = (relative_meta.as_posix(), relative_index.as_posix())
        missing_companions = set(companion_relatives) - published_paths
        if missing_companions:
            raise DataStudioToolError(
                "EMG 回放缺少 Manifest 所列 companion Artifact："
                + ", ".join(sorted(missing_companions))
            )
        signals["emg"] = _read_emg(
            _artifact_path(trial_root, emg_relative),
            meta_path=_artifact_path(trial_root, companion_relatives[0]),
            index_path=_artifact_path(trial_root, companion_relatives[1]),
            formal_t0_ns=formal_t0_ns,
            max_points=max_signal_points,
            idle_check=idle_check,
        )
        _log.info("[emg] 加载完成: time_s=%d points, values shape=%s",
                  signals["emg"].time_s.size, signals["emg"].values.shape)

    mocap: MocapPlayback | None = None
    mocap_relative = _artifact_for(manifest, modality="mocap", suffix=".h5")
    if mocap_relative is not None:
        idle_check()
        mocap = _read_hdf5_mocap(
            _artifact_path(trial_root, mocap_relative),
            formal_t0_ns=formal_t0_ns,
            max_points=max_signal_points,
        )
        _log.info(
            "[mocap] 加载完成: frames=%d, markers=%d",
            mocap.time_s.size,
            mocap.positions.shape[1],
        )

    moment: SignalPlayback | None = None
    for candidate_name in ("ground_truth.csv", "moment_truth.csv"):
        moment_path = _artifact_path(trial_root, candidate_name)
        if moment_path.is_file():
            moment = _read_moment_csv(moment_path)
            if moment is not None:
                _log.info(
                    "[moment] 加载完成: %s, values shape=%s",
                    candidate_name,
                    moment.values.shape,
                )
                break

    prompt_labels: tuple[PromptLabelPlaybackEvent, ...] = ()
    prompt_relative = _artifact_for(
        manifest,
        modality="prompt_label",
        suffix=".jsonl",
    )
    if prompt_relative is not None:
        raw_prompt_events = load_prompt_label_events(
            _artifact_path(trial_root, prompt_relative)
        )
        for event in raw_prompt_events:
            if str(event.trial_uuid) != str(manifest.trial_uuid):
                raise DataStudioToolError(
                    "人工标签 Artifact 的 Trial UUID 与 Manifest 不一致。"
                )
        prompt_labels = tuple(
            PromptLabelPlaybackEvent(
                time_s=(event.host_monotonic_ns - formal_t0_ns) / 1e9,
                source=event.source,
                label=event.label,
                key=event.key,
            )
            for event in raw_prompt_events
        )
        _log.info(
            "人工标签加载完成: total=%d subject=%d operator=%d",
            len(prompt_labels),
            sum(
                event.source is PromptLabelSource.SUBJECT
                for event in prompt_labels
            ),
            sum(
                event.source is PromptLabelSource.OPERATOR
                for event in prompt_labels
            ),
        )

    _log.info("=== load_trial_playback 完成 ===")
    return TrialPlayback(
        manifest_path=path,
        trial_uuid=str(manifest.trial_uuid),
        condition_code=manifest.condition.condition_code,
        formal_t0_host_monotonic_ns=formal_t0_ns,
        ultrasound=ultrasound,
        imu=signals["imu"],
        encoder=signals["encoder"],
        sync=signals["sync_pulse"],
        emg=signals["emg"],
        mocap=mocap,
        moment=moment,
        sync_trigger_times_s=sync_trigger_times,
        imu_sensors=imu_sensors,
        prompt_labels=prompt_labels,
    )


def compute_full_statistics(data_root: str | Path) -> FullStatistics:
    """Refresh Manifest/Catalog metadata and derive whole-dataset statistics."""

    root = Path(data_root).expanduser().resolve()
    _require_idle(root)
    snapshot = load_catalog_snapshot(root)
    _require_idle(root)
    subject_count = len(snapshot.tree)
    project_count = session_count = trial_count = artifact_count = artifact_bytes = 0
    by_quality: dict[str, int] = {}
    by_modality: dict[str, dict[str, int]] = {}
    finalized_count = 0

    def _walk(node: dict[str, Any]) -> None:
        nonlocal project_count, session_count, trial_count
        nonlocal artifact_count, artifact_bytes, finalized_count
        node_type = str(node.get("type") or "")
        if node_type == "project":
            project_count += 1
        elif node_type == "session":
            session_count += 1
        elif node_type == "trial":
            trial_count += 1
            state = str(node.get("state") or "UNKNOWN")
            if state == TrialState.FINALIZED.value:
                finalized_count += 1
            quality = str(node.get("quality_grade") or "UNASSESSED")
            by_quality[quality] = by_quality.get(quality, 0) + 1
        elif node_type == "artifact":
            artifact_count += 1
            size = int(node.get("size_bytes") or 0)
            artifact_bytes += size
            modality = str(node.get("modality") or "unknown")
            bucket = by_modality.setdefault(
                modality, {"artifact_count": 0, "size_bytes": 0}
            )
            bucket["artifact_count"] += 1
            bucket["size_bytes"] += size
        for child in node.get("children", []):
            if isinstance(child, dict):
                _walk(child)

    for subject in snapshot.tree:
        _walk(subject)
    statistics = snapshot.statistics
    return FullStatistics(
        projects=project_count,
        subjects=subject_count,
        sessions=session_count,
        trials=trial_count,
        finalized_trials=finalized_count,
        total_duration_s=float(statistics.get("total_duration_s") or 0.0),
        artifact_count=artifact_count,
        artifact_bytes=artifact_bytes,
        by_condition=dict(statistics.get("by_condition") or {}),
        by_quality=dict(sorted(by_quality.items())),
        by_modality=dict(sorted(by_modality.items())),
    )


def _sha256_with_idle_check(path: Path, data_root: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            _require_idle(data_root)
            digest.update(chunk)
    return digest.hexdigest()


def verify_trial_checksums(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> ChecksumReport:
    """Recalculate every published checksum without modifying the Trial."""

    path, trial_root, manifest = _load_finalized_trial(manifest_path)
    dataset_root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else trial_root
    )
    _require_trial_under_data_root(path, dataset_root)
    _require_idle(dataset_root)
    checksum_relative = _internal_relative(path, "checksums.sha256")
    checksum_path = _artifact_path(trial_root, checksum_relative)
    if not checksum_path.is_file() or _has_active_component(checksum_path):
        raise DataStudioToolError("Trial 缺少已发布的 checksums.sha256")

    expected_from_manifest = {
        artifact.relative_path: artifact for artifact in manifest.artifacts
    }
    items: list[ChecksumItem] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        expected, separator, relative_path = line.partition("  ")
        expected = expected.casefold()
        if not separator or len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise DataStudioToolError(
                f"checksums.sha256 第 {line_number} 行格式无效"
            )
        if relative_path in seen:
            raise DataStudioToolError(f"重复校验路径：{relative_path}")
        seen.add(relative_path)
        candidate = _artifact_path(trial_root, relative_path)
        if not candidate.is_file():
            items.append(
                ChecksumItem(
                    relative_path=relative_path,
                    expected_sha256=expected,
                    actual_sha256=None,
                    size_bytes=None,
                    passed=False,
                    message="文件缺失",
                )
            )
            continue
        actual = _sha256_with_idle_check(candidate, dataset_root)
        artifact = expected_from_manifest.get(relative_path)
        size = candidate.stat().st_size
        manifest_consistent = (
            artifact is None
            or (artifact.sha256 == expected and artifact.size_bytes == size)
        )
        passed = actual == expected and manifest_consistent
        if actual != expected:
            message = "SHA-256 不匹配"
        elif not manifest_consistent:
            message = "Manifest 中的摘要或大小不一致"
        else:
            message = "通过"
        items.append(
            ChecksumItem(
                relative_path=relative_path,
                expected_sha256=expected,
                actual_sha256=actual,
                size_bytes=size,
                passed=passed,
                message=message,
            )
        )

    required = set(expected_from_manifest) | {_internal_relative(path, "manifest.json")}
    for missing in sorted(required - seen):
        items.append(
            ChecksumItem(
                relative_path=missing,
                expected_sha256=(
                    expected_from_manifest[missing].sha256
                    if missing in expected_from_manifest
                    else ""
                ),
                actual_sha256=None,
                size_bytes=None,
                passed=False,
                message="checksums.sha256 未覆盖该文件",
            )
        )
    return ChecksumReport(
        manifest_path=path,
        trial_uuid=str(manifest.trial_uuid),
        items=tuple(items),
    )


def _read_small_text(path: Path, *, limit_bytes: int = 5 * 1024 * 1024) -> str:
    size = path.stat().st_size
    if size > limit_bytes:
        raise DataStudioToolError(
            f"报告文件超过 {limit_bytes:,} B 安全限制：{path.name}"
        )
    return path.read_text(encoding="utf-8-sig")


def load_quality_audit(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> QualityAudit:
    """Load published quality summaries without changing raw data or review state."""

    path, trial_root, manifest = _load_finalized_trial(manifest_path)
    dataset_root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else trial_root
    )
    _require_trial_under_data_root(path, dataset_root)
    _require_idle(dataset_root)

    report_document: dict[str, Any] = {}
    quality_relative = _internal_artifact_named(manifest, path, "quality_report.json")
    if quality_relative is not None:
        quality_path = _artifact_path(trial_root, quality_relative)
        if quality_path.is_file():
            loaded = json.loads(_read_small_text(quality_path))
            if not isinstance(loaded, dict):
                raise DataStudioToolError("quality_report.json 根节点必须是对象")
            report_document = loaded
    _require_idle(dataset_root)

    def csv_rows(filename: str) -> tuple[dict[str, str], ...]:
        listed = _internal_artifact_named(manifest, path, filename)
        if listed is None:
            return ()
        report_path = _artifact_path(trial_root, listed)
        if not report_path.is_file():
            return ()
        rows = csv.DictReader(_read_small_text(report_path).splitlines())
        return tuple(dict(row) for row in rows)

    devices = csv_rows("device_status.csv")
    _require_idle(dataset_root)
    sync_checks = csv_rows("sync_check.csv")
    _require_idle(dataset_root)
    warnings_relative = _internal_artifact_named(manifest, path, "warnings.txt")
    warnings_text = ""
    if warnings_relative is not None:
        warnings_path = _artifact_path(trial_root, warnings_relative)
        if warnings_path.is_file():
            warnings_text = _read_small_text(warnings_path)

    manifest_issues = tuple(
        issue.model_dump(mode="json") for issue in manifest.quality.issues
    )
    report_issues = report_document.get("issues")
    issues = (
        tuple(dict(item) for item in report_issues if isinstance(item, dict))
        if isinstance(report_issues, list)
        else manifest_issues
    )
    computed = (
        manifest.quality.computed_grade.value
        if manifest.quality.computed_grade is not None
        else str(report_document.get("computed_grade") or "UNASSESSED")
    )
    reviewed = (
        manifest.quality.reviewed_grade.value
        if manifest.quality.reviewed_grade is not None
        else None
    )
    reviewed_by = manifest.quality.reviewed_by
    reviewed_at_utc = (
        manifest.quality.reviewed_at_utc.isoformat().replace("+00:00", "Z")
        if manifest.quality.reviewed_at_utc is not None
        else None
    )
    review_reason = manifest.quality.review_reason
    review_count = 1 if reviewed is not None else 0
    # Finalized Manifests are immutable. Later human decisions therefore live
    # in an append-only, hash-chained Data Studio ledger anchored to Manifest
    # SHA-256 instead of silently rewriting the original Trial record.
    try:
        from .quality_reviews import list_quality_reviews

        review_records = list_quality_reviews(dataset_root, path)
    except Exception as exc:
        from .quality_reviews import QualityReviewError

        if isinstance(exc, QualityReviewError):
            raise DataStudioToolError(str(exc)) from exc
        raise
    if review_records:
        latest = review_records[-1].record
        reviewed = latest.reviewed_grade.value
        reviewed_by = latest.reviewer
        reviewed_at_utc = latest.reviewed_at_utc.isoformat().replace("+00:00", "Z")
        review_reason = latest.reason
        review_count += len(review_records)
    soft_metrics = report_document.get("soft_metrics")
    return QualityAudit(
        manifest_path=path,
        trial_uuid=str(manifest.trial_uuid),
        computed_grade=computed,
        reviewed_grade=reviewed,
        reviewed_by=reviewed_by,
        reviewed_at_utc=reviewed_at_utc,
        review_reason=review_reason,
        review_count=review_count,
        required_artifacts_complete=manifest.quality.required_artifacts_complete,
        integrity_checks_passed=manifest.quality.integrity_checks_passed,
        algorithm_version=(
            manifest.quality.algorithm_version
            or (
                str(report_document["algorithm_version"])
                if report_document.get("algorithm_version")
                else None
            )
        ),
        issues=issues,
        devices=devices,
        sync_checks=sync_checks,
        warnings_text=warnings_text,
        soft_metrics=dict(soft_metrics) if isinstance(soft_metrics, dict) else {},
    )


def inspect_trial_artifacts(
    manifest_path: str | Path,
    *,
    data_root: str | Path | None = None,
    max_preview_rows: int = 20,
    max_stat_rows: int = 10_000,
) -> TrialInspection:
    """Return a read-only, bounded inspection of every published Artifact.

    This is intentionally generic over modality: HDF5 signal artifacts share
    the :class:`~exo_collection.writers.hdf5_signal.Hdf5SignalWriter` schema, so
    a single reader covers imu/encoder/sync_pulse/mocap/emg alike.  Unlike
    :func:`load_trial_playback`, nothing is downsampled away — sample counts,
    dtype, sample_shape, the ``closed_cleanly`` flag, per-channel statistics,
    and the file structure are all surfaced verbatim.
    """

    if min(max_preview_rows, max_stat_rows) <= 0:
        raise ValueError("inspection limits must be positive")
    path, trial_root, manifest = _load_finalized_trial(manifest_path)
    dataset_root = (
        Path(data_root).expanduser().resolve() if data_root is not None else trial_root
    )
    _require_trial_under_data_root(path, dataset_root)
    _require_idle(dataset_root)

    artifacts: list[ArtifactInspection] = []
    for artifact in manifest.artifacts:
        relative = artifact.relative_path
        try:
            absolute = _artifact_path(trial_root, relative)
        except DataStudioToolError as exc:
            artifacts.append(
                ArtifactInspection(
                    modality=artifact.modality,
                    relative_path=relative,
                    size_bytes=int(artifact.size_bytes or 0),
                    kind="other",
                    message=str(exc),
                )
            )
            continue
        if not absolute.is_file():
            artifacts.append(
                ArtifactInspection(
                    modality=artifact.modality,
                    relative_path=relative,
                    size_bytes=int(artifact.size_bytes or 0),
                    kind="other",
                    message="文件缺失",
                )
            )
            continue
        size_bytes = int(absolute.stat().st_size)
        lowered = relative.casefold()
        if lowered.endswith(".h5"):
            artifacts.append(
                _inspect_hdf5(
                    absolute,
                    modality=artifact.modality,
                    relative_path=relative,
                    size_bytes=size_bytes,
                    max_preview_rows=max_preview_rows,
                    max_stat_rows=max_stat_rows,
                )
            )
        elif lowered.endswith(".bin"):
            artifacts.append(
                _inspect_ultrasound(
                    absolute,
                    trial_root=trial_root,
                    modality=artifact.modality,
                    relative_path=relative,
                    size_bytes=size_bytes,
                )
            )
        elif lowered.endswith(".jsonl"):
            artifacts.append(
                _inspect_jsonl(
                    absolute,
                    modality=artifact.modality,
                    relative_path=relative,
                    size_bytes=size_bytes,
                    max_preview_rows=max_preview_rows,
                )
            )
        else:
            artifacts.append(
                ArtifactInspection(
                    modality=artifact.modality,
                    relative_path=relative,
                    size_bytes=size_bytes,
                    kind="other",
                    message="无内建预览",
                )
            )

    return TrialInspection(
        manifest_path=path,
        trial_uuid=str(manifest.trial_uuid),
        condition_code=manifest.condition.condition_code,
        artifact_count=len(artifacts),
        artifacts=tuple(artifacts),
    )


def _inspect_hdf5(
    path: Path,
    *,
    modality: str,
    relative_path: str,
    size_bytes: int,
    max_preview_rows: int,
    max_stat_rows: int,
) -> ArtifactInspection:
    try:
        with h5py.File(path, "r") as handle:
            closed_cleanly = bool(handle.attrs.get("closed_cleanly", False))
            raw_rate = handle.attrs.get("nominal_rate_hz")
            nominal_rate_hz = float(raw_rate) if raw_rate is not None else None
            root_attrs = tuple(
                (str(key), _attr_to_text(handle.attrs[key]))
                for key in handle.attrs.keys()
            )
            structure = _collect_hdf5_structure(handle)
            discontinuity_count = (
                int(handle["events/discontinuities"].shape[0])
                if "events/discontinuities" in handle
                else 0
            )
            event_count = (
                int(handle["events/records"].shape[0])
                if "events/records" in handle
                else 0
            )
            channels = (
                _decode_strings(handle["metadata/channels"][:])
                if "metadata/channels" in handle
                else ()
            )
            units = (
                _decode_strings(handle["metadata/units"][:])
                if "metadata/units" in handle
                else ()
            )
            device_metadata = _read_json_dataset(handle, "metadata/device")
            trial_metadata = _read_json_dataset(handle, "metadata/trial")
            clock_model = _read_json_dataset(handle, "metadata/clock_model")

            if "samples/data" not in handle:
                return ArtifactInspection(
                    modality=modality,
                    relative_path=relative_path,
                    size_bytes=size_bytes,
                    kind="hdf5",
                    message="缺少 samples/data",
                )
            data = handle["samples/data"]
            count = int(data.shape[0])
            sample_shape = tuple(int(value) for value in data.shape[1:])
            dtype = str(data.dtype)

            if count > 0:
                stat_selector = _even_indices(count, max_stat_rows)
                sampled = np.asarray(data[stat_selector])
                flat = sampled.reshape((sampled.shape[0], -1))
                preview_rows = _build_preview_rows(data, max_preview_rows)
                labels, expanded_units = _flatten_channel_labels(
                    channels, units, int(flat.shape[1])
                )
                stats = _compute_channel_stats(flat, labels, expanded_units)
            else:
                labels, expanded_units = _flatten_channel_labels(
                    channels, units, len(channels) or 1
                )
                stats = ()
                preview_rows = ()

            return ArtifactInspection(
                modality=modality,
                relative_path=relative_path,
                size_bytes=size_bytes,
                kind="hdf5",
                hdf5=Hdf5Inspection(
                    relative_path=relative_path,
                    size_bytes=size_bytes,
                    closed_cleanly=closed_cleanly,
                    sample_count=count,
                    dtype=dtype,
                    sample_shape=sample_shape,
                    nominal_rate_hz=nominal_rate_hz,
                    channels=channels,
                    units=units,
                    device_metadata=device_metadata,
                    trial_metadata=trial_metadata,
                    clock_model=clock_model,
                    root_attrs=root_attrs,
                    structure=structure,
                    stats=stats,
                    preview_columns=labels,
                    preview_rows=preview_rows,
                    discontinuity_count=discontinuity_count,
                    event_count=event_count,
                ),
            )
    except (OSError, KeyError, ValueError) as exc:
        return ArtifactInspection(
            modality=modality,
            relative_path=relative_path,
            size_bytes=size_bytes,
            kind="hdf5",
            message=f"无法读取 HDF5：{exc}",
        )


def _inspect_ultrasound(
    path: Path,
    *,
    trial_root: Path,
    modality: str,
    relative_path: str,
    size_bytes: int,
) -> ArtifactInspection:
    meta_rel, index_rel = companion_paths(relative_path)
    meta_path = _artifact_path(trial_root, meta_rel.as_posix())
    index_path = _artifact_path(trial_root, index_rel.as_posix())
    metadata: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            metadata = loaded
    try:
        with BlockBinaryReader(
            path,
            meta_path=meta_path,
            index_path=index_path,
            validate_crc=False,
            auto_rebuild_index=False,
        ) as reader:
            block_count = int(reader.block_count)
            if not metadata:
                metadata = dict(reader.metadata or {})
    except (OSError, ValueError) as exc:
        return ArtifactInspection(
            modality=modality,
            relative_path=relative_path,
            size_bytes=size_bytes,
            kind="ultrasound",
            message=f"无法读取超声 .bin：{exc}",
        )
    return ArtifactInspection(
        modality=modality,
        relative_path=relative_path,
        size_bytes=size_bytes,
        kind="ultrasound",
        ultrasound=UltrasoundInspection(
            relative_path=relative_path,
            size_bytes=size_bytes,
            meta_path=meta_rel.as_posix(),
            index_path=index_rel.as_posix(),
            metadata=metadata,
            block_count=block_count,
        ),
    )


def _inspect_jsonl(
    path: Path,
    *,
    modality: str,
    relative_path: str,
    size_bytes: int,
    max_preview_rows: int,
) -> ArtifactInspection:
    preview: list[dict[str, Any]] = []
    event_count = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            event_count += 1
            if len(preview) < max_preview_rows:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    preview.append({"_raw": line})
                else:
                    preview.append(
                        payload if isinstance(payload, dict) else {"_value": payload}
                    )
    except OSError as exc:
        return ArtifactInspection(
            modality=modality,
            relative_path=relative_path,
            size_bytes=size_bytes,
            kind="jsonl",
            message=f"无法读取 JSONL：{exc}",
        )
    return ArtifactInspection(
        modality=modality,
        relative_path=relative_path,
        size_bytes=size_bytes,
        kind="jsonl",
        jsonl=JsonlInspection(
            relative_path=relative_path,
            size_bytes=size_bytes,
            event_count=event_count,
            preview=tuple(preview),
        ),
    )


def _collect_hdf5_structure(handle: h5py.File, max_depth: int = 3) -> tuple[str, ...]:
    lines: list[str] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and name.count("/") <= max_depth:
            lines.append(f"{name}  shape={obj.shape}  dtype={obj.dtype}")

    handle.visititems(visit)
    return tuple(lines)


def _read_json_dataset(handle: h5py.File, key: str) -> dict[str, Any]:
    if key not in handle:
        return {}
    raw = handle[key][()]
    if isinstance(raw, np.ndarray) and raw.size == 1:
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _compute_channel_stats(
    flat: NDArray[np.generic],
    labels: tuple[str, ...],
    units: tuple[str, ...],
) -> tuple[Hdf5ChannelStat, ...]:
    column_count = int(flat.shape[1])
    stats: list[Hdf5ChannelStat] = []
    for index in range(column_count):
        column = np.asarray(flat[:, index], dtype=np.float64)
        nan_count = int(np.count_nonzero(np.isnan(column)))
        finite = column[np.isfinite(column)]
        if finite.size:
            minimum = float(np.min(finite))
            maximum = float(np.max(finite))
            mean = float(np.mean(finite))
            std = float(np.std(finite))
        else:
            minimum = maximum = mean = std = float("nan")
        stats.append(
            Hdf5ChannelStat(
                channel=labels[index] if index < len(labels) else f"ch_{index + 1}",
                unit=units[index] if index < len(units) else "",
                min=minimum,
                max=maximum,
                mean=mean,
                std=std,
                nan_count=nan_count,
            )
        )
    return tuple(stats)


def _build_preview_rows(
    data: h5py.Dataset, max_rows: int
) -> tuple[tuple[Any, ...], ...]:
    preview = np.asarray(data[: min(int(data.shape[0]), max_rows)])
    if preview.size == 0:
        return ()
    flat = preview.reshape((preview.shape[0], -1))
    return tuple(
        tuple(_py_scalar(value) for value in row) for row in flat
    )


def _py_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attr_to_text(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


__all__ = [
    "AcquisitionBecameActiveError",
    "ArtifactInspection",
    "ChecksumItem",
    "ChecksumReport",
    "DataStudioToolError",
    "FullStatistics",
    "Hdf5ChannelStat",
    "Hdf5Inspection",
    "JsonlInspection",
    "PromptLabelPlaybackEvent",
    "QualityAudit",
    "SignalPlayback",
    "TrialInspection",
    "TrialPlayback",
    "UltrasoundInspection",
    "UltrasoundPlayback",
    "compute_full_statistics",
    "inspect_trial_artifacts",
    "load_trial_playback",
    "load_quality_audit",
    "verify_trial_checksums",
]
