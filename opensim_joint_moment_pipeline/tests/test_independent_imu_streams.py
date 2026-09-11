import json
from pathlib import Path

import h5py
import numpy as np

from pipeline.synchronization.clock import (
    clock_health,
    imu_sensor_candidates,
    imu_sensor_on_c3d_time,
)


def test_interleaved_independent_stream_uses_selected_sensor_clock(tmp_path: Path):
    path = tmp_path / "imu.h5"
    n = 100
    host = np.arange(n * 3, dtype=np.int64) * 3_333_333
    data = np.full((n * 3, 3, 12), np.nan, dtype=np.float32)
    for sensor in range(3):
        rows = np.arange(sensor, n * 3, 3)
        data[rows, sensor, :] = sensor + np.arange(len(rows), dtype=np.float32)[:, None]
    with h5py.File(path, "w") as f:
        f.create_dataset("samples/host_monotonic_ns", data=host)
        f.create_dataset("samples/data", data=data)

    with h5py.File(path, "r") as f:
        all_health = clock_health(f["samples/host_monotonic_ns"][:])
        selected_t, selected = imu_sensor_on_c3d_time(f, 0, sensor_index=1, axis_slice=slice(0, 12))
        selected_ns = (selected_t * 1e9).astype(np.int64)
        selected_health = clock_health(selected_ns)

    # The merged rows are not a 100 Hz stream; selected sensor is.
    assert np.isclose(1.0 / (all_health.median_period_ns / 1e9), 300.0, rtol=0.01)
    assert selected.shape == (n, 12)
    assert selected_health.n_gaps == 0
    assert np.isclose(1.0 / np.median(np.diff(selected_t)), 100.0, rtol=0.01)


def _write_imu(tmp_path: Path, preview_labels: list[str] | None, n_sensors: int) -> Path:
    path = tmp_path / "imu.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("samples/host_monotonic_ns", data=np.arange(10, dtype=np.int64))
        data = np.full((10, n_sensors, 3), np.nan, dtype=np.float32)
        f.create_dataset("samples/data", data=data)
        meta = {"preview_labels": preview_labels} if preview_labels is not None else {}
        f.create_dataset("metadata/device", data=json.dumps(meta).encode("utf-8"))
    return path


def test_imu_sensor_candidates_returns_preview_labels(tmp_path: Path):
    path = _write_imu(tmp_path, ["imu_left_leg", "imu_right_leg", "imu_pelvis"], 3)
    with h5py.File(path, "r") as f:
        assert imu_sensor_candidates(f) == [
            (0, "imu_left_leg"),
            (1, "imu_right_leg"),
            (2, "imu_pelvis"),
        ]


def test_imu_sensor_candidates_skips_non_imu_and_falls_back_to_count(tmp_path: Path):
    # preview_labels 缺失或没有 imu_ 前缀时，按 samples/data 第 1 维传感器数兜底。
    path = _write_imu(tmp_path, None, 3)
    with h5py.File(path, "r") as f:
        assert imu_sensor_candidates(f) == [(0, "sensor_0"), (1, "sensor_1"), (2, "sensor_2")]
