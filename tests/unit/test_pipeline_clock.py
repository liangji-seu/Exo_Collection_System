"""pipeline.synchronization.clock 的纯逻辑测试：传感器定位 + IMU 单传感器信号提取。

不 import opensim、不读真实数据；用 h5py 构造「三传感器独立流交织」的 imu.h5，
锁定 ``imu_sensor_on_c3d_time`` 丢弃其它传感器 NaN 行、只保留目标传感器数据的契约。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

# clock.py 只依赖 numpy + 标准库，把 pipeline 根目录挂进 sys.path 即可 import。
_REPO = Path(__file__).resolve().parents[2]
_PIPELINE = _REPO / "opensim_joint_moment_pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from pipeline.synchronization.clock import (  # noqa: E402
    find_imu_sensor,
    imu_sensor_on_c3d_time,
)

_LABELS = ["imu_left_leg", "imu_right_leg", "imu_pelvis"]


def _write_imu_h5(
    path: Path,
    *,
    n_rows: int = 30,
    labels: tuple[str, ...] = _LABELS,
) -> Path:
    """构造真实布局的 imu.h5：每行只属于一个传感器，其余为 NaN。"""
    n_sensors = len(labels)
    data = np.full((n_rows, n_sensors, 12), np.nan, dtype=np.float32)
    host_ns = (1_000_000_000 + np.arange(n_rows, dtype=np.uint64) * 3_333_333)
    for i in range(n_rows):
        sensor = i % n_sensors
        data[i, sensor, :] = float(i)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("samples/data", data=data)
        handle.create_dataset("samples/host_monotonic_ns", data=host_ns)
        handle.create_dataset(
            "metadata/device", data=json.dumps({"preview_labels": list(labels)})
        )
    return path


def test_find_imu_sensor_locates_right_leg_by_label(tmp_path: Path) -> None:
    path = _write_imu_h5(tmp_path / "imu.h5")
    with h5py.File(path, "r") as handle:
        index, label = find_imu_sensor(handle, side="right")
    assert index == 1
    assert label == "imu_right_leg"


def test_imu_sensor_on_c3d_time_drops_other_sensor_rows(tmp_path: Path) -> None:
    path = _write_imu_h5(tmp_path / "imu.h5")
    with h5py.File(path, "r") as handle:
        index, _ = find_imu_sensor(handle, side="right")
        time_s, signal = imu_sensor_on_c3d_time(
            handle, 1_000_000_000, sensor_index=index, axis_slice=slice(0, 12)
        )

    # 只应保留右腿（sensor 1）的行：30 行中每 3 行一行 → 10 行，且无 NaN。
    assert signal.shape == (10, 12)
    assert not np.isnan(signal).any()
    # 保留行的值应等于其原始行号 i（sensor 1 占 i=1,4,7,…）。
    np.testing.assert_allclose(signal[:, 0], [1.0, 4.0, 7.0, 10.0, 13.0,
                                              16.0, 19.0, 22.0, 25.0, 28.0])
    # 时间轴与信号同步裁剪。
    assert time_s.shape == (10,)
    assert np.all(np.diff(time_s) > 0.0)


def test_imu_sensor_on_c3d_time_respects_axis_slice(tmp_path: Path) -> None:
    path = _write_imu_h5(tmp_path / "imu.h5")
    with h5py.File(path, "r") as handle:
        _, signal = imu_sensor_on_c3d_time(
            handle, 1_000_000_000, sensor_index=1, axis_slice=slice(0, 3)
        )
    assert signal.shape == (10, 3)


def test_imu_sensor_on_c3d_time_empty_when_sensor_absent(tmp_path: Path) -> None:
    # 全 NaN 的传感器（例如该传感器从未收到包）→ 空信号，不抛异常。
    path = _write_imu_h5(tmp_path / "imu.h5")
    with h5py.File(path, "r+") as handle:
        handle["samples/data"][:, 1, :] = np.nan
    with h5py.File(path, "r") as handle:
        time_s, signal = imu_sensor_on_c3d_time(
            handle, 1_000_000_000, sensor_index=1, axis_slice=slice(0, 3)
        )
    assert signal.size == 0
    assert time_s.size == 0
