"""「导出真值数据」的契约测试：IMU 12 通道 + 关节力矩对齐写 CSV。

不 import opensim、不联网；用合成时间轴验证 ``align_ground_truth`` 的插值对齐
与 ``write_ground_truth_csv`` 的表头/行数/数值，锁定训练输入格式。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from exo_collection.apps.calculate.ground_truth import (
    IMU_CHANNEL_NAMES,
    align_ground_truth,
    write_ground_truth_csv,
)


def test_align_interpolates_imu_onto_moment_grid() -> None:
    # 力矩 100 Hz（0~1 s，101 帧）；IMU 120 Hz（0~1 s，121 帧）。
    t_m = np.linspace(0.0, 1.0, 101)
    moments = np.column_stack([np.sin(t_m), np.cos(t_m)])
    t_i = np.linspace(0.0, 1.0, 121)
    # 第 0 通道做成精确线性，便于断言插值值。
    imu = np.zeros((t_i.size, len(IMU_CHANNEL_NAMES)))
    imu[:, 0] = 2.0 * t_i + 1.0

    time_s, imu_aligned, moments_aligned = align_ground_truth(t_m, moments, t_i, imu)

    assert time_s.shape == (101,)
    assert imu_aligned.shape == (101, 12)
    assert moments_aligned.shape == (101, 2)
    # 线性通道插值后应与 2*t+1 一致（容差内）。
    assert np.allclose(imu_aligned[:, 0], 2.0 * time_s + 1.0, atol=1e-6)
    assert np.allclose(moments_aligned[:, 0], np.sin(time_s))


def test_align_restricts_to_imu_coverage() -> None:
    # 力矩时间轴超出 IMU 覆盖范围的部分应被裁掉。
    t_m = np.linspace(-1.0, 2.0, 31)
    moments = np.zeros((31, 1))
    t_i = np.linspace(0.0, 1.0, 13)
    imu = np.ones((13, len(IMU_CHANNEL_NAMES)))

    time_s, imu_aligned, moments_aligned = align_ground_truth(t_m, moments, t_i, imu)

    assert time_s.min() >= 0.0
    assert time_s.max() <= 1.0
    assert time_s.shape[0] < 31


def test_align_sorts_and_dedupes_imu_time() -> None:
    t_m = np.array([0.0, 0.5, 1.0])
    moments = np.zeros((3, 1))
    # 乱序 + 重复时间戳。
    t_i = np.array([1.0, 0.0, 0.0, 0.5])
    imu = np.zeros((4, len(IMU_CHANNEL_NAMES)))
    imu[:, 0] = np.array([10.0, 0.0, 1.0, 5.0])

    time_s, imu_aligned, _ = align_ground_truth(t_m, moments, t_i, imu)
    # 0.0 处去重后取首次出现的 0.0；0.5 处线性插值。
    assert imu_aligned.shape == (3, 12)
    assert imu_aligned[0, 0] == 0.0


def test_align_raises_on_no_overlap() -> None:
    t_m = np.linspace(0.0, 1.0, 11)
    moments = np.zeros((11, 1))
    t_i = np.linspace(5.0, 6.0, 13)
    imu = np.ones((13, len(IMU_CHANNEL_NAMES)))
    with pytest.raises(ValueError):
        align_ground_truth(t_m, moments, t_i, imu)


def test_align_raises_on_wrong_channel_count() -> None:
    t_m = np.linspace(0.0, 1.0, 11)
    moments = np.zeros((11, 1))
    t_i = np.linspace(0.0, 1.0, 13)
    imu = np.ones((13, 3))  # 只有 3 通道，应为 12。
    with pytest.raises(ValueError):
        align_ground_truth(t_m, moments, t_i, imu)


def test_write_csv_header_and_rows(tmp_path: Path) -> None:
    t_m = np.linspace(0.0, 1.0, 11)
    moments = np.column_stack([np.sin(t_m), np.cos(t_m)])
    t_i = np.linspace(0.0, 1.0, 13)
    imu = np.ones((13, len(IMU_CHANNEL_NAMES)))
    moment_names = ["hip_flexion_r", "hip_flexion_l"]

    time_s, imu_aligned, moments_aligned = align_ground_truth(t_m, moments, t_i, imu)
    out = tmp_path / "ground_truth.csv"
    write_ground_truth_csv(out, time_s, imu_aligned, moments_aligned,
                           moment_names=moment_names)

    with out.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)

    assert header == ["time_s", *(f"imu_{n}" for n in IMU_CHANNEL_NAMES),
                      "hip_flexion_r", "hip_flexion_l"]
    assert len(rows) == 11
    assert len(rows[0]) == 1 + 12 + 2
    # 首行时间列可解析为浮点。
    assert float(rows[0][0]) == pytest.approx(0.0)
