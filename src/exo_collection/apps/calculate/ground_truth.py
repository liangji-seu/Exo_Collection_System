"""导出真值数据：IMU 12 通道 + 关节力矩对齐到同一时间轴后写 CSV。

训练输入格式：每一行 = 一个 C3D 时刻的右腿 IMU 特征（12 通道）+ 该时刻的
关节力矩真值。力矩来自 ``viewer/*.npy``（100 Hz，C3D 时间），IMU 来自
``imu.h5``（120 Hz），两者通过主机单调时钟都映射到 C3D 时间；这里把 IMU
线性插值到力矩采样点上，保证逐行对齐、可直接 ``pandas.read_csv`` 训练。

本模块不 import opensim、不访问网络，纯 NumPy + 标准库，可在 UI 外单测。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# 与 imu.h5 ``samples/data`` 第 3 维的 12 通道顺序一致：
# 0-2 加速度(m/s²) 3-5 角速度(rad/s) 6-8 磁力计 9-11 roll/pitch/yaw。
IMU_CHANNEL_NAMES = [
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "mag_x", "mag_y", "mag_z",
    "roll", "pitch", "yaw",
]


def align_ground_truth(
    moments_time_s: np.ndarray,
    moments: np.ndarray,
    imu_time_s: np.ndarray,
    imu_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 IMU 12 通道线性插值到力矩时间轴，返回 ``(time_s, imu_aligned, moments)``。

    两个时间轴都必须已是 C3D 时间（IMU 通过 ``clock.imu_sensor_on_c3d_time`` 得到）。
    IMU 时间戳先排序去重保证 ``np.interp`` 单调；只保留落在 IMU 覆盖范围内的力矩帧。
    """
    moments_time_s = np.asarray(moments_time_s, dtype=np.float64)
    moments = np.asarray(moments, dtype=np.float64)
    imu_time_s = np.asarray(imu_time_s, dtype=np.float64)
    imu_signal = np.asarray(imu_signal, dtype=np.float64)

    if imu_signal.ndim != 2 or imu_signal.shape[1] != len(IMU_CHANNEL_NAMES):
        raise ValueError(
            f"IMU 信号应为 (n, {len(IMU_CHANNEL_NAMES)})，实际 {imu_signal.shape}"
        )
    if moments.shape[0] != moments_time_s.shape[0]:
        raise ValueError("力矩行数与时间轴长度不一致")
    if imu_signal.shape[0] != imu_time_s.shape[0]:
        raise ValueError("IMU 行数与时间轴长度不一致")
    if imu_time_s.size < 2:
        raise ValueError("IMU 样本不足，无法插值对齐")

    imu_time_s, uniq_idx = np.unique(imu_time_s, return_index=True)
    imu_signal = imu_signal[uniq_idx]

    lo, hi = float(imu_time_s.min()), float(imu_time_s.max())
    mask = (moments_time_s >= lo) & (moments_time_s <= hi)
    if not mask.any():
        raise ValueError("IMU 与力矩时间轴无重叠，无法对齐导出")

    time_s = moments_time_s[mask]
    moments_aligned = moments[mask]
    imu_aligned = np.column_stack([
        np.interp(time_s, imu_time_s, imu_signal[:, ch])
        for ch in range(imu_signal.shape[1])
    ])
    return time_s, imu_aligned, moments_aligned


def write_ground_truth_csv(
    path: Path,
    time_s: np.ndarray,
    imu_aligned: np.ndarray,
    moments: np.ndarray,
    *,
    moment_names: list[str] | None = None,
) -> Path:
    """把对齐结果写成一个 CSV；首行表头为 time + 12 IMU 通道 + 关节力矩列。"""
    if moment_names is None:
        moment_names = [f"moment_{i}" for i in range(moments.shape[1])]
    if len(moment_names) != moments.shape[1]:
        raise ValueError("力矩列名数量与数据列数不一致")

    header = ["time_s", *(f"imu_{name}" for name in IMU_CHANNEL_NAMES), *moment_names]
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i in range(time_s.shape[0]):
            writer.writerow(
                [f"{time_s[i]:.6f}"]
                + [f"{v:.6f}" for v in imu_aligned[i]]
                + [f"{v:.6f}" for v in moments[i]]
            )
    return path


__all__ = [
    "IMU_CHANNEL_NAMES",
    "align_ground_truth",
    "write_ground_truth_csv",
]
