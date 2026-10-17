"""导出真值数据：IMU 12 通道 + 关节力矩对齐到同一时间轴后写 CSV。

训练输入格式：每一行 = 一个 C3D 时刻的右腿 IMU 特征（12 通道）+ 该时刻的
关节力矩真值。力矩来自 ``viewer/*.npy``（100 Hz，C3D 时间），IMU 来自
``imu.h5``（MTw 每个传感器 nominal 100 Hz），两者通过主机单调时钟都映射到 C3D 时间；这里把 IMU
线性插值到力矩采样点上，保证逐行对齐、可直接 ``pandas.read_csv`` 训练。

本模块不 import opensim、不访问网络，纯 NumPy + 标准库，可在 UI 外单测。
"""

from __future__ import annotations

import csv
import tempfile
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
    *,
    validity_mask: np.ndarray | None = None,
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

    if not np.isfinite(moments_time_s).all() or np.any(np.diff(moments_time_s) <= 0):
        raise ValueError("力矩时间必须有限且严格递增；禁止排序或填补原始时间")
    imu_finite = np.isfinite(imu_time_s)
    imu_time_s, imu_signal = imu_time_s[imu_finite], imu_signal[imu_finite]
    imu_time_s, uniq_idx = np.unique(imu_time_s, return_index=True)
    imu_signal = imu_signal[uniq_idx]

    if imu_time_s.size < 2:
        raise ValueError("有限 IMU 时间样本不足")
    lo, hi = float(imu_time_s.min()), float(imu_time_s.max())
    mask = (moments_time_s >= lo) & (moments_time_s <= hi)
    mask &= np.isfinite(moments).all(axis=1)
    if validity_mask is not None:
        validity_mask = np.asarray(validity_mask)
        if validity_mask.shape != moments_time_s.shape or validity_mask.dtype != np.bool_:
            raise ValueError("有效性 mask 必须为逐帧布尔值")
        mask &= validity_mask
    if not mask.any():
        raise ValueError("IMU 与力矩时间轴无重叠，无法对齐导出")

    time_s = moments_time_s[mask]
    moments_aligned = moments[mask]
    imu_aligned = np.column_stack([
        np.interp(time_s, imu_time_s, imu_signal[:, ch])
        for ch in range(imu_signal.shape[1])
    ])
    # Interpolation may propagate invalid IMU channels. Also do not bridge a missing
    # native IMU interval: retain only immediate finite neighbours at nominal cadence.
    right = np.clip(np.searchsorted(imu_time_s, time_s), 0, len(imu_time_s) - 1)
    left = np.maximum(right - 1, 0)
    exact = np.isclose(imu_time_s[right], time_s, rtol=0, atol=1e-9)
    typical_dt = float(np.median(np.diff(imu_time_s)))
    finite = np.isfinite(imu_aligned).all(axis=1)
    finite &= exact | ((imu_time_s[right] - imu_time_s[left]) <= 1.5 * typical_dt)
    if not finite.any():
        raise ValueError("无有限且连续 IMU 覆盖的有效力矩行")
    return time_s[finite], imu_aligned[finite], moments_aligned[finite]


def write_ground_truth_csv(
    path: Path,
    time_s: np.ndarray,
    imu_aligned: np.ndarray,
    moments: np.ndarray,
    *,
    moment_names: list[str] | None = None,
) -> Path:
    """把对齐结果写成一个 CSV；首行表头为 time + 12 IMU 通道 + 关节力矩列。"""
    if not (np.isfinite(time_s).all() and np.isfinite(imu_aligned).all() and np.isfinite(moments).all()):
        raise ValueError("禁止导出 NaN/Inf 真值")
    if len(time_s) == 0 or len(time_s) != len(imu_aligned) or len(time_s) != len(moments):
        raise ValueError("导出数据为空或行数不一致")
    if moment_names is None:
        moment_names = [f"moment_{i}" for i in range(moments.shape[1])]
    if len(moment_names) != moments.shape[1]:
        raise ValueError("力矩列名数量与数据列数不一致")

    header = ["time_s", *(f"imu_{name}" for name in IMU_CHANNEL_NAMES), *moment_names]
    path = Path(path)
    # Stage the complete CSV beside its destination so a failed write never
    # destroys a previous export and replacement is on the same filesystem.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as fh:
            temporary = Path(fh.name)
            writer = csv.writer(fh)
            writer.writerow(header)
            for i in range(time_s.shape[0]):
                writer.writerow(
                    [f"{time_s[i]:.6f}"]
                    + [f"{v:.6f}" for v in imu_aligned[i]]
                    + [f"{v:.6f}" for v in moments[i]]
                )
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


__all__ = [
    "IMU_CHANNEL_NAMES",
    "align_ground_truth",
    "write_ground_truth_csv",
]


def export_validity_audit(data, exported_time_s: np.ndarray) -> dict:
    """Receipt counts and intervals preserve C3D timestamps, including exclusions."""
    from .direction_policy import EXPORT_VALIDITY_VERSION, GRF_INPUT_POLICY_VERSION
    mask = np.asarray(data.export_mask, dtype=bool)
    times = np.asarray(data.time_s)
    selected = np.isin(times, exported_time_s)
    edges = np.diff(np.r_[False, selected, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return {"export_validity_version": EXPORT_VALIDITY_VERSION,
            "grf_input_policy_version": GRF_INPUT_POLICY_VERSION,
            "excluded_native_wrench_invalid": int((~np.all(data.support_mask, axis=1)).sum()) if data.support_mask is not None else len(times),
            "excluded_analysis_unreviewed": int((~np.all(data.analysis_mask, axis=1)).sum()) if data.analysis_mask is not None else len(times),
            "total_frames": len(times), "valid_analysis_frames": int(mask.sum()),
            "exported_frames": int(len(exported_time_s)),
            "excluded_invalid_or_unreviewed": int((~mask).sum()),
            "excluded_imu_coverage_or_nonfinite": int(mask.sum() - len(exported_time_s)),
            "coverage_fraction": float(len(exported_time_s) / len(times)) if len(times) else 0.0,
            "exported_intervals_s": [[float(times[a]), float(times[b])] for a,b in zip(starts, stops)],
            "note": "仅有效片段，非完整瞬态真值；保留原始时间戳，未填补缺失载荷或间隙"}
