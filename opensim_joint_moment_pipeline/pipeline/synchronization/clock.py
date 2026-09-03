"""mocap.h5 / imu.h5 公共主机时钟对齐与 IMU 传感器定位。

两个 H5 的每个样本都带 ``samples/host_monotonic_ns``（同一台主机单调时钟），
对齐方式是**绝对时钟相减**，而不是把各自第一帧都归零后叠加——那样会丢掉
两个设备真正的启动先后。

uint64 时间戳直接相减在「后到样本时间戳反而更小」时会溢出成巨大正数，
因此一律先转成 ``int64`` 再差分。
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np


@dataclass(frozen=True)
class ClockHealth:
    n_samples: int
    median_period_ns: float
    n_decreasing: int            # 时间戳回退的样本数（含等值）
    n_gaps: int                  # 相邻周期明显异常的间断数
    max_gap_ns: float            # 最大相邻间隔

    @property
    def monotonic(self) -> bool:
        return self.n_decreasing == 0


def read_host_monotonic_ns(handle) -> np.ndarray:
    """读取 ``samples/host_monotonic_ns`` 并转为安全的 ``int64``。"""
    raw = handle["samples/host_monotonic_ns"][:]
    if raw.dtype == np.uint64:
        return raw.astype(np.int64)
    return raw.astype(np.int64)


def clock_health(times_ns: np.ndarray) -> ClockHealth:
    """检查时间戳单调性与采样周期（``times_ns`` 为 int64 纳秒）。"""
    t = np.asarray(times_ns, dtype=np.int64)
    if t.size < 2:
        return ClockHealth(int(t.size), float("nan"), 0, 0, 0.0)
    diff = np.diff(t)
    median = float(np.median(diff))
    n_decreasing = int(np.sum(diff <= 0))
    # 周期偏差超过中位数 20% 视为一次间断（采样周期异常或丢帧）。
    threshold = max(abs(median) * 0.2, 1.0)
    n_gaps = int(np.sum(np.abs(diff - median) > threshold))
    return ClockHealth(
        n_samples=int(t.size),
        median_period_ns=median,
        n_decreasing=n_decreasing,
        n_gaps=n_gaps,
        max_gap_ns=float(diff.max()),
    )


def _device_metadata(handle) -> dict:
    raw = handle["metadata/device"][()]
    if isinstance(raw, (bytes, bytearray)):
        return json.loads(raw.decode("utf-8"))
    return raw


def find_imu_sensor(handle, *, side: str = "right") -> tuple[int, str]:
    """返回「右/左腿」IMU 在 ``samples/data`` 第 1 维的下标及其标签。

    依据 ``metadata/device.preview_labels``（形如 ``imu_right_leg``）定位，
    不再硬编码为数组下标 1。找不到对应标签时按 ``sensor_slots`` 顺序兜底
    返回第 2 个传感器（历史布局），并显式返回 ``<fallback>`` 标签供审计。
    """
    meta = _device_metadata(handle)
    labels = [str(x).casefold() for x in meta.get("preview_labels", [])]
    needle = f"imu_{side}_leg"
    for idx, label in enumerate(labels):
        if needle in label:
            return idx, str(meta.get("preview_labels", [])[idx])

    # 兜底：物理位置映射 / sensor_slots 也找一遍
    mapping = meta.get("physical_location_mapping") or {}
    if isinstance(mapping, dict):
        for slot, location in mapping.items():
            loc = str(location).casefold()
            if f"{side}" in loc and "leg" in loc:
                slots = [str(x) for x in meta.get("sensor_slots", [])]
                if slot in slots:
                    return slots.index(slot), str(location)
    # 无明确标签 → 兜底第 2 个传感器（历史：index 1 = 右腿）
    return 1, "<fallback>"


def imu_sensor_on_c3d_time(
    imu_handle,
    c3d_t0_host_ns: int,
    *,
    sensor_index: int,
    axis_slice: slice = slice(0, 3),
) -> tuple[np.ndarray, np.ndarray]:
    """把某 IMU 传感器信号映射到 C3D 时间轴。

    返回 ``(time_s, signal)``，``time_s = (host_monotonic_ns - c3d_t0_host_ns)/1e9``，
    ``signal`` 为 ``samples/data[:, sensor_index, axis_slice]``。
    """
    host_ns = read_host_monotonic_ns(imu_handle)
    time_s = (host_ns - int(c3d_t0_host_ns)) / 1e9
    signal = imu_handle["samples/data"][:, sensor_index, axis_slice]
    return time_s, np.asarray(signal, dtype=np.float64)


def imu_sample_rate_hz(time_s: np.ndarray) -> float:
    t = np.asarray(time_s, dtype=np.float64)
    if t.size < 2:
        return float("nan")
    period = float(np.median(np.diff(t)))
    return float(1.0 / period) if period > 0 else float("nan")


__all__ = [
    "ClockHealth",
    "clock_health",
    "find_imu_sensor",
    "imu_sample_rate_hz",
    "imu_sensor_on_c3d_time",
    "read_host_monotonic_ns",
]
