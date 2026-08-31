"""时间同步：只检查点/模拟采样率的整数倍关系，**不擅自重采样**。

当前 OpenSim ID 只需要 marker ↔ GRF 严格同步；未来 US/IMU/encoder 通过
hardware trigger / timestamp / known offset 映射到 mocap 主时间轴。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyncInfo:
    point_rate_hz: float
    analog_rate_hz: float
    ratio: float | None
    integer_ratio: bool
    marker_time_range_s: tuple[float, float] | None
    force_time_range_s: tuple[float, float] | None

    def is_aligned(self) -> bool:
        """点/模拟同帧编号是否可直接对齐（ratio==1）。"""
        return self.ratio is not None and abs(self.ratio - 1.0) < 1e-6


def check_rate_alignment(point_rate_hz: float, analog_rate_hz: float,
                         n_frames: int) -> SyncInfo:
    pr, ar = float(point_rate_hz), float(analog_rate_hz)
    ratio = (ar / pr) if (pr > 0 and ar > 0) else None
    integer = bool(ratio is not None and abs(ratio - round(ratio)) < 1e-6)
    marker_dur = n_frames / pr if pr > 0 else None
    force_dur = n_frames / ar if ar > 0 else None
    return SyncInfo(pr, ar, ratio, integer,
                    (0.0, marker_dur) if marker_dur is not None else None,
                    (0.0, force_dur) if force_dur is not None else None)


__all__ = ["SyncInfo", "check_rate_alignment"]
