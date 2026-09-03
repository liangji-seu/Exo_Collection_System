"""静态试验稳定窗口自动选择（prompt6 §3.6）。

在静态标定 trial 里挑出「marker 完整度最高、速度最低」的连续 2~3 秒，供 Scale
与两遍静态 marker refinement 共用。纯 numpy，不 import opensim，可在 EXO 环境
直接跑、脱离 UI 单元测试。

评分原则（lexicographic，先完整性后稳定性）：
    score = completeness_weight * valid_frac - mean_velocity_mm_s

``completeness_weight`` 取 1000（mm/s 量级）时，一个缺失 marker 造成的完整度下降
（1/19 * 1000 ≈ 52.6）远大于常见速度差异（数十~数百 mm/s），因此只要存在完整度
更高的窗口就不会被「更慢但缺 marker」的窗口抢走；完整度并列时才比谁更慢。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .build_trc import extract_hh19
from .hh19_markers import HH19_MARKERS

# 静态标定期望的 19 点（HH19 协议），顺序与 build_trc.extract_hh19 一致。
EXPECTED_STATIC_MARKERS: tuple[str, ...] = tuple(HH19_MARKERS.keys())

# 默认滑窗参数（进入审计结果，prompt6 §3.6 要求阈值集中）。
DEFAULT_TARGET_DURATION_S = 2.5
DEFAULT_MIN_DURATION_S = 2.0
DEFAULT_EDGE_TRIM_S = 0.5
DEFAULT_COMPLETENESS_WEIGHT = 1000.0


@dataclass(frozen=True)
class StaticWindowResult:
    """静态稳定窗口选择结果（JSON 可序列化）。"""

    start_s: float
    end_s: float
    duration_s: float
    method: str  # "auto" | "manual"
    n_frames: int
    # 稳定度指标（进入审计）
    mean_velocity_mm_s: float | None
    valid_frac: float  # 窗口内 19 点平均完整度（0~1）
    n_markers_present: int
    missing_markers: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": float(self.start_s),
            "end_s": float(self.end_s),
            "duration_s": float(self.duration_s),
            "method": self.method,
            "n_frames": int(self.n_frames),
            "mean_velocity_mm_s": (
                None if self.mean_velocity_mm_s is None
                else float(self.mean_velocity_mm_s)
            ),
            "valid_frac": float(self.valid_frac),
            "n_markers_present": int(self.n_markers_present),
            "missing_markers": list(self.missing_markers),
            "params": self.params,
        }


def _frame_velocity(traj: np.ndarray, rate_hz: float) -> np.ndarray:
    """逐 marker 速度模长 (n_frames, n_markers)；缺失帧 NaN 传播。"""
    vel = np.gradient(traj, axis=0) * float(rate_hz)
    return np.linalg.norm(vel, axis=2)


def select_static_window(
    data,
    *,
    target_duration_s: float = DEFAULT_TARGET_DURATION_S,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    edge_trim_s: float = DEFAULT_EDGE_TRIM_S,
    completeness_weight: float = DEFAULT_COMPLETENESS_WEIGHT,
) -> StaticWindowResult:
    """在静态 C3D 里选出最稳定窗口。

    ``data`` 为 ``read_c3d`` 的 ``C3dData``。返回 ``StaticWindowResult``；
    若 trial 过短或没有足够有效数据，退回「去掉首尾 edge_trim_s 的整段」并如实
    报告缺失 marker（绝不静默伪造一个窗口）。
    """
    rate = float(data.point_rate_hz) if data.point_rate_hz > 0 else 100.0
    n_frames = data.n_frames

    present_names, traj = extract_hh19(data)  # (frames, n_markers, 3)，缺失 NaN
    present_set = set(present_names)
    missing = tuple(m for m in EXPECTED_STATIC_MARKERS if m not in present_set)
    n_present = len(present_names)

    valid = ~np.isnan(traj).any(axis=2)  # (frames, n_markers)
    valid_count = valid.sum(axis=1)  # (frames,)
    vel_mag = _frame_velocity(traj, rate)
    vel_mag = np.where(valid, vel_mag, np.nan)
    frame_vel = np.nanmean(vel_mag, axis=1)  # (frames,)

    target = max(float(target_duration_s), float(min_duration_s))
    win_frames = int(round(target * rate))
    edge = int(round(float(edge_trim_s) * rate))

    # trial 太短 / 有效窗太窄 → 退回整段（去掉首尾），不做滑窗。
    available = n_frames - 2 * edge
    if n_frames < 3 or available < int(round(float(min_duration_s) * rate)) or n_present == 0:
        lo = min(edge, n_frames // 4)
        hi = max(n_frames - lo, lo + 1)
        frac = float(valid_count[lo:hi].mean()) / max(len(EXPECTED_STATIC_MARKERS), 1)
        return StaticWindowResult(
            start_s=float(lo) / rate,
            end_s=float(hi) / rate,
            duration_s=float(hi - lo) / rate,
            method="auto",
            n_frames=int(hi - lo),
            mean_velocity_mm_s=(None if not np.isfinite(frame_vel[lo:hi]).any()
                                else float(np.nanmean(frame_vel[lo:hi]))),
            valid_frac=frac,
            n_markers_present=n_present,
            missing_markers=missing,
            params={
                "target_duration_s": float(target),
                "min_duration_s": float(min_duration_s),
                "edge_trim_s": float(edge_trim_s),
                "completeness_weight": float(completeness_weight),
                "fallback": True,
            },
        )

    win_frames = min(win_frames, available)
    best_score = -np.inf
    best_start = edge
    expected = max(len(EXPECTED_STATIC_MARKERS), 1)
    # 候选起点逐帧扫描（数据量小，O(n)；窗口内用切片求和避免重复遍历）。
    for start in range(edge, n_frames - win_frames - edge + 1):
        end = start + win_frames
        vc = valid_count[start:end]
        frac = float(vc.mean()) / expected
        fv = frame_vel[start:end]
        vel = float(np.nanmean(fv)) if np.isfinite(fv).any() else 0.0
        score = completeness_weight * frac - vel
        if score > best_score:
            best_score = score
            best_start = start

    best_end = best_start + win_frames
    win_frac = float(valid_count[best_start:best_end].mean()) / expected
    win_vel = float(np.nanmean(frame_vel[best_start:best_end])) if np.isfinite(
        frame_vel[best_start:best_end]
    ).any() else None

    return StaticWindowResult(
        start_s=float(best_start) / rate,
        end_s=float(best_end) / rate,
        duration_s=float(win_frames) / rate,
        method="auto",
        n_frames=int(win_frames),
        mean_velocity_mm_s=win_vel,
        valid_frac=win_frac,
        n_markers_present=n_present,
        missing_markers=missing,
        params={
            "target_duration_s": float(target),
            "min_duration_s": float(min_duration_s),
            "edge_trim_s": float(edge_trim_s),
            "completeness_weight": float(completeness_weight),
            "fallback": False,
        },
    )


__all__ = [
    "EXPECTED_STATIC_MARKERS",
    "DEFAULT_TARGET_DURATION_S",
    "DEFAULT_MIN_DURATION_S",
    "DEFAULT_EDGE_TRIM_S",
    "DEFAULT_COMPLETENESS_WEIGHT",
    "StaticWindowResult",
    "select_static_window",
]
