"""稳态步行分析区间自动检测（prompt6 §3.5）。

从单侧垂直地面反力（FzL 或 FzR，步行时交替接触/摆动）里识别稳定周期步行区，
排除采集开头的跺脚/站立等待、跑步机启动加速与末尾减速，推荐一段包含若干完整
步态周期的连续稳态区间。纯 numpy，不 import opensim，可脱离 UI 单元测试。

思路：
1. 用单脚 Fz 的上升沿（脚跟触地）作为步态事件；
2. 事件间隔（步态周期）在局部窗口内越稳定，越可能是稳态步行；
3. 取「最长的周期稳定连续步段」，首尾各让半个周期并夹到边缘保护带内。

加速/减速段周期单调变化、跺脚段周期杂乱，都会被周期稳定性判据排除；
站立等待段没有接触上升沿，自然不产生步态事件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 默认参数（进入审计结果，prompt6 §3.5 要求阈值集中）。
DEFAULT_CONTACT_THRESHOLD_FRAC = 0.15
DEFAULT_MIN_CYCLES = 3
DEFAULT_CYCLE_JITTER_FRAC = 0.25
DEFAULT_EDGE_GUARD_S = 1.0


@dataclass(frozen=True)
class SteadyStateResult:
    """稳态区间检测结果（JSON 可序列化）。"""

    start_s: float
    end_s: float
    method: str  # "auto" | "manual" | "fallback"
    n_steps: int
    n_cycles: int
    median_step_period_s: float | None
    step_period_cv: float | None
    reason: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": float(self.start_s),
            "end_s": float(self.end_s),
            "method": self.method,
            "n_steps": int(self.n_steps),
            "n_cycles": int(self.n_cycles),
            "median_step_period_s": (
                None if self.median_step_period_s is None else float(self.median_step_period_s)
            ),
            "step_period_cv": (
                None if self.step_period_cv is None else float(self.step_period_cv)
            ),
            "reason": self.reason,
            "params": self.params,
        }


def detect_steady_walking(
    time_s: np.ndarray,
    fz: np.ndarray,
    *,
    contact_threshold_frac: float = DEFAULT_CONTACT_THRESHOLD_FRAC,
    min_cycles: int = DEFAULT_MIN_CYCLES,
    cycle_jitter_frac: float = DEFAULT_CYCLE_JITTER_FRAC,
    edge_guard_s: float = DEFAULT_EDGE_GUARD_S,
) -> SteadyStateResult:
    """检测稳态步行区间，``time_s``/``fz`` 为同长单脚垂直力（同一时间基）。"""
    t = np.asarray(time_s, dtype=np.float64)
    f = np.asarray(fz, dtype=np.float64)
    if t.size < 3 or f.size != t.size:
        return SteadyStateResult(
            start_s=float(t[0]) if t.size else 0.0,
            end_s=float(t[-1]) if t.size else 0.0,
            method="fallback", n_steps=0, n_cycles=0,
            median_step_period_s=None, step_period_cv=None,
            reason="输入过短或长度不一致", params={},
        )

    # 稳健接触阈值：用 p95 而非 max，避免开头跺脚尖峰把阈值抬高。
    robust_max = float(np.nanpercentile(np.abs(f), 95))
    if not np.isfinite(robust_max) or robust_max <= 0:
        return _fallback(t, "无有效力信号", edge_guard_s, contact_threshold_frac,
                         min_cycles, cycle_jitter_frac)
    threshold = float(contact_threshold_frac) * robust_max

    contact = np.isfinite(f) & (f > threshold)
    # 上升沿（False→True）＝脚跟触地事件
    rising = np.flatnonzero(contact[1:] & ~contact[:-1]) + 1
    event_times = t[rising]

    need_edges = int(min_cycles) + 1
    if event_times.size < need_edges:
        return _fallback(
            t,
            f"接触上升沿仅 {event_times.size} 个（需 ≥{need_edges}）",
            edge_guard_s, contact_threshold_frac, min_cycles, cycle_jitter_frac,
        )

    periods = np.diff(event_times)
    median_period = float(np.median(periods))
    if not np.isfinite(median_period) or median_period <= 0:
        return _fallback(t, "步态周期异常", edge_guard_s, contact_threshold_frac,
                         min_cycles, cycle_jitter_frac)

    # 周期稳定掩码：单步周期与全局中位周期偏差 < 比例阈值。
    regular = np.abs(periods - median_period) <= float(cycle_jitter_frac) * median_period

    # 找最长的「连续稳定步段」（长度 ≥ min_cycles 步）。
    best_start = best_end = -1
    run_start = 0
    for i in range(1, len(regular) + 1):
        if i == len(regular) or not regular[i]:
            run_len = i - run_start
            if run_len >= int(min_cycles) and run_len > (best_end - best_start):
                best_start, best_end = run_start, i
            run_start = i + 1 if i < len(regular) else len(regular)

    if best_start < 0:
        return _fallback(
            t,
            f"未找到 ≥{min_cycles} 个周期的稳定连续步段",
            edge_guard_s, contact_threshold_frac, min_cycles, cycle_jitter_frac,
        )

    n_steps = best_end - best_start
    # 事件索引范围 [best_start, best_end] → 覆盖这些步的时间跨度，首尾各让半周期。
    half = median_period / 2.0
    lo = float(event_times[best_start]) - half
    hi = float(event_times[best_end]) + half
    lo = max(lo, float(t[0]) + float(edge_guard_s))
    hi = min(hi, float(t[-1]) - float(edge_guard_s))
    if hi <= lo:
        lo, hi = float(t[0] + edge_guard_s), float(t[-1] - edge_guard_s)

    run_periods = periods[best_start:best_end]
    cv = float(np.std(run_periods) / median_period) if median_period > 0 else None

    return SteadyStateResult(
        start_s=float(lo),
        end_s=float(hi),
        method="auto",
        n_steps=int(n_steps),
        n_cycles=int(n_steps),
        median_step_period_s=round(median_period, 4),
        step_period_cv=None if cv is None else round(cv, 4),
        reason=f"最长稳定步段 {n_steps} 步，周期 {median_period:.3f}s",
        params={
            "contact_threshold_frac": float(contact_threshold_frac),
            "min_cycles": int(min_cycles),
            "cycle_jitter_frac": float(cycle_jitter_frac),
            "edge_guard_s": float(edge_guard_s),
        },
    )


def _fallback(t, reason, edge_guard_s, threshold_frac, min_cycles, jitter_frac) -> SteadyStateResult:
    lo = float(t[0]) + float(edge_guard_s)
    hi = float(t[-1]) - float(edge_guard_s)
    if hi <= lo:
        lo, hi = float(t[0]), float(t[-1])
    return SteadyStateResult(
        start_s=lo, end_s=hi, method="fallback", n_steps=0, n_cycles=0,
        median_step_period_s=None, step_period_cv=None, reason=reason,
        params={
            "contact_threshold_frac": float(threshold_frac),
            "min_cycles": int(min_cycles),
            "cycle_jitter_frac": float(jitter_frac),
            "edge_guard_s": float(edge_guard_s),
        },
    )


__all__ = [
    "DEFAULT_CONTACT_THRESHOLD_FRAC",
    "DEFAULT_MIN_CYCLES",
    "DEFAULT_CYCLE_JITTER_FRAC",
    "DEFAULT_EDGE_GUARD_S",
    "SteadyStateResult",
    "detect_steady_walking",
]
