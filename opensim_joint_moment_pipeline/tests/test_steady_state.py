"""稳态步行区间自动检测（prompt6 §3.6 → §3.5）单元测试。

用合成单脚 Fz 序列验证：开头跺脚 / 站立等待 / 启动加速与末尾减速不会被纳入
推荐分析区间，只有周期稳定的连续步段被选中。
"""

from __future__ import annotations

import numpy as np

from pipeline.gait.steady_state import detect_steady_walking


def _half_sine_contact(t: np.ndarray, strike_times, stance_dur_s: float, amplitude: float) -> np.ndarray:
    """单脚垂直力：每次脚跟触地后一个半正弦接触段（后跟离地回落 0）。"""
    fz = np.zeros_like(t, dtype=np.float64)
    dt = float(np.median(np.diff(t)))
    n = max(int(round(stance_dur_s / dt)), 1)
    for ts in strike_times:
        idx = int(np.argmin(np.abs(t - ts)))
        for k in range(n):
            j = idx + k
            if j < len(t):
                fz[j] = amplitude * np.sin(np.pi * (k + 1) / n)
    return fz


def test_detect_steady_walking_excludes_leading_stomps_and_startup() -> None:
    rate = 100.0
    t = np.arange(0.0, 30.0, 1.0 / rate)

    # 采集开头 3 次跺脚（间隔不规则，只有 2 个周期，构不成稳定步段）
    stomps = _half_sine_contact(t, [0.4, 1.3, 2.1], stance_dur_s=0.3, amplitude=1400.0)
    # 站立等待（无接触）0s~8s 之间只有跺脚；随后连续稳态步行 8~20s（周期 1.0s）
    steady_strikes = [8.0 + i for i in range(12)]
    steady = _half_sine_contact(t, steady_strikes, stance_dur_s=0.6, amplitude=700.0)
    # 末尾减速：周期逐步拉长（21s/22.6s/24.6s…）
    decel = _half_sine_contact(t, [21.0, 22.6, 24.6], stance_dur_s=0.6, amplitude=700.0)

    fz = stomps + steady + decel
    result = detect_steady_walking(t, fz, edge_guard_s=1.0)

    assert result.method == "auto"
    assert result.n_steps >= 11
    # 开头跺脚（<2s）与减速段（>20.5s）被排除，窗口落在稳态步行区。
    assert result.start_s >= 7.0
    assert result.end_s <= 20.5
    # 周期稳定性指标有效。
    assert result.median_step_period_s is not None
    assert result.step_period_cv is not None
    assert result.step_period_cv < 0.15


def test_detect_steady_walking_fallback_on_few_events() -> None:
    t = np.arange(0.0, 5.0, 0.01)
    fz = _half_sine_contact(t, [0.5, 1.5], stance_dur_s=0.5, amplitude=700.0)
    result = detect_steady_walking(t, fz)
    assert result.method == "fallback"
    assert result.n_steps == 0
    assert "上升沿" in result.reason


def test_detect_steady_walking_no_signal_fallback() -> None:
    t = np.arange(0.0, 5.0, 0.01)
    result = detect_steady_walking(t, np.zeros_like(t))
    assert result.method == "fallback"
    assert result.n_steps == 0


def test_detect_steady_walking_params_enter_result() -> None:
    t = np.arange(0.0, 10.0, 0.01)
    steady = _half_sine_contact(t, [1.0 + i for i in range(8)], stance_dur_s=0.6, amplitude=700.0)
    result = detect_steady_walking(t, steady, min_cycles=3, cycle_jitter_frac=0.25)
    assert result.method == "auto"
    # 阈值进入审计结果（prompt6 要求阈值集中并进入审计）。
    assert result.params["min_cycles"] == 3
    assert result.params["cycle_jitter_frac"] == 0.25
    assert "contact_threshold_frac" in result.params
