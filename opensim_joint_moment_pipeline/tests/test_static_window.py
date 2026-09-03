"""静态稳定窗口选择（prompt6 §3.6）单元测试。

用合成的 19 点静态 marker 轨迹验证：选择避开「进入站位 / 调整姿势 / 离开站位」
的移动段，落在 marker 速度最低、完整度最高的中间段；缺失 marker 如实报告。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipeline.opensim_io.hh19_markers import HH19_MARKERS
from pipeline.opensim_io.static_window import select_static_window


def _static_data(
    n_frames: int = 500,
    rate: float = 100.0,
    *,
    drop_markers: tuple[str, ...] = (),
) -> SimpleNamespace:
    """合成静态 trial：中间段静止，首尾各有一段「进入/离开站位」的移动。"""
    names = [m for m in HH19_MARKERS if m not in drop_markers]
    n = len(names)
    base = np.array(
        [[200.0 + 60 * i, 300.0 + 40 * i, 500.0 + 50 * i] for i in range(n)],
        dtype=np.float64,
    )
    traj = np.tile(base[None, :, :], (n_frames, 1, 1)).astype(np.float64)
    # 首 0.5s 与末 0.5s 施加漂移（进入/离开站位），中间完全静止。
    edge = int(round(0.5 * rate))
    if edge > 0:
        traj[:edge, :, 0] += np.linspace(0.0, 250.0, edge)[:, None]
        traj[-edge:, :, 0] += np.linspace(0.0, -250.0, edge)[:, None]
    return SimpleNamespace(
        point_rate_hz=rate,
        n_frames=n_frames,
        point_labels=tuple(names),
        points_mm=traj,
        subjects=(),
    )


def test_select_static_window_avoids_moving_edges() -> None:
    data = _static_data(n_frames=500, rate=100.0)
    result = select_static_window(data, target_duration_s=2.5, min_duration_s=2.0, edge_trim_s=0.5)

    assert result.method == "auto"
    # 避开首尾各 0.5s 的移动段，窗口落在中间静止区。
    assert result.start_s >= 0.49
    assert result.end_s <= 4.51
    # 静止区内速度近似为 0（漂移段速度约 250/0.5 = 500 mm/s，会被排除）。
    assert result.mean_velocity_mm_s is not None
    assert result.mean_velocity_mm_s < 20.0
    assert result.valid_frac == 1.0
    assert result.missing_markers == ()


def test_select_static_window_reports_missing_marker() -> None:
    data = _static_data(n_frames=500, rate=100.0, drop_markers=("R.Shank",))
    result = select_static_window(data, target_duration_s=2.5, min_duration_s=2.0, edge_trim_s=0.5)

    assert "R.Shank" in result.missing_markers
    assert result.n_markers_present == 18
    assert result.valid_frac < 1.0


def test_select_static_window_fallback_on_short_trial() -> None:
    # 只有 0.5s，比 min_duration 短 → 退回「去掉首尾的整段」并如实报告 fallback。
    data = _static_data(n_frames=50, rate=100.0)
    result = select_static_window(data, target_duration_s=2.5, min_duration_s=2.0, edge_trim_s=0.5)

    assert result.method == "auto"
    assert result.params["fallback"] is True
    assert result.duration_s > 0.0


def test_select_static_window_params_enter_result() -> None:
    data = _static_data(n_frames=500, rate=100.0)
    result = select_static_window(data, target_duration_s=2.5, min_duration_s=2.0, edge_trim_s=0.5)
    assert result.params["target_duration_s"] == 2.5
    assert result.params["min_duration_s"] == 2.0
    assert result.params["edge_trim_s"] == 0.5
