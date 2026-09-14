"""gait_events 的纯 NumPy 单测：从合成动捕轨迹还原足跟触地 / 足尖离地。

这里用 ``types.SimpleNamespace`` 伪造 ``MocapPlayback``（只有 positions /
marker_names / time_s），避免引入 h5py / Qt，让检测保持可离线单测。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from exo_collection.apps.data_studio.gait_events import detect_gait_events


# -- helpers ---------------------------------------------------------------


_FOOT_NAMES = (
    "R.ASIS", "L.ASIS", "V.Sacral", "R.Heel", "R.Toe", "L.Heel", "L.Toe"
)


def _make_mocap(
    contact_right: np.ndarray,
    contact_left: np.ndarray,
    *,
    fs: float = 100.0,
    prefix: str = "010_no_exo_dynamic",
) -> SimpleNamespace:
    """骨盆固定在 z=900，足 marker 接触时 z=10、离地时 z=200（刚性足）。"""
    n_frames = contact_right.shape[0]
    positions = np.full((n_frames, len(_FOOT_NAMES), 3), np.nan, dtype=np.float64)
    # 骨盆点：高高在上、y 略分开，使垂直轴被推到 z。
    for i in range(3):
        positions[:, i, 0] = 0.0
        positions[:, i, 1] = (i - 1) * 120.0
        positions[:, i, 2] = 900.0
    # 足点：每脚 heel/toe 同 z（刚性），y 分左右。
    for foot_index, contact in ((3, contact_right), (5, contact_left)):
        z = np.where(contact, 10.0, 200.0)
        for marker in (foot_index, foot_index + 1):
            positions[:, marker, 0] = 0.0
            positions[:, marker, 1] = 100.0 if foot_index == 3 else -100.0
            positions[:, marker, 2] = z
    marker_names = tuple(f"{prefix}/{name}" for name in _FOOT_NAMES)
    time_s = np.arange(n_frames, dtype=np.float64) / fs
    return SimpleNamespace(
        positions=positions, marker_names=marker_names, time_s=time_s
    )


def _times(events, side: str, kind: str) -> list[float]:
    return [event.time_s for event in events if event.side == side and event.kind == kind]


# -- tests -----------------------------------------------------------------


def test_detect_gait_events_recovers_heel_strike_and_toe_off() -> None:
    n = 1000
    contact_right = np.zeros(n, dtype=bool)
    contact_right[100:300] = True
    contact_right[500:700] = True
    contact_left = np.zeros(n, dtype=bool)
    contact_left[300:500] = True
    contact_left[700:900] = True

    events = detect_gait_events(_make_mocap(contact_right, contact_left))

    assert _times(events, "right", "heel_strike") == pytest.approx([1.0, 5.0])
    assert _times(events, "right", "toe_off") == pytest.approx([3.0, 7.0])
    assert _times(events, "left", "heel_strike") == pytest.approx([3.0, 7.0])
    assert _times(events, "left", "toe_off") == pytest.approx([5.0, 9.0])
    # 结果按时间升序。
    assert [e.time_s for e in events] == sorted(e.time_s for e in events)


def test_detect_gait_events_matches_markers_by_suffix() -> None:
    n = 400
    contact_right = np.zeros(n, dtype=bool)
    contact_right[100:200] = True
    contact_left = np.zeros(n, dtype=bool)

    prefixed = detect_gait_events(
        _make_mocap(contact_right, contact_left, prefix="042_static")
    )

    assert _times(prefixed, "right", "heel_strike") == pytest.approx([1.0])
    assert _times(prefixed, "right", "toe_off") == pytest.approx([2.0])
    assert _times(prefixed, "left", "heel_strike") == []
    assert _times(prefixed, "left", "toe_off") == []


def test_detect_gait_events_skips_foot_without_toe_marker() -> None:
    n = 400
    contact_right = np.zeros(n, dtype=bool)
    contact_right[100:200] = True
    contact_left = np.zeros(n, dtype=bool)
    contact_left[100:200] = True
    mocap = _make_mocap(contact_right, contact_left)

    # 去掉 R.Toe → 右脚跳过，左脚照常。
    keep = [i for i, name in enumerate(mocap.marker_names) if not name.endswith("R.Toe")]
    mocap.marker_names = tuple(mocap.marker_names[i] for i in keep)
    mocap.positions = mocap.positions[:, keep, :]

    events = detect_gait_events(mocap)

    assert _times(events, "right", "heel_strike") == []
    assert _times(events, "right", "toe_off") == []
    assert _times(events, "left", "heel_strike") == pytest.approx([1.0])
    assert _times(events, "left", "toe_off") == pytest.approx([2.0])


def test_detect_gait_events_forward_fills_occlusion() -> None:
    n = 400
    contact_right = np.zeros(n, dtype=bool)
    contact_right[100:300] = True
    contact_left = np.zeros(n, dtype=bool)
    mocap = _make_mocap(contact_right, contact_left)

    # 在接触段中段遮挡 10 帧（右足 heel/toe 同时 NaN）。
    for marker in (3, 4):
        mocap.positions[200:210, marker, :] = np.nan

    events = detect_gait_events(mocap)

    # 前向填充后仍只有一次触地 / 一次离地，不掉帧伪造事件对。
    assert _times(events, "right", "heel_strike") == pytest.approx([1.0])
    assert _times(events, "right", "toe_off") == pytest.approx([3.0])


def test_detect_gait_events_returns_empty_when_no_foot_markers() -> None:
    # 只有骨盆点、没有足 marker → 无事件。
    positions = np.zeros((100, 3, 3), dtype=np.float64)
    positions[:, :, 2] = 900.0
    mocap = SimpleNamespace(
        positions=positions,
        marker_names=("R.ASIS", "L.ASIS", "V.Sacral"),
        time_s=np.arange(100, dtype=np.float64) / 100.0,
    )
    assert detect_gait_events(mocap) == ()


def test_detect_gait_events_returns_empty_for_degenerate_shapes() -> None:
    mocap = SimpleNamespace(
        positions=np.zeros((1, 4, 3)),
        marker_names=("R.Heel", "R.Toe", "L.Heel", "L.Toe"),
        time_s=np.array([0.0]),
    )
    assert detect_gait_events(mocap) == ()
