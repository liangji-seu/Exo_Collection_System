"""步态事件提取：从动捕 marker 高度轨迹还原足跟触地 / 足尖离地时刻。

可视化路径只有 ``MocapPlayback``（marker 三维轨迹），没有测力台 GRF。这里
移植 :mod:`opensim_joint_moment_pipeline.pipeline.gait.detect_contact` 的「垂直轴
推导 + 地平面 + 足底接触」逻辑，但**去掉 GRF 校验**（无 Fz 可用），并进一步从
逐帧接触布尔序列提取离散事件：

- 接触起始（上升沿）= 足跟触地（heel strike）；
- 接触结束（下降沿）= 足尖离地（toe off）。

每脚用 ``fmin(heel, toe)`` 作足底高度，对单个 marker 遮挡更稳健；遮挡/缺失帧
向前填充，避免一次掉帧伪造一对事件。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .local_tools import MocapPlayback

# 用于判定垂直轴与接触的 marker（去 subject 前缀后的短名，与 3D 视图一致）
_PELVIS_MARKERS = ("R.ASIS", "L.ASIS", "V.Sacral")
_FOOT_MARKERS = ("R.Heel", "R.Toe", "L.Heel", "L.Toe")


@dataclass(frozen=True, slots=True)
class GaitEvent:
    """A single discrete gait event on the shared playback timeline."""

    time_s: float
    side: str  # "right" | "left"
    kind: str  # "heel_strike" | "toe_off"


def _name_to_index(marker_names: tuple[str, ...]) -> dict[str, int]:
    """短名（去 ``前缀/``，大小写不敏感）→ marker 列号。"""
    return {
        name.casefold().rsplit("/", 1)[-1]: index
        for index, name in enumerate(marker_names)
    }


def _marker_xyz(positions: np.ndarray, index: int) -> np.ndarray:
    """返回单个 marker 的 (n_frames, 3) 轨迹，遮挡/缺失帧置 NaN。"""
    traj = np.asarray(positions[:, index, :], dtype=np.float64)
    missing = ~np.isfinite(traj).any(axis=1) | (np.abs(traj) < 1e-6).all(axis=1)
    traj[missing] = np.nan
    return traj


def _detect_vertical_axis(
    positions: np.ndarray, name_to_index: dict[str, int]
) -> int:
    """从骨盆点与足点质心差最大的轴推导垂直轴（0/1/2 → X/Y/Z）。"""
    pelvis: list[np.ndarray] = []
    foot: list[np.ndarray] = []
    for name in _PELVIS_MARKERS:
        index = name_to_index.get(name.casefold())
        if index is not None:
            pelvis.append(_marker_xyz(positions, index))
    for name in _FOOT_MARKERS:
        index = name_to_index.get(name.casefold())
        if index is not None:
            foot.append(_marker_xyz(positions, index))

    if pelvis and foot:
        p = np.nanmedian(np.concatenate([t for t in pelvis], axis=0), axis=0)
        f = np.nanmedian(np.concatenate([t for t in foot], axis=0), axis=0)
        return int(np.argmax(np.abs(p - f)))

    all_pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    all_pts = all_pts[np.isfinite(all_pts).all(axis=1)]
    if all_pts.size == 0:
        return 2
    span = np.nanmax(all_pts, axis=0) - np.nanmin(all_pts, axis=0)
    return int(np.argmax(span))


def _ground_level(z_values: np.ndarray, percentile: float = 1.0) -> float:
    """地平面高度 = 足部 marker 垂直值的低百分位（默认 1%）。"""
    valid = z_values[np.isfinite(z_values)]
    if valid.size == 0:
        return float("nan")
    return float(np.nanpercentile(valid, percentile))


def _forward_fill_contact(contact: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """把接触状态跨遮挡帧向前填充，避免一次掉帧伪造一对事件。"""
    filled = contact.astype(bool, copy=True)
    valid_index = np.flatnonzero(valid)
    if valid_index.size == 0:
        return filled
    first = int(valid_index[0])
    filled[:first] = filled[first]
    last_valid = np.maximum.accumulate(
        np.where(valid, np.arange(len(valid)), -1)
    )
    return filled[last_valid]


def _debounce(times: np.ndarray, min_gap_s: float) -> np.ndarray:
    """同类事件间保留至少 ``min_gap_s`` 的间隔（贪心保留）。"""
    if times.size == 0:
        return times
    kept = [float(times[0])]
    for value in times[1:]:
        if float(value) - kept[-1] >= min_gap_s:
            kept.append(float(value))
    return np.asarray(kept, dtype=np.float64)


def detect_gait_events(
    mocap: MocapPlayback,
    *,
    foot_height_threshold_mm: float = 30.0,
    min_gap_s: float = 0.25,
) -> tuple[GaitEvent, ...]:
    """从动捕 marker 轨迹提取左右脚足跟触地 / 足尖离地事件（按时间升序）。

    Parameters
    ----------
    mocap:
        已降采样、bounded 的 marker 回放数据（``positions`` 毫米，``time_s`` 秒）。
    foot_height_threshold_mm:
        足底距地平面小于此值即判定接触。
    min_gap_s:
        同脚同类事件的最小间隔，用于去抖。
    """
    positions = np.asarray(mocap.positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[0] < 2 or positions.shape[1] == 0:
        return ()
    name_to_index = _name_to_index(mocap.marker_names)
    axis = _detect_vertical_axis(positions, name_to_index)

    foot_z: dict[str, np.ndarray | None] = {}
    for name in _FOOT_MARKERS:
        index = name_to_index.get(name.casefold())
        foot_z[name] = _marker_xyz(positions, index)[:, axis] if index is not None else None

    available = [z for z in foot_z.values() if z is not None]
    if not available:
        return ()
    ground = _ground_level(np.concatenate(available))
    if not np.isfinite(ground):
        return ()

    events: list[GaitEvent] = []
    for side, heel_name, toe_name in (
        ("right", "R.Heel", "R.Toe"),
        ("left", "L.Heel", "L.Toe"),
    ):
        heel_z = foot_z[heel_name]
        toe_z = foot_z[toe_name]
        if heel_z is None or toe_z is None:
            continue
        foot_min = np.fmin(heel_z, toe_z)
        valid = np.isfinite(foot_min)
        contact = (foot_min - ground) < foot_height_threshold_mm
        contact = _forward_fill_contact(contact, valid)

        rising = np.flatnonzero(contact[1:] & ~contact[:-1]) + 1
        falling = np.flatnonzero(~contact[1:] & contact[:-1]) + 1
        for value in _debounce(mocap.time_s[rising], min_gap_s):
            events.append(GaitEvent(time_s=value, side=side, kind="heel_strike"))
        for value in _debounce(mocap.time_s[falling], min_gap_s):
            events.append(GaitEvent(time_s=value, side=side, kind="toe_off"))

    events.sort(key=lambda event: event.time_s)
    return tuple(events)


__all__ = ["GaitEvent", "detect_gait_events"]
