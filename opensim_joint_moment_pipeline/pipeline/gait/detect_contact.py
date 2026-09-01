"""步态接触检测：判定每帧左右脚是否接触地面。

单块跑步机测力台下，**无法**直接得到左右脚独立的 GRF。这里用两个信号融合判断：

1. **足部 marker 高度**（相对地平面的高度）：Heel / Toe 靠近地面 → 该脚着地。
2. **总垂直 GRF（Fz_total）**：Fz 高于阈值 → 至少有一只脚着地（验证器）。

垂直轴（哪个轴是"向上"）**不硬编码 Z-up**，而是从静态 trial 的 marker
几何自动推导：骨盆点（ASIS/Sacral）与足点（Heel/Toe）差异最大的轴即垂直轴。

输出每帧的 ``right_contact`` / ``left_contact`` 布尔序列，供
:mod:`detect_single_support` 与 :mod:`build_support_mask` 使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..c3d.reader import C3dData

# 用于判定垂直轴与接触的 marker（去 subject 前缀后的短名）
_PELVIS_MARKERS = ("R.ASIS", "L.ASIS", "V.Sacral")
_FOOT_MARKERS = ("R.Heel", "R.Toe", "L.Heel", "L.Toe")


def _short_name(data: C3dData, label: str) -> str:
    for s in data.subjects:
        if label.startswith(s.prefix):
            return label[len(s.prefix):]
    return label


def _find_marker_index(data: C3dData, short_name: str) -> int | None:
    for i, label in enumerate(data.point_labels):
        if _short_name(data, label) == short_name:
            return i
    return None


def _marker_trajectory(data: C3dData, short_name: str) -> np.ndarray:
    """返回 (n_frames, 3) 的 marker 轨迹；找不到抛 KeyError。"""
    idx = _find_marker_index(data, short_name)
    if idx is None:
        raise KeyError(f"marker {short_name!r} 不存在于 C3D")
    traj = data.points_mm[:, idx, :].astype(np.float64)
    # 遮挡/缺失帧：C3D 用 0 或 NaN 占位，统一置 NaN
    missing = np.all(np.abs(traj) < 1e-6, axis=1) | np.isnan(traj).any(axis=1)
    traj[missing] = np.nan
    return traj


def detect_vertical_axis(data: C3dData) -> int:
    """从静态 trial 推导 mocap 全局的垂直轴（0/1/2 -> X/Y/Z）。

    用骨盆点（ASIS/Sacral）与足点（Heel/Toe）质心差最大的轴作为垂直轴。
    找不到骨盆/足点时退化为各点总跨度最大的轴。
    """
    pelvis, foot = [], []
    for name in _PELVIS_MARKERS:
        try:
            pelvis.append(_marker_trajectory(data, name))
        except KeyError:
            pass
    for name in _FOOT_MARKERS:
        try:
            foot.append(_marker_trajectory(data, name))
        except KeyError:
            pass

    if pelvis and foot:
        p = np.nanmedian(np.concatenate([t for t in pelvis], axis=0), axis=0)
        f = np.nanmedian(np.concatenate([t for t in foot], axis=0), axis=0)
        return int(np.argmax(np.abs(p - f)))

    # 退化：取所有点总跨度最大的轴
    all_pts = data.points_mm.reshape(-1, 3)
    rng = np.nanmax(all_pts, axis=0) - np.nanmin(all_pts, axis=0)
    return int(np.argmax(rng))


def _ground_level(z_values: np.ndarray, percentile: float = 1.0) -> float:
    """地平面高度 = 足部 marker 在垂直轴上的低百分位（默认 1%）。"""
    valid = z_values[np.isfinite(z_values)]
    if valid.size == 0:
        return float("nan")
    return float(np.nanpercentile(valid, percentile))


@dataclass
class ContactResult:
    right_contact: np.ndarray   # (n_frames,) bool
    left_contact: np.ndarray    # (n_frames,) bool
    any_contact: np.ndarray     # (n_frames,) bool（用于验证：Fz 是否 > 阈值）
    vertical_axis: int
    ground_z_mm: float
    config: dict


def detect_contacts(
    data: C3dData,
    *,
    vertical_axis: int | None = None,
    force_threshold_N: float = 50.0,
    foot_height_threshold_mm: float = 30.0,
    total_fz: np.ndarray | None = None,
) -> ContactResult:
    """融合足部高度 + 总垂直 GRF，输出左右脚接触布尔序列。

    Parameters
    ----------
    data:
        已解析的动态 trial。
    vertical_axis:
        垂直轴（0/1/2）。为 None 时自动推导。
    force_threshold_N:
        Fz_total 低于此值认为"无人着地"。
    foot_height_threshold_mm:
        足部 marker 距地平面小于此值即判定为接触。
    total_fz:
        总垂直 GRF（N）。为 None 时从 C3D analog 里自动取 Fz 通道。
    """
    axis = vertical_axis if vertical_axis is not None else detect_vertical_axis(data)

    # 足部 marker 在垂直轴上的高度
    heel_r = _marker_trajectory(data, "R.Heel")[:, axis]
    toe_r = _marker_trajectory(data, "R.Toe")[:, axis]
    heel_l = _marker_trajectory(data, "L.Heel")[:, axis]
    toe_l = _marker_trajectory(data, "L.Toe")[:, axis]

    # 地平面：左右脚共用全局最小值附近（单块跑台，地面平坦）
    ground = _ground_level(np.concatenate([heel_r, toe_r, heel_l, toe_l]), percentile=1.0)

    right_foot_min = np.fmin(heel_r, toe_r)
    left_foot_min = np.fmin(heel_l, toe_l)

    right_contact = (right_foot_min - ground) < foot_height_threshold_mm
    left_contact = (left_foot_min - ground) < foot_height_threshold_mm

    # 总 GRF 验证：无人着地时强制 NO_CONTACT
    if total_fz is None:
        total_fz = _extract_total_fz(data)
    any_contact = np.abs(total_fz) > force_threshold_N
    # Fz 有效（有有限值）时才用它做 gate；全 NaN 时退回纯 marker 高度判定
    if np.isfinite(total_fz).any():
        right_contact = right_contact & any_contact
        left_contact = left_contact & any_contact

    return ContactResult(
        right_contact=right_contact,
        left_contact=left_contact,
        any_contact=any_contact,
        vertical_axis=axis,
        ground_z_mm=ground,
        config={
            "vertical_axis": axis,
            "force_threshold_N": force_threshold_N,
            "foot_height_threshold_mm": foot_height_threshold_mm,
        },
    )


def _extract_total_fz(data: C3dData) -> np.ndarray:
    """从 analog 通道里取总垂直力 Fz（返回 (n_frames,) N）。"""
    from ..c3d.extract_forces import classify_channel

    for i, label in enumerate(data.analog_labels):
        c = classify_channel(label)
        if c.kind == "Fz" and c.side == "total":
            return data.analogs[:, i].astype(np.float64)
    return np.full(data.n_frames, np.nan)


__all__ = [
    "ContactResult",
    "detect_contacts",
    "detect_vertical_axis",
    "_extract_total_fz",
]
