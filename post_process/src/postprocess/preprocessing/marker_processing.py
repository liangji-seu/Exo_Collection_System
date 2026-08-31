"""Marker 预处理：把 C3D marker（mm，mocap 全局）转到 OpenSim ground（m）。

坐标变换在 preprocessing 显式完成，write_trc 只负责写盘。
mocap→OpenSim 的轴对应是 BLOCKING（R_mocap_to_opensim 未确认时禁止转换）。
"""

from __future__ import annotations

import numpy as np

from .coordinate_transform import Transform3D
from .units import mm_to_m


def markers_to_opensim_ground(
    markers_mm: np.ndarray,          # (n_frames, n_markers, 3)
    R_mocap_to_opensim: np.ndarray,  # 3x3 旋转（Z-up → Y-up 等）
) -> np.ndarray:
    """旋转到 OpenSim ground 并 mm→m。"""
    markers_mm = np.asarray(markers_mm, dtype=np.float64)
    shape = markers_mm.shape
    flat = markers_mm.reshape(-1, 3)
    rot = flat @ np.asarray(R_mocap_to_opensim, dtype=np.float64).T
    return (rot * mm_to_m(1.0)).reshape(shape)


def transform_markers(markers_mm: np.ndarray, transform: Transform3D) -> np.ndarray:
    """通用：用 Transform3D 转 marker（点），不改变单位。"""
    return transform.apply_position(markers_mm)


__all__ = ["markers_to_opensim_ground", "transform_markers"]
