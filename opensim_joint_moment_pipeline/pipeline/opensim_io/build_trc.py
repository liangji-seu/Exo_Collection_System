"""从 C3D 提取 HH19 marker 并写成 OpenSim ground 帧的 TRC。

流程：
1. 按短名抽取 HH19 真实 marker（剔除 V_* 虚拟 marker / 静态副本）；
2. mocap 全局 → OpenSim ground（``transforms.mocap_to_opensim_points``）；
3. 缺失帧（C3D 的 [0,0,0] 占位）→ NaN，避免 IK 误判到原点；
4. 写盘（units=mm，OpenSim 读入时转 m）。

动态 C3D 会内嵌静态 subject 的同名全零 marker，抽取时按「有效帧数最多」去重。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..transforms import mocap_to_opensim_points
from ..filtering import lowpass_zero_phase
from .hh19_markers import HH19_MARKERS
from .write_trc import write_trc


def _short(label: str, data) -> str:
    for s in data.subjects:
        if label.startswith(s.prefix):
            return label[len(s.prefix):]
    return label


def _missing_mask(points_mm: np.ndarray) -> np.ndarray:
    p = points_mm.astype(np.float64)
    return np.all(np.abs(p) < 1e-6, axis=2) | np.isnan(p).any(axis=2)


def extract_hh19(data) -> tuple[list[str], np.ndarray]:
    """返回 (marker_names, traj_mm) ，traj 为 (n_frames, n_markers, 3) 的 mocap 帧数据。

    marker 顺序固定为 HH19_MARKERS 的 key 顺序；缺失帧置 NaN。
    """
    names = list(HH19_MARKERS.keys())
    n_valid = (~_missing_mask(data.points_mm)).sum(axis=0)  # (n_points,)
    by_short: dict[str, int] = {}
    for i, label in enumerate(data.point_labels):
        s = _short(label, data)
        # 只收「至少有一帧有效数据」的 marker。动态 trial 里内嵌的 static 副本
        # （如 medial 膝/踝，只在静态标定时戴）全零/全缺失，应排除而不是写出
        # 一整列 NaN——否则 IK 会把 medial 关节中心钉到原点。
        if s in HH19_MARKERS and n_valid[i] > 0:
            if s not in by_short or n_valid[i] > n_valid[by_short[s]]:
                by_short[s] = i

    present = [m for m in names if m in by_short]
    cols = [by_short[m] for m in present]
    if not cols:
        raise ValueError("C3D 中未找到任何 HH19 marker")

    traj = data.points_mm[:, cols, :].astype(np.float64)  # (frames, n_markers, 3)
    traj[_missing_mask(traj)] = np.nan
    return present, traj


def build_trc(
    data,
    out_path: str | Path,
    *,
    opensim_frame: bool = True,
    cutoff_hz: float | None = None,
) -> tuple[list[str], np.ndarray]:
    """写 TRC 文件，返回 (marker_names, traj) 其中 traj 为写盘前坐标。

    ``opensim_frame=True`` 时坐标已转 OpenSim ground（mm）；否则保持 mocap 帧。
    """
    names, traj = extract_hh19(data)
    if opensim_frame:
        # (frames, n, 3) 逐个点变换
        traj_osim = np.full_like(traj, np.nan)
        for j in range(traj.shape[1]):
            valid = ~np.isnan(traj[:, j, :]).any(axis=1)
            if valid.any():
                traj_osim[valid, j, :] = mocap_to_opensim_points(traj[valid, j, :])
        traj = traj_osim
    if cutoff_hz is not None:
        traj = lowpass_zero_phase(
            traj, data.point_rate_hz, float(cutoff_hz), preserve_missing=True)
    write_trc(out_path, data.time_s, names, traj, rate_hz=data.point_rate_hz, units="mm")
    return names, traj


__all__ = ["extract_hh19", "build_trc"]
