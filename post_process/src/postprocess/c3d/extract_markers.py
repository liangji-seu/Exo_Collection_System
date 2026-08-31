"""Marker 提取：把某个 subject 的 marker 抽成 (frame, marker, xyz) 数组 + 元数据。

给 IK 用的 marker 集合策略由 config 决定，这里只提供"取真实 marker"的原语：
- 虚拟关节中心（V_R.Hip_JC 等）默认**不**作为 IK marker。
- V.Sacral 是否保留由 config 的 ``marker.use_virtual_sacral`` 决定。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reader import C3dData


@dataclass
class MarkerSet:
    names: tuple[str, ...]             # 完整 label（含 subject 前缀）
    data_mm: np.ndarray                # (n_frames, n_markers, 3)
    time_s: np.ndarray                 # (n_frames,)
    is_virtual: np.ndarray             # (n_markers,) bool


def subject_markers(data: C3dData, subject_name: str) -> MarkerSet:
    subject = data.subject(subject_name)
    idx = list(subject.marker_indices)
    names = tuple(data.point_labels[i] for i in idx)
    virt = np.asarray([data.is_virtual(n) for n in names], dtype=bool)
    return MarkerSet(names, data.points_mm[:, idx, :], data.time_s, virt)


def dynamic_markers(data: C3dData) -> MarkerSet:
    """取动态 subject 的 marker（通常是唯一非 static 的 subject）。"""
    for s in data.subjects:
        if not s.is_static:
            return subject_markers(data, s.name)
    return subject_markers(data, data.subjects[0].name)


def filter_markers_for_ik(
    ms: MarkerSet,
    *,
    exclude_joint_centers: bool = True,
    use_virtual_sacral: bool = False,
) -> MarkerSet:
    """返回用于 IK 的 marker 子集。

    - 默认排除所有虚拟关节中心/偏移点（V_*JC / Toe_Offset / Pelvis_Origin 等）。
    - ``use_virtual_sacral`` 控制是否保留 ``V.Sacral``（默认 False，由 config 决定）。
    """
    def keep(name: str) -> bool:
        short = name.split(":")[-1]
        if not ms.is_virtual[ms.names.index(name)]:
            return True  # 真实 marker 一律保留
        if exclude_joint_centers:
            # 虚拟点中只有 V.Sacral 允许按 config 保留，其余一律排除
            if short == "V.Sacral":
                return use_virtual_sacral
            return False
        return True

    mask = np.asarray([keep(n) for n in ms.names], dtype=bool)
    idx = np.where(mask)[0]
    return MarkerSet(
        tuple(ms.names[i] for i in idx),
        ms.data_mm[:, idx, :],
        ms.time_s,
        ms.is_virtual[idx],
    )


def marker_metadata_table(data: C3dData) -> list[dict]:
    """(label, subject, short, is_virtual, n_valid_frames, n_frames) 清单。"""
    rows = []
    for i, label in enumerate(data.point_labels):
        subject = next((s for s in data.subjects if label.startswith(s.prefix)), None)
        traj = data.points_mm[:, i, :]
        valid = int(np.any(np.abs(traj) > 1e-6, axis=1).sum())
        rows.append({
            "index": i,
            "label": label,
            "short_name": label.split(":")[-1],
            "subject": subject.name if subject else "",
            "is_virtual": bool(data.is_virtual(label)),
            "n_valid_frames": valid,
            "n_frames": data.n_frames,
        })
    return rows


__all__ = ["MarkerSet", "subject_markers", "dynamic_markers",
           "filter_markers_for_ik", "marker_metadata_table"]
