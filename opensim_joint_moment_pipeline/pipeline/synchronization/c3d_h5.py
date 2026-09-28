"""C3D 与 mocap.h5 的逐帧精确匹配。

C3D 由 XINGYING/CAP 单独录制，mocap.h5 由 Exo Collector 从同一 SDK 实时
落盘。正常情况下二者保存的是同一批 SDK 浮点样本，因此能找到「C3D 第 0 帧
对应 H5 第几帧」的精确起点（RMS ≈ 0 mm）。本模块用**多个公共 marker** 的
XYZ 时间序列寻找该起点，输出起点帧号、重叠帧数、RMS、最大误差与唯一性，
不依赖固定 marker 顺序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .marker_names import build_marker_index, normalize_marker_name

_EXACT_RMS_MM = 1e-4      # SDK 逐帧相等时的 RMS 上界（mm）
_UNIQUE_GAP_MM = 1.0      # 唯一匹配要求：次优候选 RMS 至少比最优大这么多（mm）
_UNIQUE_MIN_FRAME_DIST = 2  # 次优候选须距最优至少 2 帧，才算「真正不同的位置」
_MIN_UNIQUE_MARKERS = 3   # 至少 3 个公共 marker 才敢判「唯一」
_MIN_UNIQUE_OVERLAP = 10  # 至少 10 帧有效重叠才有足够证据判唯一
_MIN_QUERY_COORDS = 3     # 首帧至少 3 个有效坐标（=1 个 marker 的 XYZ）才用首帧定位
_REFINE_CANDIDATES = 30
_REFINE_CAP_FRAMES = 4000
_SENTINEL_MM = 1e6        # 缺失 marker 的哨兵值（|x| ≥ 1e6 视为缺失）
_MAX_TIME_MAPPING_RMS_MM = 5.0  # 防止明显错误匹配进入主机时间映射


def _valid_mask(x: np.ndarray) -> np.ndarray:
    """数值有效（有限、非哨兵、非完整零点）的掩码。

    坐标轴上的合法 0 保留有效；只有同一 marker 的 XYZ 全为 0 时，
    才把该 marker 样本整体视为缺失。
    """
    values = np.asarray(x)
    valid = np.isfinite(values) & (np.abs(values) < _SENTINEL_MM)
    if values.ndim >= 1 and values.shape[-1] == 3:
        all_zero = np.all(values == 0.0, axis=-1, keepdims=True)
        valid &= ~all_zero
    return valid


def _marker_has_data(points: np.ndarray, cols: list[int]) -> list[bool]:
    """每个 marker 列在 C3D/H5 中是否至少有有效且非零的数据。"""
    result: list[bool] = []
    for col in cols:
        sub = points[:, int(col), :]
        valid = _valid_mask(sub)
        result.append(bool(np.any(valid)))
    return result


@dataclass(frozen=True)
class C3dH5Match:
    start_frame: int                 # C3D 第 0 帧对应的 H5 帧号
    overlap_frames: int              # 两序列重叠的帧数
    rms_mm: float                    # 重叠区间所有公共 marker 的 RMS 误差（只计有效坐标）
    max_error_mm: float              # 重叠区间最大逐点误差
    matched_markers: tuple[str, ...]  # 参与匹配的短名 marker
    unique: bool                     # 起点是否唯一（复合判定，见下）
    second_best_rms_mm: float = float("inf")   # 次优候选 RMS（审计）
    second_best_frame: int | None = None        # 次优候选起点（审计）

    @property
    def exact(self) -> bool:
        return self.rms_mm <= _EXACT_RMS_MM


def _h5_marker_names(handle) -> list[str]:
    """从 mocap.h5 的 ``metadata/device.marker_names`` 解析并规范化为短名。"""
    import json

    raw = handle["metadata/device"][()]
    if isinstance(raw, (bytes, bytearray)):
        raw = json.loads(raw.decode("utf-8"))
    return [normalize_marker_name(name) for name in raw.get("marker_names", [])]


def match_c3d_to_h5(
    c3d_points_mm: np.ndarray,
    c3d_labels: Iterable[str],
    h5_points_mm: np.ndarray,
    h5_marker_names: Iterable[str],
    *,
    preferred_markers: Iterable[str] | None = None,
) -> C3dH5Match:
    """返回 C3D 第 0 帧在 mocap.h5 中的位置与匹配质量。

    ``c3d_points_mm``: (n_frames, n_points, 3)；``h5_points_mm``: (n_frames, n_points, 3)。
    ``preferred_markers`` 为参与匹配的短名（默认取 C3D/H5 全部公共 marker）。
    """
    labels = [normalize_marker_name(x) for x in c3d_labels]
    h5_names = [normalize_marker_name(x) for x in h5_marker_names]
    index = build_marker_index(labels, h5_names)

    if preferred_markers is not None:
        wanted = {normalize_marker_name(x) for x in preferred_markers}
        index = {k: v for k, v in index.items() if k in wanted}

    if not index:
        raise ValueError("C3D 与 mocap.h5 没有可匹配的公共 marker")

    short_names = sorted(index)  # 稳定顺序，结果可复现
    c3d_cols = [index[name][0] for name in short_names]
    h5_cols = [index[name][1] for name in short_names]

    c3d_common = np.asarray(c3d_points_mm, dtype=np.float64)[:, c3d_cols, :]
    h5_common = np.asarray(h5_points_mm, dtype=np.float64)[:, h5_cols, :]

    # 只保留「两边都有真实数据」的 marker：动态导出内嵌的静态 marker（如
    # R/L Knee.Medial、R/L Ankle.Medial）在 C3D 里是全零、在 H5 里是哨兵
    # 9999999，二者并不一致，不能参与精确匹配（见 reader 的 static 副本说明）。
    usable = [
        c and h
        for c, h in zip(
            _marker_has_data(c3d_common, range(len(short_names))),
            _marker_has_data(h5_common, range(len(short_names))),
        )
    ]
    if not any(usable):
        raise ValueError("C3D 与 mocap.h5 的公共 marker 均无有效数据")

    short_names = [name for name, keep in zip(short_names, usable) if keep]
    usable_mask = np.asarray(usable, dtype=bool)
    c3d = c3d_common[:, usable_mask, :]
    h5 = h5_common[:, usable_mask, :]

    # 首帧定位：只用 C3D 首帧的**有效坐标**做查询，避免首帧某个 marker 缺失时
    # 距离全为 NaN、argsort 退化成任意取候选（prompt6 §3.7 第 2 条）。
    first = c3d[0]                       # (n_common, 3)
    first_valid = _valid_mask(first)
    n_first_valid = int(first_valid.sum())
    if n_first_valid < _MIN_QUERY_COORDS:
        raise ValueError(
            f"C3D 第 0 帧有效坐标不足（仅 {n_first_valid} 个），无法用首帧定位起点"
        )
    distances = np.full(h5.shape[0], np.inf, dtype=np.float64)
    for frame_index, h5_frame in enumerate(h5):
        h5_valid = _valid_mask(h5_frame) & first_valid
        if int(h5_valid.sum()) < _MIN_QUERY_COORDS:
            continue
        difference = h5_frame[h5_valid] - first[h5_valid]
        # Compare candidates by RMS coordinate distance so a frame with fewer
        # valid coordinates is not preferred merely because its norm sums fewer
        # terms.
        distances[frame_index] = np.sqrt(np.mean(np.square(difference)))
    finite_candidates = np.flatnonzero(np.isfinite(distances))
    candidates = finite_candidates[np.argsort(distances[finite_candidates])][:_REFINE_CANDIDATES]
    if not candidates.size:
        raise RuntimeError(
            "C3D 与 mocap.h5 无法匹配（首帧没有至少 3 个同时有效的坐标）"
        )

    # 收集全体候选的 (RMS, 起点帧)。RMS 只在两边均有效的坐标上计算，分母是
    # 有效坐标数，绝不把无效值当 0 计入平均（prompt6 §3.7 第 1 条）。
    refined: list[tuple[float, int]] = []
    for start in candidates:
        start = int(start)
        n = min(_REFINE_CAP_FRAMES, c3d.shape[0], h5.shape[0] - start)
        if n <= 0:
            continue
        a = c3d[:n]
        b = h5[start:start + n]
        valid = _valid_mask(a) & _valid_mask(b)
        if not valid.any():
            continue
        delta = a - b
        rms = float(np.sqrt(np.mean(np.square(delta[valid]))))
        refined.append((rms, start))

    if not refined:
        raise RuntimeError("C3D 与 mocap.h5 无法匹配（无有效重叠帧）")

    # 最佳与次优（prompt6 §3.7 第 3 条）：次优 = 除最佳外 RMS 最小的候选。
    best = min(refined)
    rms, start = best
    others = [e for e in refined if e != best]
    second_best = min(others) if others else None
    second_best_rms = second_best[0] if second_best is not None else float("inf")
    second_best_frame = second_best[1] if second_best is not None else None

    overlap = int(min(c3d.shape[0], h5.shape[0] - start))
    a = c3d[:overlap]
    b = h5[start:start + overlap]
    valid = _valid_mask(a) & _valid_mask(b)
    if valid.any():
        delta = np.where(valid, a - b, np.nan)
        max_error = float(np.nanmax(np.abs(delta)))
    else:
        max_error = float("inf")

    # 唯一性不能只靠 ``rms <= 1e-4``：周期动作会让两个不同起点都近似零 RMS。
    # 复合判定还要看「与最优在时间上真正不同的候选」的次优差距、公共 marker 数
    # 与重叠长度（prompt6 §3.7 第 4 条）。
    #   - 次优差距不能拿「相邻帧」算：受试者起始站立时相邻帧亚毫米抖动，会让
    #     相邻候选 RMS≈0，把唯一的精确匹配误判成歧义。故只统计距离 ≥
    #     ``_UNIQUE_MIN_FRAME_DIST`` 的「真正不同位置」候选。
    #   - 精确匹配（RMS ≤ 1e-4）本质是 SDK 逐帧相等，正确起点只有一个：只要没有
    #     另一个「不同位置」也逐帧相等，就是唯一。
    distinct = [
        e for e in refined
        if abs(e[1] - start) >= _UNIQUE_MIN_FRAME_DIST
    ]
    second_distinct_rms = min(e[0] for e in distinct) if distinct else float("inf")

    if rms <= _EXACT_RMS_MM:
        unique = (
            second_distinct_rms > _EXACT_RMS_MM
            and len(short_names) >= _MIN_UNIQUE_MARKERS
            and overlap >= _MIN_UNIQUE_OVERLAP
        )
    else:
        unique = (
            second_distinct_rms - rms > _UNIQUE_GAP_MM
            and len(short_names) >= _MIN_UNIQUE_MARKERS
            and overlap >= _MIN_UNIQUE_OVERLAP
        )
    return C3dH5Match(
        start_frame=start,
        overlap_frames=overlap,
        rms_mm=rms,
        max_error_mm=max_error,
        matched_markers=tuple(short_names),
        unique=unique,
        second_best_rms_mm=second_best_rms,
        second_best_frame=second_best_frame,
    )


def validate_match_for_time_mapping(
    match: C3dH5Match,
    *,
    max_rms_mm: float = _MAX_TIME_MAPPING_RMS_MM,
) -> None:
    """Reject an obviously bad C3D↔H5 match before using its start frame.

    The matcher intentionally returns all quality levels for audit.  Consumers
    that convert H5 host timestamps to C3D time must apply this separate guard;
    a large RMS means the selected start frame cannot define a trustworthy time
    origin, even when the candidate was unique.  The 5 mm default is a broad
    coarse-error safety limit, not a synchronization-accuracy acceptance rule.
    """
    if not is_reliable_match_rms(match.rms_mm, max_rms_mm=max_rms_mm):
        raise ValueError(
            "C3D与mocap.h5无法可靠对应，请检查文件，不能通过IMU/Gaitway手动offset补偿"
        )


def is_reliable_match_rms(
    rms_mm: object,
    *,
    max_rms_mm: float = _MAX_TIME_MAPPING_RMS_MM,
) -> bool:
    """Return whether an RMS value passes the coarse time-mapping guard."""
    try:
        value = float(rms_mm)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and value <= max_rms_mm)


__all__ = [
    "C3dH5Match",
    "match_c3d_to_h5",
    "is_reliable_match_rms",
    "validate_match_for_time_mapping",
]
