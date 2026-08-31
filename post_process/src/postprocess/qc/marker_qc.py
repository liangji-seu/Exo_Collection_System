"""Marker QC：缺失率、gap 段、离群点（速度异常）。"""

from __future__ import annotations

import numpy as np


def _gap_runs(missing: np.ndarray) -> list[tuple[int, int]]:
    """返回连续缺失段的 (start, end) 帧号。"""
    gaps = []
    n = missing.shape[0]
    i = 0
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            gaps.append((i, j - 1))
            i = j
        else:
            i += 1
    return gaps


def marker_qc(data_mm: np.ndarray, marker_names: list[str], time_s: np.ndarray) -> dict:
    """data_mm (n_frames, n_markers, 3)。"""
    data = np.asarray(data_mm, dtype=np.float64)
    n_frames, n_markers, _ = data.shape
    per_marker = []
    for j, name in enumerate(marker_names):
        traj = data[:, j, :]
        missing = np.isnan(traj).any(axis=1) | np.all(np.abs(traj) < 1e-6, axis=1)
        gaps = _gap_runs(missing)
        n_valid = int((~missing).sum())
        rms = float(np.sqrt(np.mean(np.sum(traj[~missing] ** 2, axis=1)))) if n_valid else np.nan

        # 速度离群点（相邻有效帧位移 > 5 倍中位位移）
        outlier_count = 0
        valid_idx = np.where(~missing)[0]
        if valid_idx.size > 2:
            disp = np.linalg.norm(np.diff(traj[valid_idx], axis=0), axis=1)
            med = np.median(disp)
            if med > 1e-9:
                outlier_count = int((disp > 5 * med).sum())

        per_marker.append({
            "marker": name,
            "missing_ratio": float(missing.mean()),
            "n_gaps": len(gaps),
            "longest_gap_frames": max((e - s + 1) for s, e in gaps) if gaps else 0,
            "n_valid_frames": n_valid,
            "rms_mm": rms,
            "velocity_outliers": outlier_count,
        })

    overall_missing = float(np.mean([m["missing_ratio"] for m in per_marker]))
    return {
        "n_frames": n_frames,
        "n_markers": n_markers,
        "overall_missing_ratio": overall_missing,
        "per_marker": per_marker,
    }


__all__ = ["marker_qc"]
