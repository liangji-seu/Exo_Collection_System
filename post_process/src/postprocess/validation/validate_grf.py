"""GRF 校验：Fz 范围、COP 是否落在力台内、stance/swing 合理性。"""

from __future__ import annotations

import numpy as np


def validate_grf(force_N: np.ndarray, cop_local: np.ndarray,
                 plate_corners: np.ndarray | None = None,
                 *, fz_stance_threshold: float = 20.0) -> dict:
    """force_N (n,3)，cop_local (n,2) 或 (n,3)，plate_corners (3, n_corners) mm。"""
    force = np.asarray(force_N, dtype=np.float64)
    cop = np.asarray(cop_local, dtype=np.float64)
    if force.ndim == 1:
        force = force.reshape(-1, 3)

    fz = force[:, 2]
    warnings: list[str] = []
    if np.any(~np.isfinite(fz)):
        warnings.append("Fz 含 NaN/Inf")
    if force.shape[0] and np.nanmax(np.abs(fz)) > 5000:
        warnings.append("Fz 量级异常（>5000 N），检查单位是否 N")

    stance = np.abs(fz) > fz_stance_threshold
    stance_ratio = float(stance.mean()) if stance.size else 0.0

    cop_in_bounds = None
    if plate_corners is not None:
        corners = np.asarray(plate_corners, dtype=np.float64)
        xmin, xmax = corners[0].min(), corners[0].max()
        ymin, ymax = corners[1].min(), corners[1].max()
        x, y = cop[:, 0], cop[:, 1]
        in_bounds = (x >= xmin - 5) & (x <= xmax + 5) & (y >= ymin - 5) & (y <= ymax + 5)
        cop_in_bounds = float(in_bounds.mean()) if in_bounds.size else 0.0
        if cop_in_bounds < 0.8:
            warnings.append(f"COP 落在力台外的比例过高（{100 * cop_in_bounds:.1f}% 在内）")

    return {
        "ok": not warnings,
        "warnings": warnings,
        "fz_range": [float(np.nanmin(fz)), float(np.nanmax(fz))] if fz.size else None,
        "stance_ratio": stance_ratio,
        "cop_in_bounds_ratio": cop_in_bounds,
    }


__all__ = ["validate_grf"]
