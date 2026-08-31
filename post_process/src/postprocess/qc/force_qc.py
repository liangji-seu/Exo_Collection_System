"""Force QC：Fz 范围、stance/swing、COP 落台比例。复用 validate_grf。"""

from __future__ import annotations

import numpy as np

from ..validation.validate_grf import validate_grf


def force_qc(force_N: np.ndarray, cop_local: np.ndarray,
             plate_corners: np.ndarray | None = None,
             *, fz_stance_threshold: float = 20.0) -> dict:
    v = validate_grf(force_N, cop_local, plate_corners,
                     fz_stance_threshold=fz_stance_threshold)
    return {
        "ok": v["ok"],
        "warnings": v["warnings"],
        "fz_range_N": v["fz_range"],
        "stance_ratio": v["stance_ratio"],
        "cop_in_bounds_ratio": v["cop_in_bounds_ratio"],
    }


__all__ = ["force_qc"]
