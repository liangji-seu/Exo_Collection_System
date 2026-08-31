"""变换校验：旋转矩阵正交性 R.T@R ≈ I、det(R) ≈ +1。"""

from __future__ import annotations

import numpy as np


def validate_rotation(R, *, ortho_tol: float = 1e-4) -> dict:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return {"ok": False, "reason": f"R 必须是 3x3，得到 {R.shape}",
                "orthogonality_error": None, "det": None}
    ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
    det = float(np.linalg.det(R))
    ok = ortho_err < ortho_tol and abs(det - 1.0) < ortho_tol
    reason = ""
    if ortho_err >= ortho_tol:
        reason = f"非正交：max|R.T@R - I| = {ortho_err:.2e}（阈值 {ortho_tol:.0e}）"
    if abs(det - 1.0) >= ortho_tol:
        reason = (reason + "；" if reason else "") + f"det(R)={det:.4f}（含镜像/缩放）"
    return {"ok": ok, "reason": reason, "orthogonality_error": ortho_err, "det": det}


def validate_transform(rotation_matrix, translation) -> dict:
    """校验 (R, t) 变换：R 正交 + t 是 (3,)。"""
    R = np.asarray(rotation_matrix, dtype=np.float64) if rotation_matrix is not None else None
    if R is None:
        return {"ok": False, "reason": "rotation_matrix 未提供（BLOCKING）"}
    rot = validate_rotation(R)
    if not rot["ok"]:
        return rot
    t = np.asarray(translation, dtype=np.float64)
    if t.shape != (3,):
        return {"ok": False, "reason": f"translation 必须是 (3,)，得到 {t.shape}"}
    return {"ok": True, "reason": "", "orthogonality_error": rot["orthogonality_error"],
            "det": rot["det"]}


__all__ = ["validate_rotation", "validate_transform"]
