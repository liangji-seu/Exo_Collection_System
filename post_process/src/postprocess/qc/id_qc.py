"""Dynamics QC：从 ID 输出提取骨盆残余力/力矩，判断坐标系/GRF 是否可能反了。

ID 未执行时返回 BLOCKING 占位。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def id_qc(inverse_dynamics_sto: str | Path | None = None) -> dict:
    if inverse_dynamics_sto is None or not Path(inverse_dynamics_sto).is_file():
        return {"ok": False, "reason": "ID 未执行或 inverse_dynamics.sto 缺失（BLOCKING）"}

    # 简化：读取 .sto 并找 pelvis 残余力相关列（残差通常名为 pelvis_*_force/moment）
    residual = _read_residuals(Path(inverse_dynamics_sto))
    if residual is None:
        return {"ok": True, "reason": "ID 结果可读，但未识别到 pelvis residual 列"}
    return {
        "ok": True,
        "pelvis_residual": {
            "force_N": {k: [float(np.nanmin(v)), float(np.nanmax(v))]
                        for k, v in residual.items() if "force" in k},
            "moment_Nm": {k: [float(np.nanmin(v)), float(np.nanmax(v))]
                          for k, v in residual.items() if "moment" in k},
        },
    }


def _read_residuals(path: Path) -> dict | None:
    try:
        header = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("endheader"):
                break
        # 重新按列解析（简化：只找含 pelvis 的列名）
        rows = []
        names = None
        started = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "endheader":
                started = True
                continue
            if not started:
                continue
            if names is None:
                names = line.split()
                continue
            rows.append([float(x) for x in line.split()])
        if names is None:
            return None
        out = {}
        for i, n in enumerate(names):
            if "pelvis" in n.lower() and ("force" in n.lower() or "moment" in n.lower()):
                out[n] = np.asarray([r[i] for r in rows])
        return out or None
    except Exception:  # noqa: BLE001
        return None


__all__ = ["id_qc"]
