"""IK QC：mean/RMS/max marker error，per-marker RMS。

从 IK 输出的 marker error 文件（若存在）读取；否则返回占位（BLOCKING）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def ik_qc(marker_error_csv: str | Path | None = None) -> dict:
    if marker_error_csv is None or not Path(marker_error_csv).is_file():
        return {"ok": False, "reason": "IK 未执行或 marker error 文件缺失（BLOCKING）"}

    # 简化：读 CSV（time, marker1, marker2, ...）
    try:
        arr = np.genfromtxt(marker_error_csv, delimiter=",", names=True, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"无法解析 marker error 文件：{exc}"}

    cols = [n for n in arr.dtype.names if n.lower() != "time"]
    if not cols:
        return {"ok": False, "reason": "marker error 文件无 marker 列"}
    per_marker = {}
    all_vals = []
    for c in cols:
        v = np.asarray(arr[c], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size:
            per_marker[c] = float(np.sqrt(np.mean(v ** 2)))
            all_vals.append(v)
    if not all_vals:
        return {"ok": False, "reason": "marker error 无有效数值"}
    allv = np.concatenate(all_vals)
    return {
        "ok": True,
        "mean_error_mm": float(np.mean(allv)),
        "rms_error_mm": float(np.sqrt(np.mean(allv ** 2))),
        "max_error_mm": float(np.max(allv)),
        "per_marker_rms_mm": per_marker,
    }


__all__ = ["ik_qc"]
