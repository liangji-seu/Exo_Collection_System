"""从 inverse_dynamics.sto 导出髋关节净力矩 CSV。

**不依赖固定列号**：按列名自动匹配 ``hip_flexion_r`` / ``hip_flexion_l``
（以及可选的 adduction/rotation）。同时输出 Nm 与 Nm/kg。

注意：OpenSim generalized force 的正方向由 coordinate 定义决定，导出时在
QC 报告中写明 flexion/extension 正方向，避免后续把符号解释反。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from ..opensim_io.read_sto import read_sto


def _match_hip_column(name: str) -> tuple[str, str] | None:
    """返回 (side, dof)，例如 ('r', 'flexion')。匹配 hip_flexion_r / hip_adduction_l 等。"""
    n = name.lower()
    m = re.search(r"hip_(flexion|adduction|rotation|abduction|internal_rotation|external_rotation)[_\.]([rl])", n)
    if m:
        return m.group(2), m.group(1)
    m = re.search(r"([rl])[_\.]hip_(flexion|adduction|rotation)", n)
    if m:
        return m.group(1), m.group(2)
    return None


def export_hip_moment(
    inverse_dynamics_sto: str | Path,
    out_csv: str | Path,
    *,
    mass_kg: float | None = None,
) -> dict:
    time, columns, mat = read_sto(inverse_dynamics_sto)

    found: dict[str, int] = {}
    for i, col in enumerate(columns):
        hit = _match_hip_column(col)
        if hit:
            found[col] = i

    if not found:
        raise ValueError(
            f"未在 inverse_dynamics.sto 中找到 hip 关节力矩列。"
            f"现有列：{columns}"
        )

    header = ["time"]
    for col, i in found.items():
        header += [f"{col}_moment_Nm"]
        if mass_kg:
            header += [f"{col}_moment_Nm_per_kg"]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in range(mat.shape[0]):
            row = [f"{time[r]:.6f}"]
            for col, i in found.items():
                v = mat[r, i]
                row.append(f"{v:.6f}")
                if mass_kg:
                    row.append(f"{v / mass_kg:.6f}")
            w.writerow(row)

    return {"columns_exported": list(found.keys()), "n_rows": mat.shape[0],
            "mass_kg": mass_kg, "out_csv": str(out_csv)}


__all__ = ["export_hip_moment"]
