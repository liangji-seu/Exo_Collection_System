"""OpenSim Storage(.sto/.mot) 最小读取器（纯 Python，无绑定）。

只解析：header 到 ``endheader``，其后第一行为列名，再后为数据。
用于读 inverse_dynamics.sto 等结果文件，供 export / qc 使用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_sto(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """返回 (time, column_names, data_matrix)。

    time 是第一列（通常名为 time），data_matrix 形状 (n_rows, n_cols)。
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    # 找 endheader
    header_end = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break
    if header_end is None:
        raise ValueError(f"{path} 不是合法的 Storage 文件（无 endheader）")

    # 列名 = endheader 后第一个非空行
    col_line = None
    for line in lines[header_end + 1:]:
        if line.strip():
            col_line = line
            break
    if col_line is None:
        raise ValueError(f"{path} 无列名行")
    columns = col_line.split()

    rows = []
    for line in lines[header_end + 1:]:
        if not line.strip():
            continue
        parts = line.split()
        if parts == columns:
            continue  # 跳过列名行本身
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    mat = np.asarray(rows, dtype=np.float64)
    time = mat[:, 0] if mat.shape[1] else np.empty(0)
    return time, columns, mat


__all__ = ["read_sto"]
