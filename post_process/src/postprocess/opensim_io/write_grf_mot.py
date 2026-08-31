"""GRF .mot 写出（OpenSim ExternalLoads 用）。

每只脚 9 列：``{i}_ground_force_v{x,y,z}``、``{i}_ground_force_p{x,y,z}``、
``{i}_ground_torque_{x,y,z}``，全部必须已经转换到 OpenSim ground 坐标系、单位 SI
（力 N，作用点 m，自由力矩 Nm）。

**这里不做任何坐标变换/单位换算**——输入应当已由 preprocessing.grf_processing
处理成 OpenSim ground 表达。写出的列名用 ``{plate_index}_`` 前缀，与
write_external_loads 里的 force identifier 一一对应。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_grf_mot(
    path: str | Path,
    time_s: np.ndarray,
    feet: list[dict],  # 每项: {"name": str, "force": (n,3), "point": (n,3), "torque": (n,3)}
) -> None:
    """feet 顺序即力台编号：第 1 只脚 -> ``1_`` 前缀，第 2 只 -> ``2_``。"""
    if not feet:
        raise ValueError("feet 不能为空")
    n_frames = time_s.shape[0]
    for f in feet:
        for key in ("force", "point", "torque"):
            arr = np.asarray(f[key], dtype=np.float64)
            if arr.shape != (n_frames, 3):
                raise ValueError(f"foot '{f['name']}' {key} shape {arr.shape} != {(n_frames, 3)}")

    # 显式构造列名（force 用 vx/vy/vz，point 用 px/py/pz，torque 用 x/y/z）
    columns: list[str] = ["time"]
    for i in range(1, len(feet) + 1):
        for ax in ("vx", "vy", "vz"):
            columns.append(f"{i}_ground_force_{ax}")
        for ax in ("px", "py", "pz"):
            columns.append(f"{i}_ground_force_{ax}")
        for ax in ("x", "y", "z"):
            columns.append(f"{i}_ground_torque_{ax}")

    name = Path(path).stem
    header = [
        f"{name}",
        "version=1",
        f"nRows={n_frames}",
        f"nColumns={len(columns)}",
        "inDegrees=no",
        "endheader",
    ]
    lines = header + ["\t".join(columns)]

    for r in range(n_frames):
        row = [f"{time_s[r]:.6f}"]
        for f in feet:
            for ax in range(3):
                row.append(f"{f['force'][r, ax]:.8f}")
            for ax in range(3):
                row.append(f"{f['point'][r, ax]:.8f}")
            for ax in range(3):
                row.append(f"{f['torque'][r, ax]:.8f}")
        lines.append("\t".join(row))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["write_grf_mot"]
