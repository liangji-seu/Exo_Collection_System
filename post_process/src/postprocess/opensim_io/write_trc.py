"""TRC 写出（Motion Analysis .trc，OpenSim TRCFileAdapter 可读）。

**关键约束**：
1. 时间严格来自 C3D（不重排、不插值）。
2. 坐标变换/单位换算**不在这里做**——write_trc 只负责把已经预处理好的
   (frame, marker, xyz) 按 TRC 格式写盘。坐标变换属于 preprocessing 阶段。
3. 默认以 mm 写入（``Units`` 字段 = ``mm``），OpenSim 读入时按 Units 转 m。

TRC 头部：
    第1行  PathFileType  4  (X/Y/Z)  <name>.trc
    第2行  DataRate  CameraRate  NumFrames  NumMarkers  Units  OrigDataRate  OrigDataStartFrame  OrigNumFrames
    第3行  Frame#  Time  <m1>    <m2>    ...
    第4行  (空)  (空)  X1 Y1 Z1  X2 Y2 Z2  ...
    第5行起 数据行
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_trc(
    path: str | Path,
    time_s: np.ndarray,
    marker_names: list[str],
    marker_data: np.ndarray,   # (n_frames, n_markers, 3)
    *,
    rate_hz: float,
    units: str = "mm",
) -> None:
    n_frames, n_markers, _ = marker_data.shape
    if len(marker_names) != n_markers:
        raise ValueError(f"marker_names {len(marker_names)} != data {n_markers}")
    if time_s.shape[0] != n_frames:
        raise ValueError(f"time_s {time_s.shape[0]} != frames {n_frames}")

    name = Path(path).stem
    lines: list[str] = []
    lines.append(f"PathFileType\t4\t(X/Y/Z)\t{name}.trc")
    lines.append(
        f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
        f"OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    )
    lines.append(
        f"{rate_hz:.5f}\t{rate_hz:.5f}\t{n_frames}\t{n_markers}\t{units}\t"
        f"{rate_hz:.5f}\t1\t{n_frames}"
    )
    # marker 名行：每个名字占 3 列（名字 + 2 空列）
    header = ["Frame#", "Time"]
    for m in marker_names:
        header += [m, "", ""]
    lines.append("\t".join(header))
    # 坐标标签行
    sub = ["", ""]
    for i in range(1, n_markers + 1):
        sub += [f"X{i}", f"Y{i}", f"Z{i}"]
    lines.append("\t".join(sub))
    # 数据行
    for i in range(n_frames):
        row = [str(i + 1), f"{time_s[i]:.6f}"]
        for j in range(n_markers):
            x, y, z = marker_data[i, j]
            row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
        lines.append("\t".join(row))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["write_trc"]
