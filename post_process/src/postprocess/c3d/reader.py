"""C3D reader —— 把 XINGYING / NOKOV 导出的 ``.c3d`` 解析成 NumPy 数组。

只依赖 ``ezc3d`` + ``numpy``，与 OpenSim 解耦。已验证的关键事实
（对 ezc3d 1.7.2 实测）：

- ``c3d['data']['points']`` 是 ``(4, n_points, n_frames)`` = ``(x, y, z, residual)``，
  ``c3d['data']['analogs']`` 是 ``(n_subframes, n_channels, n_frames)``。
  ezc3d 已经应用 POINT:SCALE / ANALOG:SCALE，读出来就是物理值（mm / N）。
- XINGYING 用 ``SUBJECTS:USES_PREFIXES=1`` 把 marker 分成 ``*_static``（静态标定）
  与 ``*_dynamic``（动态步态）。静态 subject 在动态导出里常常全 0。
- 虚拟 marker 两种前缀：``V_``（V_Mid_ASIS）与 ``V.``（V.Sacral）。
- 测力台 analog 是原始台面输出 Fx/Fy/Fz + COPx/COPy + Tz（自由力矩），
  不是六维力 Mx/My/Mz。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import ezc3d
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要 ezc3d：pip install ezc3d 或 conda install -c conda-forge ezc3d") from exc


def _ezc3d_path(path: Path) -> str:
    """返回 ezc3d 能打开的路径。

    ezc3d 的 C++ 层在 Windows 上打不开含非 ASCII（中文）字符的**绝对**路径，
    但相对 ASCII 路径可以。因此：
    1. 路径本身是 ASCII → 原样返回；
    2. 含非 ASCII → 尝试转成相对当前工作目录的 ASCII 相对路径；
    3. 仍不行（跨盘）→ 复制到临时 ASCII 目录读取。
    """
    s = str(path)
    try:
        s.encode("ascii")
        return s
    except UnicodeEncodeError:
        pass
    try:
        rel = os.path.relpath(path, os.getcwd())
        rel.encode("ascii")
        return rel
    except (ValueError, UnicodeEncodeError):
        pass
    tmpdir = tempfile.mkdtemp(prefix="c3d_")
    dest = Path(tmpdir) / path.name
    shutil.copy(path, dest)
    return str(dest)


@dataclass(frozen=True)
class SubjectInfo:
    name: str
    prefix: str
    is_static: bool
    marker_indices: tuple[int, ...]
    marker_names: tuple[str, ...]


@dataclass(frozen=True)
class ForcePlatformInfo:
    index: int
    type: int
    corners: np.ndarray      # (3, n) mm
    origin: np.ndarray       # (3,) mm
    channels: tuple[int, ...]  # 1-based analog channel indices
    cal_matrix: np.ndarray | None


@dataclass
class C3dData:
    path: Path
    manufacturer: str
    software: str
    software_version: str

    point_rate_hz: float
    analog_rate_hz: float
    n_frames: int
    data_start: int

    point_labels: tuple[str, ...]
    point_units: str
    points_mm: np.ndarray      # (n_frames, n_points, 3) mm
    residuals: np.ndarray      # (n_frames, n_points)
    subjects: tuple[SubjectInfo, ...]

    analog_labels: tuple[str, ...]
    analog_units: tuple[str, ...]
    analog_scale: tuple[float, ...]
    analogs: np.ndarray        # (n_frames, n_channels)

    force_platforms: tuple[ForcePlatformInfo, ...]

    time_s: np.ndarray = field(init=False)
    frame_index: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        step = 1.0 / self.point_rate_hz if self.point_rate_hz > 0 else 1.0
        self.time_s = np.arange(self.n_frames, dtype=np.float64) * step
        self.frame_index = self.data_start + np.arange(self.n_frames, dtype=np.int64)

    # -- 便捷查询 ----------------------------------------------------- #
    def subject(self, name: str | None = None, *, static: bool | None = None) -> SubjectInfo:
        for s in self.subjects:
            if name is not None and s.name != name:
                continue
            if static is not None and s.is_static != static:
                continue
            return s
        raise KeyError(name if name else f"static={static}")

    def is_virtual(self, label: str) -> bool:
        stripped = label
        for s in self.subjects:
            if label.startswith(s.prefix):
                stripped = label[len(s.prefix):]
                break
        return stripped.startswith(("V_", "V."))

    def marker_mm(self, name: str) -> np.ndarray:
        return self.points_mm[:, self.point_labels.index(name), :]


def _param(c3d: Any, group: str, name: str, default: Any = None) -> Any:
    try:
        return c3d["parameters"][group][name]["value"]
    except (KeyError, TypeError):
        return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float64).ravel()
    return float(arr[0]) if arr.size else default


def _text(value: Any) -> str:
    if value is None:
        return "unknown"
    seq = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]
    return str(seq[0]) if seq else "unknown"


def read_c3d(path: str | Path) -> C3dData:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    c3d = ezc3d.c3d(_ezc3d_path(source))

    manufacturer = _text(_param(c3d, "MANUFACTURER", "COMPANY"))
    software = _text(_param(c3d, "MANUFACTURER", "SOFTWARE"))
    version = _text(_param(c3d, "MANUFACTURER", "VERSION"))

    point_rate = _as_float(_param(c3d, "POINT", "RATE"), 0.0)
    analog_rate = _as_float(_param(c3d, "ANALOG", "RATE"), 0.0)
    data_start = int(_as_float(_param(c3d, "POINT", "DATA_START"), 0.0))

    raw_points = np.asarray(c3d["data"]["points"], dtype=np.float64)
    if raw_points.ndim != 3 or raw_points.shape[0] != 4:
        raise ValueError(f"unexpected point array shape {raw_points.shape}")
    n_points = raw_points.shape[1]
    n_frames = raw_points.shape[2]
    points_mm = np.ascontiguousarray(np.transpose(raw_points[:3], (2, 1, 0)), dtype=np.float32)
    residuals = np.ascontiguousarray(raw_points[3].T, dtype=np.float32)

    point_units = _text(_param(c3d, "POINT", "UNITS"))
    labels_raw = _param(c3d, "POINT", "LABELS", [])
    point_labels = tuple(str(x) for x in labels_raw)
    if not point_labels:
        point_labels = tuple(f"point_{i + 1:02d}" for i in range(n_points))
    if len(point_labels) != n_points:
        raise ValueError(f"point label count {len(point_labels)} != data points {n_points}")

    # subjects
    subjects: list[SubjectInfo] = []
    subjects_used = int(_as_float(_param(c3d, "SUBJECTS", "USED"), 0.0))
    subject_names = _param(c3d, "SUBJECTS", "NAMES", [])
    subject_prefixes = _param(c3d, "SUBJECTS", "LABEL_PREFIXES", [])
    for index in range(subjects_used):
        name = str(subject_names[index]) if index < len(subject_names) else f"subject_{index + 1}"
        prefix = str(subject_prefixes[index]) if index < len(subject_prefixes) else f"{name}:"
        owned = tuple(i for i, label in enumerate(point_labels) if label.startswith(prefix))
        subjects.append(SubjectInfo(name, prefix, "static" in name.casefold(), owned,
                                    tuple(point_labels[i] for i in owned)))
    if not subjects:
        subjects = [SubjectInfo("default", "", False, tuple(range(n_points)), point_labels)]

    # analog
    raw_analogs = np.asarray(c3d["data"]["analogs"], dtype=np.float64)
    if raw_analogs.ndim == 3:
        analogs = np.ascontiguousarray(raw_analogs[0].T, dtype=np.float32)
    elif raw_analogs.ndim == 2:
        analogs = np.ascontiguousarray(raw_analogs.T, dtype=np.float32)
    else:
        analogs = np.empty((n_frames, 0), dtype=np.float32)
    analog_labels = tuple(str(x) for x in (_param(c3d, "ANALOG", "LABELS", []) or []))
    analog_units = tuple(str(x) for x in (_param(c3d, "ANALOG", "UNITS", []) or []))
    analog_scale = tuple(float(x) for x in np.asarray(_param(c3d, "ANALOG", "SCALE", [])).ravel())
    if not analog_labels:
        analog_labels = tuple(f"analog_{i + 1:02d}" for i in range(analogs.shape[1]))

    # force platforms
    force_platforms: list[ForcePlatformInfo] = []
    fp_used = int(_as_float(_param(c3d, "FORCE_PLATFORM", "USED"), 0.0))
    fp_type = _param(c3d, "FORCE_PLATFORM", "TYPE", [])
    fp_corners = _param(c3d, "FORCE_PLATFORM", "CORNERS", None)
    fp_origin = _param(c3d, "FORCE_PLATFORM", "ORIGIN", None)
    fp_channel = _param(c3d, "FORCE_PLATFORM", "CHANNEL", None)
    fp_cal = _param(c3d, "FORCE_PLATFORM", "CAL_MATRIX", None)
    for index in range(fp_used):
        corners = (np.squeeze(np.asarray(fp_corners, dtype=np.float64))
                   if fp_corners is not None else np.empty((3, 0)))
        if corners.ndim == 1 and corners.size == 3:
            corners = corners.reshape(3, 1)
        origin = (np.squeeze(np.asarray(fp_origin, dtype=np.float64))
                  if fp_origin is not None else np.zeros(3))
        channels = (tuple(int(ch) for ch in np.asarray(fp_channel).ravel())
                    if fp_channel is not None else ())
        cal = np.asarray(fp_cal, dtype=np.float64) if fp_cal is not None else None
        ptype = (int(np.asarray(fp_type).ravel()[index])
                 if fp_type is not None and np.asarray(fp_type).size > index else 0)
        force_platforms.append(ForcePlatformInfo(index + 1, ptype, corners, origin, channels, cal))

    return C3dData(
        path=source, manufacturer=manufacturer, software=software, software_version=version,
        point_rate_hz=point_rate, analog_rate_hz=analog_rate, n_frames=n_frames,
        data_start=data_start, point_labels=point_labels, point_units=point_units,
        points_mm=points_mm, residuals=residuals, subjects=tuple(subjects),
        analog_labels=analog_labels, analog_units=analog_units, analog_scale=analog_scale,
        analogs=analogs, force_platforms=tuple(force_platforms),
    )


__all__ = ["C3dData", "SubjectInfo", "ForcePlatformInfo", "read_c3d"]
