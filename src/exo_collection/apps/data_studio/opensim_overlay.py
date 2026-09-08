"""OpenSim 解算结果的「验证三联图」数据加载：右脚 Fz、右髋屈曲角、右髋屈曲力矩。

把 ``derived/opensim/run_*/`` 里同一次解算导出的三个量落到**同一条 c3d 相对时间轴**
上，用于核对「脚跟触地（Fz 从 ≈0 起跳）↔ 伸髋力矩谷值 ↔ 髋屈曲角」之间的时序。

纯 NumPy + 标准库，不依赖 opensim / Qt，便于离线单测。

坐标约定（已从真实数据核实）：
- ``viewer/grf.npy`` 形状 ``(n, 2, 3)``，``cop_order = [right, left]``，分量 ``[x, y, z]``。
  其中 **y 分量是竖直方向**（约 0–900 N，着地时为正），x/z 是水平（内外/前后）。
  因此「右脚 Fz（竖直地面反力）」= ``grf[:, 0, 1]``。
- ``viewer/moments.npy`` 形状 ``(n, 6)``，列 0 = ``hip_flexion_r``（右髋屈曲力矩，N·m）。
- 动态 IK ``result.json["files"]["ik"]`` 指向 ``*_ik.mot``，其 ``hip_flexion_r`` 列为
  右髋屈曲角（度）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_RIGHT_FOOT_INDEX = 0   # grf.npy 的 cop_order = [right, left]
_VERTICAL_AXIS = 1      # grf.npy 分量 [x, y, z]，y 为竖直
_HIP_MOMENT_COLUMN = 0  # moments.npy 列 0 = hip_flexion_r


@dataclass(frozen=True, slots=True)
class HipValidation:
    """同一条时间轴上的三个验证信号（全部为 c3d 相对时间）。"""

    time_s: np.ndarray
    fz_r: np.ndarray         # 右脚竖直地面反力 (N)
    hip_angle_r: np.ndarray  # 右髋屈曲角 (deg)
    hip_moment_r: np.ndarray  # 右髋屈曲力矩 (N·m)
    run_dir: Path


def _run_dirs(trial_root: Path) -> list[Path]:
    """``trial_root/derived/opensim/`` 下按名字（时间戳）升序的 ``run_*`` 目录。"""
    opensim_dir = Path(trial_root) / "derived" / "opensim"
    if not opensim_dir.is_dir():
        return []
    return sorted(p for p in opensim_dir.glob("run_*") if p.is_dir())


def find_latest_run_dir(trial_root: Path) -> Path | None:
    """返回最新的 ``run_*`` 目录；没有则 ``None``。"""
    run_dirs = _run_dirs(trial_root)
    return run_dirs[-1] if run_dirs else None


def _parse_mot(path: Path) -> tuple[list[str], np.ndarray]:
    """解析 OpenSim ``.mot`` / ``.sto``，返回 ``(列名, 数据)``（数据含 time 列）。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    end_index = next(
        (i for i, line in enumerate(lines) if line.strip() == "endheader"), None
    )
    if end_index is None:
        raise ValueError(f"{path} 缺少 endheader，不是合法的 .mot 文件")
    column_names = lines[end_index + 1].split()
    rows = [
        [float(value) for value in line.split()]
        for line in lines[end_index + 2 :]
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path} 没有数据行")
    return column_names, np.asarray(rows, dtype=np.float64)


def load_hip_validation(trial_root: Path) -> HipValidation | None:
    """从最新一次**完整** OpenSim 解算目录读取验证三信号；缺数据返回 ``None``。

    按 run 目录从新到旧尝试，跳过缺 ``result.json`` / viewer 导出的半成品
    （OpenSim 解算已跑完但 viewer 导出中断的 run）。
    """
    for run_dir in reversed(_run_dirs(trial_root)):
        validation = _load_from_run_dir(run_dir)
        if validation is not None:
            return validation
    return None


def _load_from_run_dir(run_dir: Path) -> HipValidation | None:
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    ik_path = result.get("files", {}).get("ik")
    viewer_dir = result.get("viewer", {}).get("viewer_dir")
    if not ik_path:
        ik_path = run_dir / "hh19_static_calibrated_ik.mot"
    ik_path = Path(ik_path)
    viewer_dir = Path(viewer_dir) if viewer_dir else run_dir / "viewer"

    grf_path = viewer_dir / "grf.npy"
    moments_path = viewer_dir / "moments.npy"
    time_path = viewer_dir / "time_s.npy"
    if not (grf_path.is_file() and moments_path.is_file() and time_path.is_file()):
        return None
    if not ik_path.is_file():
        return None

    time_s = np.load(time_path).astype(np.float64)
    grf = np.load(grf_path)
    moments = np.load(moments_path)
    if (
        grf.ndim != 3
        or grf.shape[1] < 2
        or moments.ndim != 2
        or moments.shape[1] <= _HIP_MOMENT_COLUMN
    ):
        return None

    column_names, ik_data = _parse_mot(ik_path)
    if "hip_flexion_r" not in column_names:
        return None
    ik_time = ik_data[:, 0]
    hip_angle_r = ik_data[:, column_names.index("hip_flexion_r")]

    # IK .mot 与 viewer 通常同网格，但保险起见插值到 viewer 时间轴。
    if ik_time.shape != time_s.shape or not np.allclose(ik_time, time_s):
        hip_angle_r = np.interp(time_s, ik_time, hip_angle_r)

    fz_r = np.asarray(grf[:, _RIGHT_FOOT_INDEX, _VERTICAL_AXIS], dtype=np.float64)
    hip_moment_r = np.asarray(moments[:, _HIP_MOMENT_COLUMN], dtype=np.float64)

    return HipValidation(
        time_s=time_s,
        fz_r=fz_r,
        hip_angle_r=np.asarray(hip_angle_r, dtype=np.float64),
        hip_moment_r=hip_moment_r,
        run_dir=run_dir,
    )


__all__ = ["HipValidation", "find_latest_run_dir", "load_hip_validation"]
