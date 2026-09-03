"""把 OpenSim 解算结果导出为 **EXO 环境离线 viewer** 可消费的纯 NumPy + JSON。

主界面进程（EXO 环境）**不 import opensim**，因此本模块在 opensim 环境里把
「模型 marker / 骨架 body 原点 / 实验 marker / 左右 COP / 左右 GRF / 矢状面关节力矩」
全部对齐到 IK 时间网格，写成 ``<out>/viewer/*.npy`` + ``viewer_meta.json``。

viewer 只读这些纯数据文件，不解析 .mot/.trc、不接触 OpenSim 模型，从而完全离线、
无 CDN、无 opensim 依赖。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import opensim as osim

# 矢状面 6 个关节力矩列（ID 输出列名 = 坐标名 + "_moment"）
_SAGITTAL_MOMENTS = [
    "hip_flexion_r", "hip_flexion_l",
    "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l",
]

# 骨架 body（gait2392）：躯干 + 左右腿链；只取模型里真实存在的 body。
_SKELETON_BODIES = [
    "pelvis", "torso",
    "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
]

# 内侧 4 点：动态 trial 摘除，只作为「模型预测点」显示，绝不画成实测点。
_MEDIAL = {"R.Knee.Medial", "L.Knee.Medial", "R.Ankle.Medial", "L.Ankle.Medial"}

_MM = 1000.0  # m -> mm


def _read_mot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = lines.index("endheader") + 1
    return lines[header].split(), np.loadtxt(lines[header + 1:])


def _read_trc(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """返回 (marker_names, time_s, xyz_m)，xyz 单位米。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    names = [name for name in lines[3].split("\t")[2::3] if name]
    values = np.genfromtxt(lines[5:])
    return names, values[:, 1], values[:, 2:].reshape(len(values), len(names), 3) / _MM


def _apply_pose(model: osim.Model, state, columns: list[str], row: np.ndarray) -> None:
    """把一行 IK motion 写入模型状态并 realize 位形（与 run_precision_opensim 同源）。"""
    coordinates = model.getCoordinateSet()
    lookup = {
        coordinates.get(i).getName(): coordinates.get(i)
        for i in range(coordinates.getSize())
    }
    for j, name in enumerate(columns[1:], 1):
        if name not in lookup:
            continue
        value = float(row[j])
        if name not in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
            value = math.radians(value)
        lookup[name].setValue(state, value, False)
    state.setTime(float(row[0]))
    model.realizePosition(state)


def _resample(
    src_t: np.ndarray, src_vals: np.ndarray, dst_t: np.ndarray, fill: float = np.nan
) -> np.ndarray:
    """逐列线性插值到 dst_t，界外填 ``fill``（默认 NaN）。"""
    src_vals = np.asarray(src_vals, dtype=np.float64)
    if src_vals.ndim == 1:
        src_vals = src_vals[:, None]
    out = np.empty((len(dst_t), src_vals.shape[1]), dtype=np.float64)
    for c in range(src_vals.shape[1]):
        out[:, c] = np.interp(dst_t, src_t, src_vals[:, c], left=fill, right=fill)
    return out


def _mot_subset(columns: list[str], values: np.ndarray, prefix: str) -> np.ndarray:
    """取 ``<prefix>_v{x,y,z}`` 三列（写出的力/点列名用 vx/vy/vz 或 px/py/pz 约定）。"""
    idx = [columns.index(f"{prefix}_{suf}") for suf in ("vx", "vy", "vz")]
    return values[:, idx].astype(np.float64)


def export_viewer_data(
    *,
    model_path: Path,
    ik_path: Path,
    id_path: Path,
    trc_path: Path,
    grf_path: Path,
    out_dir: Path,
    mass_kg: float,
) -> dict[str, Any]:
    """导出 viewer 数据到 ``out_dir/viewer/``，返回摘要 dict。"""
    viewer_dir = Path(out_dir) / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)

    ik_cols, ik = _read_mot(ik_path)
    id_cols, id_vals = _read_mot(id_path)
    exp_names, trc_time, experimental_m = _read_trc(trc_path)
    grf_cols, grf_vals = _read_mot(grf_path)

    model = osim.Model(str(model_path))
    state = model.initSystem()
    marker_set = model.getMarkerSet()
    marker_lookup = {
        marker_set.get(i).getName(): marker_set.get(i)
        for i in range(marker_set.getSize())
    }
    model_marker_names = [marker_set.get(i).getName() for i in range(marker_set.getSize())]
    body_set = model.getBodySet()
    body_lookup = {
        body_set.get(i).getName(): body_set.get(i) for i in range(body_set.getSize())
    }
    body_names = [b for b in _SKELETON_BODIES if b in body_lookup]

    n = len(ik)
    time_s = ik[:, 0].astype(np.float64)
    frame_rate = 1.0 / float(np.median(np.diff(time_s))) if n > 1 else 100.0

    # 模型 marker + body 原点：逐帧 realize 位形后取 ground 坐标（m → mm）。
    model_markers = np.zeros((n, len(model_marker_names), 3), dtype=np.float32)
    body_origins = np.zeros((n, len(body_names), 3), dtype=np.float32)
    for i in range(n):
        _apply_pose(model, state, ik_cols, ik[i])
        for j, name in enumerate(model_marker_names):
            v = marker_lookup[name].getLocationInGround(state)
            model_markers[i, j] = (v.get(0) * _MM, v.get(1) * _MM, v.get(2) * _MM)
        for j, name in enumerate(body_names):
            v = body_lookup[name].getPositionInGround(state)
            body_origins[i, j] = (v.get(0) * _MM, v.get(1) * _MM, v.get(2) * _MM)

    # 实验 marker（动态 TRC，15 点）插值到 IK 网格（m → mm）。
    experimental_mm = _resample(
        trc_time, experimental_m.reshape(len(experimental_m), -1), time_s, fill=np.nan
    ).reshape(n, len(exp_names), 3).astype(np.float32)

    # 左右 GRF / COP（grf.mot：plate 1 = right，plate 2 = left；点 m → mm）。
    force_r = _mot_subset(grf_cols, grf_vals, "1_ground_force")
    force_l = _mot_subset(grf_cols, grf_vals, "2_ground_force")
    point_r = grf_vals[:, [grf_cols.index("1_ground_force_px"),
                           grf_cols.index("1_ground_force_py"),
                           grf_cols.index("1_ground_force_pz")]].astype(np.float64)
    point_l = grf_vals[:, [grf_cols.index("2_ground_force_px"),
                           grf_cols.index("2_ground_force_py"),
                           grf_cols.index("2_ground_force_pz")]].astype(np.float64)

    force = np.stack(
        [_resample(grf_vals[:, 0], force_r, time_s, fill=0.0),
         _resample(grf_vals[:, 0], force_l, time_s, fill=0.0)],
        axis=1,
    ).astype(np.float32)  # (n, 2, 3) N
    cop = np.stack(
        [_resample(grf_vals[:, 0], point_r, time_s, fill=0.0) * _MM,
         _resample(grf_vals[:, 0], point_l, time_s, fill=0.0) * _MM],
        axis=1,
    ).astype(np.float32)  # (n, 2, 3) mm

    # 矢状面关节力矩（ID）插值到 IK 网格。
    moment_names = [f"{c}_moment" for c in _SAGITTAL_MOMENTS]
    moment_cols = [id_cols.index(mn) for mn in moment_names]
    moments = _resample(id_vals[:, 0], id_vals[:, moment_cols], time_s, fill=np.nan).astype(
        np.float32
    )

    np.save(viewer_dir / "time_s.npy", time_s)
    np.save(viewer_dir / "model_markers.npy", model_markers)
    np.save(viewer_dir / "experimental_markers.npy", experimental_mm)
    np.save(viewer_dir / "body_origins.npy", body_origins)
    np.save(viewer_dir / "cop.npy", cop)
    np.save(viewer_dir / "grf.npy", force)
    np.save(viewer_dir / "moments.npy", moments)

    meta = {
        "schema_version": "1.0.0",
        "n_frames": int(n),
        "frame_rate_hz": float(frame_rate),
        "mass_kg": float(mass_kg),
        "units": {"position": "mm", "force": "N", "moment": "Nm"},
        "model_marker_names": list(model_marker_names),
        "experimental_marker_names": list(exp_names),
        "medial_marker_names": sorted(_MEDIAL),
        "body_names": list(body_names),
        "skeleton_segments": [
            ["pelvis", "torso"],
            ["pelvis", "femur_r"], ["femur_r", "tibia_r"], ["tibia_r", "talus_r"],
            ["talus_r", "calcn_r"], ["calcn_r", "toes_r"],
            ["pelvis", "femur_l"], ["femur_l", "tibia_l"], ["tibia_l", "talus_l"],
            ["talus_l", "calcn_l"], ["calcn_l", "toes_l"],
        ],
        "moment_names": list(_SAGITTAL_MOMENTS),
        "moment_curve_labels": [
            "hip_flexion_r", "hip_flexion_l",
            "knee_angle_r", "knee_angle_l",
            "ankle_angle_r", "ankle_angle_l",
        ],
        "cop_order": ["right", "left"],
    }
    (viewer_dir / "viewer_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "viewer_dir": str(viewer_dir),
        "n_frames": int(n),
        "frame_rate_hz": float(frame_rate),
        "n_model_markers": len(model_marker_names),
        "n_experimental_markers": len(exp_names),
        "n_bodies": len(body_names),
    }
