"""把一次「静态 + 动态 + Gaitway」会话预处理成 OpenSim 下游可消费的中间产物。

与 ``scripts/prep_opensim.py`` 同源，但**不读 YAML 配置**、不依赖 cwd —— 输入全部
显式传参（来自 Exo Calculate 的 Session 发现 + 同步结论），输出写进指定的 ASCII
工作目录并生成 ``manifest.json``。因此既能被后台 Worker 直接调用，也能单独 CLI 驱动。

产出（都在 ``out_dir`` 下）：
    static.trc            静态 trial（19 marker，OpenSim ground 帧，mm）
    dynamic.trc           动态 trial（15 marker，medial 剔除，OpenSim ground 帧，mm）
    grf.mot               双侧地面反力（Gaitway 原生左右分解，OpenSim ground 帧，SI）
    external_loads.xml    ExternalLoads（左右 calcn）
    support_mask.npy      (n_frames, 2) bool [right_valid, left_valid]
    gait2392_simbody.osim  通用模型副本（自包含，OpenSim 以裸名相对加载）
    manifest.json         供 opensim 环境 ``scripts/process_session.py`` 读取

本模块**不 import opensim**，在 EXO 环境（numpy 1.x + ezc3d）直接运行。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..c3d.reader import read_c3d
from ..gait.steady_state import detect_steady_walking
from ..gaitway import build_bilateral_grf, read_gaitway_ascii
from ..transforms import R_FP_TO_MOCAP_DEFAULT, R_MOCAP_TO_OPENSIM
from .build_trc import build_trc
from .grf import write_external_loads_xml
from .static_window import select_static_window
from .write_grf_mot import write_grf_mot


def _file_sha256(path: Path, chunk: int = 1 << 20) -> str | None:
    """全文件 SHA-256；文件不存在返回 None（STALE 判定用，prompt6 §3.10）。"""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _input_fingerprint(path: str | Path) -> dict[str, Any]:
    """记录一个输入的路径/大小/mtime_ns/SHA-256，供历史 run STALE 判定。"""
    p = Path(path)
    try:
        stat = p.stat()
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        size = None
        mtime_ns = None
    return {
        "path": str(p),
        "size_bytes": size,
        "mtime_ns": mtime_ns,
        "sha256": _file_sha256(p),
    }


def prepare_session(
    *,
    static_c3d_path: str | Path,
    dynamic_c3d_path: str | Path,
    gaitway_txt_path: str | Path,
    generic_model_path: str | Path,
    out_dir: str | Path,
    subject_id: str,
    mass_kg: float,
    height_m: float,
    gaitway_offset_s: float,
    marker_cutoff_hz: float | None = None,
    grf_cutoff_hz: float = 20.0,
    opensim_x_sign: float = -1.0,
    opensim_z_sign: float = -1.0,
    analysis_time_range_s: tuple[float, float] | None = None,
    static_time_range_s: tuple[float, float] | None = None,
    sync_confidence: str | None = None,
    sync_quality: dict[str, Any] | None = None,
    marker_adjustment_expert_confirmed: bool = False,
    force_threshold_N: float = 50.0,
    R_fp_to_mocap: np.ndarray | None = None,
) -> dict[str, Any]:
    """预处理并写 manifest，返回摘要 dict（含 ``manifest_path``）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if R_fp_to_mocap is None:
        R_fp_to_mocap = R_FP_TO_MOCAP_DEFAULT
    R_fp = np.asarray(R_fp_to_mocap, dtype=np.float64)

    static_data = read_c3d(static_c3d_path)
    dynamic_data = read_c3d(dynamic_c3d_path)
    gaitway = read_gaitway_ascii(gaitway_txt_path)

    # 0. 静态稳定窗口（prompt6 §3.6）：用户手动指定优先，否则自动滑窗选择。
    if static_time_range_s is not None:
        s0, s1 = float(static_time_range_s[0]), float(static_time_range_s[1])
        static_window = {
            "start_s": s0,
            "end_s": s1,
            "duration_s": s1 - s0,
            "method": "manual",
            "n_frames": None,
            "mean_velocity_mm_s": None,
            "valid_frac": None,
            "n_markers_present": None,
            "missing_markers": [],
            "params": {"manual": True},
        }
    else:
        static_window = select_static_window(static_data).to_dict()

    # 0b. 稳态分析区间（prompt6 §3.5）：手动指定优先，否则自动检测稳定步行段。
    # 检测在 gaitway 原生时间进行，再按 t_c3d = t_gaitway - offset 映射回 C3D 时间。
    if analysis_time_range_s is not None:
        a0, a1 = float(analysis_time_range_s[0]), float(analysis_time_range_s[1])
        steady_state = {
            "start_s": a0, "end_s": a1, "method": "manual",
            "n_steps": None, "n_cycles": None,
            "median_step_period_s": None, "step_period_cv": None,
            "reason": "手动指定", "params": {"manual": True},
        }
        analysis_range = (a0, a1)
    else:
        fz_side = gaitway.columns["FzR(N)"]
        if not np.isfinite(fz_side).any():
            fz_side = gaitway.columns["FzL(N)"]
        steady_gait = detect_steady_walking(gaitway.time_s, fz_side)
        # gaitway 时间 → C3D 时间
        a0 = steady_gait.start_s - gaitway_offset_s
        a1 = steady_gait.end_s - gaitway_offset_s
        analysis_range = (a0, a1)
        steady_state = steady_gait.to_dict()
        steady_state["start_s"] = a0
        steady_state["end_s"] = a1

    # 1. 静态 / 动态 TRC（OpenSim ground 帧，mm）
    static_names, _ = build_trc(
        static_data, out / "static.trc", opensim_frame=True, cutoff_hz=marker_cutoff_hz
    )
    dyn_names, _ = build_trc(
        dynamic_data, out / "dynamic.trc", opensim_frame=True, cutoff_hz=marker_cutoff_hz
    )

    # 2. 双侧 GRF（Gaitway 原生左右分解，无需单支撑拆分）
    feet, decomposed_valid, gaitway_qc = build_bilateral_grf(
        gaitway,
        dynamic_data.time_s,
        gaitway_offset_s,
        R_fp,
        force_threshold_N=force_threshold_N,
        cutoff_hz=grf_cutoff_hz,
        opensim_x_sign=opensim_x_sign,
        opensim_z_sign=opensim_z_sign,
    )
    write_grf_mot(out / "grf.mot", dynamic_data.time_s, feet)
    write_external_loads_xml(out / "external_loads.xml", "grf.mot")
    # 双侧有效：两腿 ID 结果在分解有效区间内都有效（含摆动与双支撑）。
    right_valid = decomposed_valid.copy()
    left_valid = decomposed_valid.copy()
    np.save(out / "support_mask.npy", np.column_stack([right_valid, left_valid]))

    # 3. 通用模型副本（自包含，OpenSim 用裸名相对加载）
    generic_dest = out / "gait2392_simbody.osim"
    shutil.copyfile(generic_model_path, generic_dest)

    # 3b. 输入指纹：历史 run 据此判定 STALE_INPUTS（prompt6 §3.10 第 4 条）。
    inputs = {
        "static_c3d": _input_fingerprint(static_c3d_path),
        "dynamic_c3d": _input_fingerprint(dynamic_c3d_path),
        "gaitway_txt": _input_fingerprint(gaitway_txt_path),
        "generic_model": _input_fingerprint(generic_model_path),
    }

    # 4. manifest
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "subject": {
            "id": subject_id,
            "mass_kg": float(mass_kg),
            "height_m": float(height_m),
        },
        "generic_model": str(generic_dest),
        "static_trc": str(out / "static.trc"),
        "dynamic_trc": str(out / "dynamic.trc"),
        "external_loads": str(out / "external_loads.xml"),
        "grf_mot": str(out / "grf.mot"),
        "support_mask": str(out / "support_mask.npy"),
        "inputs": inputs,
        "analysis_time_range_s": list(analysis_range),
        "steady_state": steady_state,
        "static_time_range_s": [static_window["start_s"], static_window["end_s"]],
        "static_window": static_window,
        "processing": {
            "marker_cutoff_hz": marker_cutoff_hz,
            "grf_cutoff_hz": grf_cutoff_hz,
            "opensim_force_x_sign": float(opensim_x_sign),
            "opensim_force_z_sign": float(opensim_z_sign),
            "marker_adjustment_expert_confirmed": bool(marker_adjustment_expert_confirmed),
        },
        "gaitway": gaitway_qc,
        "sync": {
            **({} if sync_quality is None else sync_quality),
            "gaitway_offset_s": float(gaitway_offset_s),
            "confidence": sync_confidence,
        },
        "out_dir": str(out),
        "mocap_to_opensim_R": R_MOCAP_TO_OPENSIM.tolist(),
        "static_n_markers": len(static_names),
        "dynamic_n_markers": len(dyn_names),
        "static_n_frames": static_data.n_frames,
        "dynamic_n_frames": dynamic_data.n_frames,
        "dynamic_point_rate_hz": float(dynamic_data.point_rate_hz),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "manifest_path": str(manifest_path),
        "static_n_markers": len(static_names),
        "dynamic_n_markers": len(dyn_names),
        "static_n_frames": static_data.n_frames,
        "dynamic_n_frames": dynamic_data.n_frames,
        "n_valid_decomposed_frames": int(decomposed_valid.sum()),
        "static_window": static_window,
        "steady_state": steady_state,
        "gaitway": gaitway_qc,
    }


__all__ = ["prepare_session"]
