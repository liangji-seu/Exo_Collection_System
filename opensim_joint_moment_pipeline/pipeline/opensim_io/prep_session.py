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
from ..gaitway.ascii_export import audit_native_bilateral_wrench
from ..gaitway.ascii_export import GRF_INPUT_POLICY_VERSION
from ..transforms import R_FP_TO_MOCAP_DEFAULT, R_MOCAP_TO_OPENSIM, FORCE_TRANSFORM_VERSION, mocap_to_opensim_points
from .build_trc import build_trc, extract_hh19
from .walking_direction import DIRECTION_POLICY_VERSION, is_backward_trial, body_heading, anatomical_load_mapping, rotate_loads
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
    grf_cutoff_hz: float = 6.0,
    opensim_x_sign: float = -1.0,
    opensim_z_sign: float = -1.0,
    analysis_time_range_s: tuple[float, float] | None = None,
    static_time_range_s: tuple[float, float] | None = None,
    sync_confidence: str | None = None,
    sync_quality: dict[str, Any] | None = None,
    marker_adjustment_expert_confirmed: bool = False,
    quiet_standing: bool = False,
    force_threshold_N: float = 50.0,
    R_fp_to_mocap: np.ndarray | None = None,
    backpack: dict[str, Any] | None = None,
    condition_code: str = "",
) -> dict[str, Any]:
    """预处理并写 manifest，返回摘要 dict（含 ``manifest_path``）。"""
    condition = condition_code or Path(dynamic_c3d_path).stem
    backward = is_backward_trial(condition)
    if backward and "STAND_BWALK_STAND" in condition.upper() and analysis_time_range_s is None:
        raise ValueError("站立→倒走→站立属于瞬态工况，不能自动截取稳态：请在 Calculate 显式设置分析起止时间，确认对应起停/行走段后再解算；批处理请跳过该试次")
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
    elif quiet_standing:
        # 静止站立没有周期性脚跟着地事件。取 C3D 与测力台真正重叠的整段，并
        # 去掉首尾各 1 秒，不能让步态检测器把它误判为“无有效力信号”。
        overlap_start = max(
            float(dynamic_data.time_s[0]),
            float(gaitway.time_s[0]) - float(gaitway_offset_s),
        )
        overlap_end = min(
            float(dynamic_data.time_s[-1]),
            float(gaitway.time_s[-1]) - float(gaitway_offset_s),
        )
        if overlap_end <= overlap_start:
            raise ValueError("站立工况的 C3D 与 Gaitway 时间没有重叠")
        guard = min(1.0, max(0.0, (overlap_end - overlap_start) / 4.0))
        a0, a1 = overlap_start + guard, overlap_end - guard
        analysis_range = (a0, a1)
        steady_state = {
            "start_s": a0,
            "end_s": a1,
            "method": "quiet_standing",
            "n_steps": 0,
            "n_cycles": 0,
            "median_step_period_s": None,
            "step_period_cv": None,
            "reason": "静止站立：采用 C3D/Gaitway 重叠区间并去除首尾边缘",
            "params": {"edge_guard_s": guard},
        }
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

    if backward:
        steady_state["event_semantics"] = "vertical_force_contact_onset; does not imply heel-first landing"

    # Body heading and walking direction are independent. Keep forward yaw;
    # reverse heading uses a constant proper half-turn about ground Y.
    static_geom_names, static_geom = extract_hh19(static_data)
    dynamic_geom_names, dynamic_geom = extract_hh19(dynamic_data)
    static_geom = mocap_to_opensim_points(static_geom)
    dynamic_geom = mocap_to_opensim_points(dynamic_geom)
    static_selected = (static_data.time_s >= static_window["start_s"]) & (static_data.time_s <= static_window["end_s"])
    dynamic_selected = (dynamic_data.time_s >= analysis_range[0]) & (dynamic_data.time_s <= analysis_range[1])
    static_rotation, static_heading = body_heading(static_geom_names, static_geom, static_selected, required=backward)
    dynamic_rotation, dynamic_heading = body_heading(dynamic_geom_names, dynamic_geom, dynamic_selected, required=backward)

    # 1. 静态 / 动态 TRC（model ground 帧，mm）；viewer reads these exact TRCs.
    static_names, _ = build_trc(
        static_data, out / "static.trc", opensim_frame=True, cutoff_hz=marker_cutoff_hz,
        ground_rotation=static_rotation,
    )
    dyn_names, dyn_traj_mm = build_trc(
        dynamic_data, out / "dynamic.trc", opensim_frame=True, cutoff_hz=marker_cutoff_hz,
        ground_rotation=dynamic_rotation,
    )

    standing_centers = None
    if quiet_standing:
        # Keep filtering and ordinary forward standing outputs unchanged. The
        # standing estimator expects lab ground, before any heading half-turn.
        lab_dyn_traj_mm = (dyn_traj_mm if np.array_equal(dynamic_rotation, np.eye(3))
                           else dyn_traj_mm @ dynamic_rotation)
        marker_index = {name: index for index, name in enumerate(dyn_names)}
        required = ("R.Heel", "R.Toe", "L.Heel", "L.Toe")
        missing = [name for name in required if name not in marker_index]
        if missing:
            raise ValueError(f"站立载荷估计缺少脚部 marker：{missing}")
        standing_centers = {
            "right": np.nanmean(
                lab_dyn_traj_mm[:, [marker_index["R.Heel"], marker_index["R.Toe"]], :],
                axis=1,
            ) / 1000.0,
            "left": np.nanmean(
                lab_dyn_traj_mm[:, [marker_index["L.Heel"], marker_index["L.Toe"]], :],
                axis=1,
            ) / 1000.0,
        }

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
        require_grade=True,
        standing_foot_centers_opensim_m=standing_centers,
    )
    mapping_audit = None
    diagnostic_times = dynamic_data.time_s[dynamic_selected] + float(gaitway_offset_s)
    if gaitway_qc["decomposition_method"] == "native_bilateral" and len(diagnostic_times):
        gaitway_qc["native_wrench_consistency"] = audit_native_bilateral_wrench(
            gaitway, time_range_s=(float(diagnostic_times[0]), float(diagnostic_times[-1])),
            diagnostic_contact_threshold_N=50.0,
        )
    if (backward or dynamic_heading["heading"] == "reverse") and gaitway_qc["decomposition_method"] == "native_bilateral":
        feet, mapping_audit = anatomical_load_mapping(
            feet, dynamic_geom_names, dynamic_geom, dynamic_selected,
            force_threshold_N=force_threshold_N,
        )
        gaitway_qc["anatomical_mapping"] = mapping_audit
        if mapping_audit["native_to_anatomical"]["R"] == "left":
            gaitway_qc["n_right_contact_frames"], gaitway_qc["n_left_contact_frames"] = (
                gaitway_qc["n_left_contact_frames"], gaitway_qc["n_right_contact_frames"])
    # Incline is already applied in the lab frame. Rotate all spatial load
    # quantities together, retaining the anatomical joint-moment signs.
    feet = rotate_loads(feet, dynamic_rotation)
    if backward:
        gaitway_qc["backward_decomposition_unvalidated"] = True
        gaitway_qc["assumptions"].append("倒走单支撑映射检查不验证厂商双支撑分力算法；倒走结果必须标记WARN并复核")
    write_grf_mot(out / "grf.mot", dynamic_data.time_s, feet)
    write_external_loads_xml(out / "external_loads.xml", "grf.mot")
    # 双侧有效：两腿 ID 结果在分解有效区间内都有效（含摆动与双支撑）。
    right_valid = feet[0].get("valid", decomposed_valid).copy()
    left_valid = feet[1].get("valid", decomposed_valid).copy()
    support_mask = np.column_stack([right_valid, left_valid])
    gaitway_qc["n_right_valid_frames"] = int(right_valid.sum())
    gaitway_qc["n_left_valid_frames"] = int(left_valid.sum())
    np.save(out / "support_mask.npy", support_mask)
    np.save(out / "analysis_mask.npy", support_mask & dynamic_selected[:, None])
    np.save(out / "contact_mask.npy", np.column_stack([foot.get("contact", decomposed_valid) for foot in feet]))

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
        "analysis_mask": str(out / "analysis_mask.npy"),
        "contact_mask": str(out / "contact_mask.npy"),
        "inputs": inputs,
        "analysis_time_range_s": list(analysis_range),
        "steady_state": steady_state,
        "static_time_range_s": [static_window["start_s"], static_window["end_s"]],
        "static_window": static_window,
        "processing": {
            "force_transform_version": FORCE_TRANSFORM_VERSION,
            "grf_input_policy_version": GRF_INPUT_POLICY_VERSION,
            "support_mask_semantics": "per_anatomical_side_finite_native_wrench_and_filter_edge_quality; not contact threshold",
            "analysis_mask_semantics": "support_mask intersect dynamic C3D analysis interval; truth requires both sides",
            "force_sign_application": "native_before_rotation",
            "marker_cutoff_hz": marker_cutoff_hz,
            "grf_cutoff_hz": grf_cutoff_hz,
            "opensim_force_x_sign": float(opensim_x_sign),
            "opensim_force_z_sign": float(opensim_z_sign),
            "marker_adjustment_expert_confirmed": bool(marker_adjustment_expert_confirmed),
            "quiet_standing": bool(quiet_standing),
            "walking_direction": "backward" if backward else "unspecified_or_forward",
            "direction_policy_version": DIRECTION_POLICY_VERSION,
            "condition_code": condition,
            "body_orientation": {"static": static_heading, "dynamic": dynamic_heading},
            "joint_moment_sign_convention": "OpenSim anatomical coordinate; no direction-dependent negation",
            "viewer_coordinate_frame": "same normalized ground as dynamic.trc/grf.mot",
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
    if backpack is not None:
        manifest["backpack"] = backpack
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "manifest_path": str(manifest_path),
        "support_mask": str(out / "support_mask.npy"),
        "analysis_mask": str(out / "analysis_mask.npy"),
        "contact_mask": str(out / "contact_mask.npy"),
        "grf_input_policy_version": GRF_INPUT_POLICY_VERSION,
        "processing": manifest["processing"],
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
