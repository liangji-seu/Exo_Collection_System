"""Config 校验：把 config 里的 TODO/BLOCKING 字段转成 Preflight 状态。

这是 blocking-state 机制的落地。任何未确认的信息（标定矩阵、坐标轴、左右脚
GRF、受试者质量、模型路径……）都被显式标成 BLOCKING，绝不让下游"假装能跑"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..blocking import Preflight, Status
from .validate_transform import validate_rotation


def _is_todo(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        return s == "" or s.startswith("todo") or s in ("null", "none")
    if isinstance(value, (list, dict, tuple)):
        return len(value) == 0
    return False


def _file_status(value: Any) -> tuple[Status, str]:
    if _is_todo(value):
        return Status.BLOCKING, "未提供路径（TODO_BLOCKING）"
    p = Path(str(value))
    if p.is_file():
        return Status.READY, str(p)
    return Status.BLOCKING, f"路径不存在：{p}"


def validate_config(config: dict[str, Any], *,
                    inspection: dict[str, Any] | None = None) -> Preflight:
    pre = Preflight()
    files = config.get("files", {}) or {}
    subject = config.get("subject", {}) or {}
    marker = config.get("marker", {}) or {}
    transforms = config.get("transforms", {}) or {}
    force = config.get("force", {}) or {}

    # --- files / subject -------------------------------------------- #
    for key, label in (("static_c3d", "static C3D"), ("dynamic_c3d", "dynamic C3D")):
        status, reason = _file_status(files.get(key))
        pre.add(key, status, f"{label}: {reason}")

    model_path = files.get("generic_model", {}).get("path") if isinstance(files.get("generic_model"), dict) else files.get("generic_model")
    status, reason = _file_status(model_path)
    pre.add("generic_model", status, f"generic model (gait2392): {reason}")

    mass = subject.get("mass_kg")
    if _is_todo(mass):
        pre.add("subject_mass", Status.BLOCKING, "受试者体重缺失（Scale/ID 必需）")
    else:
        pre.add("subject_mass", Status.READY, f"{mass} kg")

    height = subject.get("height_m")
    pre.add("subject_height", Status.WARNING if _is_todo(height) else Status.READY,
            "身高（recommended）" + ("" if not _is_todo(height) else " 缺失，部分缩放策略受影响"))

    # --- marker ------------------------------------------------------ #
    unit = marker.get("input_unit")
    pre.add("marker_input_unit",
            Status.BLOCKING if _is_todo(unit) else Status.READY,
            f"marker 输入单位：{unit if not _is_todo(unit) else 'TODO_BLOCKING'}")

    mapping = marker.get("mapping", {}) or {}
    mapped = {k: v for k, v in mapping.items() if not _is_todo(v)}
    pre.add("marker_mapping",
            Status.BLOCKING if not mapped else Status.WARNING,
            f"HH19→OpenSim marker mapping：{len(mapped)}/{len(mapping)} 已填"
            if mapping else "HH19→OpenSim marker mapping 未定义（BLOCKING）")

    # --- transforms -------------------------------------------------- #
    fp2mocap = transforms.get("forceplate_to_mocap", {}) or {}
    fp_status = fp2mocap.get("status", "")
    R = fp2mocap.get("rotation_matrix")
    t = fp2mocap.get("translation_m")
    if _is_todo(fp_status) and R is None:
        pre.add("forceplate_to_mocap_transform", Status.BLOCKING,
                "力台→mocap 标定矩阵未标定（需实测 R, t）")
    elif R is None or t is None:
        pre.add("forceplate_to_mocap_transform", Status.BLOCKING,
                "rotation_matrix / translation_m 缺失")
    else:
        rot = validate_rotation(R)
        pre.add("forceplate_to_mocap_transform",
                Status.READY if rot["ok"] else Status.BLOCKING,
                "" if rot["ok"] else rot["reason"])

    m2os = transforms.get("mocap_to_opensim", {}) or {}
    m2os_status = m2os.get("status", "")
    R2 = m2os.get("rotation_matrix")
    if _is_todo(m2os_status) and R2 is None:
        pre.add("mocap_to_opensim_transform", Status.BLOCKING,
                "mocap→OpenSim 轴对应未确认（需实测 Z-up/Y-up 等）")
    elif R2 is None:
        pre.add("mocap_to_opensim_transform", Status.BLOCKING, "rotation_matrix 缺失")
    else:
        rot = validate_rotation(R2)
        pre.add("mocap_to_opensim_transform",
                Status.READY if rot["ok"] else Status.BLOCKING,
                "" if rot["ok"] else rot["reason"])

    # --- force / GRF -------------------------------------------------- #
    convention = force.get("convention")
    pre.add("grf_force_convention",
            Status.BLOCKING if _is_todo(convention) else Status.READY,
            f"力方向约定：{convention if not _is_todo(convention) else 'TODO_BLOCKING'}")

    if inspection is not None:
        grf_mode = inspection.get("force", {}).get("grf_mode", "UNKNOWN")
        if grf_mode == "LEFT_RIGHT":
            pre.add("grf_left_right", Status.READY, "GRF_MODE=LEFT_RIGHT，左右脚已分解")
        elif grf_mode == "TOTAL_ONLY":
            pre.add("grf_left_right", Status.BLOCKING,
                    "GRF_MODE=TOTAL_ONLY，只有合力，双支撑阶段无法唯一确定左右 external load")
        else:
            pre.add("grf_left_right", Status.BLOCKING, f"GRF_MODE={grf_mode}")
        analog = inspection.get("analog", {})
        pre.add("analog_channels",
                Status.READY if analog.get("n_channels") else Status.BLOCKING,
                f"analog 通道数 {analog.get('n_channels', 0)}")
    else:
        pre.add("analog_channels", Status.WARNING, "尚未运行 inspection，通道未知")
        pre.add("grf_left_right", Status.WARNING, "尚未运行 inspection，GRF_MODE 未知")

    # --- OpenSim 绑定 -------------------------------------------------- #
    try:
        import opensim  # noqa: F401
        pre.add("opensim_available", Status.READY, "OpenSim 绑定可用")
    except Exception as exc:  # noqa: BLE001
        pre.add("opensim_available", Status.BLOCKING,
                f"OpenSim 绑定不可用（{type(exc).__name__}），Scale/IK/ID 无法执行")

    # --- 下游 stage 的输出（初始必然 BLOCKING）-------------------------- #
    pre.add("scaled_model", Status.BLOCKING, "尚未运行 Scale")
    pre.add("ik_motion", Status.BLOCKING, "尚未运行 IK")
    pre.add("grf_mot", Status.BLOCKING, "尚未生成 GRF（依赖标定 + 左右脚分离）")
    pre.add("id_results", Status.BLOCKING, "尚未运行 ID")

    return pre


__all__ = ["validate_config", "_is_todo"]
