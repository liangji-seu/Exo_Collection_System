"""Pipeline 编排器：加载配置 → preflight → 执行可运行 stage。

第一阶段（prompt2 §30）目标不是强行得到髋力矩，而是：

> 把 pipeline 搭到「只差真实标定参数即可运行」，并用现有 C3D
> 自动识别可靠的**单支撑区间**。

当前可运行的 stage：
- C3D inspection（静态 + 动态）
- marker / force 提取
- 单支撑检测 → support_phase.csv / segments.json / phase plot
- TRC 写出（marker，mocap 全局 frame，**未**做 mocap→OpenSim 变换）

BLOCKING（缺依赖，见 preflight）：
- forceplate→mocap、mocap→opensim 变换（等你给矩阵）
- Scale / IK / ID（缺 OpenSim 绑定 + gait2392 模型）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .blocking import BlockingReport, Status
from .c3d.reader import read_c3d
from .c3d.inspect_c3d import inspect as inspect_c3d, build_markdown
from .gait.detect_contact import detect_contacts
from .gait.build_support_mask import (
    build_support_mask,
    write_support_csv,
    write_segments_json,
)
from .opensim_io.write_trc import write_trc

# 单支撑检测可用的动态足部 marker 短名
_TRC_MARKERS = [
    "R.ASIS", "L.ASIS", "V.Sacral",
    "R.Thigh", "L.Thigh",
    "R.Knee", "L.Knee",
    "R.Shank", "L.Shank",
    "R.Ankle", "L.Ankle",
    "R.Heel", "L.Heel",
    "R.Toe", "L.Toe",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_preflight(config: dict[str, Any]) -> BlockingReport:
    """根据配置与运行环境生成 preflight 报告。"""
    rep = BlockingReport()
    subject = config.get("subject", {})
    rep.add("subject_height", Status.READY, f"{subject.get('height_m')} m")
    rep.add("subject_mass", Status.READY, f"{subject.get('mass_kg')} kg")

    static = config.get("files", {}).get("static_c3d", "TODO_BLOCKING")
    dynamic = config.get("files", {}).get("dynamic_c3d", "TODO_BLOCKING")
    rep.add("static_c3d", Status.READY if static != "TODO_BLOCKING" else Status.BLOCKING,
            str(static))
    rep.add("dynamic_c3d", Status.READY if dynamic != "TODO_BLOCKING" else Status.BLOCKING,
            str(dynamic))

    rep.add("c3d_inspection", Status.READY)
    rep.add("marker_extraction", Status.READY)
    rep.add("force_extraction", Status.READY)
    rep.add("single_support_detection", Status.READY)
    rep.add("trc_writer", Status.READY, "marker 写盘（mocap frame，未变换）")
    rep.add("grf_writer", Status.READY, "框架就绪，但变换前不写正式 GRF")

    t = config.get("transforms", {})
    fp2mocap = t.get("forceplate_to_mocap", {})
    mocap2osim = t.get("mocap_to_opensim", {})
    rep.add("forceplate_to_mocap_transform",
            Status.READY if fp2mocap.get("rotation_matrix") is not None else Status.BLOCKING,
            "已提供（三点标定 R/t）" if fp2mocap.get("rotation_matrix") is not None else "待你提供测力台位姿 R/t")
    rep.add("mocap_to_opensim_transform",
            Status.READY if mocap2osim.get("rotation_matrix") is not None else Status.BLOCKING,
            "已提供（R=[[0,-1,0],[0,0,1],[-1,0,0]]）" if mocap2osim.get("rotation_matrix") is not None else "待确认 mocap 全局轴 → OpenSim ground 轴")

    model = config.get("files", {}).get("generic_model", "TODO_BLOCKING")
    opensim_ok = _opensim_available()
    rep.add("opensim_bindings", Status.READY if opensim_ok else Status.BLOCKING,
            "import opensim" if opensim_ok else "装在独立 opensim 环境（非本 EXO）")
    rep.add("gait2392_model", Status.READY if model != "TODO_BLOCKING" else Status.BLOCKING,
            str(model))

    rep.add("scale", Status.READY if (opensim_ok and model != "TODO_BLOCKING") else Status.BLOCKING,
            "依赖 opensim + 模型 + 静态数据")
    rep.add("ik", Status.READY if (opensim_ok and model != "TODO_BLOCKING") else Status.BLOCKING)
    rep.add("id", Status.READY if (
        opensim_ok and model != "TODO_BLOCKING"
        and fp2mocap.get("rotation_matrix") is not None
        and mocap2osim.get("rotation_matrix") is not None
    ) else Status.BLOCKING, "依赖标定矩阵 + opensim + 模型")
    return rep


def _opensim_available() -> bool:
    try:
        import opensim  # noqa: F401
        opensim.GetVersionAndDate()  # 防 namespace 空包误判
        return True
    except Exception:
        return False


def _short(label: str, data) -> str:
    for s in data.subjects:
        if label.startswith(s.prefix):
            return label[len(s.prefix):]
    return label


def _missing_mask(points_mm: np.ndarray) -> np.ndarray:
    """(n_frames, n_points) bool：该帧该点是否缺失（全零占位或 NaN）。"""
    p = points_mm.astype(np.float64)
    return np.all(np.abs(p) < 1e-6, axis=2) | np.isnan(p).any(axis=2)


def _dynamic_trc(data) -> tuple[list[str], np.ndarray]:
    """取动态 trial 的 14 个真实下肢 marker（去 medial），返回 (names, (n,3))。

    动态 C3D 会内嵌 static subject 的同名 marker（全缺失，占位 [0,0,0]）。
    因此按短名去重时不能简单取最后一个，而是取「有效帧数最多」的那个，
    避免选到空的 static 副本。
    """
    n_valid = (~_missing_mask(data.points_mm)).sum(axis=0)  # (n_points,)
    by_short: dict[str, int] = {}
    for i, label in enumerate(data.point_labels):
        s = _short(label, data)
        if s not in by_short or n_valid[i] > n_valid[by_short[s]]:
            by_short[s] = i
    names, cols = [], []
    for m in _TRC_MARKERS:
        if m in by_short:
            names.append(m)
            cols.append(by_short[m])
    traj = data.points_mm[:, cols, :].astype(np.float64) if cols else \
        np.empty((data.n_frames, 0, 3))
    # 缺失帧（c3d 的 [0,0,0] 占位）→ NaN，避免 IK 把 marker 误判到原点
    if cols:
        traj[_missing_mask(traj)] = np.nan
    return names, traj


def run_pipeline(config: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """执行可运行 stage，返回结果摘要 dict。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = config["files"]
    static_path = Path(files["static_c3d"])
    dynamic_path = Path(files["dynamic_c3d"])
    subject_id = config["subject"]["id"]

    result: dict[str, Any] = {"subject": subject_id, "stages": {}}

    # 1. inspection
    insp_dir = out / "inspection"
    insp_dir.mkdir(parents=True, exist_ok=True)
    static_data = read_c3d(static_path)
    dynamic_data = read_c3d(dynamic_path)

    static_rep = inspect_c3d(static_data)
    (insp_dir / "static_inspection_report.json").write_text(
        json.dumps(static_rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (insp_dir / "static_inspection_report.md").write_text(
        build_markdown(static_rep), encoding="utf-8")

    dyn_rep = inspect_c3d(dynamic_data)
    (insp_dir / "dynamic_inspection_report.json").write_text(
        json.dumps(dyn_rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (insp_dir / "dynamic_inspection_report.md").write_text(
        build_markdown(dyn_rep), encoding="utf-8")
    result["stages"]["inspection"] = {
        "static_markers": static_rep["marker_class_counts"],
        "dynamic_markers": dyn_rep["marker_class_counts"],
        "grf_mode": dyn_rep["force"]["grf_mode"],
    }

    # 2. single-support detection（动态 trial）
    ss_cfg = config.get("single_support", {})
    gait_dir = out / "gait"
    contact = detect_contacts(
        dynamic_data,
        vertical_axis=None,
        force_threshold_N=float(ss_cfg.get("vertical_force_threshold_N", 50.0)),
        foot_height_threshold_mm=float(ss_cfg.get("foot_height_threshold_mm", 30.0)),
    )
    mask = build_support_mask(
        dynamic_data.time_s,
        contact.right_contact,
        contact.left_contact,
        contact.any_contact,
        trim_boundary_ms=float(ss_cfg.get("trim_boundary_ms", 20.0)),
    )
    write_support_csv(mask, gait_dir / "support_phase.csv")
    write_segments_json(mask, gait_dir / "segments.json")
    _plot_phase(gait_dir / "phase_plot.png", dynamic_data, contact, mask)

    stats = mask.statistics()
    result["stages"]["single_support"] = {
        "vertical_axis": contact.vertical_axis,
        "ground_z_mm": contact.ground_z_mm,
        **{k: v for k, v in stats.items() if k != "segments"},
    }

    # 3. TRC（marker，mocap frame）
    int_dir = out / "intermediate"
    int_dir.mkdir(parents=True, exist_ok=True)
    names, traj = _dynamic_trc(dynamic_data)
    write_trc(int_dir / "dynamic.trc", dynamic_data.time_s, names, traj,
              rate_hz=dynamic_data.point_rate_hz, units="mm")
    result["stages"]["trc"] = {"markers": names, "frames": dynamic_data.n_frames}

    # 4. preflight 报告（落盘）
    preflight = build_preflight(config)
    (out / "preflight.txt").write_text(preflight.render(), encoding="utf-8")
    result["preflight"] = {s.key: s.status.value for s in preflight.stages}
    result["max_executable_stage"] = preflight.max_executable()

    return result


def _plot_phase(path: Path, data, contact, mask) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = data.time_s
    # 足部高度（垂直轴，相对地平面）
    axis = contact.vertical_axis
    heel_r = _mz(data, "R.Heel", axis) - contact.ground_z_mm
    heel_l = _mz(data, "L.Heel", axis) - contact.ground_z_mm
    fz = np.abs(contact.any_contact) if contact.any_contact is not None else np.zeros_like(t)

    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, heel_r, label="R.Heel height", lw=1)
    ax[0].plot(t, heel_l, label="L.Heel height", lw=1)
    ax[0].axhline(0, color="k", ls="--", lw=0.5)
    ax[0].set_ylabel("height (mm)")
    ax[0].legend(loc="upper right")

    ax[1].plot(t, fz, label="|Fz_total|", lw=1)
    ax[1].set_ylabel("|Fz| (N)")
    ax[1].legend(loc="upper right")

    ax[2].fill_between(t, mask.valid_for_id.astype(float), step="mid",
                       color="green", alpha=0.3)
    ax[2].set_ylabel("valid_for_id")
    ax[2].set_ylim(-0.1, 1.1)

    # phase 转数值用于着色
    phase_map = {"RIGHT_SINGLE_SUPPORT": 0, "LEFT_SINGLE_SUPPORT": 1,
                 "DOUBLE_SUPPORT": 2, "NO_CONTACT": 3, "UNKNOWN": 4}
    phase_num = np.array([phase_map.get(p, 4) for p in mask.phase])
    ax[3].plot(t, phase_num, lw=0.5)
    ax[3].set_yticks(list(phase_map.values()))
    ax[3].set_yticklabels(list(phase_map.keys()), fontsize=7)
    ax[3].set_xlabel("time (s)")

    fig.suptitle("Support phase detection")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _mz(data, short_name: str, axis: int) -> np.ndarray:
    from .gait.detect_contact import _marker_trajectory
    return _marker_trajectory(data, short_name)[:, axis]


__all__ = ["load_config", "build_preflight", "run_pipeline"]
