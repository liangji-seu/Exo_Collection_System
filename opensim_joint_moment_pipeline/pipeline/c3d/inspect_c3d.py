"""C3D inspection：产出结构化 JSON + Markdown 报告。

在**不安装 OpenSim、没有标定矩阵**的情况下也能运行（只依赖 ezc3d + numpy）。

报告四块：
A. C3D 基本信息（POINT / ANALOG / FORCE_PLATFORM 原始 metadata）
B. Marker 清单（real / suspected_virtual / unknown，missing ratio，NaN，有效帧范围）
C. Force / analog 通道（识别 Fx..Tz、左右脚、GRF_MODE）
D. 时间同步（point vs analog rate、整数倍关系、trigger 通道）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .extract_forces import force_channel_summary
from .reader import C3dData, read_c3d

# Helen Hayes 19 下肢模型的已知真实 marker（去掉 subject 前缀后精确匹配）
_HH19_REAL = {
    "R.ASIS", "L.ASIS",
    "R.Thigh", "L.Thigh",
    "R.Knee", "L.Knee", "R.Knee.Medial", "L.Knee.Medial",
    "R.Shank", "L.Shank",
    "R.Ankle", "L.Ankle", "R.Ankle.Medial", "L.Ankle.Medial",
    "R.Heel", "L.Heel",
    "R.Toe", "L.Toe",
}
# 已知虚拟点（HH 模型算出的关节中心/骶骨/骨盆原点等）
_HH19_VIRTUAL = {
    "V.Sacral", "V_Mid_ASIS", "V_Pelvis_Origin", "V_Mid_Hip",
    "V_R.Hip_JC", "V_L.Hip_JC", "V_R.Knee_JC", "V_L.Knee_JC",
    "V_R.Ankle_JC", "V_L.Ankle_JC",
    "V_R.Toe_Offset", "V_L.Toe_Offset",
}


def _strip_subject(label: str, data: C3dData) -> str:
    for s in data.subjects:
        if label.startswith(s.prefix):
            return label[len(s.prefix):]
    return label


def _classify_marker(short: str) -> str:
    if short in _HH19_REAL:
        return "real"
    if short in _HH19_VIRTUAL or short.startswith(("V_", "V.")):
        return "suspected_virtual"
    return "unknown"


def _marker_stats(data: C3dData, index: int) -> dict[str, Any]:
    traj = data.points_mm[:, index, :]          # (n_frames, 3)
    res = data.residuals[:, index]
    is_nan = np.isnan(traj).any(axis=1)
    is_zero = np.all(np.abs(traj) < 1e-6, axis=1)
    missing = is_nan | is_zero
    valid_idx = np.where(~missing)[0]
    n_frames = data.n_frames
    return {
        "nan_count": int(is_nan.sum()),
        "zero_count": int(is_zero.sum()),
        "missing_count": int(missing.sum()),
        "missing_ratio": float(missing.mean()) if n_frames else float("nan"),
        "n_valid_frames": int(valid_idx.size),
        "first_valid_frame": int(valid_idx[0]) if valid_idx.size else None,
        "last_valid_frame": int(valid_idx[-1]) if valid_idx.size else None,
        "residual_min": float(np.nanmin(res)) if res.size else float("nan"),
        "residual_max": float(np.nanmax(res)) if res.size else float("nan"),
        "n_negative_residual": int((res < 0).sum()),
    }


def _fp_metadata(data: C3dData) -> list[dict[str, Any]]:
    out = []
    for fp in data.force_platforms:
        out.append({
            "index": fp.index,
            "type": fp.type,
            "channels": list(fp.channels),
            "corners_mm": np.round(fp.corners, 3).tolist(),
            "origin_mm": np.round(fp.origin, 3).tolist(),
            "cal_matrix": (np.round(fp.cal_matrix, 5).tolist()
                           if fp.cal_matrix is not None else None),
        })
    return out


def inspect(data: C3dData) -> dict[str, Any]:
    """返回 JSON 可序列化的 inspection 结果 dict。"""
    force = force_channel_summary(data)

    markers = []
    for index, label in enumerate(data.point_labels):
        short = _strip_subject(label, data)
        subject = next((s for s in data.subjects if label.startswith(s.prefix)), None)
        stats = _marker_stats(data, index)
        markers.append({
            "index": index,
            "label": label,
            "short_name": short,
            "subject": subject.name if subject else "-",
            "classification": _classify_marker(short),
            **stats,
        })

    # 时间同步
    pr, ar = data.point_rate_hz, data.analog_rate_hz
    ratio = (ar / pr) if (pr > 0 and ar > 0) else None
    integer_ratio = bool(ratio is not None and abs(ratio - round(ratio)) < 1e-6)
    marker_dur = data.n_frames / pr if pr > 0 else None
    force_dur = data.n_frames / ar if ar > 0 else None

    return {
        "path": str(data.path),
        "manufacturer": data.manufacturer,
        "software": data.software,
        "software_version": data.software_version,

        "point": {
            "rate_hz": pr,
            "units": data.point_units,
            "n_points": len(data.point_labels),
            "n_frames": data.n_frames,
            "data_start": data.data_start,
            "first_frame": data.frame_index[0].item(),
            "last_frame": data.frame_index[-1].item(),
            "duration_s": marker_dur,
        },
        "analog": {
            "rate_hz": ar,
            "units": list(data.analog_units),
            "scale": list(data.analog_scale),
            "n_channels": len(data.analog_labels),
            "channel_names": list(data.analog_labels),
        },
        "force_platforms": _fp_metadata(data),

        "subjects": [
            {
                "name": s.name,
                "prefix": s.prefix,
                "is_static": s.is_static,
                "n_markers": len(s.marker_names),
            }
            for s in data.subjects
        ],
        "markers": markers,
        "marker_class_counts": {
            "real": sum(1 for m in markers if m["classification"] == "real"),
            "suspected_virtual": sum(1 for m in markers if m["classification"] == "suspected_virtual"),
            "unknown": sum(1 for m in markers if m["classification"] == "unknown"),
        },

        "force": force,

        "sync": {
            "point_rate_hz": pr,
            "analog_rate_hz": ar,
            "ratio": ratio,
            "integer_ratio": integer_ratio,
            "marker_time_range_s": [0.0, marker_dur] if marker_dur is not None else None,
            "force_time_range_s": [0.0, force_dur] if force_dur is not None else None,
        },
    }


# --------------------------------------------------------------------- #
# Markdown 报告
# --------------------------------------------------------------------- #
def build_markdown(rep: dict[str, Any]) -> str:
    L: list[str] = []
    a = L.append

    a("# C3D Inspection Report")
    a("")
    a(f"- 文件：`{rep['path']}`")
    a(f"- 厂商：{rep['manufacturer']} · 软件：{rep['software']} {rep['software_version']}")
    a("")

    p, an, sy = rep["point"], rep["analog"], rep["sync"]
    a("## A. C3D 基本信息")
    a("")
    a("### POINT")
    a(f"- rate: {p['rate_hz']} Hz · 单位: {p['units']} · 点数: {p['n_points']}")
    a(f"- 帧: {p['n_frames']} · DATA_START: {p['data_start']} · "
      f"首帧 {p['first_frame']} → 末帧 {p['last_frame']} · 时长 ≈ {p['duration_s']:.2f} s")
    a("")
    a("### ANALOG")
    a(f"- rate: {an['rate_hz']} Hz · 通道数: {an['n_channels']}")
    a(f"- 通道: {', '.join(an['channel_names'])}")
    a(f"- 单位: {an['units']} · scale: {an['scale']}")
    a("")
    a("### FORCE_PLATFORM metadata")
    if rep["force_platforms"]:
        for fp in rep["force_platforms"]:
            a(f"- 力台 {fp['index']}（type {fp['type']}）: 通道 {fp['channels']}")
            a(f"  - CORNERS (mm): {fp['corners_mm']}")
            a(f"  - ORIGIN (mm): {fp['origin_mm']}")
            a(f"  - CAL_MATRIX: {fp['cal_matrix']}")
    else:
        a("- 无 FORCE_PLATFORM 参数组")
    a("")

    a("## B. Marker 清单")
    a("")
    a("| # | label | subject | 分类 | missing% | NaN | 有效帧 | 首/末有效帧 |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in rep["markers"]:
        first = m["first_valid_frame"]
        last = m["last_valid_frame"]
        rng = f"{first}–{last}" if first is not None else "—"
        a(f"| {m['index']} | `{m['label']}` | {m['subject']} | {m['classification']} | "
          f"{100 * m['missing_ratio']:.1f} | {m['nan_count']} | {m['n_valid_frames']} | {rng} |")
    a("")
    a(f"分类统计：real={rep['marker_class_counts']['real']}, "
      f"suspected_virtual={rep['marker_class_counts']['suspected_virtual']}, "
      f"unknown={rep['marker_class_counts']['unknown']}")
    a("")
    a("> 注意：分类仅根据命名推断（V_/V. 前缀 = suspected_virtual），"
      "不做 100% 真实/虚拟判定。")
    a("")

    f = rep["force"]
    a("## C. Force / analog 通道")
    a("")
    a(f"- **GRF_MODE = `{f['grf_mode']}`**")
    a(f"- 识别到的力/力矩分量: {f['force_components_found']}")
    a(f"- 自由力矩 Tz 存在: {f['has_free_moment']} · 六维力矩 Mx/My/Mz 存在: {f['has_six_axis_moment']}")
    if f["other_channels"]:
        a(f"- 未识别通道: {f['other_channels']}")
    a("")
    a("| 通道 | kind | side | 力台编号 |")
    a("| --- | --- | --- | --- |")
    for c in f["channels"]:
        a(f"| {c['label']} | {c['kind']} | {c['side']} | {c['plate_index']} |")
    a("")
    if f["grf_mode"] in ("TOTAL_ONLY", "UNKNOWN"):
        a("> ⚠️ 当前**只有合力/无左右脚分解**：双支撑阶段无法唯一确定左右侧 external load，"
          "双侧 inverse dynamics 必须 BLOCKING。")
    a("")

    a("## D. 时间同步")
    a("")
    a(f"- point rate: {sy['point_rate_hz']} Hz · analog rate: {sy['analog_rate_hz']} Hz")
    a(f"- ratio (analog/point): {sy['ratio']} · 整数倍: {sy['integer_ratio']}")
    a(f"- marker 时间范围: {sy['marker_time_range_s']}")
    a(f"- force 时间范围: {sy['force_time_range_s']}")
    a("")
    return "\n".join(L)


def run_inspection(c3d_path: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    """读 c3d 并产出报告；若给定 out_dir，写入 inspection_report.{json,md}。"""
    data = read_c3d(c3d_path)
    rep = inspect(data)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "inspection_report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "inspection_report.md").write_text(build_markdown(rep), encoding="utf-8")
    return rep


__all__ = ["inspect", "build_markdown", "run_inspection", "read_c3d", "C3dData"]
