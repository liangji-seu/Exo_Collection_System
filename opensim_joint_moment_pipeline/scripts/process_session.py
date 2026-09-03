"""把一次已同步的会话跑完整条 OpenSim 链（**opensim 环境**：numpy 2.x + opensim 4.6）。

读取 ``prep_session.py`` 写出的 ``manifest.json``，执行：

    通用模型 + HH19 marker → Scale（ModelScaler 静态缩放）
      → 静态 marker 标定（两遍 refine，仅静态 trial）
      → 动态 IK → 双侧 GRF ID → QC 指标

以 **JSON-Lines** 向 stdout 输出进度事件，供 Exo Calculate 的
``OpenSimProcessWorker`` 流式解析；结果写 ``result.json``。

事件协议（每行一个 JSON 对象，``event`` 字段区分）：
    {"event":"start","opensim":"...","out_dir":"..."}
    {"event":"stage","stage":"scale|static_calibration|ik|id|qc","message":"..."}
    {"event":"log","level":"info","message":"..."}
    {"event":"result","exit_code":0,"result_path":"...","qc":{"...":{...}}}
    {"event":"cancelled"}
    {"event":"error","message":"..."}

**关键约定**：退出码 0 只表示「进程正常跑完」，不代表「QC 通过」。QC 结论由
``result.json`` 里的 marker/ID 指标（及 App 侧 QC 规则）单独判定。取消/异常分别
以退出码 2 / 1 返回。

用法：:

    python scripts/process_session.py --manifest <run>/manifest.json [--cancel-file <p>]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opensim as osim  # noqa: E402

from pipeline.opensim_io.export_viewer import export_viewer_data  # noqa: E402
from pipeline.opensim_io.run_opensim import (  # noqa: E402
    _id_setup_xml,
    _ik_setup_xml,
    _scale_setup_xml,
    _trc_info,
    add_hh19_markers,
)
from pipeline.qc.evaluate import evaluate_qc, grade_marker_adjustments  # noqa: E402
from scripts.run_precision_opensim import (  # noqa: E402
    _marker_qc,
    _read_table,
    _refine_marker_locations,
    _result_qc,
)

# gait2392 矢状面关节坐标（ID 输出力矩列 = 坐标名 + "_moment"）
_SAGITTAL = [
    "hip_flexion_r", "hip_flexion_l",
    "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l",
]

_CANCEL_EXIT = 2


class _Cancelled(Exception):
    pass


def _emit(obj: dict) -> None:
    # ensure_ascii=True：让 JSON-Lines 纯 ASCII，避开子进程 stdout 编码差异
    # （Windows 上子进程默认 cp936，父进程按 UTF-8 读），中文一律 \uXXXX 转义。
    sys.stdout.write(json.dumps(obj, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _default_static_window(st_range: tuple[float, float], trim_s: float = 0.5) -> tuple[float, float]:
    """静态窗默认取整段并去掉首尾各 ``trim_s``（避开进入/离开站位）。"""
    t0, t1 = st_range
    if t1 - t0 > 2 * trim_s:
        return (t0 + trim_s, t1 - trim_s)
    return (t0, t1)


def _run_ik(
    out: Path, setup_name: str, model: Path, trc: Path, output: Path,
    time_range: tuple[float, float], check_cancel=None,
) -> None:
    if check_cancel:
        check_cancel()
    marker_names, _ = _trc_info(str(trc))
    Path(setup_name).write_text(
        _ik_setup_xml(
            model.name, trc.name, marker_names, output.name,
            time_range[0], time_range[1],
        ),
        encoding="utf-8",
    )
    osim.InverseKinematicsTool(setup_name).run()


def _moment_summary(id_file: Path, mask: np.ndarray, mass_kg: float) -> dict:
    columns, values = _read_table(id_file)
    joints: dict = {}
    for coord in _SAGITTAL:
        col = f"{coord}_moment"
        if col not in columns:
            joints[coord] = {"missing": True}
            continue
        vals = values[:, columns.index(col)].astype(np.float64)
        side_mask = mask[:, 0] if coord.endswith("_r") else mask[:, 1]
        sel = np.isfinite(vals) & side_mask
        v = vals[sel]
        if v.size == 0:
            joints[coord] = {"missing": True, "n_valid": 0}
            continue
        joints[coord] = {
            "n_valid": int(sel.sum()),
            "min_Nm": float(np.min(v)),
            "max_Nm": float(np.max(v)),
            "mean_Nm": float(np.mean(v)),
            "p95_abs_Nm": float(np.percentile(np.abs(v), 95)),
            "p95_abs_Nm_per_kg": float(np.percentile(np.abs(v), 95) / mass_kg),
        }
    return joints


def _map_mask_to_id_time(full_mask: np.ndarray, id_time: np.ndarray, point_rate: float) -> np.ndarray:
    """把 ID 输出时间（C3D 时间，从 0 起）映射回 support_mask 的行索引。"""
    indices = np.clip(np.rint(id_time * point_rate).astype(int), 0, len(full_mask) - 1)
    return full_mask[indices]


def _run(manifest: dict, cancel_file: str | None) -> dict:
    out = Path(manifest["out_dir"])
    mass_kg = float(manifest["subject"]["mass_kg"])
    point_rate = float(manifest.get("dynamic_point_rate_hz", 100.0))

    static_trc = out / Path(manifest["static_trc"]).name
    dynamic_trc = out / Path(manifest["dynamic_trc"]).name
    external_loads = out / Path(manifest["external_loads"]).name
    support_mask_name = Path(manifest["support_mask"]).name
    generic_dest = out / "gait2392_simbody.osim"

    # 通用模型副本（prep 已写；这里兜底保证自包含）
    if not generic_dest.is_file():
        shutil.copyfile(manifest["generic_model"], generic_dest)

    _, st_range = _trc_info(str(static_trc))
    _, dyn_range = _trc_info(str(dynamic_trc))
    # 静态窗口优先取 prep 阶段选好的 ``static_window``（Scale 与两遍 refine 共用，§3.6），
    # 退回旧的 ``static_time_range_s``，再退回「整段去掉首尾 0.5s」。
    sw = manifest.get("static_window") or {}
    if "start_s" in sw and "end_s" in sw:
        static_window = (float(sw["start_s"]), float(sw["end_s"]))
    elif manifest.get("static_time_range_s"):
        static_window = tuple(manifest.get("static_time_range_s"))
    else:
        static_window = _default_static_window(st_range)
    # IK/ID 覆盖整段动态记录（供动画回放）；关节力矩摘要与 QC 只统计已确认的分析区间
    # （prompt6 §3.5 第 5/7 条）。
    dynamic_window = dyn_range
    analysis_window = tuple(manifest.get("analysis_time_range_s") or dyn_range)

    def check_cancel() -> None:
        if cancel_file and Path(cancel_file).exists():
            raise _Cancelled()

    # OpenSim 只吃 ASCII 相对路径：cwd 切到 out 后一律用裸文件名。
    original_cwd = os.getcwd()
    os.chdir(out)
    try:
        check_cancel()

        # 1. Scale（加 HH19 marker → ModelScaler 静态缩放）
        _emit({"event": "stage", "stage": "scale", "message": "Scale：静态缩放 + HH19 marker"})
        model = osim.Model(generic_dest.name)
        add_hh19_markers(model)
        markers_model = Path("hh19_markers.osim")
        model.printToXML(str(markers_model))
        static_markers, _ = _trc_info(static_trc.name)
        Path("scale_setup.xml").write_text(
            _scale_setup_xml(
                markers_model.name, static_trc.name, static_markers,
                "hh19_scaledOnly.osim", "hh19_scaled.osim", "hh19_static_pose.mot",
                mass_kg, static_window[0], static_window[1],
            ),
            encoding="utf-8",
        )
        osim.ScaleTool("scale_setup.xml").run()

        # 2. 静态 marker 标定（两遍 refine，仅用静态 trial）
        _emit({"event": "stage", "stage": "static_calibration",
               "message": "静态 marker 标定（两遍 refine）"})
        seed_model = Path("hh19_scaledOnly.osim")
        static_ik_1 = Path("hh19_static_calibration_ik_1.mot")
        static_model_1 = Path("hh19_static_calibrated_1.osim")
        static_ik_2 = Path("hh19_static_calibration_ik_2.mot")
        final_model = Path("hh19_static_calibrated.osim")

        _run_ik(out, "ik_static_calibration_1_setup.xml", seed_model, static_trc,
                static_ik_1, static_window, check_cancel)
        refinement_1 = _refine_marker_locations(
            seed_model, static_ik_1, static_trc, static_model_1, static_window,
            sample_stride=2, max_adjustment_m=0.15,
        )
        _run_ik(out, "ik_static_calibration_2_setup.xml", static_model_1, static_trc,
                static_ik_2, static_window, check_cancel)
        refinement_2 = _refine_marker_locations(
            static_model_1, static_ik_2, static_trc, final_model, static_window,
            sample_stride=2, max_adjustment_m=0.04,
        )

        # 3. 动态 IK
        _emit({"event": "stage", "stage": "ik", "message": "动态 IK"})
        final_ik = Path("hh19_static_calibrated_ik.mot")
        _run_ik(out, "ik_static_calibrated_setup.xml", final_model, dynamic_trc,
                final_ik, dynamic_window, check_cancel)

        # 4. 双侧 GRF ID
        _emit({"event": "stage", "stage": "id", "message": "双侧 GRF ID"})
        final_id = Path("hh19_static_calibrated_id.mot")
        Path("id_static_calibrated_setup.xml").write_text(
            _id_setup_xml(
                final_model.name, final_ik.name, external_loads.name, final_id.name,
                dynamic_window[0], dynamic_window[1],
            ),
            encoding="utf-8",
        )
        osim.InverseDynamicsTool("id_static_calibrated_setup.xml").run()

        # 5. QC 指标（仍在 out 内，用相对裸名；只算指标，不在此判定 PASS/FAIL）
        _emit({"event": "stage", "stage": "qc", "message": "计算 QC 指标"})
        _, id_values = _read_table(final_id)
        full_mask = np.load(support_mask_name)
        window_mask = _map_mask_to_id_time(full_mask, id_values[:, 0], point_rate)
        # 分析区间内才计入力矩摘要 / 残余力 QC；其余区间仅供回放。
        id_time = id_values[:, 0]
        in_analysis = (id_time >= analysis_window[0]) & (id_time <= analysis_window[1])
        analysis_mask = window_mask & in_analysis[:, None]
        np.save("support_mask_static_calibrated.npy", window_mask)
        np.save("analysis_mask_static_calibrated.npy", analysis_mask)
        marker_qc = _marker_qc(final_model, final_ik, Path(dynamic_trc.name))
        id_qc = _result_qc(final_id, analysis_mask, mass_kg)
        moments = _moment_summary(final_id, analysis_mask, mass_kg)

        # 5b. 版本化 QC 报告（pipeline.qc）：与 App 侧同一规则，落盘供审计。
        # 同步质量 + 力覆盖一并纳入（prompt6 §3.3），静态 marker 调整分级纳入
        # （prompt6 §3.6），保证 result.json / qc_report.json / 界面三者同一个结论。
        marker_adjustment = grade_marker_adjustments(refinement_1)
        qc_report = evaluate_qc(
            marker_qc_overall=marker_qc["overall"],
            id_qc=id_qc,
            mass_kg=mass_kg,
            sync=manifest.get("sync"),
            force=manifest.get("gaitway"),
            dynamic_n_frames=manifest.get("dynamic_n_frames"),
            marker_adjustment=marker_adjustment,
            marker_adjustment_expert_confirmed=bool(
                (manifest.get("processing") or {}).get("marker_adjustment_expert_confirmed", False)
            ),
        )
        Path("qc_report.json").write_text(
            json.dumps(qc_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 6. 导出离线 viewer 数据（EXO 环境回放页只读这些 npy/json，不 import opensim）
        _emit({"event": "stage", "stage": "export", "message": "导出 viewer 数据"})
        viewer_summary = export_viewer_data(
            model_path=final_model,
            ik_path=final_ik,
            id_path=final_id,
            trc_path=dynamic_trc,
            grf_path=Path(Path(manifest["grf_mot"]).name),
            out_dir=out,
            mass_kg=mass_kg,
        )
    finally:
        os.chdir(original_cwd)

    result = {
        "schema_version": "1.0.0",
        "method": "scale + two-pass static-trial marker calibration + dynamic IK + bilateral ID",
        "subject": manifest["subject"],
        "static_window_s": list(static_window),
        "static_window": manifest.get("static_window"),
        "analysis_time_range_s": list(analysis_window),
        "steady_state": manifest.get("steady_state"),
        "full_time_range_s": list(dynamic_window),
        "sync": manifest.get("sync"),
        "refinement_1": refinement_1,
        "refinement_2": refinement_2,
        "marker_adjustment": marker_adjustment,
        "marker_qc": marker_qc,
        "id_qc": id_qc,
        "moments": moments,
        "qc": qc_report,
        "viewer": viewer_summary,
        "files": {
            "model": str(out / final_model.name),
            "ik": str(out / final_ik.name),
            "id": str(out / final_id.name),
            "support_mask": str(out / "support_mask_static_calibrated.npy"),
            "analysis_mask": str(out / "analysis_mask_static_calibrated.npy"),
            "viewer_dir": viewer_summary["viewer_dir"],
        },
    }
    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenSim Scale→标定→IK→ID（opensim 环境）")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cancel-file", default=None)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)

    _emit({
        "event": "start",
        "opensim": osim.GetVersionAndDate(),
        "out_dir": manifest.get("out_dir"),
        "subject": manifest.get("subject", {}).get("id"),
    })

    try:
        result = _run(manifest, args.cancel_file)
    except _Cancelled:
        _emit({"event": "cancelled"})
        return _CANCEL_EXIT
    except Exception as exc:  # noqa: BLE001 —— CLI 边界，转成 JSON 错误事件
        _emit({"event": "error", "message": str(exc)})
        return 1

    _emit({
        "event": "result",
        "exit_code": 0,
        "result_path": result["files"]["id"],
        "viewer_dir": result["files"]["viewer_dir"],
        "marker_qc_overall": result["marker_qc"]["overall"],
        "id_qc": result["id_qc"],
        "moments": result["moments"],
        "qc": result["qc"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
