"""Refine marker locations and run a conservative single-support IK/ID result.

This script intentionally writes to a separate precision directory.  It uses a
robust median marker location over a user-selected complete-marker walking
window, then re-runs IK and ID twice to remove systematic marker offsets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import opensim as osim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.opensim_io.run_opensim import _id_setup_xml, _ik_setup_xml, _trc_info


def _read_table(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines.index("endheader") + 1
    return lines[header].split(), np.loadtxt(lines[header + 1:])


def _read_trc(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    names = [name for name in lines[3].split("\t")[2::3] if name]
    values = np.genfromtxt(lines[5:])
    return names, values[:, 1], values[:, 2:].reshape(len(values), len(names), 3) / 1000.0


def _apply_pose(model: osim.Model, state, columns: list[str], row: np.ndarray) -> None:
    coordinates = model.getCoordinateSet()
    lookup = {coordinates.get(i).getName(): coordinates.get(i) for i in range(coordinates.getSize())}
    for j, name in enumerate(columns[1:], 1):
        if name not in lookup:
            continue
        value = float(row[j])
        if name not in ("pelvis_tx", "pelvis_ty", "pelvis_tz"):
            value = math.radians(value)
        lookup[name].setValue(state, value, False)
    state.setTime(float(row[0]))
    model.realizePosition(state)


def _refine_marker_locations(
    model_file: Path,
    ik_file: Path,
    trc_file: Path,
    output_model: Path,
    time_range: tuple[float, float],
    *,
    sample_stride: int = 5,
    max_adjustment_m: float = 0.06,
) -> dict:
    columns, motion = _read_table(ik_file)
    marker_names, trc_time, experimental = _read_trc(trc_file)
    model = osim.Model(str(model_file))
    state = model.initSystem()
    markers = model.getMarkerSet()
    marker_lookup = {markers.get(i).getName(): markers.get(i) for i in range(markers.getSize())}
    ground = model.getGround()
    local_samples: dict[str, list[list[float]]] = {name: [] for name in marker_names}
    start, end = time_range

    for motion_i in range(0, len(motion), sample_stride):
        time = float(motion[motion_i, 0])
        if time < start or time > end:
            continue
        trc_i = int(np.argmin(np.abs(trc_time - time)))
        if abs(float(trc_time[trc_i]) - time) > 0.006:
            continue
        _apply_pose(model, state, columns, motion[motion_i])
        for marker_i, name in enumerate(marker_names):
            point = experimental[trc_i, marker_i]
            if not np.isfinite(point).all() or name not in marker_lookup:
                continue
            marker = marker_lookup[name]
            parent = marker.getParentFrame()
            local = ground.findStationLocationInAnotherFrame(
                state, osim.Vec3(float(point[0]), float(point[1]), float(point[2])), parent)
            local_samples[name].append([local.get(0), local.get(1), local.get(2)])

    report: dict[str, dict] = {}
    for name, samples in local_samples.items():
        if len(samples) < 10 or name not in marker_lookup:
            report[name] = {"updated": False, "n_samples": len(samples)}
            continue
        marker = marker_lookup[name]
        old = np.array([marker.get_location().get(i) for i in range(3)], dtype=np.float64)
        target = np.median(np.asarray(samples, dtype=np.float64), axis=0)
        delta = target - old
        norm = float(np.linalg.norm(delta))
        if norm > max_adjustment_m:
            target = old + delta * (max_adjustment_m / norm)
        marker.set_location(osim.Vec3(*[float(value) for value in target]))
        report[name] = {
            "updated": True,
            "n_samples": len(samples),
            "adjustment_mm": [round(float(value * 1000.0), 2) for value in (target - old)],
            "adjustment_norm_mm": round(float(np.linalg.norm(target - old) * 1000.0), 2),
        }
    model.finalizeConnections()
    model.printToXML(str(output_model))
    return report


def _marker_qc(model_file: Path, ik_file: Path, trc_file: Path) -> dict:
    columns, motion = _read_table(ik_file)
    marker_names, trc_time, experimental = _read_trc(trc_file)
    model = osim.Model(str(model_file))
    state = model.initSystem()
    markers = model.getMarkerSet()
    lookup = {markers.get(i).getName(): markers.get(i) for i in range(markers.getSize())}
    errors = {name: [] for name in marker_names}
    frame_rms: list[float] = []
    frame_max: list[float] = []
    for row in motion:
        time = float(row[0])
        trc_i = int(np.argmin(np.abs(trc_time - time)))
        _apply_pose(model, state, columns, row)
        current: list[float] = []
        for marker_i, name in enumerate(marker_names):
            point = experimental[trc_i, marker_i]
            if not np.isfinite(point).all() or name not in lookup:
                continue
            predicted = lookup[name].getLocationInGround(state)
            error = float(np.linalg.norm(np.array([
                predicted.get(0), predicted.get(1), predicted.get(2)]) - point))
            errors[name].append(error)
            current.append(error)
        if current:
            frame_rms.append(float(np.sqrt(np.mean(np.square(current)))))
            frame_max.append(float(np.max(current)))
    return {
        "overall": {
            "rms_mean_cm": float(np.mean(frame_rms) * 100.0),
            "rms_p95_cm": float(np.percentile(frame_rms, 95) * 100.0),
            "max_marker_p95_cm": float(np.percentile(frame_max, 95) * 100.0),
            "max_marker_max_cm": float(np.max(frame_max) * 100.0),
        },
        "markers": {
            name: {
                "mean_cm": float(np.mean(values) * 100.0),
                "p95_cm": float(np.percentile(values, 95) * 100.0),
                "max_cm": float(np.max(values) * 100.0),
            }
            for name, values in errors.items() if values
        },
    }


def _result_qc(id_file: Path, mask: np.ndarray, mass_kg: float) -> dict:
    columns, values = _read_table(id_file)
    right = values[:, columns.index("hip_flexion_r_moment")]
    left = values[:, columns.index("hip_flexion_l_moment")]
    residual = np.linalg.norm(values[:, [
        columns.index("pelvis_tx_force"),
        columns.index("pelvis_ty_force"),
        columns.index("pelvis_tz_force"),
    ]], axis=1)

    def stats(series: np.ndarray, valid: np.ndarray) -> dict:
        selected = series[valid & np.isfinite(series)]
        return {
            "n_valid": int(len(selected)),
            "min_Nm": float(np.min(selected)),
            "max_Nm": float(np.max(selected)),
            "p95_abs_Nm": float(np.percentile(np.abs(selected), 95)),
            "p95_abs_Nm_per_kg": float(np.percentile(np.abs(selected), 95) / mass_kg),
        }

    any_valid = mask.any(axis=1)
    return {
        "hip_flexion_r": stats(right, mask[:, 0]),
        "hip_flexion_l": stats(left, mask[:, 1]),
        "residual_force": {
            "rms_N": float(np.sqrt(np.mean(np.square(residual[any_valid])))),
            "p95_N": float(np.percentile(residual[any_valid], 95)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed-model", required=True)
    parser.add_argument("--seed-ik", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = Path(manifest["out_dir"])
    time_range = tuple(float(value) for value in manifest["analysis_time_range_s"])
    dynamic_trc = out / Path(manifest["dynamic_trc"]).name
    external_loads = out / Path(manifest["external_loads"]).name
    seed_model = Path(args.seed_model)
    seed_ik = Path(args.seed_ik)
    marker_names, _ = _trc_info(str(dynamic_trc))

    model_1 = out / "hh19_precision_refined_1.osim"
    model_2 = out / "hh19_precision_refined_2.osim"
    ik_1 = out / "hh19_precision_ik_1.mot"
    ik_final = out / "hh19_precision_ik.mot"
    id_final = out / "hh19_precision_id.mot"

    refinement_1 = _refine_marker_locations(
        seed_model, seed_ik, dynamic_trc, model_1, time_range)

    original_cwd = os.getcwd()
    os.chdir(out)
    try:
        Path("ik_precision_1_setup.xml").write_text(_ik_setup_xml(
            model_1.name, dynamic_trc.name, marker_names, ik_1.name,
            time_range[0], time_range[1]), encoding="utf-8")
        osim.InverseKinematicsTool("ik_precision_1_setup.xml").run()
        error_file = Path("hh19_ik_ik_marker_errors.sto")
        if error_file.exists():
            shutil.copyfile(error_file, "hh19_precision_ik_1_marker_errors.sto")
    finally:
        os.chdir(original_cwd)

    refinement_2 = _refine_marker_locations(
        model_1, ik_1, dynamic_trc, model_2, time_range, max_adjustment_m=0.025)

    os.chdir(out)
    try:
        Path("ik_precision_setup.xml").write_text(_ik_setup_xml(
            model_2.name, dynamic_trc.name, marker_names, ik_final.name,
            time_range[0], time_range[1]), encoding="utf-8")
        osim.InverseKinematicsTool("ik_precision_setup.xml").run()
        error_file = Path("hh19_ik_ik_marker_errors.sto")
        if error_file.exists():
            shutil.copyfile(error_file, "hh19_precision_ik_marker_errors.sto")
        Path("id_precision_setup.xml").write_text(_id_setup_xml(
            model_2.name, ik_final.name, external_loads.name, id_final.name,
            time_range[0], time_range[1]), encoding="utf-8")
        osim.InverseDynamicsTool("id_precision_setup.xml").run()
    finally:
        os.chdir(original_cwd)

    full_mask = np.load(manifest["support_mask"])
    id_columns, id_values = _read_table(id_final)
    full_times = np.arange(len(full_mask), dtype=np.float64) / 100.0
    indices = np.array([int(np.argmin(np.abs(full_times - time))) for time in id_values[:, 0]])
    window_mask = full_mask[indices]
    np.save(out / "support_mask_window.npy", window_mask)

    report = {
        "processing": manifest.get("processing", {}),
        "analysis_time_range_s": list(time_range),
        "refinement_1": refinement_1,
        "refinement_2": refinement_2,
        "marker_qc": _marker_qc(model_2, ik_final, dynamic_trc),
        "id_qc": _result_qc(id_final, window_mask, float(manifest["subject"]["mass_kg"])),
        "files": {
            "model": str(model_2),
            "ik": str(ik_final),
            "id": str(id_final),
            "support_mask": str(out / "support_mask_window.npy"),
        },
    }
    (out / "precision_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["marker_qc"]["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(report["id_qc"], ensure_ascii=False, indent=2))
    print(out / "precision_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
