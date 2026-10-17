"""Bounded read-only source audit: one forward and two backward input cases.

Writes only to the requested output folder, never to source trial folders.
Does not run OpenSim or produce ground truth.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.synchronization.sync import run_auto_sync
from pipeline.c3d.reader import read_c3d
from pipeline.gaitway import read_gaitway_ascii, build_bilateral_grf
from pipeline.gait.steady_state import detect_steady_walking
from pipeline.transforms import R_FP_TO_MOCAP_DEFAULT, mocap_to_opensim_points
from pipeline.opensim_io.build_trc import extract_hh19
from pipeline.opensim_io.prep_session import prepare_session
from pipeline.opensim_io.walking_direction import anatomical_load_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    static = next(args.data_root.glob("**/STATIC_CALIB/**/*.c3d"))
    model = Path(__file__).resolve().parents[1] / "data/models/gait2392/gait2392_simbody.osim"
    report = {"scope": "one forward and two backward inputs; this script does not run OpenSim/ID or export ground truth", "cases": []}
    for condition in ("DWALK_0P6_NOEXO", "BWALK_0P3_NOEXO", "BWALK_0P4_NOEXO"):
        trial = next(p for p in sorted(args.data_root.glob(f"**/{condition}/**/*.c3d")) if p.with_suffix(".txt").is_file())
        case = {"condition": condition, "dynamic_c3d": str(trial)}
        gaitway = read_gaitway_ascii(trial.with_suffix(".txt"))
        case["source_force_audit"] = {
            "n_rows": len(gaitway.time_s), "duration_s": float(gaitway.time_s[-1] - gaitway.time_s[0]),
            "total_fz_min_N": float(np.nanmin(gaitway.columns["GRFz vertical (N)"])),
            "total_fz_max_N": float(np.nanmax(gaitway.columns["GRFz vertical (N)"])),
            "bilateral_finite_count": {s: int(np.isfinite(gaitway.columns[f"Fz{s}(N)"]).sum()) for s in ("R", "L")},
        }
        try:
            sync = run_auto_sync(trial, trial.parent / "mocap.h5", trial.parent / "imu.h5", trial.with_suffix(".txt"))
            case["sync"] = {k: sync.get(k) for k in ("confidence", "gaitway_offset_s")}
            if sync["confidence"] != "HIGH":
                raise ValueError("Automatic sync is not HIGH; manual review required")
            offset = sync["gaitway_offset_s"]
            # Audit BOTH mappings even for the unchanged forward legacy path.
            data = read_c3d(trial)
            gaitway = read_gaitway_ascii(trial.with_suffix(".txt"))
            steady = detect_steady_walking(gaitway.time_s, gaitway.columns["FzR(N)"])
            selected = (data.time_s >= steady.start_s - offset) & (data.time_s <= steady.end_s - offset)
            names, points = extract_hh19(data)
            feet, _, _ = build_bilateral_grf(gaitway, data.time_s, offset, R_FP_TO_MOCAP_DEFAULT,
                                           opensim_x_sign=-1, opensim_z_sign=-1, force_threshold_N=50, require_grade=True)
            _, case["mapping_geometry"] = anatomical_load_mapping(feet, names, mocap_to_opensim_points(points), selected)
            summary = prepare_session(static_c3d_path=static, dynamic_c3d_path=trial,
                                      gaitway_txt_path=trial.with_suffix(".txt"), generic_model_path=model,
                                      out_dir=(args.out_dir / condition).resolve(), subject_id="501",
                                      mass_kg=float(gaitway.metadata["Patient weight (kg)"]),
                                      height_m=float(gaitway.metadata["Patient height (m)"]),
                                      gaitway_offset_s=offset, sync_confidence=sync["confidence"],
                                      sync_quality=sync, condition_code=condition)
            manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
            case["status"] = "PREPROCESS_OK"
            case["manifest_path"] = summary["manifest_path"]
            case["orientation"] = manifest["processing"]["body_orientation"]
            case["analysis_time_range_s"] = manifest["analysis_time_range_s"]
            case["backward_decomposition_unvalidated"] = manifest["gaitway"].get("backward_decomposition_unvalidated", False)
        except Exception as exc:
            case["status"] = "REJECTED"
            case["error"] = str(exc)
            if case["source_force_audit"]["duration_s"] < 1:
                case["diagnosis"] = "Gaitway TXT truncated/wrong export: cannot contain three stomps and walking; re-export complete matching trial. Does not imply stomps were not performed."
        report["cases"].append(case)
        print(json.dumps(case, ensure_ascii=True))
    (args.out_dir / "direction_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
