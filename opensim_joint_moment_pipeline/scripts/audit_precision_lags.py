"""Run ID for prebuilt lag candidates and choose the lowest-residual result."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import opensim as osim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.opensim_io.run_opensim import _id_setup_xml


def read_table(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines.index("endheader") + 1
    return lines[header].split(), np.loadtxt(lines[header + 1:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mass-kg", type=float, default=80.0)
    parser.add_argument("--time-range", nargs=2, type=float, default=[12.0, 30.0])
    args = parser.parse_args()
    out = Path(args.out)
    candidates = json.loads((out / "lag_candidates.json").read_text(encoding="utf-8"))
    model = out / "hh19_precision_refined_2.osim"
    ik = out / "hh19_precision_ik.mot"
    mask = np.load(out / "support_mask_window.npy")
    records = []
    original_cwd = os.getcwd()
    os.chdir(out)
    try:
        for candidate in candidates:
            lag = int(candidate["lag_ms"])
            result_name = f"hh19_precision_id_lag{lag}.mot"
            setup_name = f"id_precision_lag{lag}_setup.xml"
            Path(setup_name).write_text(_id_setup_xml(
                model.name, ik.name, Path(candidate["external_loads"]).name,
                result_name, args.time_range[0], args.time_range[1]), encoding="utf-8")
            osim.InverseDynamicsTool(setup_name).run()
            columns, values = read_table(Path(result_name))
            residual_components = values[:, [
                columns.index("pelvis_tx_force"),
                columns.index("pelvis_ty_force"),
                columns.index("pelvis_tz_force"),
            ]]
            residual = np.linalg.norm(residual_components, axis=1)
            valid = mask.any(axis=1)
            hips = {}
            for side, mask_col in (("r", 0), ("l", 1)):
                series = values[:, columns.index(f"hip_flexion_{side}_moment")]
                selected = series[mask[:, mask_col]]
                hips[side] = {
                    "max_abs_Nm": float(np.max(np.abs(selected))),
                    "p95_abs_Nm_per_kg": float(np.percentile(np.abs(selected), 95) / args.mass_kg),
                }
            records.append({
                "lag_ms": lag,
                "residual_rms_N": float(np.sqrt(np.mean(np.square(residual[valid])))),
                "residual_p95_N": float(np.percentile(residual[valid], 95)),
                "component_rms_N": {
                    axis: float(np.sqrt(np.mean(np.square(residual_components[valid, i]))))
                    for i, axis in enumerate(("x_forward", "y_vertical", "z_right"))
                },
                "hip": hips,
                "id_file": str(out / result_name),
            })
    finally:
        os.chdir(original_cwd)
    best = min(records, key=lambda item: item["residual_rms_N"])
    report = {"best_lag_ms": best["lag_ms"], "best": best, "candidates": records}
    (out / "lag_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
