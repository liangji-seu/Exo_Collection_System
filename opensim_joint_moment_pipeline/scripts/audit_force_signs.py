"""Run ID for horizontal GRF sign candidates (OpenSim env)."""

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
    args = parser.parse_args()
    out = Path(args.out)
    candidates = json.loads((out / "force_sign_candidates.json").read_text(encoding="utf-8"))
    model = out / "hh19_precision_refined_2.osim"
    ik = out / "hh19_precision_ik.mot"
    mask = np.load(out / "support_mask_window.npy").any(axis=1)
    records = []
    original_cwd = os.getcwd()
    os.chdir(out)
    try:
        for candidate in candidates:
            label = candidate["label"]
            result = f"hh19_precision_id_{label}.mot"
            setup = f"id_precision_{label}_setup.xml"
            Path(setup).write_text(_id_setup_xml(
                model.name, ik.name, Path(candidate["external_loads"]).name,
                result, 12.0, 30.0), encoding="utf-8")
            osim.InverseDynamicsTool(setup).run()
            columns, values = read_table(Path(result))
            components = values[:, [
                columns.index("pelvis_tx_force"),
                columns.index("pelvis_ty_force"),
                columns.index("pelvis_tz_force"),
            ]]
            total = np.linalg.norm(components, axis=1)
            records.append({
                **candidate,
                "residual_rms_N": float(np.sqrt(np.mean(np.square(total[mask])))),
                "component_rms_N": {
                    axis: float(np.sqrt(np.mean(np.square(components[mask, i]))))
                    for i, axis in enumerate(("x_forward", "y_vertical", "z_right"))
                },
                "id_file": str(out / result),
            })
    finally:
        os.chdir(original_cwd)
    best = min(records, key=lambda item: item["residual_rms_N"])
    report = {"best": best, "candidates": records}
    (out / "force_sign_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
