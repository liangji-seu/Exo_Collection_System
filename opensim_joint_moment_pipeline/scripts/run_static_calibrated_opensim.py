"""Calibrate HH19 marker locations from a static trial, then run dynamic IK/ID."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import opensim as osim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.opensim_io.run_opensim import _id_setup_xml, _ik_setup_xml, _trc_info  # noqa: E402
from scripts.run_precision_opensim import (  # noqa: E402
    _marker_qc,
    _read_table,
    _refine_marker_locations,
    _result_qc,
)


def _run_ik(
    out: Path, setup_name: str, model: Path, trc: Path, output: Path,
    time_range: tuple[float, float],
) -> None:
    marker_names, _ = _trc_info(str(trc))
    original_cwd = os.getcwd()
    os.chdir(out)
    try:
        Path(setup_name).write_text(
            _ik_setup_xml(
                model.name, trc.name, marker_names, output.name,
                time_range[0], time_range[1],
            ),
            encoding="utf-8",
        )
        osim.InverseKinematicsTool(setup_name).run()
    finally:
        os.chdir(original_cwd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed-model", required=True)
    parser.add_argument("--static-window", nargs=2, type=float, default=(3.45, 5.45))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(manifest["out_dir"])
    seed_model = Path(args.seed_model)
    static_trc = out / Path(manifest["static_trc"]).name
    dynamic_trc = out / Path(manifest["dynamic_trc"]).name
    external_loads = out / Path(manifest["external_loads"]).name
    static_window = tuple(float(value) for value in args.static_window)
    dynamic_window = tuple(float(value) for value in manifest["analysis_time_range_s"])

    static_ik_1 = out / "hh19_static_calibration_ik_1.mot"
    static_model_1 = out / "hh19_static_calibrated_1.osim"
    static_ik_2 = out / "hh19_static_calibration_ik_2.mot"
    final_model = out / "hh19_static_calibrated.osim"
    final_ik = out / "hh19_static_calibrated_ik.mot"
    final_id = out / "hh19_static_calibrated_id.mot"

    _run_ik(out, "ik_static_calibration_1_setup.xml", seed_model, static_trc,
            static_ik_1, static_window)
    refinement_1 = _refine_marker_locations(
        seed_model, static_ik_1, static_trc, static_model_1, static_window,
        sample_stride=2, max_adjustment_m=0.15,
    )
    _run_ik(out, "ik_static_calibration_2_setup.xml", static_model_1, static_trc,
            static_ik_2, static_window)
    refinement_2 = _refine_marker_locations(
        static_model_1, static_ik_2, static_trc, final_model, static_window,
        sample_stride=2, max_adjustment_m=0.04,
    )

    _run_ik(out, "ik_static_calibrated_setup.xml", final_model, dynamic_trc,
            final_ik, dynamic_window)
    original_cwd = os.getcwd()
    os.chdir(out)
    try:
        Path("id_static_calibrated_setup.xml").write_text(
            _id_setup_xml(
                final_model.name, final_ik.name, external_loads.name, final_id.name,
                dynamic_window[0], dynamic_window[1],
            ),
            encoding="utf-8",
        )
        osim.InverseDynamicsTool("id_static_calibrated_setup.xml").run()
    finally:
        os.chdir(original_cwd)

    _, id_values = _read_table(final_id)
    full_mask = np.load(manifest["support_mask"])
    indices = np.clip(np.rint(id_values[:, 0] * 100.0).astype(int), 0, len(full_mask) - 1)
    window_mask = full_mask[indices]
    mask_path = out / "support_mask_static_calibrated.npy"
    np.save(mask_path, window_mask)

    report = {
        "method": "two-pass static-trial marker localization; no dynamic marker fitting",
        "static_window_s": list(static_window),
        "analysis_time_range_s": list(dynamic_window),
        "refinement_1": refinement_1,
        "refinement_2": refinement_2,
        "marker_qc": _marker_qc(final_model, final_ik, dynamic_trc),
        "id_qc": _result_qc(
            final_id, window_mask, float(manifest["subject"]["mass_kg"])
        ),
        "files": {
            "model": str(final_model),
            "ik": str(final_ik),
            "id": str(final_id),
            "support_mask": str(mask_path),
        },
    }
    report_path = out / "static_calibrated_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["marker_qc"]["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(report["id_qc"], ensure_ascii=False, indent=2))
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
