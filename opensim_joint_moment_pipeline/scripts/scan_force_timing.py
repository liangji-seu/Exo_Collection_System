"""Scan small GRF time shifts around stomp synchronization using ID residuals."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import opensim as osim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.opensim_io.run_opensim import _id_setup_xml  # noqa: E402
from scripts.scan_force_signs import _read_storage, _write_external_loads, _write_storage  # noqa: E402


def _shift(values: np.ndarray, frames: int) -> np.ndarray:
    """Return source(t + frames*dt); positive frames advance GRF on the C3D clock."""
    result = values.copy()
    signal = values[:, 1:]
    shifted = np.zeros_like(signal)
    if frames > 0:
        shifted[:-frames] = signal[frames:]
    elif frames < 0:
        shifted[-frames:] = signal[:frames]
    else:
        shifted[:] = signal
    result[:, 1:] = shifted
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ik", required=True)
    parser.add_argument("--half-width-ms", type=int, default=100)
    parser.add_argument("--step-ms", type=int, default=10)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(manifest["out_dir"])
    columns, source, header = _read_storage(out / Path(manifest["grf_mot"]).name)
    model, ik = Path(args.model), Path(args.ik)
    time_range = tuple(float(v) for v in manifest["analysis_time_range_s"])
    base_offset = float(manifest["gaitway"]["force_time_offset_s"])
    results = []

    original_cwd = os.getcwd()
    os.chdir(out)
    try:
        for shift_ms in range(-args.half_width_ms, args.half_width_ms + 1, args.step_ms):
            frames = int(round(shift_ms / 10.0))
            tag = f"shift_{shift_ms:+d}ms".replace("+", "p").replace("-", "m")
            grf_name, loads_name = f"grf_{tag}.mot", f"external_loads_{tag}.xml"
            id_name, setup_name = f"id_{tag}.mot", f"id_{tag}_setup.xml"
            _write_storage(Path(grf_name), columns, _shift(source, frames), header)
            _write_external_loads(Path(loads_name), grf_name)
            Path(setup_name).write_text(
                _id_setup_xml(model.name, ik.name, loads_name, id_name, *time_range),
                encoding="utf-8",
            )
            osim.InverseDynamicsTool(setup_name).run()
            id_columns, values, _ = _read_storage(Path(id_name))
            residual = np.column_stack([
                values[:, id_columns.index("pelvis_tx_force")],
                values[:, id_columns.index("pelvis_ty_force")],
                values[:, id_columns.index("pelvis_tz_force")],
            ])
            norm = np.linalg.norm(residual, axis=1)
            results.append({
                "shift_ms": shift_ms,
                "candidate_offset_s": base_offset + shift_ms / 1000.0,
                "residual_component_rms_N": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
                "residual_norm_rms_N": float(np.sqrt(np.mean(norm ** 2))),
                "residual_norm_p95_N": float(np.percentile(norm, 95)),
                "id_file": str(out / id_name),
            })
    finally:
        os.chdir(original_cwd)

    results.sort(key=lambda item: item["residual_norm_rms_N"])
    report = {"base_offset_s": base_offset, "ranked_candidates": results}
    path = out / "force_timing_scan.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": results[0], "next": results[1:5]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
