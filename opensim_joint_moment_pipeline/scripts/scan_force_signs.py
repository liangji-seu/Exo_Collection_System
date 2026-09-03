"""Run four horizontal GRF sign candidates and rank them by pelvis residual force."""

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


def _read_storage(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_row = lines.index("endheader") + 1
    columns = lines[header_row].split()
    values = np.loadtxt(lines[header_row + 1:])
    return columns, values, lines[:header_row]


def _write_storage(path: Path, columns: list[str], values: np.ndarray, header: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(header) + "\n")
        stream.write("\t".join(columns) + "\n")
        np.savetxt(stream, values, delimiter="\t", fmt="%.8f")


def _write_external_loads(path: Path, mot_name: str) -> None:
    forces = []
    for name, body, index in (("RightFootGRF", "calcn_r", 1), ("LeftFootGRF", "calcn_l", 2)):
        forces.append(f"""            <ExternalForce name="{name}">
                <applied_to_body>{body}</applied_to_body>
                <force_expressed_in_body>ground</force_expressed_in_body>
                <point_expressed_in_body>ground</point_expressed_in_body>
                <force_identifier>{index}_ground_force_v</force_identifier>
                <point_identifier>{index}_ground_force_p</point_identifier>
                <torque_identifier>{index}_ground_torque_</torque_identifier>
                <datafile>{mot_name}</datafile>
            </ExternalForce>
""")
    path.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n"
        "<OpenSimDocument Version=\"40000\">\n"
        "  <ExternalLoads name=\"gait_grf\">\n"
        "    <objects>\n" + "".join(forces) + "    </objects>\n"
        f"    <datafile>{mot_name}</datafile>\n"
        "  </ExternalLoads>\n"
        "</OpenSimDocument>\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ik", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(manifest["out_dir"])
    model = Path(args.model)
    ik = Path(args.ik)
    source_grf = out / Path(manifest["grf_mot"]).name
    columns, source, header = _read_storage(source_grf)
    time_range = tuple(float(v) for v in manifest["analysis_time_range_s"])
    results = []

    original_cwd = os.getcwd()
    os.chdir(out)
    try:
        for foreaft_sign, lateral_sign in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            tag = f"fx{foreaft_sign:+d}_fz{lateral_sign:+d}".replace("+", "p").replace("-", "m")
            values = source.copy()
            for foot in (1, 2):
                values[:, columns.index(f"{foot}_ground_force_vx")] *= foreaft_sign
                values[:, columns.index(f"{foot}_ground_force_vz")] *= lateral_sign
            grf_name = f"grf_{tag}.mot"
            loads_name = f"external_loads_{tag}.xml"
            id_name = f"id_{tag}.mot"
            setup_name = f"id_{tag}_setup.xml"
            _write_storage(Path(grf_name), columns, values, header)
            _write_external_loads(Path(loads_name), grf_name)
            Path(setup_name).write_text(
                _id_setup_xml(
                    model.name, ik.name, loads_name, id_name,
                    time_range[0], time_range[1],
                ),
                encoding="utf-8",
            )
            osim.InverseDynamicsTool(setup_name).run()
            id_columns, id_values, _ = _read_storage(Path(id_name))
            residual = np.column_stack([
                id_values[:, id_columns.index("pelvis_tx_force")],
                id_values[:, id_columns.index("pelvis_ty_force")],
                id_values[:, id_columns.index("pelvis_tz_force")],
            ])
            residual_norm = np.linalg.norm(residual, axis=1)
            hips = np.column_stack([
                id_values[:, id_columns.index("hip_flexion_r_moment")],
                id_values[:, id_columns.index("hip_flexion_l_moment")],
            ])
            results.append({
                "tag": tag,
                "foreaft_sign": foreaft_sign,
                "lateral_sign": lateral_sign,
                "residual_component_rms_N": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
                "residual_norm_rms_N": float(np.sqrt(np.mean(residual_norm ** 2))),
                "residual_norm_p95_N": float(np.percentile(residual_norm, 95)),
                "hip_p95_abs_Nm": np.percentile(np.abs(hips), 95, axis=0).tolist(),
                "id_file": str(out / id_name),
            })
    finally:
        os.chdir(original_cwd)

    results.sort(key=lambda item: item["residual_norm_rms_N"])
    report = {"ranked_candidates": results}
    path = out / "force_sign_scan.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
