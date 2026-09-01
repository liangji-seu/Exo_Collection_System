"""Build zero-phase filtered GRF candidates for synchronization audit (EXO env)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.c3d.reader import read_c3d
from pipeline.gait.detect_contact import detect_contacts
from pipeline.opensim_io.grf import build_grf
from pipeline.pipeline import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lags", nargs="+", type=int, default=[100, 110, 120, 130, 140, 150, 160, 170])
    args = parser.parse_args()
    config = load_config(args.config)
    data = read_c3d(config["files"]["dynamic_c3d"])
    support = config["single_support"]
    contact = detect_contacts(
        data,
        force_threshold_N=float(support["vertical_force_threshold_N"]),
        foot_height_threshold_mm=float(support["foot_height_threshold_mm"]),
    )
    rotation = np.asarray(
        config["transforms"]["forceplate_to_mocap"]["rotation_matrix"], dtype=np.float64)
    cutoff = float(config["filtering"]["grf_cutoff_hz"])
    direction = config.get("force_direction_correction", {})
    x_sign = float(direction.get("opensim_x_sign", 1.0))
    z_sign = float(direction.get("opensim_z_sign", 1.0))
    out = Path(args.out)
    records = []
    for lag in args.lags:
        result = build_grf(
            data, contact.right_contact, contact.left_contact, rotation, out,
            mot_name=f"grf_lag{lag}.mot",
            xml_name=f"external_loads_lag{lag}.xml",
            cutoff_hz=cutoff,
            advance_ms=float(lag),
            force_x_sign=x_sign,
            force_z_sign=z_sign,
        )
        records.append({"lag_ms": lag, "grf": result["mot"], "external_loads": result["xml"]})
    (out / "lag_candidates.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out / "lag_candidates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
