"""Build four OpenSim-horizontal GRF sign candidates (EXO env)."""

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
    lag = float(config["synchronization"]["force_advance_ms"])
    out = Path(args.out)
    records = []
    for x_sign, z_sign in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        label = f"x{'p' if x_sign > 0 else 'm'}_z{'p' if z_sign > 0 else 'm'}"
        result = build_grf(
            data, contact.right_contact, contact.left_contact, rotation, out,
            mot_name=f"grf_{label}.mot", xml_name=f"external_loads_{label}.xml",
            cutoff_hz=cutoff, advance_ms=lag,
            force_x_sign=x_sign, force_z_sign=z_sign,
        )
        records.append({
            "label": label, "force_x_sign": x_sign, "force_z_sign": z_sign,
            "grf": result["mot"], "external_loads": result["xml"],
        })
    (out / "force_sign_candidates.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out / "force_sign_candidates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
