"""OpenSim 下游执行（**opensim 环境**：numpy 2.x + opensim 4.6）。

读取 ``prep_opensim.py`` 写出的 ``manifest.json``，跑
MarkerPlacer→Scale→IK→ID，并输出关节力矩摘要。

用法：:

    python scripts/run_opensim.py --manifest outputs/subject_001/walk_level/opensim/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opensim as osim  # noqa: E402

from pipeline.opensim_io.run_opensim import run_scale_ik_id  # noqa: E402

# gait2392 矢状面关节坐标（ID 输出里的力矩列 = 坐标名 + "_moment"）
_SAGITTAL = [
    "hip_flexion_r", "hip_flexion_l",
    "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l",
]


def _summarize_moments(id_mot: str, mass_kg: float, support_mask: str | None = None) -> dict:
    """统计矢状面关节力矩。仅在单支撑期（该脚承重）统计对应腿的力矩。

    单块跑台只有 total 合力，双支撑期无法拆左右脚 → 该期力矩无效，直接 mask 掉。
    ``support_mask`` 是 prep 写出的 (n,2) bool 数组 [right_single, left_single]。
    """
    table = osim.TimeSeriesTable(id_mot)
    labels = list(table.getColumnLabels())
    n = table.getNumRows()
    summary: dict = {"n_rows": n, "columns": labels, "joints": {}}

    right_mask = left_mask = None
    if support_mask is not None and Path(support_mask).exists():
        m = np.load(support_mask)
        right_mask = m[:, 0].astype(bool)[:n]
        left_mask = m[:, 1].astype(bool)[:n]
        summary["n_valid_right"] = int(right_mask.sum())
        summary["n_valid_left"] = int(left_mask.sum())
        summary["n_invalid_double_support"] = int((~right_mask & ~left_mask).sum())

    for coord in _SAGITTAL:
        col = f"{coord}_moment"
        if col not in labels:
            summary["joints"][coord] = {"missing": True}
            continue
        vals = np.asarray(table.getDependentColumn(col).to_numpy(), dtype=np.float64)[:n]
        mask = right_mask if coord.endswith("_r") else left_mask
        sel = np.isfinite(vals) if mask is None else (mask & np.isfinite(vals))
        v = vals[sel]
        summary["joints"][coord] = {
            "n_valid": int(sel.sum()),
            "min_Nm": float(np.nanmin(v)),
            "max_Nm": float(np.nanmax(v)),
            "mean_Nm": float(np.nanmean(v)),
            "min_Nm_per_kg": float(np.nanmin(v) / mass_kg),
            "max_Nm_per_kg": float(np.nanmax(v) / mass_kg),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenSim Scale→IK→ID（opensim 环境）")
    ap.add_argument("--manifest", default="outputs/subject_001/walk_level/opensim/manifest.json")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    mass_kg = float(manifest["subject"]["mass_kg"])
    out = Path(manifest["out_dir"])

    print(f"OpenSim version: {osim.GetVersionAndDate()}")
    print(f"subject {manifest['subject']['id']} · mass {mass_kg} kg")
    print(f"static TRC  : {manifest['static_trc']}")
    print(f"dynamic TRC : {manifest['dynamic_trc']}")
    print(f"external    : {manifest['external_loads']}\n")

    result = run_scale_ik_id(
        manifest["generic_model"],
        manifest["static_trc"],
        manifest["dynamic_trc"],
        manifest["external_loads"],
        out,
        mass_kg=mass_kg,
    )

    summary = _summarize_moments(result["id_mot"], mass_kg, manifest.get("support_mask"))
    (out / "id_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 52)
    print("Scale / IK / ID done")
    print("=" * 52)
    print(f"[scaled model] {result['scaled_model']}")
    print(f"[IK]          {result['ik_mot']}")
    print(f"[ID]          {result['id_mot']}")
    if "n_invalid_double_support" in summary:
        print(f"[validity]    right_single {summary['n_valid_right']} · "
              f"left_single {summary['n_valid_left']} · "
              f"double/masked {summary['n_invalid_double_support']} "
              f"(of {summary['n_rows']})")
    print(f"\nJoint moments (Nm, sagittal, single-support only):")
    for coord, m in summary["joints"].items():
        if m.get("missing"):
            print(f"  {coord:16s}  (no moment column)")
        else:
            print(f"  {coord:16s}  min {m['min_Nm']:8.2f}  max {m['max_Nm']:8.2f}  "
                  f"mean {m['mean_Nm']:8.2f}  (n={m['n_valid']})")
    print(f"\n[summary] {out / 'id_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
