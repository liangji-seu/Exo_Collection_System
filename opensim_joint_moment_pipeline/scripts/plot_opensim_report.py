"""Create compact scientific QC figures from an OpenSim ID result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_storage(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines.index("endheader") + 1
    return lines[header].split(), np.loadtxt(lines[header + 1:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mass-kg", type=float, required=True)
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns, data = _read_storage(Path(report["files"]["id"]))
    time = data[:, 0]

    coordinates = [
        ("Hip flexion", "hip_flexion_r_moment", "hip_flexion_l_moment"),
        ("Knee flexion", "knee_angle_r_moment", "knee_angle_l_moment"),
        ("Ankle dorsiflexion", "ankle_angle_r_moment", "ankle_angle_l_moment"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    result_summary: dict[str, dict] = {}
    for axis, (title, right_name, left_name) in zip(axes, coordinates):
        right = data[:, columns.index(right_name)] / args.mass_kg
        left = data[:, columns.index(left_name)] / args.mass_kg
        axis.plot(time, right, color="#d62728", lw=0.9, label="right")
        axis.plot(time, left, color="#1f77b4", lw=0.9, label="left")
        axis.axhline(0, color="black", lw=0.6)
        axis.set(ylabel="Nm/kg", title=title)
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, loc="upper right")
        result_summary[title] = {
            "right": {
                "min_Nm_per_kg": float(np.min(right)),
                "max_Nm_per_kg": float(np.max(right)),
                "p95_abs_Nm_per_kg": float(np.percentile(np.abs(right), 95)),
            },
            "left": {
                "min_Nm_per_kg": float(np.min(left)),
                "max_Nm_per_kg": float(np.max(left)),
                "p95_abs_Nm_per_kg": float(np.percentile(np.abs(left), 95)),
            },
        }
    axes[-1].set_xlabel("C3D time (s)")
    fig.suptitle("Subject 003 — bilateral inverse-dynamics joint moments", fontsize=15)
    moment_png = out / "joint_moments_bilateral.png"
    fig.savefig(moment_png, dpi=170)
    plt.close(fig)

    marker_qc = report["marker_qc"]["markers"]
    marker_names = list(marker_qc)
    means = np.array([marker_qc[name]["mean_cm"] for name in marker_names])
    p95 = np.array([marker_qc[name]["p95_cm"] for name in marker_names])
    order = np.argsort(means)
    residual_names = ["pelvis_tx_force", "pelvis_ty_force", "pelvis_tz_force"]
    residual = np.column_stack([data[:, columns.index(name)] for name in residual_names])
    residual_norm = np.linalg.norm(residual, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    y = np.arange(len(marker_names))
    axes[0].barh(y, p95[order], color="#aec7e8", label="95th percentile")
    axes[0].barh(y, means[order], color="#1f77b4", label="mean")
    axes[0].set_yticks(y, [marker_names[i] for i in order])
    axes[0].axvline(2.0, color="#ff7f0e", ls="--", lw=1.1, label="2 cm target")
    axes[0].set(xlabel="3D marker error (cm)", title="Dynamic IK marker errors")
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].plot(time, residual[:, 0], lw=0.75, label="fore-aft")
    axes[1].plot(time, residual[:, 1], lw=0.75, label="vertical")
    axes[1].plot(time, residual[:, 2], lw=0.75, label="lateral")
    axes[1].plot(time, residual_norm, color="black", lw=0.9, alpha=0.75, label="3D norm")
    axes[1].axhline(args.mass_kg * 9.80665 * 0.05, color="#ff7f0e", ls="--", lw=1,
                    label="5% body weight")
    axes[1].set(xlabel="C3D time (s)", ylabel="N", title="Inverse-dynamics pelvis residual force")
    axes[1].legend(ncol=5, fontsize=9)
    axes[1].grid(alpha=0.2)
    qc_png = out / "ik_and_residual_qc.png"
    fig.savefig(qc_png, dpi=170)
    plt.close(fig)

    result_summary["QC"] = {
        "ik_rms_mean_cm": report["marker_qc"]["overall"]["rms_mean_cm"],
        "ik_rms_p95_cm": report["marker_qc"]["overall"]["rms_p95_cm"],
        "residual_force_rms_N": float(np.sqrt(np.mean(residual_norm ** 2))),
        "residual_force_p95_N": float(np.percentile(residual_norm, 95)),
    }
    result_summary["files"] = {
        "joint_moments_plot": str(moment_png),
        "qc_plot": str(qc_png),
    }
    summary_path = out / "result_summary.json"
    summary_path.write_text(json.dumps(result_summary, indent=2), encoding="utf-8")
    print(json.dumps(result_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
