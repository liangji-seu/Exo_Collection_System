"""Phase-normalized backward-walking comparison for subject 501.

This is a presentation-only audit figure.  It reads the existing OpenSim
viewer arrays and IK output, uses the already reviewed side-specific cycle
boundaries from the diagnostic JSON, interpolates each complete cycle to 101
phase points, and does not filter or modify source values.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
REPO_ROOT = WORKSPACE_ROOT / "Exo_Collection_System"
OUT_DIR = Path(__file__).resolve().parent
RUN_REL = Path(
    "Exo_Collection_Data/501/d1/F_STEADY/BWALK_0P3_NOEXO/"
    "session5_20261014_093948/derived/opensim/run_20261015_202917_1886"
)
RUN_DIR = WORKSPACE_ROOT / RUN_REL
DIAGNOSTIC_REL = Path(
    "Exo_Collection_System/.direction_501_preprocess/"
    "bwalk_window_diagnostic_20261015.json"
)
DIAGNOSTIC_PATH = WORKSPACE_ROOT / DIAGNOSTIC_REL
PHASE = np.linspace(0.0, 100.0, 101)
COLORS = {"r": "#0072B2", "l": "#D55E00"}
SIDE_LABELS = {"r": "Right", "l": "Left"}


def read_mot(path: Path) -> pd.DataFrame:
    """Read an OpenSim .mot table without changing numeric values."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    endheader = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    return pd.read_csv(path, sep=r"\s+", skiprows=endheader + 1, engine="python")


def load_inputs() -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict]:
    diagnostic = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    time = np.load(RUN_DIR / "viewer/time_s.npy")
    moments = np.load(RUN_DIR / "viewer/moments.npy")
    grf = np.load(RUN_DIR / "viewer/grf.npy")
    ik = read_mot(RUN_DIR / "hh19_static_calibrated_ik.mot")
    ik_time = ik["time"].to_numpy(dtype=float)
    if not np.allclose(time, ik_time, atol=1e-7):
        raise ValueError("viewer time and IK time axes do not match")

    # viewer/moments.npy follows viewer_meta.json: hip_flexion_r, hip_flexion_l,
    # then knee and ankle moments.  viewer/grf.npy is (frame, side, xyz), with
    # OpenSim vertical force on the y component (index 1).
    values = {
        "r": {
            "angle": ik["hip_flexion_r"].to_numpy(dtype=float),
            "fz": grf[:, 0, 1].astype(float),
            "moment": moments[:, 0].astype(float),
        },
        "l": {
            "angle": ik["hip_flexion_l"].to_numpy(dtype=float),
            "fz": grf[:, 1, 1].astype(float),
            "moment": moments[:, 1].astype(float),
        },
    }
    return time, values, diagnostic


def normalize_cycles(
    time: np.ndarray,
    values: dict[str, dict[str, np.ndarray]],
    diagnostic: dict,
) -> dict[str, dict[str, np.ndarray]]:
    """Interpolate supplied complete cycles to 101 display phase points."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for side in ("r", "l"):
        cycles = diagnostic["sides"][side]["cycles"]
        per_variable = {name: [] for name in ("angle", "fz", "moment")}
        for cycle in cycles:
            start, end = float(cycle["start"]), float(cycle["end"])
            mask = (time >= start - 1e-9) & (time <= end + 1e-9)
            if mask.sum() < 3:
                raise ValueError(f"cycle {side} {start}-{end} has too few source samples")
            source_t = time[mask]
            phase_source = (source_t - start) / (end - start) * 100.0
            phase_source[0] = 0.0
            phase_source[-1] = 100.0
            for name in per_variable:
                per_variable[name].append(
                    np.interp(PHASE, phase_source, values[side][name][mask])
                )
        out[side] = {
            name: np.asarray(series, dtype=float) for name, series in per_variable.items()
        }
    return out


def set_common_limits(ax: mpl.axes.Axes, traces: list[np.ndarray], pad_fraction: float = 0.08) -> None:
    finite = np.concatenate([x[np.isfinite(x)] for x in traces])
    lo, hi = float(finite.min()), float(finite.max())
    span = hi - lo
    pad = pad_fraction * span if span > 0 else 1.0
    ax.set_ylim(lo - pad, hi + pad)


def plot_figure(cycles: dict[str, dict[str, np.ndarray]], diagnostic: dict) -> mpl.figure.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.3,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    # Reserve a dedicated header band so the two-line title/QC note cannot
    # overlap the first data panel.  Data limits and traces are unchanged.
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.0), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.085, top=0.86, hspace=0.10)
    variables = [
        ("angle", "Hip flexion angle (deg)"),
        ("fz", "Vertical GRF (N)"),
        ("moment", "Hip flexion moment (N m)"),
    ]
    handles = []
    for side in ("r", "l"):
        n = cycles[side]["angle"].shape[0]
        handles.extend(
            [
                Line2D([0], [0], color=COLORS[side], lw=1.0, alpha=0.45, label=f"{SIDE_LABELS[side]} cycles (n={n})"),
                Line2D([0], [0], color=COLORS[side], lw=2.8, label=f"{SIDE_LABELS[side]} mean (n={n})"),
            ]
        )

    for ax, (name, ylabel) in zip(axes, variables):
        for side in ("r", "l"):
            traces = cycles[side][name]
            for trace in traces:
                ax.plot(PHASE, trace, color=COLORS[side], lw=0.85, alpha=0.32, linestyle="-")
            ax.plot(
                PHASE,
                traces.mean(axis=0),
                color=COLORS[side],
                lw=2.5,
                linestyle="-",
                zorder=4,
            )
        ax.axhline(0.0, color="#666666", lw=0.65, linestyle=(0, (3, 3)), zorder=0)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", color="#D9D9D9", lw=0.55)
        ax.grid(axis="y", color="#ECECEC", lw=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.995,
            0.94,
            f"nR={cycles['r'][name].shape[0]}, nL={cycles['l'][name].shape[0]}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color="#333333",
        )
        set_common_limits(ax, [cycles["r"][name], cycles["l"][name]])

    axes[-1].set_xlabel("Gait cycle phase (%)")
    axes[-1].set_xlim(0, 100)
    axes[-1].set_xticks(np.arange(0, 101, 20))
    fig.suptitle(
        "Backward walking: phase-normalized side comparison\n"
        "QC WARN | nR=3, nL=4 | no extra filter | 50 N events only",
        fontsize=11.8,
        y=0.975,
        linespacing=1.35,
    )
    axes[0].legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.76),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.86,
        edgecolor="#CCCCCC",
        handlelength=2.5,
        borderpad=0.45,
    )
    return fig


def build_manifest(diagnostic: dict, cycles: dict[str, dict[str, np.ndarray]]) -> dict:
    cycle_boundaries = {
        side: [
            {
                "start_s": float(c["start"]),
                "end_s": float(c["end"]),
                "period_s": float(c["period"]),
            }
            for c in diagnostic["sides"][side]["cycles"]
        ]
        for side in ("r", "l")
    }
    return {
        "figure": "phase_comparison.png and phase_comparison.svg",
        "purpose": "Presentation-only gait-phase comparison for backward-walking moment review",
        "source_run": str(RUN_DIR),
        "source_run_relative": RUN_REL.as_posix(),
        "diagnostic_source": str(DIAGNOSTIC_PATH),
        "diagnostic_source_relative": DIAGNOSTIC_REL.as_posix(),
        "window_s": [24.18, 32.18],
        "cycle_boundary_source": "diagnostic sides[*].cycles; complete side-specific contact-onset cycles in the stated window",
        "cycle_boundaries_s": cycle_boundaries,
        "cycle_counts": {side: int(cycles[side]["angle"].shape[0]) for side in ("r", "l")},
        "event_segmentation": {
            "threshold_N": 50.0,
            "use": "event segmentation only",
            "transient_jitter": "no new split at short transient crossings; supplied diagnostic boundaries reused",
        },
        "normalization": {
            "method": "linear interpolation within each complete cycle",
            "phase_points": 101,
            "phase_range_percent": [0.0, 100.0],
            "formula": "phase=(t-start)/(end-start)*100",
            "additional_filtering": "none",
            "source_values_modified": False,
            "display_only": True,
        },
        "variables": {
            "hip_angle": {"source": "hh19_static_calibrated_ik.mot hip_flexion_r/l", "unit": "deg"},
            "vertical_grf": {"source": "viewer/grf.npy [:, side, 1]", "unit": "N", "component": "OpenSim y"},
            "hip_moment": {"source": "viewer/moments.npy hip_flexion_r/l", "unit": "N m"},
        },
        "encoding": {
            "right": {"color": COLORS["r"], "label": "Right", "n": int(cycles["r"]["angle"].shape[0])},
            "left": {"color": COLORS["l"], "label": "Left", "n": int(cycles["l"]["angle"].shape[0])},
            "thin_lines": "one semitransparent line per complete cycle",
            "thick_lines": "arithmetic mean across the displayed cycles at each phase point",
        },
        "qc": {
            "status": "WARN",
            "notes": [
                "Only 3 right and 4 left complete cycles are available.",
                "Backward decomposition is marked unvalidated in the source run manifest.",
                "This figure has not been independently validated against a separate event detector or source reprocessing.",
                "No healthy template or expected pattern was imposed.",
            ],
        },
        "units_and_signs": {
            "angle": "degrees as stored in IK .mot (inDegrees=yes)",
            "force": "N from viewer arrays",
            "moment": "N m from viewer arrays; source OpenSim anatomical sign convention retained",
        },
        "verification": {
            "independent_validation": False,
            "render_review": "Performed after export for clipping, font rendering, and panel readability",
        },
    }


def main() -> None:
    time, values, diagnostic = load_inputs()
    cycles = normalize_cycles(time, values, diagnostic)
    fig = plot_figure(cycles, diagnostic)
    png_path = OUT_DIR / "phase_comparison.png"
    svg_path = OUT_DIR / "phase_comparison.svg"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    manifest = build_manifest(diagnostic, cycles)
    manifest["generated_outputs"] = {
        "png": {
            "path": str(png_path),
            "sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
            "size_bytes": png_path.stat().st_size,
        },
        "svg": {
            "path": str(svg_path),
            "sha256": hashlib.sha256(svg_path.read_bytes()).hexdigest(),
            "size_bytes": svg_path.stat().st_size,
        },
    }
    (OUT_DIR / "phase_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"png": str(png_path), "svg": str(svg_path), "manifest": str(OUT_DIR / 'phase_comparison_manifest.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
