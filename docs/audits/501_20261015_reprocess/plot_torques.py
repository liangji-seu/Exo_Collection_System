"""Plot subject 501 torque viewer arrays for two preselected reprocessed runs.

This script is deliberately a presentation-only consumer of the saved viewer
arrays.  It does not alter source arrays, filter, smooth, average, resample,
or normalize the moments.  Frames outside the bilateral support/analysis
intersection are displayed as gaps and gray background spans.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "docs" / "audits" / "501_20261015_reprocess" / "batch_report.json"
OUT = ROOT / "docs" / "audits" / "501_20261015_reprocess" / "figures"
PYTHON = r"E:\miniconda\envs\EXO\python.exe"

TARGETS = (
    {
        "suffix": r"BWALK_0P3_NOEXO\session5_20261014_093948",
        "display_condition": "倒走 0.3 m/s",
        "direction": "backward",
    },
    {
        "suffix": r"DWALK_1P0_NOEXO\session5_20261014_091806",
        "display_condition": "正走 1.0 m/s",
        "direction": "forward",
    },
)

RIGHT = "#1769aa"  # blue, solid
LEFT = "#d97722"  # orange, dashed
INVALID = "#e6e6e6"
TEXT = "#202020"
JOINTS = (
    ("髋关节", "hip_flexion"),
    ("膝关节", "knee_angle"),
    ("踝关节", "ankle_angle"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_rows() -> list[dict]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = []
    for target in TARGETS:
        matches = [
            row
            for row in report["rows"]
            if row.get("session", "").replace("/", "\\").endswith(target["suffix"])
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one batch row for {target['suffix']}, got {len(matches)}")
        row = matches[0]
        if row.get("status") != "COMPLETED":
            raise RuntimeError(f"Target row is not COMPLETED: {target['suffix']}")
        row = dict(row)
        row.update(target)
        rows.append(row)
    return rows


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index runs, without joining across false frames."""
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def load_run(row: dict) -> dict:
    run = Path(row["run_dir"])
    viewer = run / "viewer"
    paths = {
        "time_s": viewer / "time_s.npy",
        "moments": viewer / "moments.npy",
        "viewer_meta": viewer / "viewer_meta.json",
        "support_mask": run / "support_mask.npy",
        "analysis_mask": run / "analysis_mask.npy",
    }
    if not all(p.exists() for p in paths.values()):
        missing = [str(p) for p in paths.values() if not p.exists()]
        raise FileNotFoundError("Missing viewer inputs: " + ", ".join(missing))
    time_s = np.load(paths["time_s"], allow_pickle=False)
    moments = np.load(paths["moments"], allow_pickle=False)
    support = np.load(paths["support_mask"], allow_pickle=False).astype(bool)
    analysis = np.load(paths["analysis_mask"], allow_pickle=False).astype(bool)
    meta = json.loads(paths["viewer_meta"].read_text(encoding="utf-8"))
    if meta["processing"]["grf_input_policy_version"] != "continuous_native_wrench_v2":
        raise RuntimeError("Unexpected GRF policy version")
    if float(meta["processing"]["marker_cutoff_hz"]) != 6.0 or float(meta["processing"]["grf_cutoff_hz"]) != 6.0:
        raise RuntimeError("Expected marker/GRF 6 Hz metadata")
    if moments.ndim != 2 or support.shape != (len(time_s), 2) or analysis.shape != (len(time_s), 2):
        raise RuntimeError("Viewer array shapes are inconsistent")
    names = list(meta["moment_names"])
    expected = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]
    if names != expected or moments.shape[1] != len(names):
        raise RuntimeError(f"Unexpected moment column mapping: {names}")
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Invalid time vector")
    # The display validity rule is bilateral analysis ∩ support plus finite
    # values in all six moment columns.  Source NaNs remain NaNs in display.
    valid = np.all(support & analysis, axis=1) & np.all(np.isfinite(moments), axis=1)
    # Break any real time gap even if the boolean masks happen to touch.
    jumps = np.flatnonzero(np.diff(time_s) > 1.5 * dt) + 1
    plot_valid = valid.copy()
    plot_valid[jumps] = False
    runs = contiguous_runs(plot_valid)
    if not runs:
        raise RuntimeError("No valid bilateral frames")
    longest = max(runs, key=lambda ij: (ij[1] - ij[0] + 1, -ij[0]))
    center = 0.5 * (time_s[longest[0]] + time_s[longest[1]])
    detail_start, detail_end = center - 4.0, center + 4.0
    detail_sel = (time_s >= detail_start) & (time_s <= detail_end) & plot_valid
    display_moments = np.where(plot_valid[:, None], moments, np.nan)
    return {
        "row": row,
        "run": run,
        "viewer": viewer,
        "paths": paths,
        "time_s": time_s,
        "moments": moments,
        "display_moments": display_moments,
        "support": support,
        "analysis": analysis,
        "valid": valid,
        "plot_valid": plot_valid,
        "runs": runs,
        "longest_run": longest,
        "detail_window_s": [float(detail_start), float(detail_end)],
        "detail_sel": detail_sel,
        "dt_s": dt,
        "meta": meta,
        "column_names": names,
        "hashes": {k: sha256(p) for k, p in paths.items()},
    }


def mask_runs(mask: np.ndarray, time_s: np.ndarray) -> list[tuple[float, float]]:
    return [(float(time_s[a]), float(time_s[b])) for a, b in contiguous_runs(mask)]


def panel_ylim(runs: list[dict], col_pair: tuple[int, int], detail: bool) -> tuple[float, float]:
    vals = []
    for data in runs:
        sel = data["detail_sel"] if detail else data["plot_valid"]
        z = data["moments"][sel][:, list(col_pair)]
        if z.size:
            vals.append(z[np.isfinite(z)])
    flat = np.concatenate(vals) if vals else np.array([0.0])
    lo, hi = float(np.min(flat)), float(np.max(flat))
    span = max(hi - lo, 1.0)
    pad = 0.06 * span
    return lo - pad, hi + pad


def make_figure(data: list[dict], filename: str, detail: bool) -> dict:
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), dpi=100, sharex=False, layout="constrained")
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.set_facecolor("white")
        ax.grid(True, color="#d6d6d6", linewidth=0.55, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=10, colors=TEXT)
        for spine in ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(0.7)

    for c, trial in enumerate(data):
        t = trial["time_s"]
        title = (
            f"subject501 · {trial['row']['display_condition']} · {trial['row']['condition']} · QC WARN"
        )
        for r, (joint_cn, prefix) in enumerate(JOINTS):
            ax = axes[r, c]
            cols = (trial["column_names"].index(prefix + "_r"), trial["column_names"].index(prefix + "_l"))
            y0, y1 = panel_ylim(data, cols, detail)
            ax.set_ylim(y0, y1)
            if detail:
                xsel = trial["detail_sel"]
                x = t[xsel]
                yr = trial["display_moments"][xsel, cols[0]]
                yl = trial["display_moments"][xsel, cols[1]]
                ax.set_xlim(trial["detail_window_s"])
            else:
                x = t
                yr = trial["display_moments"][:, cols[0]]
                yl = trial["display_moments"][:, cols[1]]
                # Gray explicitly marks missing load or outside-analysis time.
                invalid = ~trial["plot_valid"]
                for a, b in mask_runs(invalid, t):
                    ax.axvspan(a, b, color=INVALID, alpha=0.9, linewidth=0, zorder=0)
            (h_r,) = ax.plot(x, yr, color=RIGHT, linestyle="-", linewidth=1.15, label="右侧 / Right", zorder=3)
            (h_l,) = ax.plot(x, yl, color=LEFT, linestyle="--", linewidth=1.15, label="左侧 / Left", zorder=3)
            if r == 0:
                ax.set_title(title, fontsize=12, color=TEXT, pad=8, loc="left")
                ax.legend(
                    [h_r, h_l],
                    ["右侧 / Right", "左侧 / Left"],
                    loc="upper right",
                    fontsize=9,
                    frameon=True,
                    framealpha=0.82,
                    facecolor="white",
                    edgecolor="#cccccc",
                    borderpad=0.35,
                    handlelength=2.2,
                )
            else:
                ax.set_title(joint_cn, fontsize=12, color=TEXT, pad=5, loc="left")
            ax.set_ylabel(f"{joint_cn}力矩 (N·m)", fontsize=10, color=TEXT)
            ax.text(
                0.99,
                0.04,
                "放大：最长连续有效段中间 8 s" if detail else "整段 C3D 时间",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#555555",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
            )
            if c == 0 and r == 0:
                ax.text(0.01, 0.04, "灰色：缺载/分析范围外", transform=ax.transAxes, fontsize=9, color="#555555", va="bottom")
            if r == 2:
                ax.set_xlabel("C3D 时间 (s)", fontsize=10, color=TEXT)
    mode = "局部放大" if detail else "全程"
    fig.suptitle(
        f"subject501 力矩计算结果 · {mode}\n"
        "未额外滤波/平滑/平均/降采样；缺载或分析范围外以断线表示 · QC WARN 不作为独立验证真值",
        fontsize=14,
        color=TEXT,
        y=1.035,
    )
    fig.text(
        0.5,
        0.008,
        "脚注：显示保存的计算结果，单位 N·m（未按体重归一）；缺载/分析范围外断线，不跨 gap 连接。",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )
    out_png = OUT / f"{filename}.png"
    out_svg = OUT / f"{filename}.svg"
    fig.savefig(out_png, dpi=100, facecolor="white", edgecolor="none")
    fig.savefig(out_svg, facecolor="white", edgecolor="none")
    plt.close(fig)
    return {"png": str(out_png), "svg": str(out_svg), "width_px": 1800, "height_px": 1200}


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    rows = find_rows()
    data = [load_run(row) for row in rows]
    overview = make_figure(data, "overview", detail=False)
    detail = make_figure(data, "detail", detail=True)
    manifest = {
        "subject": "501",
        "source_report": str(REPORT),
        "policy_check": {
            "grf_input_policy_version": "continuous_native_wrench_v2",
            "marker_cutoff_hz": 6.0,
            "grf_cutoff_hz": 6.0,
            "status": "passed",
        },
        "validity_rule": "bilateral analysis_mask ∩ support_mask and all six moment columns finite; time gaps force breaks",
        "transformations": ["display mask only", "no extra filtering", "no smoothing", "no averaging", "no downsampling", "no body-mass normalization"],
        "column_mapping": {name: i for i, name in enumerate(data[0]["column_names"])},
        "runs": [],
        "outputs": {"overview": overview, "detail": detail},
        "render": {"size_px": [1800, 1200], "dpi": 100, "background": "white", "extra_image_transform": "none"},
        "figure_note": "QC WARN is the batch report status and is not independent truth validation.",
    }
    for trial in data:
        manifest["runs"].append(
            {
                "condition": trial["row"]["condition"],
                "display_condition": trial["row"]["display_condition"],
                "session": trial["row"]["session"],
                "run_dir": str(trial["run"]),
                "qc_status": trial["row"]["qc_status"],
                "sample_total": int(len(trial["time_s"])),
                "sample_valid": int(np.sum(trial["plot_valid"])),
                "valid_by_mask_before_gap_break": int(np.sum(trial["valid"])),
                "time_range_s": [float(trial["time_s"][0]), float(trial["time_s"][-1])],
                "longest_continuous_valid_segment_s": [
                    float(trial["time_s"][trial["longest_run"][0]]),
                    float(trial["time_s"][trial["longest_run"][1]]),
                ],
                "detail_window_s": trial["detail_window_s"],
                "array_sha256": trial["hashes"],
                "npy_shapes": {
                    "time_s": list(trial["time_s"].shape),
                    "moments": list(trial["moments"].shape),
                    "support_mask": list(trial["support"].shape),
                    "analysis_mask": list(trial["analysis"].shape),
                },
            }
        )
    manifest["outputs"]["overview"]["sha256"] = sha256(OUT / "overview.png")
    manifest["outputs"]["detail"]["sha256"] = sha256(OUT / "detail.png")
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overview": overview, "detail": detail, "manifest": str(OUT / 'figure_manifest.json')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
