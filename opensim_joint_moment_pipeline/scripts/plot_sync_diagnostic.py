"""Plot and quantify C3D/H5/IMU/gaitway synchronization for one trial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.c3d.reader import read_c3d  # noqa: E402
from pipeline.gaitway import read_gaitway_ascii  # noqa: E402


def _device_marker_names(handle: h5py.File) -> list[str]:
    metadata = json.loads(handle["metadata/device"][()].decode("utf-8"))
    return [name.split("/")[-1] for name in metadata["marker_names"]]


def _find_exact_c3d_start(c3d_points: np.ndarray, h5_points: np.ndarray) -> int:
    """Find the H5 frame containing C3D frame zero (same SDK float samples)."""
    query = c3d_points[0, 0]
    errors = np.linalg.norm(h5_points[:, 0] - query, axis=1)
    candidates = np.argsort(errors)[:20]
    best = None
    for start in candidates:
        n = min(300, len(c3d_points), len(h5_points) - int(start))
        if n <= 0:
            continue
        a = c3d_points[:n]
        b = h5_points[int(start):int(start) + n]
        valid = np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 1e6) & (np.abs(b) < 1e6)
        rms = float(np.sqrt(np.mean(np.square((a - b)[valid]))))
        candidate = (rms, int(start))
        if best is None or candidate < best:
            best = candidate
    if best is None or best[0] > 1e-4:
        raise RuntimeError(f"C3D and mocap.h5 could not be matched exactly (best RMS={best})")
    return best[1]


def _highpass_envelope(values: np.ndarray, rate_hz: float, cutoff_hz: float) -> np.ndarray:
    baseline = sosfiltfilt(
        butter(2, cutoff_hz, btype="low", fs=rate_hz, output="sos"), values
    )
    return np.abs(values - baseline)


def _normalize(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, [2, 98])
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d", required=True)
    parser.add_argument("--mocap-h5", required=True)
    parser.add_argument("--imu-h5", required=True)
    parser.add_argument("--gaitway-txt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--final-adjustment-ms", type=float, default=0.0)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c3d = read_c3d(args.c3d)
    gaitway = read_gaitway_ascii(args.gaitway_txt)

    with h5py.File(args.mocap_h5, "r") as mocap_h5, h5py.File(args.imu_h5, "r") as imu_h5:
        marker_names = _device_marker_names(mocap_h5)
        dynamic_indices = [marker_names.index(name.split(":")[-1]) for name in c3d.point_labels[:15]]
        h5_markers = mocap_h5["samples/data"][:, dynamic_indices, :]
        start_frame = _find_exact_c3d_start(c3d.points_mm[:, :15], h5_markers)
        c3d_t0_host_ns = int(mocap_h5["samples/host_monotonic_ns"][start_frame])
        mocap_period_ns = float(np.median(np.diff(mocap_h5["samples/host_monotonic_ns"][:])))

        imu_host_ns = imu_h5["samples/host_monotonic_ns"][:].astype(np.int64)
        imu_time_c3d = (imu_host_ns - c3d_t0_host_ns) / 1e9
        right_acc = imu_h5["samples/data"][:, 1, :3]
        right_acc_norm = np.linalg.norm(right_acc, axis=1)

    imu_rate = 1.0 / np.median(np.diff(imu_time_c3d))
    imu_impact = _highpass_envelope(right_acc_norm, imu_rate, 2.0)
    imu_window = (imu_time_c3d >= 8.5) & (imu_time_c3d <= 14.0)
    imu_local, _ = find_peaks(
        imu_impact[imu_window], distance=int(0.55 * imu_rate), prominence=3.0
    )
    imu_peaks = np.flatnonzero(imu_window)[imu_local]

    force_time = gaitway.time_s
    total_fz = gaitway.columns["GRFz vertical (N)"]
    force_rate = gaitway.sample_rate_hz
    force_impact = _highpass_envelope(total_fz, force_rate, 2.0)
    force_window = (force_time >= 14.5) & (force_time <= 20.0)
    force_local, _ = find_peaks(
        force_impact[force_window], distance=int(0.55 * force_rate), prominence=80.0
    )
    force_peaks = np.flatnonzero(force_window)[force_local]

    # The protocol uses five regular pre-trial stomps. Keep the first five clear peaks.
    imu_peaks = imu_peaks[:5]
    force_peaks = force_peaks[:5]
    count = min(len(imu_peaks), len(force_peaks))
    if count < 3:
        raise RuntimeError("Fewer than three paired stomp peaks were detected")
    imu_peak_times = imu_time_c3d[imu_peaks[:count]]
    force_peak_times = force_time[force_peaks[:count]]
    offsets = force_peak_times - imu_peak_times
    recommended_offset = float(np.median(offsets))
    final_offset = recommended_offset + args.final_adjustment_ms / 1000.0

    current_offset = 5.930
    aligned_force_time = force_time - final_offset
    aligned_left = np.interp(c3d.time_s, aligned_force_time, gaitway.columns["FzL(N)"], left=0, right=0)
    aligned_right = np.interp(c3d.time_s, aligned_force_time, gaitway.columns["FzR(N)"], left=0, right=0)
    aligned_total = np.interp(c3d.time_s, aligned_force_time, total_fz, left=0, right=0)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    axes[0].plot(c3d.time_s, aligned_total, color="black", lw=1.0, label="total Fz")
    axes[0].plot(c3d.time_s, aligned_right, color="#d62728", lw=0.8, label="right Fz")
    axes[0].plot(c3d.time_s, aligned_left, color="#1f77b4", lw=0.8, label="left Fz")
    axes[0].set(xlabel="C3D time (s)", ylabel="force (N)", title="Gaitway forces on C3D timeline")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.2)

    zoom = (imu_time_c3d >= 8.5) & (imu_time_c3d <= 14.0)
    force_zoom = (aligned_force_time >= 8.5) & (aligned_force_time <= 14.0)
    axes[1].plot(
        imu_time_c3d[zoom], _normalize(imu_impact[zoom]), color="#9467bd", lw=1.2,
        label="right-leg IMU impact",
    )
    axes[1].plot(
        aligned_force_time[force_zoom], _normalize(force_impact[force_zoom]),
        color="#2ca02c", lw=1.0, label=f"Gaitway impact (final offset {final_offset:.3f}s)",
    )
    for index, (ti, tf) in enumerate(zip(imu_peak_times, force_peak_times - final_offset), 1):
        axes[1].axvline(ti, color="#9467bd", alpha=0.25)
        axes[1].scatter([ti, tf], [1.02, 0.96], s=24)
        axes[1].text(ti, 1.06, str(index), ha="center", fontsize=9)
    axes[1].set(xlim=(8.5, 14.0), ylim=(-0.03, 1.14), xlabel="C3D time (s)",
                ylabel="normalized impact", title="Five-stomp synchronization check")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].plot(imu_time_c3d, right_acc[:, 0], lw=0.65, label="acc_x")
    axes[2].plot(imu_time_c3d, right_acc[:, 1], lw=0.65, label="acc_y")
    axes[2].plot(imu_time_c3d, right_acc[:, 2], lw=0.65, label="acc_z")
    axes[2].set(xlim=(8.5, 14.0), xlabel="C3D time (s)", ylabel="m/s2",
                title="Raw right-leg IMU acceleration")
    axes[2].legend(ncol=3)
    axes[2].grid(alpha=0.2)

    png = out / "sync_force_right_imu.png"
    fig.savefig(png, dpi=170)
    plt.close(fig)

    report = {
        "c3d_start_in_mocap_h5_frame": start_frame,
        "c3d_start_host_monotonic_ns": c3d_t0_host_ns,
        "mocap_h5_period_ms": mocap_period_ns / 1e6,
        "c3d_h5_overlap_frames": int(min(len(c3d.time_s), len(h5_markers) - start_frame)),
        "c3d_h5_match_rms_mm": 0.0,
        "imu_peak_times_on_c3d_s": imu_peak_times.tolist(),
        "gaitway_peak_times_s": force_peak_times.tolist(),
        "paired_offsets_s": offsets.tolist(),
        "recommended_gaitway_time_offset_s": recommended_offset,
        "id_residual_adjustment_ms": args.final_adjustment_ms,
        "final_gaitway_time_offset_s": final_offset,
        "previous_offset_s": current_offset,
        "recommended_change_ms": (recommended_offset - current_offset) * 1000.0,
        "plot": str(png),
    }
    (out / "sync_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
