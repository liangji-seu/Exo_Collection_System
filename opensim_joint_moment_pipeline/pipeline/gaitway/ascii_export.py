"""Parse gaitway-3D native tab-delimited exports and build bilateral GRFs.

The native export already contains ground-on-foot left/right forces and COP in
the gaitway frame.  Unlike the legacy C3D analog path, no force sign inversion
or single-support allocation is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..filtering import lowpass_zero_phase
from ..transforms import combine_transform


@dataclass(frozen=True)
class GaitwayAsciiData:
    path: Path
    metadata: dict[str, str]
    time_s: np.ndarray
    columns: dict[str, np.ndarray]

    @property
    def sample_rate_hz(self) -> float:
        value = self.metadata.get("Sample rate (Hz)")
        if value:
            return float(value)
        return float(1.0 / np.median(np.diff(self.time_s)))


def read_gaitway_ascii(path: str | Path) -> GaitwayAsciiData:
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if line.startswith("Time (s)\t")), None
    )
    if header_index is None:
        raise ValueError(f"gaitway column header not found in {source}")

    metadata: dict[str, str] = {}
    for line in lines[:header_index]:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            metadata[parts[0].strip()] = parts[1].strip()

    frame = pd.read_csv(
        source, sep="\t", skiprows=header_index, encoding="utf-8-sig", low_memory=False
    )
    required = (
        "Time (s)",
        "FzL(N)", "FyL(N)", "FxL(N)", "CoPxL(m)", "CoPyL(m)",
        "FzR(N)", "FyR(N)", "FxR(N)", "CoPxR(m)", "CoPyR(m)",
        "GRFz vertical (N)",
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"gaitway export misses columns: {missing}")
    columns = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        for name in required
    }
    return GaitwayAsciiData(source, metadata, columns.pop("Time (s)"), columns)


def _interp(time: np.ndarray, values: np.ndarray, query: np.ndarray, *, fill: float) -> np.ndarray:
    valid = np.isfinite(time) & np.isfinite(values)
    if valid.sum() < 2:
        return np.full(query.shape, fill, dtype=np.float64)
    return np.interp(query, time[valid], values[valid], left=fill, right=fill)


def build_bilateral_grf(
    gaitway: GaitwayAsciiData,
    mocap_time_s: np.ndarray,
    force_time_offset_s: float,
    R_fp_to_mocap: np.ndarray,
    *,
    force_threshold_N: float = 20.0,
    cutoff_hz: float | None = None,
    opensim_x_sign: float = 1.0,
    opensim_z_sign: float = 1.0,
) -> tuple[list[dict], np.ndarray, dict]:
    """Resample native left/right forces to mocap time and rotate to OpenSim.

    ``force_time_offset_s`` follows ``gaitway_time = mocap_time + offset``.
    Returned validity is true where the native decomposition is available.
    """
    query = np.asarray(mocap_time_s, dtype=np.float64) + float(force_time_offset_s)
    in_bounds = (query >= gaitway.time_s[0]) & (query <= gaitway.time_s[-1])
    R = combine_transform(R_fp_to_mocap)

    feet: list[dict] = []
    contacts: dict[str, np.ndarray] = {}
    for side, label in (("R", "right"), ("L", "left")):
        fz = _interp(gaitway.time_s, gaitway.columns[f"Fz{side}(N)"], query, fill=0.0)
        fy = _interp(gaitway.time_s, gaitway.columns[f"Fy{side}(N)"], query, fill=0.0)
        fx = _interp(gaitway.time_s, gaitway.columns[f"Fx{side}(N)"], query, fill=0.0)
        copx = _interp(gaitway.time_s, gaitway.columns[f"CoPx{side}(m)"], query, fill=0.0)
        copy = _interp(gaitway.time_s, gaitway.columns[f"CoPy{side}(m)"], query, fill=0.0)

        # gaitway native: X=lateral, Y=fore-aft, Z=up.  The exported forces are
        # already ground-on-foot, so rotate only (do not negate as for C3D analogs).
        force_local = np.column_stack([fy, fx, fz])
        point_local = np.column_stack([copy, copx, np.zeros_like(copx)])
        force = force_local @ R.T
        point = point_local @ R.T
        force[:, 0] *= float(opensim_x_sign)
        force[:, 2] *= float(opensim_z_sign)
        if cutoff_hz is not None:
            force = lowpass_zero_phase(
                force, 1.0 / np.median(np.diff(mocap_time_s)), float(cutoff_hz),
                preserve_missing=False,
            )

        contact = in_bounds & np.isfinite(fz) & (fz > float(force_threshold_N))
        force[~contact] = 0.0
        point[~contact] = 0.0
        contacts[label] = contact
        feet.append({
            "name": label,
            "force": force,
            "point": point,
            "torque": np.zeros_like(force),
        })

    total_fz = _interp(
        gaitway.time_s, gaitway.columns["GRFz vertical (N)"], query, fill=0.0
    )
    decomposition_valid = in_bounds & ((contacts["right"] | contacts["left"]))
    qc = {
        "force_time_offset_s": float(force_time_offset_s),
        "gaitway_sample_rate_hz": gaitway.sample_rate_hz,
        "opensim_x_sign": float(opensim_x_sign),
        "opensim_z_sign": float(opensim_z_sign),
        "n_valid_decomposed_frames": int(decomposition_valid.sum()),
        "n_right_contact_frames": int(contacts["right"].sum()),
        "n_left_contact_frames": int(contacts["left"].sum()),
        "total_fz_min_N": float(np.min(total_fz[in_bounds])),
        "total_fz_max_N": float(np.max(total_fz[in_bounds])),
    }
    return feet, decomposition_valid, qc
