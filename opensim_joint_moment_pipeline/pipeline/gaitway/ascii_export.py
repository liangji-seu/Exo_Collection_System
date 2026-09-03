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

from ..filtering import lowpass_segmented
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


def _interp_vec(time: np.ndarray, values: np.ndarray, query: np.ndarray, *, fill: float) -> np.ndarray:
    """逐列插值 (n, c) → (m, c)；无效列/越界填充 ``fill``。"""
    time = np.asarray(time, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    out = np.full((query.shape[0], values.shape[1]), fill, dtype=np.float64)
    for c in range(values.shape[1]):
        v = values[:, c]
        valid = np.isfinite(time) & np.isfinite(v)
        if valid.sum() < 2:
            continue
        out[:, c] = np.interp(query, time[valid], v[valid], left=fill, right=fill)
    return out


# 有依据的默认 GRF 抗混叠截止频率（prompt6 §3.8 第 2 条）。步行 GRF 有效带宽一般
# 在 ~10 Hz 内，20 Hz 在保留步态特征的同时，为 1000 Hz → 100 Hz 的降采样提供抗混叠。
DEFAULT_GRF_CUTOFF_HZ = 20.0


def build_bilateral_grf(
    gaitway: GaitwayAsciiData,
    mocap_time_s: np.ndarray,
    force_time_offset_s: float,
    R_fp_to_mocap: np.ndarray,
    *,
    force_threshold_N: float = 20.0,
    cutoff_hz: float | None = DEFAULT_GRF_CUTOFF_HZ,
    opensim_x_sign: float = 1.0,
    opensim_z_sign: float = 1.0,
) -> tuple[list[dict], np.ndarray, dict]:
    """把 gaitway 原生左右力抗混叠降采样到 mocap 时间并旋转到 OpenSim。

    ``force_time_offset_s`` follows ``gaitway_time = mocap_time + offset``。
    Returned validity is true where the native decomposition is available.

    抗混叠策略（prompt6 §3.8）：在 gaitway 原生采样率（1000 Hz）先做零相位低通
    （每个接触段独立滤波，避免跨接触边界振铃），再做线性重采样到 mocap 100 Hz，
    杜绝直接把 1000 Hz 信号点采样到 100 Hz 造成的混叠。
    """
    query = np.asarray(mocap_time_s, dtype=np.float64) + float(force_time_offset_s)
    in_bounds = (query >= gaitway.time_s[0]) & (query <= gaitway.time_s[-1])
    R = combine_transform(R_fp_to_mocap)
    native_t = gaitway.time_s
    native_rate = gaitway.sample_rate_hz
    apply_aa = (
        cutoff_hz is not None and float(cutoff_hz) > 0
        and float(cutoff_hz) < native_rate / 2.0
    )

    feet: list[dict] = []
    contacts: dict[str, np.ndarray] = {}
    for side, label in (("R", "right"), ("L", "left")):
        fz_n = gaitway.columns[f"Fz{side}(N)"]
        fy_n = gaitway.columns[f"Fy{side}(N)"]
        fx_n = gaitway.columns[f"Fx{side}(N)"]
        copx_n = gaitway.columns[f"CoPx{side}(m)"]
        copy_n = gaitway.columns[f"CoPy{side}(m)"]

        # gaitway native: X=lateral, Y=fore-aft, Z=up.  The exported forces are
        # already ground-on-foot, so rotate only (do not negate as for C3D analogs).
        force_n = np.column_stack([fy_n, fx_n, fz_n])
        point_n = np.column_stack([copy_n, copx_n, np.zeros_like(copx_n)])
        force_n = force_n @ R.T
        point_n = point_n @ R.T
        force_n[:, 0] *= float(opensim_x_sign)
        force_n[:, 2] *= float(opensim_z_sign)

        # 接触段（原生采样率）：只接触时力才有物理意义，非接触按 0 处理。
        contact_n = np.isfinite(fz_n) & (fz_n > float(force_threshold_N))

        # 抗混叠低通：每个接触段独立滤波，非接触帧 NaN（不跨边界振铃）。
        if apply_aa:
            force_n = lowpass_segmented(force_n, contact_n, native_rate, float(cutoff_hz))
            point_n = lowpass_segmented(point_n, contact_n, native_rate, float(cutoff_hz))

        # 滤波后再把非接触段清零（避免振铃泄漏进摆动相）。
        force_n = np.where(contact_n[:, None], force_n, 0.0)
        point_n = np.where(contact_n[:, None], point_n, 0.0)

        # 重采样到 mocap 时间（已抗混叠，线性重采样即可）。
        force = _interp_vec(native_t, force_n, query, fill=0.0)
        point = _interp_vec(native_t, point_n, query, fill=0.0)

        # mocap 采样率下的接触判定（用原始 fz 判接触，供有效区间与 QC）。
        fz_q = _interp(native_t, fz_n, query, fill=0.0)
        contact = in_bounds & np.isfinite(fz_q) & (fz_q > float(force_threshold_N))
        force = np.nan_to_num(force, nan=0.0)
        point = np.nan_to_num(point, nan=0.0)
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
        native_t, gaitway.columns["GRFz vertical (N)"], query, fill=0.0
    )
    decomposition_valid = in_bounds & ((contacts["right"] | contacts["left"]))
    qc = {
        "force_time_offset_s": float(force_time_offset_s),
        "gaitway_sample_rate_hz": gaitway.sample_rate_hz,
        "grf_cutoff_hz": (float(cutoff_hz) if cutoff_hz is not None else None),
        "opensim_x_sign": float(opensim_x_sign),
        "opensim_z_sign": float(opensim_z_sign),
        "n_valid_decomposed_frames": int(decomposition_valid.sum()),
        "n_right_contact_frames": int(contacts["right"].sum()),
        "n_left_contact_frames": int(contacts["left"].sum()),
        "total_fz_min_N": float(np.min(total_fz[in_bounds])),
        "total_fz_max_N": float(np.max(total_fz[in_bounds])),
    }
    return feet, decomposition_valid, qc
