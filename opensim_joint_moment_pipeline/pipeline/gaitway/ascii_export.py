"""Parse gaitway-3D native tab-delimited exports and build bilateral GRFs.

The native export contains bilateral forces and COP. Configured horizontal
force-channel signs are applied in the native frame before the calibrated
zero-grade pose and recorded incline. No overall action/reaction negation or
single-support allocation is applied as in the legacy C3D analog path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..filtering import lowpass_segmented
from ..transforms import (
    combine_transform, rotate_treadmill_grade, FORCE_TRANSFORM_VERSION,
    REAR_AXIS_MOCAP, O_MOCAP_MM,
)


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


def _find_header_index(lines: list[str]) -> int | None:
    """定位列名行（以 ``Time (s)\t`` 开头）；找不到返回 None。"""
    return next(
        (i for i, line in enumerate(lines) if line.startswith("Time (s)\t")), None
    )


def _header_metadata(lines: list[str], header_index: int) -> dict[str, str]:
    """把列名行之前的 ``key<TAB>value`` 头部逐行解析成字典。"""
    metadata: dict[str, str] = {}
    for line in lines[:header_index]:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            metadata[parts[0].strip()] = parts[1].strip()
    return metadata


def read_gaitway_ascii(path: str | Path) -> GaitwayAsciiData:
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    header_index = _find_header_index(lines)
    if header_index is None:
        raise ValueError(f"gaitway column header not found in {source}")
    metadata = _header_metadata(lines, header_index)

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
    if "Grade (%)" in frame.columns:
        columns["Grade (%)"] = pd.to_numeric(frame["Grade (%)"], errors="coerce").to_numpy(dtype=np.float64)
    return GaitwayAsciiData(source, metadata, columns.pop("Time (s)"), columns)


def read_gaitway_patient_info(path: str | Path) -> dict[str, Any]:
    """只读 gaitway ASCII 头部的受试者个人信息，不加载整份力数据。

    返回键：``name`` / ``sex`` / ``birth_date`` / ``weight_kg`` / ``height_m``，
    缺失的字段不出现。注意 gaitway 的 ``Patient height (m)`` 字段实际以**厘米**存储
    （如 ``187.000`` 表示 1.87 m），这里按量级归一：> 3.0 m 不可能是人身高，一律视为
    厘米 ÷ 100。读不到列名行（非 gaitway 文件）时返回空 dict。
    """
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    header_index = _find_header_index(lines)
    if header_index is None:
        return {}
    metadata = _header_metadata(lines, header_index)

    info: dict[str, Any] = {}
    for key, out in (
        ("Patient name", "name"),
        ("Patient sex", "sex"),
        ("Patient birth date", "birth_date"),
    ):
        value = metadata.get(key)
        if value:
            info[out] = value

    def _number(key: str) -> float | None:
        raw = metadata.get(key)
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if np.isfinite(value) else None

    weight = _number("Patient weight (kg)")
    if weight is not None:
        info["weight_kg"] = weight
    height = _number("Patient height (m)")
    if height is not None:
        if height > 3.0:  # gaitway 把厘米误标为米
            height /= 100.0
        info["height_m"] = height
    return info


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
    require_grade: bool = False,
    rear_axis_mocap: np.ndarray = REAR_AXIS_MOCAP,
) -> tuple[list[dict], np.ndarray, dict]:
    """把 gaitway 原生左右力抗混叠降采样到 mocap 时间并旋转到 OpenSim。

    ``force_time_offset_s`` follows ``gaitway_time = mocap_time + offset``。
    Returned validity is true where the native decomposition is available.

    Historical parameter names are retained for callers: opensim_x_sign now
    applies to native fore-aft Fy BEFORE rotation; opensim_z_sign to native
    lateral Fx BEFORE rotation. Exo Calculate defaults to (-Fy, -Fx, Fz).
    COP retains its position convention (CoPy, CoPx, 0); a force sign correction
    is not a position-axis reflection. No global-axis sign flips are performed.

    抗混叠策略（prompt6 §3.8）：在 gaitway 原生采样率（1000 Hz）先做零相位低通
    （每个接触段独立滤波，避免跨接触边界振铃），再做线性重采样到 mocap 100 Hz，
    杜绝直接把 1000 Hz 信号点采样到 100 Hz 造成的混叠。
    """
    query = np.asarray(mocap_time_s, dtype=np.float64) + float(force_time_offset_s)
    in_bounds = (query >= gaitway.time_s[0]) & (query <= gaitway.time_s[-1])
    R = combine_transform(R_fp_to_mocap)
    native_t = gaitway.time_s
    grade = gaitway.columns.get("Grade (%)")
    if grade is None:
        if require_grade:
            raise ValueError("Gaitway 文件缺少 Grade (%) 列，无法自动修正坡度；请使用原始完整导出文件")
        grade = np.zeros_like(native_t)
        grade_source = "missing_assumed_zero_legacy_caller"
    else:
        grade = np.asarray(grade, dtype=np.float64)
        grade_source = "Grade (%)"
    if grade.shape != native_t.shape or not np.isfinite(grade).all():
        raise ValueError("Gaitway Grade (%) 长度不一致或含无效值，不能确定测力台姿态")
    if opensim_x_sign not in (-1.0, 1.0) or opensim_z_sign not in (-1.0, 1.0):
        raise ValueError("Force direction signs must be +1 or -1")
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

        # Native X=lateral, Y=fore-aft, Z=normal. The application calibration
        # uses horizontal force signs (-1, -1); leave the normal force positive.
        force_n = np.column_stack([float(opensim_x_sign)*fy_n, float(opensim_z_sign)*fx_n, fz_n])
        point_n = np.column_stack([copy_n, copx_n, np.zeros_like(copx_n)])
        force_n = force_n @ R.T
        point_n = point_n @ R.T
        # One physical pose for forces and points; at the native rate, BEFORE
        # low-pass filtering and resampling (also supports changing grade).
        force_n = rotate_treadmill_grade(force_n, grade, rear_axis_mocap)
        point_n = rotate_treadmill_grade(point_n, grade, rear_axis_mocap)

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
        "force_transform_version": FORCE_TRANSFORM_VERSION,
        "grade_source": grade_source,
        "grade_percent_min": float(np.min(grade)),
        "grade_percent_max": float(np.max(grade)),
        "grade_percent_median": float(np.median(grade)),
        "grade_angle_deg_median": float(np.degrees(np.arctan(np.median(grade)/100.0))),
        "native_force_signs_walk_left_up": [float(opensim_x_sign), float(opensim_z_sign), 1.0],
        "zero_grade_rotation_fp_to_mocap": np.asarray(R_fp_to_mocap).tolist(),
        "fixed_rear_axis_mocap": np.asarray(rear_axis_mocap).tolist(),
        "origin_mocap_mm": O_MOCAP_MM.tolist(),
        "assumptions": ["标定对应显示坡度0，后沿轴在全局中固定", "保留动捕+Z竖直和模型原有重力，不根据力矩拟合重力", "双侧自由力矩缺失，仍按0处理", "原始水平力映射为局部方向后再旋转，未用新加载实验验证"],
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
