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
    # 站立工况下 Gaitway 不执行步态事件分脚，FzL/FzR 与左右 COP 会全部为 NaN，
    # 但总合力/总 COP 仍然有效。保留这些可选列，供站立专用的可审计估计使用。
    for name in (
        "GRFy fore-aft (N)",
        "GRFx lateral (N)",
        "CoPx lateral (m)",
        "CoPy fore-aft (m)",
        "Mx (N.m)", "My (N.m)", "Mz (N.m)", "Tz free moment (N.m)",
    ):
        if name in frame.columns:
            columns[name] = pd.to_numeric(frame[name], errors="coerce").to_numpy(
                dtype=np.float64
            )
    if "Grade (%)" in frame.columns:
        columns["Grade (%)"] = pd.to_numeric(frame["Grade (%)"], errors="coerce").to_numpy(dtype=np.float64)
    return GaitwayAsciiData(source, metadata, columns.pop("Time (s)"), columns)


def audit_native_bilateral_wrench(
    gaitway: GaitwayAsciiData, *, time_range_s: tuple[float, float],
    diagnostic_contact_threshold_N: float = 50.0,
) -> dict[str, Any]:
    """Audit exported native signals before signs, rotation, filtering or mapping.

    Moments use the export's native origin and z=0 COP plane. The bilateral
    representation has no measured side free moments; its difference from total
    Mz consequently includes missing free moment. These metrics never change
    loads, masks or synchronization, and have no acceptance threshold.
    """
    col = gaitway.columns
    n = len(gaitway.time_s)
    missing = np.full(n, np.nan)
    forces, points = [], []
    for side in ("L", "R"):
        force = np.column_stack([col[f"F{axis}{side}(N)"] for axis in "xyz"])
        point = np.column_stack([col[f"CoPx{side}(m)"], col[f"CoPy{side}(m)"], np.zeros(n)])
        # Undefined COP at exactly measured zero load contributes zero moment.
        point[np.isfinite(force).all(axis=1) & (np.max(np.abs(force), axis=1) <= 1e-12)] = 0.0
        forces.append(force)
        points.append(point)
    total_force = np.column_stack([col.get(key, missing) for key in (
        "GRFx lateral (N)", "GRFy fore-aft (N)", "GRFz vertical (N)")])
    total_moment = np.column_stack([col.get(f"M{axis} (N.m)", missing) for axis in "xyz"])
    total_point = np.column_stack([col.get("CoPx lateral (m)", missing),
                                   col.get("CoPy fore-aft (m)", missing), np.zeros(n)])
    summed_force = forces[0] + forces[1]
    represented_moment = np.cross(points[0], forces[0]) + np.cross(points[1], forces[1])
    total_cop_moment = np.cross(total_point, total_force)
    total_cop_with_free = total_cop_moment.copy()
    total_cop_with_free[:, 2] += col.get("Tz free moment (N.m)", missing)
    # Match production's unavailable-decomposition rule, independently of contact
    # threshold; include missing force/COP in counts rather than converting to zero.
    unsupported = (np.isfinite(total_force[:, 2]) & (total_force[:, 2] > 0)
                   & np.isfinite(summed_force[:, 2]) & (np.abs(summed_force[:, 2]) <= 1e-12))
    scope = ((gaitway.time_s >= time_range_s[0]) & (gaitway.time_s <= time_range_s[1]))
    available = scope & ~unsupported & np.isfinite(summed_force).all(axis=1)
    complete = available & np.isfinite(represented_moment).all(axis=1)
    left_contact = forces[0][:, 2] > diagnostic_contact_threshold_N
    right_contact = forces[1][:, 2] > diagnostic_contact_threshold_N

    def statistics(values: np.ndarray) -> dict[str, Any]:
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if not len(values):
            return {"n_samples": 0, "mean": None, "rms": None, "p95_abs": None, "max_abs": None}
        return {"n_samples": int(len(values)), "mean": float(np.mean(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
                "p95_abs": float(np.quantile(np.abs(values), .95)),
                "max_abs": float(np.max(np.abs(values)))}

    phases = {}
    for name, phase in (("all", np.ones(n, bool)),
                        ("single_support", left_contact ^ right_contact),
                        ("double_support", left_contact & right_contact)):
        selected = available & phase
        loaded = complete & phase
        phases[name] = {
            "n_available_native_samples": int(selected.sum()),
            "n_complete_cop_native_samples": int(loaded.sum()),
            "force_sum_minus_exported_total_N": {axis: statistics((summed_force-total_force)[selected, i]) for i, axis in enumerate("xyz")},
            "represented_moment_minus_exported_total_Nm": {axis: statistics((represented_moment-total_moment)[loaded, i]) for i, axis in enumerate("xyz")},
            "represented_moment_minus_total_cop_cross_force_Nm": {axis: statistics((represented_moment-total_cop_moment)[loaded, i]) for i, axis in enumerate("xyz")},
            "exported_total_moment_minus_cop_cross_force_and_total_free_Nm": {axis: statistics((total_moment-total_cop_with_free)[selected, i]) for i, axis in enumerate("xyz")},
            "exported_total_free_moment_Nm": statistics(col.get("Tz free moment (N.m)", missing)[selected]),
        }
    return {
        "schema_version": "1.0.0", "status": "INFO", "frame": "native export XYZ, COP z=0",
        "native_time_range_s": [float(v) for v in time_range_s],
        "reference_origin_metadata_m": {key: gaitway.metadata.get(key) for key in ("REF-X0 (m)", "REF-Y0 (m)")},
        "diagnostic_contact_threshold_N": float(diagnostic_contact_threshold_N),
        "threshold_changes_loads_or_validity": False,
        "n_unsupported_total_native_samples": int((scope & unsupported).sum()),
        "n_missing_loaded_cop_native_samples": int((available & ~complete).sum()),
        "exported_total_moment_axes_available": [axis for axis in "xyz" if f"M{axis} (N.m)" in col],
        "exported_total_free_moment_available": "Tz free moment (N.m)" in col,
        "bilateral_free_moments_measured": False,
        "exported_total_free_moment_applied_to_feet": False,
        "output_torque_semantics": "filtered wrench representation residual, not measured side free moment",
        "exported_total_moment_reference_comparability": "not assumed; inspect exported_total_moment_minus_cop_cross_force_and_total_free_Nm and origin metadata",
        "interpretation": "Represented minus total-COP cross force compares same-plane overturning moments. Exported total moment comparison also depends on matching reference origins; Mz includes omitted side free moments. Total COP and total moments are related signals, not independent validation. No unique bilateral correction follows.",
        "phases": phases,
    }


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


# Default nominal bandwidth matches the app's 6 Hz marker/ID-coordinate setting.
# The 501 two-run sensitivity check supported this choice without establishing
# a universally optimal cutoff. Explicit alternatives (including 20 Hz) remain
# supported and are recorded in processing/QC metadata.
DEFAULT_GRF_CUTOFF_HZ = 6.0
GRF_INPUT_POLICY_VERSION = "continuous_native_wrench_v2"


def _interp_finite_segments(time, values, query, valid, *, sample_rate_hz):
    """Interpolate complete vectors only within genuinely finite native runs."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros((len(query), values.shape[1]), dtype=np.float64)
    available = np.zeros(len(query), dtype=bool)
    indices = np.flatnonzero(valid & np.isfinite(values).all(axis=1))
    if not indices.size:
        return out, available
    split = np.flatnonzero((np.diff(indices) > 1) |
                          (np.diff(time[indices]) > 1.5 / float(sample_rate_hz))) + 1
    for run in np.split(indices, split):
        selected = (query >= time[run[0]]) & (query <= time[run[-1]])
        if not selected.any():
            continue
        for axis in range(values.shape[1]):
            out[selected, axis] = np.interp(query[selected], time[run], values[run, axis])
        available[selected] = True
    return out, available


def _wrench_trust_mask(time, valid, *, sample_rate_hz, cutoff_hz):
    """Exclude finite-segment filter edges from truth, without cutting loads."""
    trust = valid.copy()
    indices = np.flatnonzero(valid)
    if not len(indices):
        return trust
    split = np.flatnonzero((np.diff(indices) > 1) |
                          (np.diff(time[indices]) > 1.5 / sample_rate_hz)) + 1
    for run in np.split(indices, split):
        if cutoff_hz is not None and 0 < cutoff_hz < sample_rate_hz / 2 and len(run) >= 12:
            guard = 3.0 / cutoff_hz
            trust[run] &= ((time[run] >= time[run[0]] + guard)
                           & (time[run] <= time[run[-1]] - guard))
        else:
            # Without filtering there is no padding contamination, but an
            # isolated sample adjacent to a genuine gap is not a trusted edge.
            if run[0] > 0:
                trust[run[0]] = False
            if run[-1] < len(time) - 1:
                trust[run[-1]] = False
    return trust


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
    standing_foot_centers_opensim_m: dict[str, np.ndarray] | None = None,
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
    （每个连续有效测量段的完整wrench一致滤波），再做线性重采样到 mocap 100 Hz，
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

    bilateral_available = any(
        np.isfinite(gaitway.columns[f"Fz{side}(N)"]).any() for side in ("R", "L")
    )
    if not bilateral_available:
        if standing_foot_centers_opensim_m is None:
            raise ValueError(
                "Gaitway 左右分力 FzL/FzR 全部缺失；该记录若为静止站立，"
                "必须启用总 COP + 双脚 marker 的站立载荷估计"
            )
        feet, valid, qc = _build_standing_grf_from_total(
            gaitway,
            query,
            in_bounds,
            R,
            np.asarray(R_fp_to_mocap, dtype=np.float64),
            grade,
            grade_source,
            standing_foot_centers_opensim_m,
            force_time_offset_s=force_time_offset_s,
            force_threshold_N=force_threshold_N,
            cutoff_hz=cutoff_hz,
            opensim_x_sign=opensim_x_sign,
            opensim_z_sign=opensim_z_sign,
            rear_axis_mocap=rear_axis_mocap,
        )
        for foot in feet:
            foot["valid"] = valid.copy()
            foot["contact"] = valid.copy()
        qc.update({"grf_input_policy_version": GRF_INPUT_POLICY_VERSION,
                   "force_input_validity": "legacy_quiet_standing_estimate_WARN",
                   "event_force_threshold_N": float(force_threshold_N),
                   "wrench_filtering": "legacy_quiet_standing_estimate; not native bilateral"})
        return feet, valid, qc

    feet: list[dict] = []
    contacts: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    missing_loaded_native = 0
    edge_excluded_frames = 0
    total_native = gaitway.columns["GRFz vertical (N)"]
    sum_native = gaitway.columns["FzR(N)"] + gaitway.columns["FzL(N)"]
    unsupported_total = (np.isfinite(total_native) & (total_native > force_threshold_N)
                         & np.isfinite(sum_native) & (np.abs(sum_native) <= 1e-12))
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

        # Measurement validity is independent of the 50 N event threshold.
        # With exactly measured zero force the wrench is zero even if COP is
        # undefined. Positive/nonzero load with missing COP is NOT reconstructible.
        finite_force = np.isfinite(force_n).all(axis=1)
        measured_zero = finite_force & (np.max(np.abs(force_n), axis=1) <= 1e-12)
        finite_point = np.isfinite(point_n).all(axis=1)
        native_valid = finite_force & (finite_point | measured_zero) & ~unsupported_total
        missing_loaded_native += int((finite_force & ~measured_zero & ~finite_point).sum())
        point_n = np.where((finite_point & ~measured_zero)[:, None], point_n, 0.0)
        # Unmeasured per-foot free moment remains assumed zero. An exported
        # total Tz is retained for diagnostics, without inventing its allocation.
        # Filter force and
        # moment about ONE fixed origin together, preserving the physical wrench.
        moment_n = np.cross(point_n, force_n)
        wrench_n = np.column_stack([force_n, moment_n])
        if apply_aa:
            wrench_n = lowpass_segmented(wrench_n, native_valid, native_rate,
                                         float(cutoff_hz), time_s=native_t)
        wrench, valid = _interp_finite_segments(native_t, wrench_n, query, native_valid,
                                                sample_rate_hz=native_rate)
        # COP is a choice of representation; the residual torque below ensures
        # the same filtered moment regardless of this unfiltered native point.
        point, point_valid = _interp_finite_segments(native_t, point_n, query, native_valid,
                                                     sample_rate_hz=native_rate)
        fz_q, fz_valid = _interp_finite_segments(native_t, fz_n[:, None], query, native_valid,
                                               sample_rate_hz=native_rate)
        available = valid & point_valid & fz_valid & in_bounds
        native_trust = _wrench_trust_mask(native_t, native_valid, sample_rate_hz=native_rate,
                                         cutoff_hz=cutoff_hz)
        _, trusted = _interp_finite_segments(native_t, wrench_n, query, native_trust,
                                              sample_rate_hz=native_rate)
        valid = available & trusted
        edge_excluded_frames += int((available & ~valid).sum())
        force = wrench[:, :3]
        torque = wrench[:, 3:] - np.cross(point, force)
        # Quality guards must NOT create a new physical zero-force edge.
        # Only true unavailable source samples receive numerical placeholders.
        force[~available] = 0.0
        point[~available] = 0.0
        torque[~available] = 0.0
        contact = valid & (fz_q[:, 0] > float(force_threshold_N))
        contacts[label] = contact
        validity[label] = valid
        feet.append({
            "name": label,
            "force": force,
            "point": point,
            "torque": torque,
            "valid": valid,
            "contact": contact,
            "input_available": available,
        })

    total_fz = _interp(
        native_t, gaitway.columns["GRFz vertical (N)"], query, fill=0.0
    )
    decomposition_valid = in_bounds & validity["right"] & validity["left"]
    qc = {
        "force_transform_version": FORCE_TRANSFORM_VERSION,
        "grf_input_policy_version": GRF_INPUT_POLICY_VERSION,
        "force_input_validity": "finite_native_complete_wrench_no_gap_interpolation",
        "event_force_threshold_N": float(force_threshold_N),
        "event_threshold_applied_to_physical_force": False,
        "wrench_filtering": "joint_force_and_moment_about_fixed_origin",
        "output_torque_semantics": "filtered_origin_moment_minus_cross_output_point_force; not measured free moment",
        "n_right_valid_frames": int(validity["right"].sum()),
        "n_left_valid_frames": int(validity["left"].sum()),
        "n_missing_loaded_cop_native_side_samples": missing_loaded_native,
        "n_unsupported_total_native_frames": int(unsupported_total.sum()),
        "filter_segment_edge_guard_s": (3.0 / float(cutoff_hz) if apply_aa else 0.0),
        "n_filter_edge_excluded_side_frames": edge_excluded_frames,
        "invalid_force_values_are_placeholders": True,
        "grade_source": grade_source,
        "grade_percent_min": float(np.min(grade)),
        "grade_percent_max": float(np.max(grade)),
        "grade_percent_median": float(np.median(grade)),
        "grade_angle_deg_median": float(np.degrees(np.arctan(np.median(grade)/100.0))),
        "native_force_signs_walk_left_up": [float(opensim_x_sign), float(opensim_z_sign), 1.0],
        "zero_grade_rotation_fp_to_mocap": np.asarray(R_fp_to_mocap).tolist(),
        "fixed_rear_axis_mocap": np.asarray(rear_axis_mocap).tolist(),
        "origin_mocap_mm": O_MOCAP_MM.tolist(),
        "assumptions": ["标定对应显示坡度0，后沿轴在全局中固定", "保留动捕+Z竖直和模型原有重力，不根据力矩拟合重力", "逐脚自由力矩未知取0；已导出总Tz仅作诊断，未擅自分配；输出torque为保持已滤完整wrench的表示残差，非实测自由力矩", "原始水平力映射为局部方向后再旋转，未用新加载实验验证", "有效低载原生力保留，50N仅用于事件；正载缺COP或分力的样本不跨缺失插值"],
        "force_time_offset_s": float(force_time_offset_s),
        "gaitway_sample_rate_hz": gaitway.sample_rate_hz,
        "grf_cutoff_hz": (float(cutoff_hz) if cutoff_hz is not None else None),
        "opensim_x_sign": float(opensim_x_sign),
        "opensim_z_sign": float(opensim_z_sign),
        "n_valid_decomposed_frames": int(decomposition_valid.sum()),
        "n_right_contact_frames": int(contacts["right"].sum()),
        "n_left_contact_frames": int(contacts["left"].sum()),
        "decomposition_method": "native_bilateral",
        "bilateral_force_measured": False,
        "bilateral_force_source": "vendor_algorithm_decomposed_total_force",
        "total_fz_min_N": float(np.min(total_fz[in_bounds])),
        "total_fz_max_N": float(np.max(total_fz[in_bounds])),
    }
    return feet, decomposition_valid, qc


def _fill_marker_gaps(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐列插值短暂 marker 缺失，并返回原始帧级可用性。"""
    data = np.asarray(values, dtype=np.float64)
    valid_rows = np.isfinite(data).all(axis=1)
    filled = data.copy()
    x = np.arange(len(data), dtype=np.float64)
    for column in range(data.shape[1]):
        valid = np.isfinite(data[:, column])
        if valid.sum() < 2:
            continue
        filled[:, column] = np.interp(x, x[valid], data[valid, column])
    return filled, valid_rows


def _build_standing_grf_from_total(
    gaitway: GaitwayAsciiData,
    query: np.ndarray,
    in_bounds: np.ndarray,
    R: np.ndarray,
    R_fp_to_mocap: np.ndarray,
    grade: np.ndarray,
    grade_source: str,
    foot_centers: dict[str, np.ndarray],
    *,
    force_time_offset_s: float,
    force_threshold_N: float,
    cutoff_hz: float | None,
    opensim_x_sign: float,
    opensim_z_sign: float,
    rear_axis_mocap: np.ndarray,
) -> tuple[list[dict], np.ndarray, dict]:
    """用总 COP 在双脚中心连线上的位置估计静止站立左右载荷。

    站立导出没有 Gaitway 原生左右力，因此此处是显式估计：左右竖直载荷比例由
    总 COP 投影得到；总三维力按同一比例拆分；每脚 COP 以脚中心为基准共同平移，
    从而严格保持 ``wR*pR + wL*pL == total_COP``。该方法保持总合力和总力矩，
    但左右分配并非实测，QC 必须标为 WARN。
    """
    required = (
        "GRFz vertical (N)",
        "GRFy fore-aft (N)",
        "GRFx lateral (N)",
        "CoPx lateral (m)",
        "CoPy fore-aft (m)",
    )
    missing = [name for name in required if name not in gaitway.columns]
    if missing:
        raise ValueError(f"站立载荷估计缺少总力/总 COP 列：{missing}")

    centers: dict[str, np.ndarray] = {}
    center_valid: dict[str, np.ndarray] = {}
    for side in ("right", "left"):
        raw = np.asarray(foot_centers.get(side), dtype=np.float64)
        if raw.shape != (len(query), 3):
            raise ValueError(
                f"{side} 脚中心 shape {raw.shape}，应为 {(len(query), 3)}"
            )
        centers[side], center_valid[side] = _fill_marker_gaps(raw)

    native_t = gaitway.time_s
    fz_n = gaitway.columns["GRFz vertical (N)"]
    fy_n = gaitway.columns["GRFy fore-aft (N)"]
    fx_n = gaitway.columns["GRFx lateral (N)"]
    copx_n = gaitway.columns["CoPx lateral (m)"]
    copy_n = gaitway.columns["CoPy fore-aft (m)"]
    finite_native = (
        np.isfinite(fz_n) & np.isfinite(fy_n) & np.isfinite(fx_n)
        & np.isfinite(copx_n) & np.isfinite(copy_n)
    )
    contact_n = finite_native & (fz_n > float(force_threshold_N))

    force_n = np.column_stack([
        float(opensim_x_sign) * fy_n,
        float(opensim_z_sign) * fx_n,
        fz_n,
    ]) @ R.T
    point_n = np.column_stack([copy_n, copx_n, np.zeros_like(copx_n)]) @ R.T
    force_n = rotate_treadmill_grade(force_n, grade, rear_axis_mocap)
    point_n = rotate_treadmill_grade(point_n, grade, rear_axis_mocap)

    native_rate = gaitway.sample_rate_hz
    apply_aa = (
        cutoff_hz is not None and float(cutoff_hz) > 0
        and float(cutoff_hz) < native_rate / 2.0
    )
    if apply_aa:
        force_n = lowpass_segmented(force_n, contact_n, native_rate, float(cutoff_hz))
        point_n = lowpass_segmented(point_n, contact_n, native_rate, float(cutoff_hz))
    force_n = np.where(contact_n[:, None], force_n, 0.0)
    point_n = np.where(contact_n[:, None], point_n, 0.0)

    total_force = _interp_vec(native_t, force_n, query, fill=0.0)
    total_point = _interp_vec(native_t, point_n, query, fill=0.0)
    total_fz = _interp(native_t, fz_n, query, fill=0.0)
    contact = in_bounds & np.isfinite(total_fz) & (total_fz > float(force_threshold_N))
    marker_valid = center_valid["right"] & center_valid["left"]
    line = centers["right"] - centers["left"]
    denominator = np.sum(np.square(line), axis=1)
    geometry_valid = marker_valid & np.isfinite(denominator) & (denominator > 1e-6)
    valid = contact & geometry_valid & np.isfinite(total_point).all(axis=1)

    right_share = np.full(len(query), 0.5, dtype=np.float64)
    right_share[geometry_valid] = (
        np.sum(
            (total_point[geometry_valid] - centers["left"][geometry_valid])
            * line[geometry_valid],
            axis=1,
        )
        / denominator[geometry_valid]
    )
    # 极端外推一般来自短暂 COP/marker 异常；保留至少 5% 的每侧载荷，避免让
    # 双脚站立瞬间退化成单脚且把噪声放大。
    right_share = np.clip(right_share, 0.05, 0.95)
    left_share = 1.0 - right_share

    weighted_center = (
        right_share[:, None] * centers["right"]
        + left_share[:, None] * centers["left"]
    )
    common_shift = total_point - weighted_center
    right_point = centers["right"] + common_shift
    left_point = centers["left"] + common_shift
    torque = np.zeros_like(total_force)

    def foot(name: str, share: np.ndarray, point: np.ndarray) -> dict:
        force = total_force * share[:, None]
        force[~valid] = 0.0
        point = point.copy()
        point[~valid] = 0.0
        return {"name": name, "force": force, "point": point, "torque": torque.copy()}

    feet = [
        foot("right", right_share, right_point),
        foot("left", left_share, left_point),
    ]
    valid_share = right_share[valid]
    qc = {
        "force_transform_version": FORCE_TRANSFORM_VERSION,
        "grade_source": grade_source,
        "grade_percent_min": float(np.min(grade)),
        "grade_percent_max": float(np.max(grade)),
        "grade_percent_median": float(np.median(grade)),
        "grade_angle_deg_median": float(np.degrees(np.arctan(np.median(grade) / 100.0))),
        "native_force_signs_walk_left_up": [
            float(opensim_x_sign), float(opensim_z_sign), 1.0
        ],
        "zero_grade_rotation_fp_to_mocap": np.asarray(R_fp_to_mocap).tolist(),
        "fixed_rear_axis_mocap": np.asarray(rear_axis_mocap).tolist(),
        "origin_mocap_mm": O_MOCAP_MM.tolist(),
        "assumptions": [
            "站立时 Gaitway 左右分力缺失，使用总 COP 与双脚 Heel/Toe marker 估计左右载荷",
            "总三维力按估计的竖直载荷比例拆分",
            "左右 COP 共同平移以保持总 COP 和总外力矩",
            "该左右分配不是实测，QC 强制 WARN",
        ],
        "force_time_offset_s": float(force_time_offset_s),
        "gaitway_sample_rate_hz": gaitway.sample_rate_hz,
        "grf_cutoff_hz": float(cutoff_hz) if cutoff_hz is not None else None,
        "opensim_x_sign": float(opensim_x_sign),
        "opensim_z_sign": float(opensim_z_sign),
        "n_valid_decomposed_frames": int(valid.sum()),
        "n_right_contact_frames": int(valid.sum()),
        "n_left_contact_frames": int(valid.sum()),
        "decomposition_method": "estimated_from_total_cop_and_foot_markers",
        "bilateral_force_measured": False,
        "right_load_share_median": (
            float(np.median(valid_share)) if valid_share.size else None
        ),
        "right_load_share_min": float(np.min(valid_share)) if valid_share.size else None,
        "right_load_share_max": float(np.max(valid_share)) if valid_share.size else None,
        "total_fz_min_N": float(np.min(total_fz[in_bounds])),
        "total_fz_max_N": float(np.max(total_fz[in_bounds])),
    }
    return feet, valid, qc
