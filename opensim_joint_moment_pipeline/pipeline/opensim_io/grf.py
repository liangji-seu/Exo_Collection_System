"""GRF 生成：测力台 total 合力 → OpenSim ExternalLoads（.mot + .xml）。

单块跑步机只能得到 **total** 合力（TOTAL_ONLY），无法在双支撑期分解左右力。
因此策略：
- **单支撑期**：把 total 合力整段分配给支撑脚（另一只脚置 0）；
- **双支撑期 / 无接触**：两脚都置 0（ID 输出随后按 valid_for_id mask 舍弃）。

力的符号：C3D 是「脚对台面」，转成 OpenSim 的「地对脚」整体取反。
坐标：力台 native (Fx1=侧向, Fy1=前后向, Fz1=竖直) → OpenSim ground。
自由力矩 Tz：默认置 0（±20+ Nm 疑似非纯自由力矩，留待确认后再启用）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..filtering import lowpass_zero_phase
from ..transforms import (
    force_plate_native_to_opensim,
    cop_plate_native_to_opensim,
    free_moment_plate_native_to_opensim,
)
from .write_grf_mot import write_grf_mot


def _channel(data, name: str) -> int:
    return data.analog_labels.index(name)


def build_grf_feet(
    data,
    right_contact: np.ndarray,
    left_contact: np.ndarray,
    R_fp_to_mocap: np.ndarray,
    *,
    include_free_moment: bool = False,
    cutoff_hz: float | None = None,
    advance_ms: float = 0.0,
    force_x_sign: float = 1.0,
    force_z_sign: float = 1.0,
) -> list[dict]:
    """把 total 合力按支撑脚拆分，返回 write_grf_mot 需要的 feet 列表。

    顺序固定 [right, left]，对应 ExternalLoads 的 calcn_r / calcn_l。
    每项含 force/point/torque，单位 SI（N / m / Nm），OpenSim ground 帧。
    """
    iFx, iFy, iFz = (_channel(data, f"{k}1") for k in ("Fx", "Fy", "Fz"))
    iCx, iCy = _channel(data, "COPx1"), _channel(data, "COPy1")

    fx = data.analogs[:, iFx].astype(np.float64)
    fy = data.analogs[:, iFy].astype(np.float64)
    fz = data.analogs[:, iFz].astype(np.float64)
    copx = data.analogs[:, iCx].astype(np.float64)
    copy = data.analogs[:, iCy].astype(np.float64)

    F_osim = force_plate_native_to_opensim(fx, fy, fz, R_fp_to_mocap, ground_on_foot=True)
    P_osim = cop_plate_native_to_opensim(copx, copy, R_fp_to_mocap)  # (n,3) m

    if cutoff_hz is not None:
        F_osim = lowpass_zero_phase(
            F_osim, data.point_rate_hz, float(cutoff_hz), preserve_missing=False)
        P_osim = lowpass_zero_phase(
            P_osim, data.point_rate_hz, float(cutoff_hz), preserve_missing=False)

    advance_frames = int(round(float(advance_ms) / 1000.0 * data.point_rate_hz))
    if advance_frames > 0:
        def advance(signal: np.ndarray) -> np.ndarray:
            shifted = np.zeros_like(signal)
            shifted[:-advance_frames] = signal[advance_frames:]
            return shifted
        F_osim = advance(F_osim)
        P_osim = advance(P_osim)
    F_osim[:, 0] *= float(force_x_sign)
    F_osim[:, 2] *= float(force_z_sign)

    n = data.n_frames
    torque = np.zeros((n, 3), dtype=np.float64)
    if include_free_moment:
        iTz = _channel(data, "Tz1")
        torque = free_moment_plate_native_to_opensim(
            data.analogs[:, iTz].astype(np.float64), R_fp_to_mocap, ground_on_foot=True)

    right_only = right_contact & ~left_contact
    left_only = left_contact & ~right_contact

    def foot(active: np.ndarray) -> dict:
        a = active.astype(np.float64)[:, None]  # (n,1)
        return {
            "name": "right" if active is right_only else "left",
            "force": F_osim * a,
            "point": P_osim * a,
            "torque": torque * a,
        }

    return [foot(right_only), foot(left_only)]


def write_external_loads_xml(
    path: str | Path,
    mot_filename: str,
    *,
    right_body: str = "calcn_r",
    left_body: str = "calcn_l",
) -> None:
    """写 ExternalLoads XML，两张力分别作用于左右 calcn。"""
    def _force(name: str, body: str, idx: int) -> str:
        return (
            f'\t\t\t<ExternalForce name="{name}">\n'
            f'\t\t\t\t<applied_to_body>{body}</applied_to_body>\n'
            f'\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>\n'
            f'\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>\n'
            f'\t\t\t\t<force_identifier>{idx}_ground_force_v</force_identifier>\n'
            f'\t\t\t\t<point_identifier>{idx}_ground_force_p</point_identifier>\n'
            f'\t\t\t\t<torque_identifier>{idx}_ground_torque_</torque_identifier>\n'
            f'\t\t\t\t<datafile>{mot_filename}</datafile>\n'
            f'\t\t\t</ExternalForce>\n'
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<OpenSimDocument Version="40000">\n'
        '\t<ExternalLoads name="gait_grf">\n'
        '\t\t<objects>\n'
        + _force("RightFootGRF", right_body, 1)
        + _force("LeftFootGRF", left_body, 2)
        + '\t\t</objects>\n'
        f'\t\t<datafile>{mot_filename}</datafile>\n'
        '\t</ExternalLoads>\n'
        '</OpenSimDocument>\n'
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(xml, encoding="utf-8")


def build_grf(
    data,
    right_contact: np.ndarray,
    left_contact: np.ndarray,
    R_fp_to_mocap: np.ndarray,
    out_dir: str | Path,
    *,
    mot_name: str = "grf.mot",
    xml_name: str = "external_loads.xml",
    include_free_moment: bool = False,
    cutoff_hz: float | None = None,
    advance_ms: float = 0.0,
    force_x_sign: float = 1.0,
    force_z_sign: float = 1.0,
) -> dict:
    """一站式：写 grf.mot + external_loads.xml，返回 feet 摘要。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feet = build_grf_feet(data, right_contact, left_contact, R_fp_to_mocap,
                          include_free_moment=include_free_moment,
                          cutoff_hz=cutoff_hz, advance_ms=advance_ms,
                          force_x_sign=force_x_sign, force_z_sign=force_z_sign)
    write_grf_mot(out / mot_name, data.time_s, feet)
    write_external_loads_xml(out / xml_name, mot_name)
    return {
        "mot": str(out / mot_name),
        "xml": str(out / xml_name),
        "feet": [f["name"] for f in feet],
    }


__all__ = ["build_grf_feet", "write_external_loads_xml", "build_grf"]
