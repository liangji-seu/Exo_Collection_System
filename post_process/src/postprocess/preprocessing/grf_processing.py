"""GRF 处理：把力台原始输出转成 OpenSim ExternalLoads 需要的表达。

力台原始输出（XINGYING/Gaitway）通常是：
    Fx/Fy/Fz [N] + COPx/COPy [mm] + Tz [Nmm]（自由力矩，不是 Mx/My/Mz）

OpenSim ExternalLoads 每只脚需要（全在 OpenSim ground 系、SI 单位）：
    Force  (N)   → R @ F_local
    Point  (m)   → R @ p_local + t
    Torque (Nm)  → R @ (0,0,Tz)   只放自由力矩；力×力臂已由"力+作用点"隐含

**严禁**：既用 COP 表示力臂、又把含 r×F 的原点力矩重复加入。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinate_transform import Transform3D
from .units import convert


class GrfBlockingError(RuntimeError):
    """缺少标定/约定信息，无法生成 GRF。"""


@dataclass
class FootGrf:
    name: str
    force_N: np.ndarray      # (n, 3) OpenSim ground
    point_m: np.ndarray      # (n, 3) OpenSim ground
    torque_Nm: np.ndarray    # (n, 3) free moment, OpenSim ground


def apply_force_plate_transform(
    force_local_N: np.ndarray,
    cop_local: np.ndarray,          # (n, 2) 或 (n, 3)，台面平面内
    free_moment_local: np.ndarray,  # (n, 3)，通常只有 (0,0,Tz)
    transform: Transform3D,         # 力台局部 → OpenSim ground（t 单位 m）
    *,
    unit_cop: str = "mm",
    unit_moment: str = "Nmm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """核心变换：点与向量严格区分。返回 (force_N, point_m, torque_Nm)。"""
    force_local_N = np.asarray(force_local_N, dtype=np.float64)
    cop = np.asarray(cop_local, dtype=np.float64)
    free_moment = np.asarray(free_moment_local, dtype=np.float64)

    if cop.shape[-1] == 2:
        cop = np.concatenate([cop, np.zeros((*cop.shape[:-1], 1))], axis=-1)

    # 单位换算（显式，不藏在变换里）
    cop_m = convert(cop, unit_cop, "m")
    moment_Nm = convert(free_moment, unit_moment, "Nm")

    force_ground = transform.apply_force(force_local_N)       # 无平移
    point_ground = transform.apply_position(cop_m)            # R@p + t
    torque_ground = transform.apply_free_moment(moment_Nm)    # 无平移

    return force_ground, point_ground, torque_ground


def resolve_force_convention(force_N: np.ndarray, convention: str) -> np.ndarray:
    """按约定调整力的符号。

    - ``ground_on_foot``  设备输出 = 地面作用在脚的力（GRF 反力）→ 原样
    - ``foot_on_ground``  设备输出 = 脚作用在地面的力 → 取负
    """
    if convention == "ground_on_foot":
        return force_N
    if convention == "foot_on_ground":
        return -force_N
    raise GrfBlockingError(
        "force.convention 未确认（必须显式指定 ground_on_foot 或 foot_on_ground），"
        "无法确定 GRF 符号。"
    )


__all__ = ["FootGrf", "GrfBlockingError", "apply_force_plate_transform",
           "resolve_force_convention"]
