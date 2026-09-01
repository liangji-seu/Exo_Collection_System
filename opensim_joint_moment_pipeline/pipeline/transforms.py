"""坐标变换：mocap 全局 / 测力台 → OpenSim ground。

纯 numpy，EXO 与 opensim 两个环境通用（不 import opensim）。

三个坐标系（均已实测确认，见 configs/subject_001.yaml 注释）：

1. **mocap 全局**（NOKOV，mm）
   +X=人体左、+Y=人体后、+Z=竖直向上。

2. **OpenSim ground**（m）
   +X=人体前、+Y=竖直向上、+Z=人体右（右手系）。

3. **测力台 native**（C3D analog 通道）
   XINGYING 输出的通道名与力台标定轴是**反的**（已用单支撑帧实证）：
     ``Fx1`` / ``COPx1`` = **侧向**（力台局部 Y）
     ``Fy1`` / ``COPy1`` = **前后向/行走向**（力台局部 X）
     ``Fz1``            = 竖直
   因此用标定矩阵前，先做 (walk, lat, up) = (Fy1, Fx1, Fz1) 的重排。

力台标定矩阵 R_fp2mocap（把「局部 X=行走向、Y=侧向、Z=上」映射到 mocap）
从 config 的 ``transforms.forceplate_to_mocap`` 读入，不作为本模块常量。
"""

from __future__ import annotations

import numpy as np

# mocap 全局 → OpenSim ground 的旋转矩阵。
# 行 = OpenSim 轴在 mocap 下的坐标（det=1 右手）：
#   OpenSim X(前) = -mocapY, OpenSim Y(上) = +mocapZ, OpenSim Z(右) = -mocapX
R_MOCAP_TO_OPENSIM = np.array(
    [[0.0, -1.0, 0.0],
     [0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)

# 测力台（跑步机）右下角原点在 mocap 全局下的坐标（mm）。
# 选它作平移参考，使台面（mocap Z=-251）映射到 OpenSim Y=0、原点→(0,0,0)。
O_MOCAP_MM = np.array([-5078.5, 2086.5, -251.0], dtype=np.float64)

# 测力台表面在 mocap 下的 Z 高度（mm），用于「台面 → Y=0」的竖直平移（等价于用 O 平移）。
_TREADMILL_SURFACE_Z_MM = -251.0


def mocap_to_opensim_points(points_mm: np.ndarray) -> np.ndarray:
    """mocap 点（mm）→ OpenSim ground 点（mm）。

    p_osim = R_MOCAP_TO_OPENSIM @ (p_mocap - O_MOCAP_MM)
    台面 → Y=0，台原点 → 原点。输入 (..., 3)，输出同形。
    """
    p = np.asarray(points_mm, dtype=np.float64)
    return (p - O_MOCAP_MM) @ R_MOCAP_TO_OPENSIM.T


def force_plate_native_to_opensim(
    fx: np.ndarray, fy: np.ndarray, fz: np.ndarray,
    R_fp_to_mocap: np.ndarray,
    *,
    ground_on_foot: bool = True,
) -> np.ndarray:
    """测力台 native 力 (Fx1,Fy1,Fz1) → OpenSim ground 力。

    步骤：
    1. 重排 native → 标定局部帧 (walk, lat, up) = (fy, fx, fz)；
    2. 旋转 标定局部帧 → mocap：F_mocap = R_fp2mocap @ F_local；
    3. 旋转 mocap → opensim：F_osim = R_mocap2osim @ F_mocap；
    4. 符号：C3D 是「脚对台面」，OpenSim ID 要「地对脚」，``ground_on_foot=True``
       时整体取反。

    输入形状 (n,) 或 (n,1)，输出 (n,3)。
    """
    fx = np.asarray(fx, dtype=np.float64).reshape(-1)
    fy = np.asarray(fy, dtype=np.float64).reshape(-1)
    fz = np.asarray(fz, dtype=np.float64).reshape(-1)
    R_fp2mocap = np.asarray(R_fp_to_mocap, dtype=np.float64)
    R_fp2osim = R_MOCAP_TO_OPENSIM @ R_fp2mocap

    # native (fx=lateral, fy=walking, fz=up) -> 标定局部 (walk, lat, up)
    F_local = np.stack([fy, fx, fz], axis=1)  # (n,3)
    F_osim = F_local @ R_fp2osim.T
    if ground_on_foot:
        F_osim = -F_osim
    return F_osim


def cop_plate_native_to_opensim(
    copx: np.ndarray, copy: np.ndarray,
    R_fp_to_mocap: np.ndarray,
) -> np.ndarray:
    """测力台 native COP (COPx1,COPy1) → OpenSim ground 作用点（m）。

    COP 在台面（native z=0）。重排 + 旋转同力：
      COP_local = (copy, copx, 0)   # (walk, lat, 0)
      COP_osim  = R_fp2osim @ COP_local   （mm）
    台原点 → OpenSim 原点（无需额外平移，见 O_MOCAP_MM 的选择）。

    输入 (n,)，输出 (n,3) 米。
    """
    copx = np.asarray(copx, dtype=np.float64).reshape(-1)
    copy = np.asarray(copy, dtype=np.float64).reshape(-1)
    R_fp2mocap = np.asarray(R_fp_to_mocap, dtype=np.float64)
    R_fp2osim = R_MOCAP_TO_OPENSIM @ R_fp2mocap

    n = copx.shape[0]
    P_local = np.stack([copy, copx, np.zeros(n)], axis=1)  # (n,3) mm
    P_osim_mm = P_local @ R_fp2osim.T
    return P_osim_mm / 1000.0  # mm -> m


def free_moment_plate_native_to_opensim(
    tz: np.ndarray,
    R_fp_to_mocap: np.ndarray,
    *,
    ground_on_foot: bool = True,
) -> np.ndarray:
    """测力台 native 自由力矩 Tz1（绕竖直轴，Nmm）→ OpenSim ground 力矩（Nm）。

    自由力矩只有竖直分量（native z），旋转后落到 OpenSim 的对应轴向。
    """
    tz = np.asarray(tz, dtype=np.float64).reshape(-1)
    R_fp2mocap = np.asarray(R_fp_to_mocap, dtype=np.float64)
    R_fp2osim = R_MOCAP_TO_OPENSIM @ R_fp2mocap

    n = tz.shape[0]
    M_local = np.stack([np.zeros(n), np.zeros(n), tz], axis=1)  # (n,3) Nmm
    M_osim = M_local @ R_fp2osim.T
    if ground_on_foot:
        M_osim = -M_osim
    return M_osim / 1000.0  # Nmm -> Nm


def combine_transform(R_fp_to_mocap: np.ndarray) -> np.ndarray:
    """测力台标定局部帧 → OpenSim ground 的总旋转矩阵 R_fp2osim。"""
    return R_MOCAP_TO_OPENSIM @ np.asarray(R_fp_to_mocap, dtype=np.float64)


__all__ = [
    "R_MOCAP_TO_OPENSIM",
    "O_MOCAP_MM",
    "mocap_to_opensim_points",
    "force_plate_native_to_opensim",
    "cop_plate_native_to_opensim",
    "free_moment_plate_native_to_opensim",
    "combine_transform",
]
