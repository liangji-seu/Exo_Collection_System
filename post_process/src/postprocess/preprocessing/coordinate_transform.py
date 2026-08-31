"""3D 坐标变换：严格区分"点"与"向量"。

点（position / COP）     :  p_global = R @ p_local + t
向量（force / free moment）: v_global = R @ v_local      （平移**不**作用）

禁止把 homogeneous transform 无脑作用到所有量——平移作用到力上会把
力的作用点搬走，导致力矩算错。这是整个工程最容易错的地方，务必用
tests/test_coordinate_transform.py 覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transform3D:
    """R: 3x3 旋转矩阵；t: (3,) 平移（目标系单位）。"""
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=np.float64)
        t = np.asarray(self.t, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"R 必须是 3x3，得到 {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"t 必须是 (3,)，得到 {t.shape}")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    # -- 点 ----------------------------------------------------------- #
    def apply_position(self, p: np.ndarray) -> np.ndarray:
        """点：R@p + t。输入 (..., 3)。"""
        p = np.asarray(p, dtype=np.float64)
        shape = p.shape
        flat = p.reshape(-1, 3)
        out = (flat @ self.R.T) + self.t
        return out.reshape(shape)

    # -- 向量（力 / 自由力矩）----------------------------------------- #
    def apply_vector(self, v: np.ndarray) -> np.ndarray:
        """向量：R@v，无平移。输入 (..., 3)。"""
        v = np.asarray(v, dtype=np.float64)
        shape = v.shape
        flat = v.reshape(-1, 3)
        out = flat @ self.R.T
        return out.reshape(shape)

    # 别名，语义清晰
    apply_force = apply_vector
    apply_free_moment = apply_vector

    # -- 组合 / 逆 ----------------------------------------------------- #
    def compose(self, other: "Transform3D") -> "Transform3D":
        """先 self 再 other：T = other ∘ self。"""
        return Transform3D(self.R @ other.R, other.t + other.R @ self.t)

    def inverse(self) -> "Transform3D":
        Rinv = self.R.T
        return Transform3D(Rinv, -Rinv @ self.t)

    @classmethod
    def identity(cls) -> "Transform3D":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_rotation_translation(cls, R: np.ndarray, t: np.ndarray) -> "Transform3D":
        return cls(np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64))

    @classmethod
    def from_axes_origin(cls, x_axis, y_axis, z_axis, origin) -> "Transform3D":
        """由"目标系原点 + 三轴单位方向（在源系下的表达）"构造 源→目标 变换。

        x_axis/y_axis/z_axis 是目标系三轴在源系坐标下的单位向量（列向量）。
        构造 R = [x y z]（列拼接），origin 是目标系原点在源系下的坐标。
        这样得到的是 目标系→源系 的变换，取逆得到 源系→目标系。
        调用方更常用的是：给出"力台原点在全局的坐标 + 力台三轴在全局的方向"，
        即构造 全局→力台，再取逆得到 力台→全局。
        """
        R = np.column_stack([np.asarray(x_axis, dtype=np.float64),
                             np.asarray(y_axis, dtype=np.float64),
                             np.asarray(z_axis, dtype=np.float64)])
        t = np.asarray(origin, dtype=np.float64)
        return cls(R, t)


__all__ = ["Transform3D"]
