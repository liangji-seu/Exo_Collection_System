"""坐标变换核心测试：点与向量严格区分。

- translation 改变点（COP），不改变向量（force/torque）
- rotation 同时作用于点与向量
"""

import numpy as np

from postprocess.preprocessing.coordinate_transform import Transform3D


def test_translation_affects_position_but_not_force():
    T = Transform3D.from_rotation_translation(np.eye(3), np.array([10.0, 20.0, 30.0]))
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(T.apply_position(p), [11.0, 22.0, 33.0])
    assert np.allclose(T.apply_force(p), [1.0, 2.0, 3.0])       # 平移不作用于力
    assert np.allclose(T.apply_free_moment(p), [1.0, 2.0, 3.0])  # 平移不作用于力矩


def test_rotation_affects_both_position_and_force():
    # 绕 Z 轴 90°
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T = Transform3D.from_rotation_translation(R, np.zeros(3))
    v = np.array([1.0, 0.0, 0.0])
    assert np.allclose(T.apply_vector(v), [0.0, 1.0, 0.0])
    assert np.allclose(T.apply_position(v), [0.0, 1.0, 0.0])


def test_batch_shape_preserved():
    T = Transform3D.from_rotation_translation(np.eye(3), np.array([1.0, 2.0, 3.0]))
    pts = np.random.rand(100, 3)
    out = T.apply_position(pts)
    assert out.shape == (100, 3)


def test_inverse_roundtrip():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T = Transform3D.from_rotation_translation(R, np.array([5.0, -3.0, 2.0]))
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(T.inverse().apply_position(T.apply_position(p)), p)
