"""单支撑检测 / 相位分类 单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.gait.detect_single_support import (
    PHASE_RIGHT_SS,
    PHASE_LEFT_SS,
    PHASE_DOUBLE,
    PHASE_NO_CONTACT,
    PHASE_UNKNOWN,
    classify_phase,
    valid_for_id_mask,
)
from pipeline.gait.build_support_mask import build_support_mask, extract_segments
from pipeline.gait.detect_contact import detect_contacts
from pipeline.c3d.reader import C3dData, SubjectInfo


# --------------------------------------------------------------------------- #
# 相位分类（纯逻辑）
# --------------------------------------------------------------------------- #
def test_classify_phase_four_states():
    right = [True, False, True, False, None]
    left = [False, True, True, False, False]
    phase = classify_phase(right, left)
    assert list(phase) == [PHASE_RIGHT_SS, PHASE_LEFT_SS, PHASE_DOUBLE, PHASE_NO_CONTACT, PHASE_UNKNOWN]


def test_classify_phase_nan_is_unknown():
    phase = classify_phase([True, np.nan], [False, False])
    assert phase[0] == PHASE_RIGHT_SS
    assert phase[1] == PHASE_UNKNOWN


def test_classify_phase_shape_mismatch_raises():
    with pytest.raises(ValueError):
        classify_phase([True, False], [False])


def test_valid_for_id_mask():
    phase = [PHASE_RIGHT_SS, PHASE_LEFT_SS, PHASE_DOUBLE, PHASE_NO_CONTACT, PHASE_UNKNOWN]
    mask = valid_for_id_mask(phase)
    assert list(mask) == [True, True, False, False, False]


# --------------------------------------------------------------------------- #
# 分段
# --------------------------------------------------------------------------- #
def test_extract_segments_trims_boundary():
    time = np.arange(100, dtype=np.float64) * 0.01  # 1 s, dt=10ms
    phase = np.array([PHASE_RIGHT_SS] * 100, dtype=object)
    phase[0:5] = PHASE_DOUBLE
    phase[95:100] = PHASE_DOUBLE
    segs = extract_segments(time, phase, trim_boundary_ms=20.0, min_segment_frames=3)
    assert len(segs) == 1
    seg = segs[0]
    assert seg["foot"] == "right"
    # trim 20ms = 2 帧，原始 5..94（含），裁剪后 7..92
    assert seg["start_frame"] == 7
    assert seg["end_frame"] == 92


def test_extract_segments_drops_too_short():
    time = np.arange(10, dtype=np.float64) * 0.01
    phase = np.array([PHASE_LEFT_SS] * 10, dtype=object)
    segs = extract_segments(time, phase, trim_boundary_ms=20.0, min_segment_frames=10)
    assert segs == []


def test_build_support_mask_end_to_end():
    time = np.arange(50, dtype=np.float64) * 0.01
    right = np.array([True] * 20 + [False] * 30)
    left = np.array([False] * 20 + [True] * 20 + [False] * 10)
    anyc = np.ones(50, dtype=bool)
    mask = build_support_mask(time, right, left, anyc, trim_boundary_ms=0.0)
    assert mask.valid_for_id.sum() == 40
    stats = mask.statistics()
    assert stats["n_right_segments"] == 1
    assert stats["n_left_segments"] == 1


# --------------------------------------------------------------------------- #
# detect_contacts（合成 C3D）
# --------------------------------------------------------------------------- #
def _fake_c3d(right_z: float, left_z: float, fz: float = 785.0) -> C3dData:
    names = ["R.ASIS", "L.ASIS", "V.Sacral", "R.Heel", "R.Toe", "L.Heel", "L.Toe"]
    prefix = "100_no_exo_dynamic:"
    n = 40
    pts = np.zeros((n, len(names), 3), dtype=np.float32)
    # pelvis 高，脚在地面（x/y 给非零值，避免与 c3d 的 [0,0,0] 遮挡占位冲突）
    for i, nm in enumerate(names):
        if nm in ("R.ASIS", "L.ASIS", "V.Sacral"):
            pts[:, i, :] = [0.0, 0.0, 800.0]
        elif nm in ("R.Heel", "R.Toe"):
            pts[:, i, :] = [100.0, 50.0, right_z]
        else:
            pts[:, i, :] = [100.0, -50.0, left_z]
    subjects = (SubjectInfo("dynamic", prefix, False, tuple(range(len(names))),
                            tuple(prefix + x for x in names)),)
    return C3dData(
        path="fake.c3d", manufacturer="x", software="x", software_version="x",
        point_rate_hz=100.0, analog_rate_hz=100.0, n_frames=n, data_start=0,
        point_labels=tuple(prefix + x for x in names), point_units="mm",
        points_mm=pts, residuals=np.zeros((n, len(names)), dtype=np.float32),
        subjects=subjects,
        analog_labels=("Fx1", "Fy1", "Fz1", "COPx1", "COPy1", "Tz1"),
        analog_units=("N", "N", "N", "mm", "mm", "Nmm"),
        analog_scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        analogs=np.tile([0.0, 0.0, fz, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        force_platforms=(),
    )


def test_detect_contacts_right_only():
    data = _fake_c3d(right_z=0.0, left_z=200.0)
    c = detect_contacts(data, vertical_axis=2, force_threshold_N=50.0)
    assert c.vertical_axis == 2
    assert c.right_contact.all()
    assert not c.left_contact.any()


def test_detect_contacts_double_support():
    data = _fake_c3d(right_z=0.0, left_z=0.0)
    c = detect_contacts(data, vertical_axis=2, force_threshold_N=50.0)
    assert c.right_contact.all() and c.left_contact.all()


def test_detect_contacts_no_contact_gated_by_fz():
    data = _fake_c3d(right_z=0.0, left_z=0.0, fz=0.0)  # 无人着地
    c = detect_contacts(data, vertical_axis=2, force_threshold_N=50.0)
    assert not c.right_contact.any()
    assert not c.left_contact.any()


def test_detect_vertical_axis_z():
    data = _fake_c3d(right_z=0.0, left_z=0.0)
    assert detect_contacts(data, vertical_axis=None).vertical_axis == 2
