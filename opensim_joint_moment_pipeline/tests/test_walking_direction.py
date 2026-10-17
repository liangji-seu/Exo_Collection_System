import numpy as np
import pytest
from pipeline.opensim_io.walking_direction import (
    HALF_TURN, body_heading, anatomical_load_mapping, rotate_loads, is_backward_trial,
)
from pipeline.qc.evaluate import _force_checks


def geometry():
    names = ["R.Heel", "R.Toe", "L.Heel", "L.Toe"]
    points = np.tile([[0, 0, -100], [300, 0, -100], [0, 0, 100], [300, 0, 100]], (120, 1, 1)).astype(float)
    contacts = np.arange(120) % 20 < 10
    feet = []
    for side, active, z in (("right", contacts, -.1), ("left", ~contacts, .1)):
        force = np.zeros((120, 3)); force[active, 1] = 800
        point = np.tile([.15, 0, z], (120, 1)); point[~active] = 0
        feet.append({"name": side, "force": force, "point": point, "torque": np.zeros_like(force)})
    return names, points, feet


def test_forward_yaw_is_not_normalized():
    names, points, _ = geometry()
    points[:, 1, 2] += 40
    rotation, audit = body_heading(names, points, required=True)
    np.testing.assert_array_equal(rotation, np.eye(3))
    assert audit["yaw_normalization_deg"] == 0


def test_reverse_static_and_dynamic_are_independently_normalized():
    names, points, _ = geometry()
    static_rotation, _ = body_heading(names, points, required=True)
    dynamic_rotation, audit = body_heading(names, points @ HALF_TURN.T, required=True)
    np.testing.assert_array_equal(static_rotation, np.eye(3))
    np.testing.assert_array_equal(dynamic_rotation, HALF_TURN)
    assert audit["heading"] == "reverse"


def test_mixed_heading_refuses_median_only_decision():
    names, points, _ = geometry()
    points[:30] = points[:30] @ HALF_TURN.T
    with pytest.raises(ValueError, match="转身"):
        body_heading(names, points, required=True)


def test_half_turn_is_proper_and_moment_covariant():
    assert np.linalg.det(HALF_TURN) == 1
    point, force = np.array([1., .2, .3]), np.array([10., 800., 30.])
    np.testing.assert_allclose(np.cross(point @ HALF_TURN.T, force @ HALF_TURN.T), np.cross(point, force) @ HALF_TURN.T)
    feet = [{"name": "right", "point": point[None], "force": force[None], "torque": np.array([[1., 2., 3.]])}]
    rotated = rotate_loads(feet, HALF_TURN)[0]
    for key in ("point", "force", "torque"):
        np.testing.assert_array_equal(rotated[key], feet[0][key] @ HALF_TURN.T)
    assert rotate_loads(feet, np.eye(3)) is feet


@pytest.mark.parametrize("swap", [False, True])
def test_single_support_maps_native_channels_by_anatomy(swap):
    names, points, feet = geometry()
    native = [dict(feet[1], name="right"), dict(feet[0], name="left")] if swap else feet
    mapped, audit = anatomical_load_mapping(native, names, points, np.ones(120, bool))
    assert audit["native_to_anatomical"]["R"] == ("left" if swap else "right")
    np.testing.assert_array_equal(mapped[0]["force"], feet[0]["force"])
    assert audit["evidence"]["R"]["n_bouts"] == 6
    assert audit["double_support_decomposition_validated"] is False


@pytest.mark.parametrize("failure", ["too_few_bouts", "ambiguous_cop", "inconsistent_mapping"])
def test_mapping_fails_closed(failure):
    names, points, feet = geometry()
    selected = np.ones(120, bool)
    if failure == "too_few_bouts":
        selected[40:] = False
    elif failure == "ambiguous_cop":
        for foot in feet:
            foot["point"][:, 2] = 0
    else:
        feet[0]["point"][:, 2] *= -1
    with pytest.raises(ValueError, match="映射未通过"):
        anatomical_load_mapping(feet, names, points, selected)


def test_backward_force_quality_can_never_pass_from_geometry_alone():
    checks = _force_checks({"n_valid_decomposed_frames": 120, "n_right_contact_frames": 60,
                            "n_left_contact_frames": 60, "backward_decomposition_unvalidated": True}, 120)
    assert next(x for x in checks if x["key"] == "backward_force_decomposition")["status"] == "WARN"
    assert is_backward_trial("501_STAND_BWALK_STAND_NOEXO_r1")


def test_backward_transient_requires_explicit_window_before_reading_inputs(tmp_path):
    from pipeline.opensim_io.prep_session import prepare_session
    with pytest.raises(ValueError, match="显式设置分析起止时间"):
        prepare_session(static_c3d_path="missing", dynamic_c3d_path="missing", gaitway_txt_path="missing",
                        generic_model_path="missing", out_dir=tmp_path, subject_id="501", mass_kg=80, height_m=1.87,
                        gaitway_offset_s=0, condition_code="STAND_BWALK_STAND_NOEXO")
