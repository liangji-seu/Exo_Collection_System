from pathlib import Path

import numpy as np
import pytest

from pipeline.gaitway import GaitwayAsciiData, build_bilateral_grf, read_gaitway_ascii
from pipeline.transforms import rotate_treadmill_grade, R_MOCAP_TO_OPENSIM


def trial(grade):
    grade = np.asarray(grade, dtype=float)
    a = np.arctan(grade/100)
    z = np.zeros(len(a))
    columns = {'Grade (%)': grade, 'GRFz vertical (N)': 800*np.cos(a)}
    for side in 'RL':
        # Known vertical 400 N load per foot, expressed in inclined native axes.
        columns.update({f'Fz{side}(N)': 400*np.cos(a), f'Fy{side}(N)': -400*np.sin(a),
                        f'Fx{side}(N)': z.copy(), f'CoPx{side}(m)': z.copy(),
                        f'CoPy{side}(m)': np.ones(len(a))})
    return GaitwayAsciiData(Path('synthetic.txt'), {'Sample rate (Hz)': '1000'},
                            np.arange(len(a))/1000, columns)


def build(g):
    # Exactly level synthetic plate: x=forward, y=left, z=up.
    R = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    return build_bilateral_grf(g, g.time_s, 0., R, cutoff_hz=None,
                               opensim_x_sign=-1., opensim_z_sign=-1.,
                               require_grade=True, rear_axis_mocap=np.array([1., 0., 0.]))


def test_known_vertical_load_stays_vertical_through_up_and_down_grades():
    grade = np.array([0., 4.3661, 8.7489, -8.7489, 0.])
    feet, valid, qc = build(trial(grade))
    angle = np.arctan(grade/100)
    for foot in feet:
        np.testing.assert_allclose(foot['force'], np.tile([0., 400., 0.], (5, 1)), atol=1e-10)
        np.testing.assert_allclose(foot['point'][:, 0], np.cos(angle), atol=1e-12)
        np.testing.assert_allclose(foot['point'][:, 1], np.sin(angle), atol=1e-12)
    assert valid.all()
    assert qc['grade_source'] == 'Grade (%)'
    assert qc['force_transform_version'] == 'rear_axis_native_signs_v1'


def test_real_rear_axis_invariance_and_moment_covariance():
    axis = np.array([814.8, 0., 2.1]); axis /= np.linalg.norm(axis)
    ground_axis = R_MOCAP_TO_OPENSIM @ axis
    grades = np.array([0., 4.3661, 8.7489])
    p = np.array([[1., 0., 0.], [1., 2., 3.], [-1., 3., 2.]])
    f = np.array([[0., 800., 0.], [1., -2., 4.], [7., 3., -1.]])
    rot = lambda v: rotate_treadmill_grade(v, grades)
    np.testing.assert_allclose(rot(np.tile(ground_axis, (3, 1))), np.tile(ground_axis, (3, 1)), atol=1e-12)
    np.testing.assert_allclose(np.cross(rot(p), rot(f)), rot(np.cross(p, f)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(rot(f), axis=1), np.linalg.norm(f, axis=1))
    np.testing.assert_allclose(rotate_treadmill_grade(rot(p), -grades), p, atol=1e-12)


def test_zero_contact_stays_zero_with_grade():
    g = trial([8.7489, 8.7489])
    g.columns['FzR(N)'][:] = 0
    feet, valid, _ = build(g)
    assert valid.all()  # Left foot still loaded.
    np.testing.assert_array_equal(feet[0]['force'], 0)
    np.testing.assert_array_equal(feet[0]['point'], 0)


def test_missing_or_nonfinite_grade_is_not_silently_flat():
    g = trial([0., 0.])
    del g.columns['Grade (%)']
    with pytest.raises(ValueError, match='Grade'):
        build(g)
    g.columns['Grade (%)'] = np.array([0., np.nan])
    with pytest.raises(ValueError, match='Grade'):
        build(g)


def test_reader_preserves_grade_column(tmp_path):
    g = trial([0., 8.7489])
    path = tmp_path/'grade.txt'
    names = ['Time (s)', *g.columns]
    values = np.column_stack([g.time_s, *g.columns.values()])
    np.savetxt(path, values, delimiter='\t', header='Sample rate (Hz)\t1000\n'+'\t'.join(names), comments='')
    loaded = read_gaitway_ascii(path)
    np.testing.assert_array_equal(loaded.columns['Grade (%)'], [0., 8.7489])
