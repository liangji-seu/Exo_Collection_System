from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.gaitway import GaitwayAsciiData, build_bilateral_grf, read_gaitway_ascii
from pipeline.transforms import R_MOCAP_TO_OPENSIM
from pipeline.filtering import lowpass_segmented


def test_parse_and_build_native_bilateral_grf(tmp_path):
    path = tmp_path / "trial.txt"
    path.write_text(
        "Sample rate (Hz)\t1000\n"
        "Time (s)\tFzL(N)\tFyL(N)\tFxL(N)\tCoPxL(m)\tCoPyL(m)\t"
        "FzR(N)\tFyR(N)\tFxR(N)\tCoPxR(m)\tCoPyR(m)\tGRFz vertical (N)\n"
        "0.000\t300\t10\t20\t0.5\t0.7\t400\t30\t40\t0.3\t0.8\t700\n"
        "0.001\t310\t11\t21\t0.5\t0.7\t410\t31\t41\t0.3\t0.8\t720\n",
        encoding="utf-8",
    )

    parsed = read_gaitway_ascii(path)
    assert parsed.sample_rate_hz == 1000.0
    feet, valid, qc = build_bilateral_grf(
        parsed,
        np.array([0.0, 0.001]),
        0.0,
        np.eye(3),
        force_threshold_N=50.0,
        opensim_x_sign=-1.0,
        opensim_z_sign=-1.0,
    )

    assert [foot["name"] for foot in feet] == ["right", "left"]
    np.testing.assert_allclose(feet[0]["force"][:, 1], [400.0, 410.0])
    # With identity plate-to-mocap rotation, the two configured horizontal sign
    # corrections yield OpenSim [native Fx, Fz, native Fy].
    np.testing.assert_allclose(feet[0]["force"][0], [40.0, 400.0, 30.0])
    assert valid.tolist() == [True, True]
    assert qc["opensim_x_sign"] == -1.0
    assert qc["opensim_z_sign"] == -1.0


def test_quiet_standing_estimates_bilateral_load_from_total_cop_and_markers():
    time = np.array([0.0, 0.01])
    nan = np.full(2, np.nan)
    zeros = np.zeros(2)
    gaitway = GaitwayAsciiData(
        Path("standing.txt"),
        {"Sample rate (Hz)": "100"},
        time,
        {
            "FzL(N)": nan.copy(), "FyL(N)": nan.copy(), "FxL(N)": nan.copy(),
            "CoPxL(m)": nan.copy(), "CoPyL(m)": nan.copy(),
            "FzR(N)": nan.copy(), "FyR(N)": nan.copy(), "FxR(N)": nan.copy(),
            "CoPxR(m)": nan.copy(), "CoPyR(m)": nan.copy(),
            "GRFz vertical (N)": np.array([800.0, 800.0]),
            "GRFy fore-aft (N)": zeros.copy(),
            "GRFx lateral (N)": zeros.copy(),
            "CoPx lateral (m)": np.array([0.0, 0.1]),
            "CoPy fore-aft (m)": zeros.copy(),
            "Grade (%)": zeros.copy(),
        },
    )
    centers = {
        "right": np.tile([0.0, 0.2, 0.0], (2, 1)),
        "left": np.tile([0.0, -0.2, 0.0], (2, 1)),
    }

    feet, valid, qc = build_bilateral_grf(
        gaitway,
        time,
        0.0,
        R_MOCAP_TO_OPENSIM.T,  # combined transform = identity for this fixture
        cutoff_hz=None,
        standing_foot_centers_opensim_m=centers,
    )

    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(feet[0]["force"][:, 2], [400.0, 600.0])
    np.testing.assert_allclose(feet[1]["force"][:, 2], [400.0, 200.0])
    total_force = feet[0]["force"] + feet[1]["force"]
    np.testing.assert_allclose(total_force[:, 2], 800.0)
    shares = feet[0]["force"][:, 2] / total_force[:, 2]
    reconstructed_cop = (
        shares[:, None] * feet[0]["point"]
        + (1.0 - shares)[:, None] * feet[1]["point"]
    )
    np.testing.assert_allclose(reconstructed_cop[:, 1], [0.0, 0.1])
    assert qc["decomposition_method"] == "estimated_from_total_cop_and_foot_markers"
    assert qc["bilateral_force_measured"] is False
    assert qc["right_load_share_median"] == 0.625


def test_missing_bilateral_force_requires_explicit_standing_mode():
    time = np.array([0.0, 0.01])
    nan = np.full(2, np.nan)
    zeros = np.zeros(2)
    gaitway = GaitwayAsciiData(
        Path("standing.txt"), {"Sample rate (Hz)": "100"}, time,
        {
            **{name: nan.copy() for name in (
                "FzL(N)", "FyL(N)", "FxL(N)", "CoPxL(m)", "CoPyL(m)",
                "FzR(N)", "FyR(N)", "FxR(N)", "CoPxR(m)", "CoPyR(m)",
            )},
            "GRFz vertical (N)": np.full(2, 800.0),
            "Grade (%)": zeros,
        },
    )
    with pytest.raises(ValueError, match="站立载荷估计"):
        build_bilateral_grf(gaitway, time, 0.0, np.eye(3), cutoff_hz=None)


def _continuous_fixture():
    time = np.arange(2001, dtype=float) / 1000
    columns = {}
    for side in ("R", "L"):
        columns[f"Fz{side}(N)"] = 40 + 30 * np.sin(2*np.pi*time) if side == "R" else np.full(len(time), 700.)
        columns[f"Fy{side}(N)"] = 12 * np.cos(2*np.pi*time)
        columns[f"Fx{side}(N)"] = np.full(len(time), 3.)
        columns[f"CoPx{side}(m)"] = np.full(len(time), .2 if side == "R" else .5)
        columns[f"CoPy{side}(m)"] = 1 + .3*time
    columns["GRFz vertical (N)"] = columns["FzR(N)"] + columns["FzL(N)"]
    columns["Grade (%)"] = np.zeros(len(time))
    return GaitwayAsciiData(Path("finite.txt"), {"Sample rate (Hz)":"1000"}, time, columns)


def test_event_threshold_does_not_cut_valid_low_load_wrench():
    data = _continuous_fixture()
    query = data.time_s[::10]
    a, valid, qc = build_bilateral_grf(data, query, 0, R_MOCAP_TO_OPENSIM.T, force_threshold_N=50)
    b, _, _ = build_bilateral_grf(data, query, 0, R_MOCAP_TO_OPENSIM.T, force_threshold_N=20)
    for fa, fb in zip(a,b):
        for key in ("force", "point", "torque"):
            np.testing.assert_array_equal(fa[key], fb[key])
    assert not np.array_equal(a[0]["contact"], b[0]["contact"])
    i = np.argmin(abs(query-1.0))  # Fz=40 N, physically loaded despite no 50 N event.
    assert valid[i] and not a[0]["contact"][i]
    assert np.linalg.norm(a[0]["force"][i]) > 30
    assert qc["event_threshold_applied_to_physical_force"] is False


def test_filter_and_resample_preserve_complete_origin_wrench():
    data = _continuous_fixture()
    query = data.time_s[::10]
    feet, valid, _ = build_bilateral_grf(data, query, 0, R_MOCAP_TO_OPENSIM.T, cutoff_hz=20)
    force = np.column_stack([data.columns[f"{axis}R(N)"] for axis in ("Fy","Fx","Fz")])
    point = np.column_stack([data.columns["CoPyR(m)"],data.columns["CoPxR(m)"],np.zeros(len(query)*10-9)])
    target = lowpass_segmented(np.column_stack([force,np.cross(point,force)]),
                              np.ones(len(data.time_s),bool), 1000, 20, time_s=data.time_s)[::10]
    foot = feet[0]
    np.testing.assert_allclose(foot["force"], target[:,:3], atol=1e-12)
    reconstructed = np.cross(foot["point"],foot["force"]) + foot["torque"]
    np.testing.assert_allclose(reconstructed, target[:,3:], atol=1e-12)
    # Nonzero output couple is a representation residual, not fabricated raw free moment.
    assert np.max(abs(foot["torque"][valid])) > 1e-5


def test_missing_loaded_cop_and_its_filter_edges_are_not_truth_or_interpolated():
    data = _continuous_fixture()
    data.columns["CoPxR(m)"][1000] = np.nan
    query = np.array([.7,.9,.9995,1.,1.0005,1.1,1.3])
    feet, valid, qc = build_bilateral_grf(data, query, 0, R_MOCAP_TO_OPENSIM.T, cutoff_hz=20)
    assert valid.tolist() == [True,False,False,False,False,False,True]
    assert feet[0]["input_available"].tolist() == [True,True,False,False,False,True,True]
    np.testing.assert_array_equal(feet[0]["force"][2:5],0)
    assert np.linalg.norm(feet[0]["force"][1]) > 0  # Guard cannot recut a finite physical load.
    assert qc["n_missing_loaded_cop_native_side_samples"] == 1


def test_explicit_zero_wrench_with_undefined_cop_is_valid_swing():
    data = _continuous_fixture()
    for side in ("R","L"):
        for axis in ("Fx","Fy","Fz"):
            data.columns[f"{axis}{side}(N)"][:] = 0
        for axis in ("CoPx","CoPy"):
            data.columns[f"{axis}{side}(m)"][:] = np.nan
    data.columns["GRFz vertical (N)"][:] = 0
    feet, valid, _ = build_bilateral_grf(data,np.array([1.]),0,R_MOCAP_TO_OPENSIM.T)
    assert valid.tolist() == [True]
    for foot in feet:
        assert foot["valid"].all() and not foot["contact"].any()
        np.testing.assert_array_equal(foot["force"],0)


def test_true_timestamp_gap_is_not_bridged():
    data = _continuous_fixture()
    data.time_s[1000:] += 1
    feet, valid, _ = build_bilateral_grf(data,np.array([.5,1.5,2.5]),0,R_MOCAP_TO_OPENSIM.T,cutoff_hz=20)
    assert valid.tolist() == [True,False,True]
    np.testing.assert_array_equal(feet[0]["force"][1],0)


def test_segment_filter_keeps_all_invalid_nan_and_handles_short_finite_segments():
    values = np.arange(12,dtype=float)
    assert np.isnan(lowpass_segmented(values,np.zeros(12,bool),1000,20)).all()
    assert np.isfinite(lowpass_segmented(values,np.ones(12,bool),1000,20)).all()


def test_default_grf_bandwidth_is_six_hz_and_explicit_twenty_is_preserved():
    from pipeline.gaitway.ascii_export import DEFAULT_GRF_CUTOFF_HZ
    from pipeline.opensim_io.prep_session import prepare_session
    import inspect
    data = _continuous_fixture()
    assert DEFAULT_GRF_CUTOFF_HZ == 6
    assert inspect.signature(prepare_session).parameters['grf_cutoff_hz'].default == 6
    _, _, default_qc = build_bilateral_grf(data,np.array([1.]),0,R_MOCAP_TO_OPENSIM.T)
    _, _, explicit_qc = build_bilateral_grf(data,np.array([1.]),0,R_MOCAP_TO_OPENSIM.T,cutoff_hz=20)
    assert default_qc['grf_cutoff_hz'] == 6
    assert explicit_qc['grf_cutoff_hz'] == 20
