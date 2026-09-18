from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.gaitway import GaitwayAsciiData, build_bilateral_grf, read_gaitway_ascii
from pipeline.transforms import R_MOCAP_TO_OPENSIM


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
