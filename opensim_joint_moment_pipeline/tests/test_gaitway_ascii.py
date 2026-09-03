from __future__ import annotations

import numpy as np

from pipeline.gaitway import build_bilateral_grf, read_gaitway_ascii


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
