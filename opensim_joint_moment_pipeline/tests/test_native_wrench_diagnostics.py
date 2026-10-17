"""Physical signal identities and missing-data semantics, independent of ID."""
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.gaitway.ascii_export import (
    GaitwayAsciiData, audit_native_bilateral_wrench, read_gaitway_ascii,
    build_bilateral_grf,
)
from pipeline.qc.evaluate import _force_checks


def fixture():
    columns = {}
    for side, load, x in (("L", [0., 400., 0., 400.], [.2]*4),
                          ("R", [800., 400., 0., 400.], [.4, .6, .4, .6])):
        columns[f"Fz{side}(N)"] = np.array(load)
        columns[f"Fy{side}(N)"] = np.zeros(4)
        columns[f"Fx{side}(N)"] = np.zeros(4)
        columns[f"CoPx{side}(m)"] = np.array(x)
        columns[f"CoPy{side}(m)"] = np.array([1.02, 1., 1., 1.])
    columns["CoPxL(m)"][0] = np.nan  # exactly unloaded: zero moment
    columns["CoPyL(m)"][0] = np.nan
    columns["CoPxL(m)"][3] = np.nan  # loaded: cannot reconstruct
    columns.update({
        "GRFx lateral (N)": np.zeros(4), "GRFy fore-aft (N)": np.zeros(4),
        "GRFz vertical (N)": np.full(4, 800.),
        "CoPx lateral (m)": np.full(4, .4), "CoPy fore-aft (m)": np.full(4, 1.02),
        "Mx (N.m)": np.full(4, 816.), "My (N.m)": np.full(4, -320.),
        "Mz (N.m)": np.full(4, 5.), "Tz free moment (N.m)": np.full(4, 5.),
        "Grade (%)": np.zeros(4),
    })
    return GaitwayAsciiData(Path("test.txt"), {"Sample rate (Hz)": "1000"}, np.arange(4)/1000, columns)


def test_native_diagnostics_detect_overturning_inconsistency_and_preserve_missingness():
    g = fixture()
    original = {k: v.copy() for k, v in g.columns.items()}
    audit = audit_native_bilateral_wrench(g, time_range_s=(0, .003))
    assert audit["n_unsupported_total_native_samples"] == 1
    assert audit["n_missing_loaded_cop_native_samples"] == 1
    ss = audit["phases"]["single_support"]
    ds = audit["phases"]["double_support"]
    assert ss["represented_moment_minus_exported_total_Nm"]["x"]["max_abs"] == 0
    assert ds["force_sum_minus_exported_total_N"]["z"]["max_abs"] == 0
    assert ds["represented_moment_minus_exported_total_Nm"]["x"]["mean"] == -16
    assert ds["represented_moment_minus_exported_total_Nm"]["z"]["mean"] == -5
    assert ds["n_available_native_samples"] == 2
    assert ds["n_complete_cop_native_samples"] == 1
    assert audit["exported_total_free_moment_available"] is True
    assert audit["exported_total_free_moment_applied_to_feet"] is False
    for key in original:
        np.testing.assert_array_equal(original[key], g.columns[key])


def test_optional_moment_columns_roundtrip_and_do_not_change_grf(tmp_path):
    g = fixture()
    path = tmp_path / "trial.txt"
    pd.DataFrame({"Time (s)": g.time_s, **g.columns}).to_csv(path, sep="\t", index=False)
    parsed = read_gaitway_ascii(path)
    np.testing.assert_array_equal(parsed.columns["Tz free moment (N.m)"], [5]*4)
    legacy = GaitwayAsciiData(g.path, g.metadata, g.time_s, {
        k: v for k, v in g.columns.items() if k not in ["Mx (N.m)", "My (N.m)", "Mz (N.m)", "Tz free moment (N.m)"]})
    a, va, _ = build_bilateral_grf(g, g.time_s, 0, np.eye(3), cutoff_hz=None)
    b, vb, _ = build_bilateral_grf(legacy, g.time_s, 0, np.eye(3), cutoff_hz=None)
    np.testing.assert_array_equal(va, vb)
    for fa, fb in zip(a, b):
        for key in ["force", "point", "torque", "valid"]:
            np.testing.assert_array_equal(fa[key], fb[key])
    old = audit_native_bilateral_wrench(legacy, time_range_s=(0, .003))
    assert old["exported_total_moment_axes_available"] == []
    assert old["phases"]["double_support"]["represented_moment_minus_exported_total_Nm"]["x"]["rms"] is None


def test_diagnostic_threshold_does_not_mask_low_load_and_qc_uses_info():
    g = fixture()
    first = audit_native_bilateral_wrench(g, time_range_s=(.001, .001), diagnostic_contact_threshold_N=50)
    other = audit_native_bilateral_wrench(g, time_range_s=(.001, .001), diagnostic_contact_threshold_N=500)
    assert first["phases"]["all"] == other["phases"]["all"]
    force = {"n_valid_decomposed_frames": 4, "native_wrench_consistency": first,
             "backward_decomposition_unvalidated": True}
    checks = _force_checks(force, 4)
    assert all(c["status"] == "INFO" for c in checks if c["key"].startswith("native_"))
    backward = next(c for c in checks if c["key"] == "backward_force_decomposition")
    assert backward["status"] == "WARN"
    assert "x=16Nm" in backward["detail"]
    assert not any(c["status"] == "FAIL" for c in checks)


def test_exported_moment_reference_offset_does_not_redefine_cop_discrepancy():
    g = fixture()
    g.columns["Mx (N.m)"] += 100.0
    audit = audit_native_bilateral_wrench(g, time_range_s=(.001, .001))
    ds = audit["phases"]["double_support"]
    assert ds["exported_total_moment_minus_cop_cross_force_and_total_free_Nm"]["x"]["mean"] == 100
    assert ds["represented_moment_minus_exported_total_Nm"]["x"]["mean"] == -116
    assert ds["represented_moment_minus_total_cop_cross_force_Nm"]["x"]["mean"] == -16
    checks = _force_checks({"n_valid_decomposed_frames": 1,
                           "native_wrench_consistency": audit,
                           "backward_decomposition_unvalidated": True}, 1)
    warning = next(c for c in checks if c["key"] == "backward_force_decomposition")
    assert "x=16Nm" in warning["detail"]
    assert "116" not in warning["detail"]
