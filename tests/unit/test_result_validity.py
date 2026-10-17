"""Validity regressions: native measured zeros, analysis gaps, finite CSV and sync reuse."""
import csv
import hashlib
import json
from pathlib import Path
from dataclasses import replace

import h5py
import numpy as np
import pytest

from exo_collection.apps.calculate.ground_truth import align_ground_truth, write_ground_truth_csv, export_validity_audit
from exo_collection.apps.calculate.viewer import load_viewer_data
from exo_collection.apps.process.batch import load_confirmed_sync
from test_viewer import _write_synthetic_viewer
from test_process import _make_session
from exo_collection.apps.calculate.models import SessionFiles


def _current_viewer(path):
    _write_synthetic_viewer(path, n=6)
    meta_path = path / "viewer_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["processing"] = {"grf_input_policy_version": "continuous_native_wrench_v2"}
    meta_path.write_text(json.dumps(meta))
    support = np.ones((6, 2), dtype=bool)
    support[3, 0] = False
    analysis = np.ones((6, 2), dtype=bool)
    analysis[[0, 5], :] = False
    np.save(path / "support_mask.npy", support)
    np.save(path / "analysis_mask.npy", analysis)
    np.save(path / "grf.npy", np.zeros((6, 2, 3)))
    return load_viewer_data(path)


def test_native_measured_zero_is_valid_but_missing_side_breaks_all_moments(tmp_path):
    data = _current_viewer(tmp_path)
    assert data.export_mask.tolist() == [False, True, True, False, True, False]
    assert np.isfinite(data.moments[[1, 2, 4]]).all()
    assert np.isnan(data.moments[[0, 3, 5]]).all()
    assert np.isnan(data.grf[3, 0]).all()
    assert (data.grf[1] == 0).all(), "A measured swing zero is a valid wrench"


def test_legacy_viewer_retains_skeleton_but_cannot_export(tmp_path):
    _write_synthetic_viewer(tmp_path)
    data = load_viewer_data(tmp_path)
    assert not data.export_mask.any()
    assert np.isnan(data.moments).all()
    assert np.isfinite(data.body_origins).all()


def test_csv_keeps_native_times_and_gaps_and_receipt_exclusions(tmp_path):
    data = _current_viewer(tmp_path / "viewer")
    times, imu, moments = align_ground_truth(data.time_s, data.moments, data.time_s,
                                           np.ones((6, 12)), validity_mask=data.export_mask)
    np.testing.assert_array_equal(times, data.time_s[[1, 2, 4]])
    output = write_ground_truth_csv(tmp_path / "ground_truth.csv", times, imu, moments,
                                   moment_names=data.moment_names)
    rows = list(csv.reader(output.open()))
    assert len(rows) == 4
    assert not any("nan" in value.lower() for row in rows for value in row)
    audit = export_validity_audit(data, times)
    assert audit["excluded_invalid_or_unreviewed"] == 3
    assert audit["coverage_fraction"] == .5
    assert len(audit["exported_intervals_s"]) == 2


def test_csv_refuses_nan_before_opening_destination(tmp_path):
    destination = tmp_path / "old.csv"
    destination.write_text("preserve")
    with pytest.raises(ValueError, match="NaN"):
        write_ground_truth_csv(destination, np.array([0.]), np.ones((1, 12)), np.array([[np.nan]]))
    assert destination.read_text() == "preserve"


def test_imu_gap_is_not_filled_with_fabricated_training_input():
    imu_times = np.array([0., .1, .2, .5, .6])
    times, _, _ = align_ground_truth(np.arange(7) / 10, np.ones((7, 2)), imu_times, np.ones((5, 12)))
    assert .3 not in times and .4 not in times


def _calibration(tmp_path):
    files = {}
    for key, name in (("c3d", "trial.c3d"), ("gaitway_txt", "trial.txt"),
                      ("mocap_h5", "mocap.h5"), ("imu_h5", "imu.h5")):
        path = tmp_path / name
        if key == "mocap_h5":
            with h5py.File(path, "w") as h:
                h["samples/host_monotonic_ns"] = np.arange(10, dtype=np.int64)
        else:
            path.write_bytes(b"source")
        files[key] = path
    record = _make_session(tmp_path, files=SessionFiles(c3d_path=files["c3d"], txt_path=files["gaitway_txt"],
                          mocap_h5_path=files["mocap_h5"], imu_h5_path=files["imu_h5"]))
    doc = {"dynamic_session_uuid": record.session_uuid, "trial_uuid": record.trial_uuid,
           "method": "MANUAL_PAIRED", "confirmed_at_utc": None,
           "result": {"manual": True, "n_pairs": 3, "confidence": "MEDIUM", "mad_s": .01,
                      "gaitway_offset_s": 6.125, "c3d_start_in_mocap_h5_frame": 2,
                      "c3d_h5_match_rms_mm": .1},
           "inputs": {key: {"path": str(path), "size_bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for key, path in files.items()}}
    sidecar = record.session_dir / "derived" / "opensim" / "sync_calibration.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps(doc))
    return record, sidecar, doc


def test_manual_applied_calibration_reuses_exact_offset_without_confirm_timestamp(tmp_path):
    record, _, _ = _calibration(tmp_path)
    result = load_confirmed_sync(record)
    assert result["gaitway_offset_s"] == 6.125
    assert result["confirmation_basis"] == "manual_applied"
    assert result["source_sha256_verified"]


@pytest.mark.parametrize("mutation", ["draft", "source", "uuid", "frame", "mad"])
def test_sync_reuse_rejects_unconfirmed_or_changed_sources(tmp_path, mutation):
    record, path, doc = _calibration(tmp_path)
    if mutation == "draft":
        doc["method"] = "AUTO_HIGH"
        doc["result"]["manual"] = False
    elif mutation == "source":
        record.files.txt_path.write_bytes(b"changed")
    elif mutation == "uuid":
        doc["trial_uuid"] = "other trial"
    elif mutation == "frame":
        doc["result"]["c3d_start_in_mocap_h5_frame"] = 10
    else:
        doc["result"]["mad_s"] = float("nan")
    path.write_text(json.dumps(doc))
    assert load_confirmed_sync(record) is None


def test_qc_fail_preserves_prior_csv_before_export(tmp_path, monkeypatch):
    import exo_collection.apps.process.batch as batch
    from test_process import _complete_files, _static_session, _high_sync
    import pipeline.opensim_io.prep_session as prep
    import pipeline.synchronization.sync as sync
    record = _make_session(tmp_path, files=_complete_files())
    csv = record.session_dir / "ground_truth.csv"
    csv.write_text("old export")
    monkeypatch.setattr(sync, "run_auto_sync", lambda *a, **k: _high_sync())
    monkeypatch.setattr(prep, "prepare_session", lambda **k: {"manifest_path": str(k["out_dir"] / "manifest.json")})
    monkeypatch.setattr(batch, "_run_opensim", lambda *a, **k: {"qc": {"status": "FAIL", "summary": "bad residual"}})
    monkeypatch.setattr(batch, "_export_ground_truth", lambda *a: pytest.fail("FAIL run reached CSV export"))
    with pytest.raises(batch.QCFailed, match="bad residual") as failure:
        batch.solve_one_session(record, _static_session(tmp_path), Path("python"), Path("model"),
                                75, 1.75, cancel_check=lambda: False, progress=lambda _: None,
                                analysis_time_range_s=(0., 5.))
    assert failure.value.run_dir.is_relative_to(record.session_dir)
    assert csv.read_text() == "old export"


def test_csv_write_failure_leaves_previous_export_and_no_staging_file(tmp_path, monkeypatch):
    import exo_collection.apps.calculate.ground_truth as gt
    destination = tmp_path / "ground_truth.csv"
    destination.write_text("old export")
    class BrokenWriter:
        def writerow(self, row):
            raise OSError("disk unavailable")
    monkeypatch.setattr(gt.csv, "writer", lambda handle: BrokenWriter())
    with pytest.raises(OSError, match="disk unavailable"):
        gt.write_ground_truth_csv(destination, np.array([0.]), np.ones((1, 12)), np.ones((1, 2)))
    assert destination.read_text() == "old export"
    assert list(tmp_path.glob("*.tmp")) == []
