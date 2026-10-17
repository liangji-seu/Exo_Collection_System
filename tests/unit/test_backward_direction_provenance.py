import json
from dataclasses import replace
from pathlib import Path

from exo_collection.apps.calculate.direction_policy import current_direction_policy, backward_export_is_current
from exo_collection.apps.calculate.ground_truth_status import write_ground_truth_qc
from exo_collection.apps.calculate.history import list_history_runs
from exo_collection.apps.calculate.models import SessionRecord, SessionFiles
from exo_collection.apps.process.batch import session_solve_status, SessionSolveState


def record(tmp_path):
    return SessionRecord(manifest_path=tmp_path / "manifest.json", session_dir=tmp_path,
                         session_name="session1", subject_code="501", subject_uuid="501", project_code="F",
                         project_name="walk", condition_code="BWALK_0P4_NOEXO", condition_name="backward",
                         condition_level=None, repeat_index=1, trial_uuid="trial", session_uuid="session",
                         state="complete", started_at_utc="", files=SessionFiles(c3d_path=tmp_path / "trial.c3d",
                         txt_path=tmp_path / "trial.txt", mocap_h5_path=tmp_path / "mocap.h5", imu_h5_path=tmp_path / "imu.h5"))


def manifest():
    from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path
    ensure_pipeline_on_path()
    from pipeline.opensim_io.walking_direction import DIRECTION_POLICY_VERSION
    return {"processing": {"direction_policy_version": DIRECTION_POLICY_VERSION, "grf_input_policy_version": "continuous_native_wrench_v2", "condition_code": "BWALK_0P4_NOEXO"},
            "gaitway": {"anatomical_mapping": {"native_to_anatomical": {"R": "left", "L": "right"}}}}


def test_old_backward_csv_requires_recalculation_without_deletion(tmp_path):
    csv = tmp_path / "ground_truth.csv"; csv.write_text("preserve old result")
    assert session_solve_status(record(tmp_path)) == SessionSolveState.UNSOLVED
    assert csv.read_text() == "preserve old result"
    assert session_solve_status(replace(record(tmp_path), condition_code="DWALK_0P6_NOEXO")) == SessionSolveState.UNSOLVED


def test_backward_current_export_cannot_display_qc_pass(tmp_path):
    run = tmp_path / "run_20261014_120000"; run.mkdir()
    (run / "manifest.json").write_text(json.dumps(manifest()))
    csv = tmp_path / "ground_truth.csv"; csv.write_text("csv")
    write_ground_truth_qc(csv, "PASS", run, validity={"export_validity_version": "bilateral_analysis_v1"})
    assert backward_export_is_current(tmp_path)
    assert session_solve_status(record(tmp_path)) == SessionSolveState.QC_WARN
    csv.write_text("changed CSV")
    assert not backward_export_is_current(tmp_path)


def test_old_history_marked_stale_by_condition_even_when_sources_unchanged(tmp_path):
    run = tmp_path / "run_20261014_120000"; run.mkdir()
    doc = manifest(); del doc["processing"]["direction_policy_version"]
    (run / "manifest.json").write_text(json.dumps(doc))
    (run / "result.json").write_text(json.dumps({"qc": {"status": "PASS"}}))
    result = list_history_runs(tmp_path)[0]
    assert result.stale
    assert "旧算法" in result.stale_reason
    assert not current_direction_policy(doc)


def test_current_policy_requires_mapping_receipt():
    doc = manifest()
    assert current_direction_policy(doc)
    del doc["gaitway"]["anatomical_mapping"]
    assert not current_direction_policy(doc)
