from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImageReader  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from exo_collection.apps.calculate.models import SessionFiles, SessionRecord  # noqa: E402
from exo_collection.apps.process.quality_report import (  # noqa: E402
    collect_day_report,
    group_subject_days,
    render_quality_report_png,
)

_APP = QApplication.instance() or QApplication([])


def _record(
    data_root: Path,
    *,
    subject: str = "103",
    day: str = "d1",
    condition: str = "WALK_1P0",
    session: str = "session1",
    static: bool = False,
) -> SessionRecord:
    session_dir = data_root / subject / day / "F_BASE" / condition / session
    session_dir.mkdir(parents=True, exist_ok=True)
    files = SessionFiles(
        c3d_path=session_dir / "trial.c3d",
        txt_path=None if static else session_dir / "trial.txt",
        mocap_h5_path=None if static else session_dir / "mocap.h5",
        imu_h5_path=None if static else session_dir / "imu.h5",
    )
    return SessionRecord(
        manifest_path=session_dir / ".exo" / "manifest.json",
        session_dir=session_dir,
        session_name=session,
        subject_code=subject,
        subject_uuid="subject-uuid",
        project_code="P",
        project_name="P",
        condition_code="STATIC_CALIB" if static else condition,
        condition_name="STATIC_CALIB" if static else condition,
        condition_level=None,
        repeat_index=1,
        trial_uuid="trial-uuid",
        session_uuid=f"uuid-{session}",
        state="FINALIZED",
        started_at_utc="2026-09-08T00:00:00Z",
        files=files,
    )


def _write_solved(record: SessionRecord, data_root: Path, *, static_subject: str = "103") -> None:
    run_dir = record.session_dir / "derived" / "opensim" / "run_test"
    run_dir.mkdir(parents=True)
    qc = {
        "status": "WARN",
        "checks": [
            {"key": "marker_rms_mean_cm", "status": "PASS", "value": 1.8},
            {"key": "marker_max_p95_cm", "status": "WARN", "value": 6.5},
            {"key": "residual_force_rms_N", "status": "PASS", "value": 100.0},
            {"key": "force_coverage", "status": "WARN", "value": 0.75},
            {"key": "hip_flexion_r_p95_Nm_per_kg", "status": "INFO", "value": 0.8},
            {"key": "hip_flexion_l_p95_Nm_per_kg", "status": "INFO", "value": 0.7},
            {"key": "sync_n_pairs", "status": "PASS", "value": 4},
            {"key": "sync_mad_s", "status": "PASS", "value": 0.004},
            {"key": "sync_imu_clock", "status": "WARN", "value": 999},
        ],
    }
    static_path = data_root / static_subject / "d1" / "T" / "STATIC_CALIB" / "static.c3d"
    manifest = {
        "subject": {"id": record.subject_code, "mass_kg": 80.0},
        "inputs": {"static_c3d": {"path": str(static_path)}},
    }
    (run_dir / "qc_report.json").write_text(json.dumps(qc), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    csv_path = record.session_dir / "ground_truth.csv"
    csv_path.write_text("time_s,hip_flexion_r\n0,0\n", encoding="utf-8")
    stat = csv_path.stat()
    csv_path.with_suffix(".qc.json").write_text(
        json.dumps(
            {
                "qc_status": "WARN",
                "run_dir": str(run_dir),
                "csv_size": stat.st_size,
                "csv_mtime_ns": stat.st_mtime_ns,
            }
        ),
        encoding="utf-8",
    )


def test_group_subject_days_separates_d_directories(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, day="d2", session="session2"),
        _record(tmp_path, day="d1", session="session1"),
        _record(tmp_path, subject="102", day="d1", session="other"),
    ]

    grouped = group_subject_days(records, tmp_path, "103")

    assert list(grouped) == ["d1", "d2"]
    assert grouped["d1"][0] == tmp_path / "103" / "d1"
    assert [item.session_name for item in grouped["d2"][1]] == ["session2"]


def test_collect_report_uses_qc_bound_to_exported_csv(tmp_path: Path) -> None:
    record = _record(tmp_path)
    _write_solved(record, tmp_path, static_subject="102")

    report = collect_day_report(
        [record], tmp_path, "103", "d1", tmp_path / "103" / "d1", recheck_sync=False
    )
    row = report.rows[0]

    assert row.qc.status == "WARN"
    assert row.marker_rms.text == "1.80 cm"
    assert row.residual.text.endswith("12.7%BW")
    assert row.static_source.text == "102 · 跨受试者"
    assert row.training.status == "FAIL"


def test_render_quality_report_writes_readable_png(tmp_path: Path) -> None:
    solved = _record(tmp_path, session="session1")
    static = _record(tmp_path, condition="STATIC_CALIB", session="static", static=True)
    _write_solved(solved, tmp_path)
    report = collect_day_report(
        [solved, static],
        tmp_path,
        "103",
        "d1",
        tmp_path / "103" / "d1",
        recheck_sync=False,
    )
    output = tmp_path / "report.png"

    render_quality_report_png(report, output)

    reader = QImageReader(str(output))
    assert reader.canRead()
    assert reader.size().width() == 2460
    assert reader.size().height() >= 1200


def test_process_window_dispatches_report_for_selected_subject(
    tmp_path: Path, monkeypatch
) -> None:
    from exo_collection.apps.process import window as window_module

    record = _record(tmp_path)
    monkeypatch.setattr(window_module, "discover_sessions", lambda _: [record])
    monkeypatch.setattr(window_module.ProcessWindow, "_resolve_opensim_env", lambda _: None)
    window = window_module.ProcessWindow(tmp_path, settings=None)
    started = []

    class Pool:
        def start(self, worker) -> None:
            started.append(worker)

    try:
        window._thread_pool = Pool()
        window._on_report_clicked()
        assert len(started) == 1
        assert started[0]._subject == "103"
        assert not window._report_button.isEnabled()
        assert "数据质量报告" in window._log.toPlainText()
        assert "未分日" in window._log.toPlainText()
    finally:
        window.close()
