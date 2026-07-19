from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from exo_collection.apps.data_studio import DataStudioWindow, load_catalog_snapshot
from exo_collection.apps.data_studio.external_import_dialog import ExternalImportDialog
from exo_collection.apps.data_studio.external_import_worker import ExternalImportWorker
from exo_collection.apps.data_studio.local_dialogs import QualityAuditDialog
from exo_collection.apps.data_studio.local_tools import load_quality_audit
from exo_collection.apps.data_studio.quality_reviews import list_quality_reviews
from exo_collection.apps.data_studio.recovery_dialog import RecoveryDialog
from exo_collection.apps.data_studio import window as window_module
from exo_collection.domain.models import ArtifactKind, Condition, QualityGrade
from exo_collection.domain.states import TrialState
from exo_collection.external import (
    ExternalImportRequest,
    ExternalImportResult,
    ExternalModality,
)
from exo_collection.storage.activity import AcquisitionActivity, AcquisitionLock
from exo_collection.storage.manifest import (
    ConfigurationSnapshot,
    ManifestArtifact,
    QualitySummary,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
    save_manifest,
)


def _make_manifest(*, condition_code: str = "WALK_LEVEL") -> TrialManifest:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    trial_uuid = uuid4()
    return TrialManifest(
        project_uuid=uuid4(),
        project_code="T",
        project_name="Test",
        subject_uuid=uuid4(),
        subject_code="001",
        session_uuid=uuid4(),
        trial_uuid=trial_uuid,
        state=TrialState.FINALIZED,
        condition=Condition(
            condition_code=condition_code,
            condition_name="Level walking",
            condition_level=2,
            parameters={"speed_mps": 0.8},
            repeat_index=1,
            protocol_version="1.0.0",
            selected_at_utc=now,
        ),
        timing=TrialTiming(
            started_at_utc=now,
            stopped_at_utc=now + timedelta(seconds=4),
            finalized_at_utc=now + timedelta(seconds=5),
            start_host_monotonic_ns=1_000,
            stop_host_monotonic_ns=4_000_001_000,
            finalize_host_monotonic_ns=5_000_001_000,
        ),
        software=SoftwareProvenance(
            application="Exo Collector",
            application_version="0.1.0",
            core_version="0.1.0",
            git_commit="test-commit",
        ),
        configuration=ConfigurationSnapshot(
            config_version="1.0.0",
            protocol_version="1.0.0",
            condition_definition_version="1.0.0",
            content_sha256="b" * 64,
        ),
        artifacts=[
            ManifestArtifact(
                artifact_uuid=uuid4(),
                trial_uuid=trial_uuid,
                modality="imu",
                kind=ArtifactKind.RAW,
                media_type="application/x-hdf5",
                relative_path="raw/imu.h5",
                size_bytes=1234,
                sha256="a" * 64,
                created_at_utc=now,
                finalized_at_utc=now + timedelta(seconds=5),
            )
        ],
        quality=QualitySummary(
            computed_grade=QualityGrade.A,
            required_artifacts_complete=True,
            integrity_checks_passed=True,
            algorithm_version="quality-0.1.0",
            assessed_at_utc=now + timedelta(seconds=5),
        ),
    )


def _publish_manifest(root: Path, manifest: TrialManifest) -> Path:
    trial_dir = (
        root
        / str(manifest.project_uuid)
        / str(manifest.subject_uuid)
        / str(manifest.session_uuid)
        / "trials"
        / str(manifest.trial_uuid)
    )
    return save_manifest(trial_dir / "manifest.json", manifest)


def _wait_until(app: QApplication, predicate: object, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:  # type: ignore[operator]
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()  # type: ignore[operator]


def test_snapshot_scans_only_published_manifests_and_never_artifacts(
    tmp_path: Path, monkeypatch: object
) -> None:
    manifest = _make_manifest()
    path = _publish_manifest(tmp_path, manifest)
    partial_payload = path.parent / "raw" / "ultrasound.bin.partial"
    partial_payload.parent.mkdir(parents=True)
    partial_payload.write_bytes(b"must not be opened")

    recording_manifest = _make_manifest(condition_code="SHOULD_NOT_BE_INDEXED")
    recording_path = (
        path.parent.parent
        / f"{recording_manifest.trial_uuid}.recording"
        / "manifest.json"
    )
    recording_path.parent.mkdir(parents=True)
    recording_path.write_text(
        recording_manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    partial_manifest = _make_manifest(condition_code="PARTIAL_NOT_INDEXED")
    partial_manifest_path = (
        path.parent.parent
        / f"{partial_manifest.trial_uuid}.partial"
        / "manifest.json"
    )
    partial_manifest_path.parent.mkdir(parents=True)
    partial_manifest_path.write_text(
        partial_manifest.model_dump_json(indent=2), encoding="utf-8"
    )

    original_open = Path.open

    def guarded_open(candidate: Path, *args: object, **kwargs: object) -> object:
        if candidate.suffix == ".partial" or any(
            part.endswith(".recording") or part.endswith(".partial")
            for part in candidate.parts
        ):
            raise AssertionError(f"Data Studio opened active data: {candidate}")
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)  # type: ignore[attr-defined]
    snapshot = load_catalog_snapshot(tmp_path)

    assert snapshot.scan_report.indexed == 1
    assert snapshot.statistics == {
        "trial_count": 1,
        "finalized_count": 1,
        "total_duration_s": 4.0,
        "by_condition": {
            "WALK_LEVEL": {"trial_count": 1, "duration_s": 4.0}
        },
    }
    assert snapshot.tree[0]["children"][0]["children"][0]["children"][0][
        "children"
    ][0]["label"] == "raw/imu.h5"


def test_trial_artifacts_are_grouped_by_modality_only_for_display() -> None:
    artifacts = [
        {
            "type": "artifact",
            "uuid": "us-data",
            "label": "raw/ultrasound.bin",
            "modality": "ultrasound",
            "size_bytes": 100,
            "children": [],
        },
        {
            "type": "artifact",
            "uuid": "us-index",
            "label": "raw/ultrasound.idx",
            "modality": "ultrasound",
            "size_bytes": 20,
            "children": [],
        },
        {
            "type": "artifact",
            "uuid": "imu-data",
            "label": "raw/imu.h5",
            "modality": "imu",
            "size_bytes": 80,
            "children": [],
        },
        {
            "type": "artifact",
            "uuid": "quality",
            "label": ".exo/quality_report.json",
            "modality": "trial",
            "size_bytes": 10,
            "children": [],
        },
    ]

    grouped = DataStudioWindow._group_trial_artifacts(artifacts)

    assert [(node["type"], node["label"]) for node in grouped] == [
        ("modality", "IMU"),
        ("modality", "超声"),
        ("supporting_files", "系统资料"),
    ]
    assert grouped[1]["artifact_count"] == 2
    assert grouped[1]["size_bytes"] == 120
    assert [item["uuid"] for item in grouped[1]["children"]] == [
        "us-data",
        "us-index",
    ]


def test_window_refreshes_in_background_and_enforces_lightweight_mode(
    tmp_path: Path,
) -> None:
    manifest = _make_manifest()
    _publish_manifest(tmp_path, manifest)
    # Index once before acquisition. While the activity lock exists, Data
    # Studio must browse this Catalog snapshot without walking all Manifests.
    assert load_catalog_snapshot(tmp_path).statistics["trial_count"] == 1
    app = QApplication.instance() or QApplication(["test-data-studio"])

    with AcquisitionLock(tmp_path, manifest.trial_uuid):
        lightweight_snapshot = load_catalog_snapshot(tmp_path)
        assert lightweight_snapshot.scan_report.indexed == 0
        assert lightweight_snapshot.statistics["trial_count"] == 1
        window = DataStudioWindow(tmp_path, autostart_refresh=False)
        completions: list[bool] = []
        window.refresh_finished.connect(completions.append)
        window.refresh_catalog()

        assert window.refresh_in_progress
        assert window.isEnabled()  # refresh must not disable/block the whole UI
        _wait_until(app, lambda: bool(completions))

        assert completions == [True]
        assert window.tree_widget.topLevelItemCount() == 1
        assert window.statistics["trial_count"] == 1
        assert window.condition_table.item(0, 0).text() == "WALK_LEVEL"
        assert window.lightweight_mode
        assert "轻量模式" in window.activity_banner.text()
        assert all(not action.isEnabled() for action in window._restricted_actions)

    _wait_until(app, lambda: not window.lightweight_mode, timeout_s=2.0)
    assert all(action.isEnabled() for action in window._restricted_actions)

    completions.clear()
    window.refresh_catalog()
    _wait_until(app, lambda: bool(completions))
    assert completions == [True]
    assert not window.lightweight_mode
    window.close()


def test_unreadable_activity_lock_banner_never_claims_pid_zero(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication(["test-unreadable-lock-banner"])
    window = DataStudioWindow(tmp_path, autostart_refresh=False)
    sentinel = AcquisitionActivity(
        pid=0,
        hostname="unreadable-lock",
        trial_uuid=None,
        heartbeat_monotonic_ns=time.perf_counter_ns(),
        heartbeat_utc_ns=time.time_ns(),
        owner_token="unreadable-lock",
    )

    window._apply_activity(sentinel)

    assert window.lightweight_mode
    assert "活动锁不可安全解析" in window.activity_banner.text()
    assert "PID 0" not in window.activity_banner.text()
    window.close()
    app.processEvents()


def test_open_recovery_dialog_becomes_read_only_when_collector_appears(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication(["test-recovery-activity-revoke"])
    dialog = RecoveryDialog(tmp_path)
    activity = AcquisitionActivity(
        pid=12345,
        hostname="collector-host",
        trial_uuid=str(uuid4()),
        heartbeat_monotonic_ns=time.perf_counter_ns(),
        heartbeat_utc_ns=time.time_ns(),
        owner_token="collector-owner",
    )

    dialog.set_acquisition_activity(activity)

    assert not dialog.rescan_button.isEnabled()
    assert not dialog.repair_button.isEnabled()
    assert not dialog.finalize_button.isEnabled()
    assert not dialog.abort_button.isEnabled()
    assert not dialog.table.isEnabled()
    assert "只读禁用" in dialog.status_label.text()
    dialog.set_acquisition_activity(None)
    assert dialog.rescan_button.isEnabled()
    dialog.close()
    app.processEvents()


def test_data_studio_close_waits_nonblocking_for_running_qrunnable(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication(["test-thread-shutdown"])
    window = DataStudioWindow(tmp_path, autostart_refresh=False)
    started = threading.Event()
    release = threading.Event()

    def slow_operation() -> str:
        started.set()
        release.wait(timeout=5.0)
        return "done"

    window._start_local_tool("slow-test", slow_operation, lambda _result: None)
    _wait_until(app, started.is_set)
    began = time.monotonic()
    closed = window.close()

    assert not closed
    assert time.monotonic() - began < 0.5
    assert window._closing
    assert window._thread_pool.activeThreadCount() == 1
    # Even after an arbitrary elapsed deadline, destroying the window while a
    # QRunnable still owns Python/SQLAlchemy/Qt objects is unsafe.
    window._close_started_at = time.monotonic() - 30.0
    assert not window.close()
    assert window._thread_pool.activeThreadCount() == 1
    release.set()
    _wait_until(app, lambda: window._thread_pool.activeThreadCount() == 0)
    window.close()
    app.processEvents()


def test_data_studio_close_terminates_and_closes_process_workers(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication(["test-process-shutdown"])
    window = DataStudioWindow(tmp_path, autostart_refresh=False)

    class HangingWorker:
        def __init__(self) -> None:
            self.started = False
            self.alive = False
            self.closed = False
            self.exitcode: int | None = None

        @property
        def is_alive(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.started = True
            self.alive = True

        def poll_result(self) -> None:
            return None

        def terminate(self, timeout: float = 3.0) -> None:
            del timeout
            self.alive = False
            self.exitcode = -15

        def join(self, timeout: float | None = None) -> int | None:
            del timeout
            return self.exitcode

        def close(self) -> None:
            assert not self.alive
            self.closed = True

    worker = HangingWorker()
    window._register_process_worker(
        "hung-test",
        worker,
        lambda _result: None,
    )
    assert worker.started and worker.alive

    window.close()

    assert not worker.alive
    assert worker.closed
    assert not window._process_tasks
    app.processEvents()


def test_external_import_dialog_builds_manual_and_csv_requests(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication(["test-external-import-dialog"])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    source = tmp_path / "force.vendor"
    source.write_bytes(b"opaque")
    pulse_csv = tmp_path / "pulses.csv"
    pulse_csv.write_text("trigger_time;value\n0;1\n", encoding="utf-8")

    dialog = ExternalImportDialog(manifest_path)
    dialog.source_edit.setText(str(source))
    dialog.manual_pulses_edit.setPlainText("0.0, 1.5\n3.0")
    manual = dialog.take_request(tmp_path)
    assert manual.modality is ExternalModality.FORCE_PLATE
    assert manual.external_pulse_times == [0.0, 1.5, 3.0]
    assert manual.pulse_csv_path is None

    dialog.modality_combo.setCurrentIndex(2)
    dialog.other_modality_edit.setText("pressure insole")
    dialog.pulse_mode_combo.setCurrentIndex(1)
    dialog.pulse_csv_edit.setText(str(pulse_csv))
    dialog.pulse_column_edit.setText("trigger_time")
    dialog.delimiter_edit.setText(";")
    csv_request = dialog.take_request(tmp_path)
    assert csv_request.modality is ExternalModality.OTHER
    assert csv_request.other_modality_label == "pressure insole"
    assert csv_request.external_pulse_times is None
    assert csv_request.pulse_csv_path == pulse_csv
    assert csv_request.pulse_csv_column == "trigger_time"
    assert csv_request.csv_delimiter == ";"
    dialog.close()
    app.processEvents()


def test_external_import_worker_uses_spawn_process(tmp_path: Path) -> None:
    request = ExternalImportRequest(
        dataset_root=tmp_path / "missing-dataset",
        trial_manifest_path=tmp_path / "missing-manifest.json",
        source_path=tmp_path / "missing-force.bin",
        modality=ExternalModality.FORCE_PLATE,
        external_pulse_times=[0.0],
    )
    worker = ExternalImportWorker(request)
    worker.start()
    try:
        result: tuple[str, object] | None = None
        deadline = time.monotonic() + 20.0
        while result is None and time.monotonic() < deadline:
            result = worker.poll_result()
            time.sleep(0.02)
        assert result is not None
        status, details = result
        assert status == "failed"
        assert "ExternalImportError" in str(details)
        worker.join(5.0)
        assert worker.exitcode == 0
        assert not worker.is_alive
    finally:
        if worker.is_alive:
            worker.terminate(timeout=3.0)
        worker.join(0)
        worker.close()


def test_quality_dialog_appends_review_without_rewriting_manifest(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication(["test-quality-review-wiring"])
    manifest = _make_manifest()
    manifest_path = _publish_manifest(tmp_path, manifest)
    manifest_before = manifest_path.read_bytes()
    audit = load_quality_audit(manifest_path, data_root=tmp_path)
    window = DataStudioWindow(tmp_path, autostart_refresh=False)

    window._show_quality_audit(audit)
    dialog = window._result_dialogs[-1]
    assert isinstance(dialog, QualityAuditDialog)
    assert dialog.save_review_button.isEnabled()
    assert dialog._review_submit is not None
    updated = dialog._review_submit("B", "reviewer-001", "同步与动作均人工复核通过")

    assert updated.reviewed_grade == "B"
    assert updated.reviewed_by == "reviewer-001"
    assert updated.review_count == 1
    assert manifest_path.read_bytes() == manifest_before
    assert len(list_quality_reviews(tmp_path, manifest_path)) == 1
    window.close()
    app.processEvents()


def test_window_routes_recovery_and_external_import_to_real_integrations(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    app = QApplication.instance() or QApplication(["test-data-studio-integrations"])
    manifest = _make_manifest()
    manifest_path = _publish_manifest(tmp_path, manifest)
    source = tmp_path / "force.bin"
    source.write_bytes(b"external")
    request = ExternalImportRequest(
        dataset_root=tmp_path,
        trial_manifest_path=manifest_path,
        source_path=source,
        modality=ExternalModality.FORCE_PLATE,
        external_pulse_times=[0.0],
    )
    result = ExternalImportResult(
        annex_uuid=uuid4(),
        trial_uuid=manifest.trial_uuid,
        annex_directory=tmp_path / "external_annexes" / str(manifest.trial_uuid) / "annex",
        annex_manifest_path=tmp_path / "annex_manifest.json",
        mapping_path=tmp_path / "mapping.json",
        copied_artifact_path=tmp_path / "copied.bin",
        base_manifest_sha256="a" * 64,
        copied_artifact_sha256="b" * 64,
        quality="UNAVAILABLE",
        offset_only=True,
        anchor_count=1,
    )

    class FakeExternalDialog:
        def __init__(self, selected: Path, _parent: object) -> None:
            assert selected == manifest_path

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

        def take_request(self, _root: Path) -> ExternalImportRequest:
            return request

        def deleteLater(self) -> None:
            return None

    class FakeExternalWorker:
        def __init__(self, selected: ExternalImportRequest) -> None:
            assert selected == request
            self.started = False
            self.returned = False
            self.closed = False

        @property
        def is_alive(self) -> bool:
            return False

        @property
        def exitcode(self) -> int | None:
            return 0 if self.started else None

        def start(self) -> None:
            self.started = True

        def poll_result(self) -> tuple[str, object] | None:
            if self.returned:
                return None
            self.returned = True
            return "completed", result

        def join(self, _timeout: float | None = None) -> int:
            return 0

        def terminate(self, timeout: float = 3.0) -> None:
            del timeout

        def close(self) -> None:
            self.closed = True

    class FakeRecoveryDialog(QDialog):
        def __init__(self, selected_root: Path, parent: object) -> None:
            super().__init__(parent)  # type: ignore[arg-type]
            assert selected_root == tmp_path

    monkeypatch.setattr(window_module, "ExternalImportDialog", FakeExternalDialog)  # type: ignore[attr-defined]
    monkeypatch.setattr(window_module, "RecoveryDialog", FakeRecoveryDialog)  # type: ignore[attr-defined]
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)  # type: ignore[attr-defined]

    workers: list[FakeExternalWorker] = []

    def worker_factory(selected: ExternalImportRequest) -> FakeExternalWorker:
        worker = FakeExternalWorker(selected)
        workers.append(worker)
        return worker

    window = DataStudioWindow(
        tmp_path,
        autostart_refresh=False,
        external_import_worker_factory=worker_factory,  # type: ignore[arg-type]
    )
    item = window._make_tree_item(
        {
            "type": "trial",
            "label": str(manifest.trial_uuid),
            "uuid": str(manifest.trial_uuid),
            "manifest_path": str(manifest_path),
            "state": "FINALIZED",
            "children": [],
        }
    )
    window.tree_widget.addTopLevelItem(item)
    window.tree_widget.setCurrentItem(item)
    completions: list[tuple[str, bool]] = []
    window.local_tool_finished.connect(lambda name, ok: completions.append((name, ok)))

    window.external_import_action.trigger()
    assert len(workers) == 1 and workers[0].started
    window._poll_process_tools()
    assert completions == [("外部模态导入", True)]
    assert workers[0].closed
    assert not window._process_tasks

    window.recovery_action.trigger()
    assert any(isinstance(dialog, FakeRecoveryDialog) for dialog in window._result_dialogs)
    window.close()
    app.processEvents()
