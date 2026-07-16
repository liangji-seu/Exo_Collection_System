from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QTableWidget

from exo_collection.apps.data_studio.management import (
    AnnexArtifactSummary,
    AnnexScanResult,
    AnnexValidationStatus,
    ExternalAnnexSummary,
    InventoryExportResult,
    ManagementRefreshResult,
    ManagementSummaryResult,
    build_management_index,
    export_manifest_inventory_checked,
    load_management_summary,
)
from exo_collection.apps.data_studio.management_dialog import ManagementSummaryDialog
from exo_collection.apps.data_studio.process_workers import DataStudioProcessWorker
from exo_collection.apps.data_studio.service import load_catalog_snapshot
from exo_collection.apps.data_studio.window import DataStudioWindow
from exo_collection.domain.models import ArtifactKind, Condition, QualityGrade
from exo_collection.domain.states import TrialState
from exo_collection.storage.activity import AcquisitionLock
from exo_collection.storage.checksum import sha256_file, write_checksum_manifest
from exo_collection.storage.manifest import (
    ConfigurationSnapshot,
    ManifestArtifact,
    QualitySummary,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
    save_manifest,
)


UTC = timezone.utc
BASE_TIME = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)


def _publish_trial(
    root: Path,
    *,
    project_uuid: UUID,
    subject_uuid: UUID,
    session_uuid: UUID,
    condition_code: str,
    condition_name: str,
    repeat_index: int,
    quality: QualityGrade,
    started_at: datetime,
) -> Path:
    trial_uuid = uuid4()
    trial_root = (
        root
        / "F"
        / str(subject_uuid)
        / str(session_uuid)
        / "trials"
        / str(trial_uuid)
    )
    artifact_path = trial_root / "raw" / "imu.h5"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(f"{condition_code}-{repeat_index}".encode())
    stopped_at = started_at + timedelta(seconds=3)
    finalized_at = stopped_at + timedelta(milliseconds=100)
    manifest = TrialManifest(
        project_uuid=project_uuid,
        project_code="F",
        project_name="正式",
        subject_uuid=subject_uuid,
        subject_code="001",
        session_uuid=session_uuid,
        trial_uuid=trial_uuid,
        state=TrialState.FINALIZED,
        condition=Condition(
            condition_code=condition_code,
            condition_name=condition_name,
            condition_level=1,
            parameters={},
            repeat_index=repeat_index,
            protocol_version="1.0.0",
            selected_at_utc=started_at,
        ),
        timing=TrialTiming(
            started_at_utc=started_at,
            stopped_at_utc=stopped_at,
            finalized_at_utc=finalized_at,
            start_host_monotonic_ns=1_000_000_000,
            stop_host_monotonic_ns=4_000_000_000,
            finalize_host_monotonic_ns=4_100_000_000,
        ),
        software=SoftwareProvenance(
            application="Exo Collector",
            application_version="0.1.0",
            core_version="0.1.0",
            git_commit="management-ui-test",
        ),
        configuration=ConfigurationSnapshot(
            config_version="1.0.0",
            protocol_version="1.0.0",
            condition_definition_version="1.0.0",
            content_sha256="e" * 64,
        ),
        artifacts=[
            ManifestArtifact(
                artifact_uuid=uuid4(),
                trial_uuid=trial_uuid,
                modality="imu",
                kind=ArtifactKind.RAW,
                media_type="application/x-hdf5",
                relative_path="raw/imu.h5",
                size_bytes=artifact_path.stat().st_size,
                sha256=sha256_file(artifact_path),
                created_at_utc=started_at,
                finalized_at_utc=finalized_at,
            )
        ],
        quality=QualitySummary(
            computed_grade=quality,
            required_artifacts_complete=True,
            integrity_checks_passed=True,
            algorithm_version="quality-test",
            assessed_at_utc=finalized_at,
        ),
    )
    manifest_path = save_manifest(trial_root / "manifest.json", manifest)
    write_checksum_manifest(trial_root, ("raw/imu.h5", "manifest.json"))
    return manifest_path


def _dataset(root: Path) -> dict[str, Path]:
    project_uuid = uuid4()
    subject_uuid = uuid4()
    session_uuid = uuid4()
    return {
        "stand": _publish_trial(
            root,
            project_uuid=project_uuid,
            subject_uuid=subject_uuid,
            session_uuid=session_uuid,
            condition_code="STAND",
            condition_name="静止站立",
            repeat_index=1,
            quality=QualityGrade.A,
            started_at=BASE_TIME,
        ),
        "walk_1": _publish_trial(
            root,
            project_uuid=project_uuid,
            subject_uuid=subject_uuid,
            session_uuid=session_uuid,
            condition_code="WALK_LEVEL",
            condition_name="平地行走",
            repeat_index=1,
            quality=QualityGrade.B,
            started_at=BASE_TIME + timedelta(days=1),
        ),
        "walk_2": _publish_trial(
            root,
            project_uuid=project_uuid,
            subject_uuid=subject_uuid,
            session_uuid=session_uuid,
            condition_code="WALK_LEVEL",
            condition_name="平地行走",
            repeat_index=2,
            quality=QualityGrade.A,
            started_at=BASE_TIME + timedelta(days=2),
        ),
    }


def _annex_summary(
    root: Path,
    manifest_path: Path,
    *,
    valid: bool,
) -> ExternalAnnexSummary:
    trial_uuid = manifest_path.parent.name
    annex_uuid = str(uuid4())
    status = (
        AnnexValidationStatus.VERIFIED if valid else AnnexValidationStatus.INVALID
    )
    return ExternalAnnexSummary(
        annex_directory=root / "external_annexes" / trial_uuid / annex_uuid,
        annex_manifest_path=(
            root / "external_annexes" / trial_uuid / annex_uuid / "annex_manifest.json"
        ),
        validation_status=status,
        annex_uuid=annex_uuid,
        trial_uuid=trial_uuid,
        modality="force_plate",
        modality_label="force_plate",
        source_system="test",
        imported_at_utc=BASE_TIME,
        mapping_quality="GOOD" if valid else "POOR",
        mapping_offset_only=False,
        mapping_anchor_count=3,
        file_count=1,
        total_bytes=4096,
        files=(
            AnnexArtifactSummary(
                artifact_uuid=str(uuid4()),
                role="external_original",
                relative_path="artifacts/force.csv",
                media_type="text/csv",
                size_bytes=4096,
                sha256="a" * 64,
            ),
        ),
        errors=() if valid else ("SHA-256 mismatch",),
    )


class _ImmediateWorker:
    def __init__(self, result: object) -> None:
        self.result = result
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
        if not self.started or self.returned:
            return None
        self.returned = True
        return "completed", self.result

    def join(self, _timeout: float | None = None) -> int:
        return 0

    def terminate(self, timeout: float = 3.0) -> None:
        del timeout

    def close(self) -> None:
        self.closed = True


def _wait_until(app: QApplication, predicate: Any, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()


def _trial_items(window: DataStudioWindow) -> list[Any]:
    found: list[Any] = []

    def visit(item: Any) -> None:
        if item.data(1, Qt.ItemDataRole.UserRole) == "trial":
            found.append(item)
        for index in range(item.childCount()):
            visit(item.child(index))

    for index in range(window.tree_widget.topLevelItemCount()):
        visit(window.tree_widget.topLevelItem(index))
    return found


def _window_with_management(
    root: Path,
    paths: dict[str, Path],
    app: QApplication,
) -> DataStudioWindow:
    snapshot = load_catalog_snapshot(root)
    index = build_management_index(snapshot)
    annex_scan = AnnexScanResult(
        root,
        (
            _annex_summary(root, paths["walk_1"], valid=True),
            _annex_summary(root, paths["stand"], valid=False),
        ),
    )
    refresh_result = ManagementRefreshResult(index=index, annex_scan=annex_scan)

    def worker_factory(operation: str, **arguments: object) -> _ImmediateWorker:
        if operation == "catalog_refresh":
            return _ImmediateWorker(snapshot)
        if operation == "management_refresh":
            return _ImmediateWorker(refresh_result)
        if operation == "management_summary":
            return _ImmediateWorker(load_management_summary(arguments["data_root"]))
        if operation == "management_export":
            return _ImmediateWorker(export_manifest_inventory_checked(**arguments))
        raise AssertionError(operation)

    window = DataStudioWindow(
        root,
        autostart_refresh=False,
        process_worker_factory=worker_factory,
    )
    management_completions: list[bool] = []
    catalog_completions: list[bool] = []
    window.management_refresh_finished.connect(management_completions.append)
    window.refresh_finished.connect(catalog_completions.append)
    window.refresh_catalog()
    # refresh_catalog starts both the management process and the Catalog
    # QRunnable.  Tests must not return a window while either background owner
    # is still active, otherwise the next test can collect the wrapper while a
    # native Qt thread is executing.
    _wait_until(
        app,
        lambda: management_completions == [True]
        and catalog_completions == [True],
    )
    return window


def test_management_filters_preserve_hierarchy_and_show_annex_integrity(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication(["management-filter-ui"])
    paths = _dataset(tmp_path)
    window = _window_with_management(tmp_path, paths, app)

    assert len(_trial_items(window)) == 3
    assert window.project_filter.count() == 2
    assert window.subject_filter.count() == 2
    assert window.session_filter.count() == 2
    walk_index = window.condition_filter.findData("WALK_LEVEL")
    window.condition_filter.setCurrentIndex(walk_index)
    assert len(_trial_items(window)) == 2
    assert window.tree_widget.topLevelItemCount() == 1
    assert window.tree_widget.topLevelItem(0).childCount() == 1
    assert window.tree_widget.topLevelItem(0).child(0).childCount() == 1

    annex_items = []
    for trial_item in _trial_items(window):
        for child_index in range(trial_item.childCount()):
            child = trial_item.child(child_index)
            if child.data(1, Qt.ItemDataRole.UserRole) == "external_annex":
                annex_items.append(child)
    assert len(annex_items) == 1
    assert "完整性 VERIFIED" in annex_items[0].text(2)
    assert "映射 GOOD" in annex_items[0].text(2)
    assert annex_items[0].child(0).data(1, Qt.ItemDataRole.UserRole) == "external_artifact"

    window.clear_management_filters()
    stand_index = window.condition_filter.findData("STAND")
    window.condition_filter.setCurrentIndex(stand_index)
    invalid_annex = next(
        child
        for trial in _trial_items(window)
        for child in (trial.child(index) for index in range(trial.childCount()))
        if child.data(1, Qt.ItemDataRole.UserRole) == "external_annex"
    )
    assert "完整性 INVALID" in invalid_annex.text(2)
    assert "SHA-256 mismatch" in invalid_annex.toolTip(2)

    window.clear_management_filters()
    window.start_date_enabled.setChecked(True)
    window.start_date_edit.setDate(QDate(2026, 7, 16))
    assert len(_trial_items(window)) == 2
    window.quality_filter.setCurrentIndex(window.quality_filter.findData("B"))
    assert len(_trial_items(window)) == 1
    selected_uuid = _trial_items(window)[0].data(0, Qt.ItemDataRole.UserRole)
    window.clear_management_filters()
    window.text_filter.setText(str(selected_uuid))
    assert len(_trial_items(window)) == 1

    with AcquisitionLock(tmp_path, uuid4()):
        window._poll_activity()
        assert not window.management_summary_action.isEnabled()
        assert not window.export_inventory_action.isEnabled()
        assert all(not widget.isEnabled() for widget in window._filter_inputs)
    window._poll_activity()
    window.close()
    app.processEvents()


def test_management_summary_and_filtered_export_are_wired_to_workers(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    app = QApplication.instance() or QApplication(["management-summary-ui"])
    paths = _dataset(tmp_path)
    window = _window_with_management(tmp_path, paths, app)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: QMessageBox.StandardButton.Ok,
    )

    window.management_summary_action.trigger()
    _wait_until(
        app,
        lambda: window.findChild(ManagementSummaryDialog, "management_summary_dialog")
        is not None,
    )
    dialog = window.findChild(ManagementSummaryDialog, "management_summary_dialog")
    assert dialog is not None
    assert isinstance(dialog.result, ManagementSummaryResult)
    assert "FINALIZED：3" in dialog.state_summary_label.text()
    coverage = dialog.findChild(QTableWidget, "management_coverage_table")
    assert coverage is not None and coverage.rowCount() == 2
    walk_rows = [
        row
        for row in range(coverage.rowCount())
        if "WALK_LEVEL" in coverage.item(row, 3).text()
    ]
    assert len(walk_rows) == 1
    assert coverage.item(walk_rows[0], 8).text() == "1, 2"

    window.condition_filter.setCurrentIndex(
        window.condition_filter.findData("WALK_LEVEL")
    )
    destination = tmp_path / "exports" / "walk_inventory"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(destination), ""),
    )
    window.export_inventory_action.trigger()
    _wait_until(app, lambda: destination.with_suffix(".json").is_file())
    payload = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["record_count"] == 2
    assert {item["condition_code"] for item in payload["records"]} == {"WALK_LEVEL"}
    assert destination.with_suffix(".csv").is_file()

    window.close()
    app.processEvents()


def test_management_refresh_operation_runs_through_spawn_worker(tmp_path: Path) -> None:
    _dataset(tmp_path)
    snapshot = load_catalog_snapshot(tmp_path)
    worker = DataStudioProcessWorker("management_refresh", snapshot=snapshot)
    worker.start()
    response: tuple[str, object] | None = None
    deadline = time.monotonic() + 20.0
    while response is None and time.monotonic() < deadline:
        response = worker.poll_result()
        time.sleep(0.02)
    assert response is not None
    status, result = response
    assert status == "completed"
    assert isinstance(result, ManagementRefreshResult)
    assert len(result.index.records) == 3
    worker.join(5.0)
    assert worker.exitcode == 0
    worker.close()

    summary_worker = DataStudioProcessWorker(
        "management_summary",
        data_root=str(tmp_path),
    )
    summary_worker.start()
    summary_response: tuple[str, object] | None = None
    deadline = time.monotonic() + 20.0
    while summary_response is None and time.monotonic() < deadline:
        summary_response = summary_worker.poll_result()
        time.sleep(0.02)
    assert summary_response is not None
    assert summary_response[0] == "completed"
    assert isinstance(summary_response[1], ManagementSummaryResult)
    summary_worker.join(5.0)
    summary_worker.close()

    destination = tmp_path / "exports" / "spawned_inventory"
    export_worker = DataStudioProcessWorker(
        "management_export",
        data_root=str(tmp_path),
        records=result.index.records,
        destination_stem=str(destination),
        overwrite=False,
    )
    export_worker.start()
    export_response: tuple[str, object] | None = None
    deadline = time.monotonic() + 20.0
    while export_response is None and time.monotonic() < deadline:
        export_response = export_worker.poll_result()
        time.sleep(0.02)
    assert export_response is not None
    assert export_response[0] == "completed"
    assert isinstance(export_response[1], InventoryExportResult)
    export_worker.join(5.0)
    export_worker.close()
    assert destination.with_suffix(".csv").is_file()
    assert destination.with_suffix(".json").is_file()
