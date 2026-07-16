from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QApplication

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.apps.collector import CollectorWindow
from exo_collection.apps.collector.window import ExperimentMetadataDialog
from exo_collection.configuration import SharedAppSettings
from exo_collection.orchestration.models import (
    MeasuredConditionMetadata,
    TrialExperimentMetadata,
    TrialRunRequest,
)


class FakeCollectorWorker:
    def __init__(self, request: TrialRunRequest) -> None:
        self.request = request
        self.events: list[WorkerEvent] = []
        self.started = False
        self.alive = False
        self.stop_requests = 0
        self.join_timeouts: list[float | None] = []
        self.closed = False
        self._exitcode: int | None = None

    @property
    def is_alive(self) -> bool:
        return self.alive

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    def start(self) -> None:
        self.started = True
        self.alive = True

    def request_stop(self) -> None:
        self.stop_requests += 1

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]:
        result = self.events[:limit]
        del self.events[:limit]
        return result

    def join(self, timeout: float | None = None) -> int | None:
        self.join_timeouts.append(timeout)
        return self._exitcode

    def close(self) -> None:
        assert not self.alive
        self.closed = True

    def terminate_for_recovery(self, timeout: float = 5.0) -> int:
        del timeout
        self.alive = False
        self._exitcode = -15
        return self._exitcode

    def finish(self, exitcode: int = 0) -> None:
        self._exitcode = exitcode
        self.alive = False


def _wait_until(
    app: QApplication, predicate: Callable[[], bool], timeout_s: float = 3.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()


def _window_with_fake(
    tmp_path: Path,
    *,
    preflight: Callable[[], dict[str, str]] | None = None,
) -> tuple[QApplication, CollectorWindow, list[FakeCollectorWorker]]:
    app = QApplication.instance() or QApplication(["test-exo-collector"])
    created: list[FakeCollectorWorker] = []

    def factory(request: TrialRunRequest) -> FakeCollectorWorker:
        worker = FakeCollectorWorker(request)
        created.append(worker)
        return worker

    class ImmediatePreflightWorker:
        def __init__(self, result: dict[str, str]) -> None:
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

        def join(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def terminate(self, timeout: float = 1.0) -> int:
            del timeout
            return 0

        def close(self) -> None:
            self.closed = True

    def preflight_factory(_root: Path) -> ImmediatePreflightWorker:
        result = (
            preflight()
            if preflight is not None
            else {
                "ultrasound": "READY",
                "imu": "READY",
                "encoder": "READY",
                "sync_pulse": "READY",
            }
        )
        return ImmediatePreflightWorker(result)

    window = CollectorWindow(
        tmp_path,
        settings=SharedAppSettings(
            QSettings(
                str(tmp_path / "ui-settings.ini"),
                QSettings.Format.IniFormat,
            )
        ),
        worker_factory=factory,
        preflight_worker_factory=preflight_factory,
        poll_interval_ms=5,
        controlled_stop_timeout_s=0.05,
    )
    return app, window, created


def test_collector_locks_condition_polls_events_and_finalizes(tmp_path: Path) -> None:
    app, window, created = _window_with_fake(tmp_path)
    assert [window.project_combo.itemText(index) for index in range(2)] == [
        "F — 正式",
        "T — 测试",
    ]
    assert window.project_combo.currentData()["project_code"] == "T"
    assert window.subject_code_edit.text() == "001"
    validation, _text, _position = window.subject_code_edit.validator().validate("A01", 0)
    assert validation is QValidator.State.Invalid
    assert not hasattr(window, "operator_edit")
    assert not hasattr(window, "duration_spin")
    assert window.overall_status == "未连接"
    assert "模拟设备" in window.device_profile_label.text()
    assert not window.start_button.isEnabled()

    window.subject_code_edit.setText("7")
    window.normalize_subject_code()
    assert window.subject_code_edit.text() == "007"
    window.condition_combo.setCurrentIndex(1)
    window.repeat_spin.setValue(3)
    window.preflight_button.click()

    assert window.preflight_ready
    assert window.overall_status == "可采集"
    assert window.start_button.isEnabled()
    for row in range(window.health_table.rowCount()):
        assert window.health_table.item(row, 1).text() == "READY"

    window.start_button.click()
    assert len(created) == 1
    worker = created[0]
    request = worker.request
    assert worker.started
    assert request.data_root == tmp_path.resolve()
    assert request.project_name == "测试"
    assert request.project_code == "T"
    assert request.subject_code == "007"
    assert request.operator == "not_recorded"
    assert request.condition_code == "WALK_LEVEL"
    assert request.repeat_index == 3
    assert request.duration_s is None
    assert window.configuration_locked
    assert not window.project_combo.isEnabled()
    assert not window.condition_combo.isEnabled()
    assert not window.repeat_spin.isEnabled()
    assert not window.preflight_button.isEnabled()
    assert window.stop_button.isEnabled()
    assert window._poll_timer.isActive()

    worker.events.extend(
        [
            WorkerEvent(
                event_type=WorkerEventType.STATE,
                payload={"state": "WAITING_SYNC"},
            ),
            WorkerEvent(
                event_type=WorkerEventType.SYNC,
                payload={
                    "status": "WAITING_SYNC",
                    "quality": "WAITING",
                    "trigger_count": 0,
                    "first_trigger_host_monotonic_ns": None,
                    "trigger_time_utc": None,
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.SYNC,
                payload={
                    "status": "TRIGGERED",
                    "quality": "PASS",
                    "trigger_count": 1,
                    "first_trigger_host_monotonic_ns": 123_456,
                    "trigger_time_utc": "2026-07-15T10:00:00Z",
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.STATE,
                payload={"state": "RECORDING"},
            ),
            WorkerEvent(
                event_type=WorkerEventType.HEALTH,
                modality="imu",
                payload={
                    "device_id": "imu_sim",
                    "status": "DEGRADED",
                    "actual_sample_rate_hz": 198.5,
                    "dropped_packets": 3,
                    "queue_depth": 2,
                    "queue_capacity": 64,
                    "sampled_at_utc": "2026-07-15T10:00:01Z",
                    "sample_count": 90,
                },
                message="preview delay",
            ),
            WorkerEvent(
                event_type=WorkerEventType.METRIC,
                payload={
                    "modality_counts": {
                        "ultrasound": 8,
                        "imu": 123,
                        "encoder": 60,
                        "sync_pulse": 500,
                    },
                    "pulse_event_count": 1,
                    "status": "TRIGGERED",
                    "quality": "PASS",
                    "trigger_count": 1,
                    "first_trigger_host_monotonic_ns": 123_456,
                    "trigger_time_utc": "2026-07-15T10:00:00Z",
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.PREVIEW,
                modality="ultrasound",
                payload={
                    "host_monotonic_ns": 1_000,
                    "values": [2, 4, 8, 4],
                    "shape": [4],
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.PREVIEW,
                modality="ultrasound",
                payload={
                    "host_monotonic_ns": 2_000,
                    "values": [3, 6, 9, 6],
                    "shape": [4],
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.PREVIEW,
                modality="imu",
                payload={
                    "host_monotonic_ns": 2_000,
                    "x": [0.0, 0.1, 0.2],
                    "values": [0.1, 0.2, 0.3],
                    "channel": "accel_norm",
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.PREVIEW,
                modality="encoder",
                payload={
                    "host_monotonic_ns": 3_000,
                    "x": [0.0, 0.1],
                    "values": [10.0, 11.0],
                    "channel": "angle_deg",
                },
            ),
        ]
    )
    _wait_until(app, lambda: window.overall_status == "采集中")

    imu_row = window._health_rows["imu"]
    assert window.health_table.item(imu_row, 1).text() == "DEGRADED"
    assert window.health_table.item(imu_row, 2).text() == "123"
    assert window.health_table.item(imu_row, 3).text() == "198.5 Hz"
    assert window.health_table.item(imu_row, 4).text() == "3"
    assert window.health_table.item(imu_row, 5).text() == "2/64"
    assert window.health_table.item(imu_row, 6).text() == "2026-07-15T10:00:01Z"
    assert "preview delay" in window.alerts_edit.toPlainText()
    assert list(window.ultrasound_curve.getData()[1]) == [3.0, 6.0, 9.0, 6.0]
    assert window.ultrasound_waterfall_image.image.shape == (2, 4)
    assert list(window.imu_curve.getData()[1]) == [0.1, 0.2, 0.3]
    assert list(window.encoder_curve.getData()[1]) == [10.0, 11.0]
    assert window.sync_status_label.text() == "已同步"
    assert window.sync_quality_label.text() == "PASS"
    assert window.trigger_count_label.text() == "1"
    assert "123456" in window.first_trigger_label.text()
    assert len(window.timeline_curve.getData()[0]) >= 4

    window.stop_button.click()
    assert worker.stop_requests == 1
    assert window.overall_status == "保存中"
    assert not window.stop_button.isEnabled()

    manifest_path = tmp_path / "trial" / "manifest.json"
    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.COMPLETED,
            message="Trial package finalized",
            payload={"state": "FINALIZED", "manifest_path": str(manifest_path)},
        )
    )
    worker.finish(0)
    _wait_until(app, lambda: window.worker is None)

    assert window.overall_status == "未连接"
    assert str(manifest_path) in window.manifest_label.text()
    assert not window.configuration_locked
    assert window.condition_combo.isEnabled()
    assert not window.start_button.isEnabled()
    assert worker.closed
    assert worker.join_timeouts == [0]
    window.close()


def test_collector_shows_failed_worker_error_without_blocking_ui(tmp_path: Path) -> None:
    app, window, created = _window_with_fake(tmp_path)
    window.run_preflight()
    window.start_trial()
    worker = created[0]
    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.FAILED,
            message="OSError: simulated disk full",
            payload={"traceback": "not rendered into the status bar"},
        )
    )
    worker.finish(1)

    _wait_until(app, lambda: window.worker is None)
    assert window.overall_status == "失败"
    assert "simulated disk full" in window.alerts_edit.toPlainText()
    assert not window.start_button.isEnabled()
    assert worker.closed
    window.close()


def test_collector_rejects_terminal_event_from_another_trial(tmp_path: Path) -> None:
    app, window, created = _window_with_fake(tmp_path)
    window.run_preflight()
    window.start_trial()
    worker = created[0]
    wrong_uuid = uuid4()
    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.COMPLETED,
            trial_uuid=str(wrong_uuid),
            payload={
                "trial_uuid": str(wrong_uuid),
                "state": "FINALIZED",
                "manifest_path": str(tmp_path / "wrong" / "manifest.json"),
            },
        )
    )
    worker.finish(0)

    _wait_until(app, lambda: window.worker is None)
    assert window.overall_status == "失败"
    assert "已拒绝不属于当前 Trial" in window.alerts_edit.toPlainText()
    assert "未发布 COMPLETED/FAILED" in window.alerts_edit.toPlainText()
    window.close()


def test_collector_forces_hung_controlled_stop_and_preserves_recovery_semantics(
    tmp_path: Path,
) -> None:
    _app, window, created = _window_with_fake(tmp_path)
    window.run_preflight()
    window.start_trial()
    worker = created[0]
    window.request_controlled_stop()
    assert window._stop_requested_at is not None
    window._stop_requested_at = time.monotonic() - 1.0

    window.poll_worker_events()

    assert window.worker is None
    assert worker.exitcode == -15
    assert worker.closed
    assert window.overall_status == "失败"
    alerts = window.alerts_edit.toPlainText()
    assert "受控停止等待超时" in alerts
    assert ".recording" in alerts
    assert "FINALIZED" in alerts
    window.close()


def test_preflight_gates_start_and_reports_missing_critical_device(tmp_path: Path) -> None:
    _app, window, created = _window_with_fake(
        tmp_path,
        preflight=lambda: {
            "ultrasound": "READY",
            "imu": "READY",
            "encoder": "READY",
            "sync_pulse": "MISSING",
        },
    )

    window.start_trial()
    assert not created
    assert "请先完成设备预检" in window.alerts_edit.toPlainText()

    window.preflight_button.click()
    assert not window.preflight_ready
    assert not window.start_button.isEnabled()
    assert window.overall_status == "失败"
    assert "sync_pulse=MISSING" in window.alerts_edit.toPlainText()
    window.close()


def test_missing_sync_trigger_is_prominent_and_never_looks_recording(
    tmp_path: Path,
) -> None:
    app, window, created = _window_with_fake(tmp_path)
    window.run_preflight()
    window.start_trial()
    worker = created[0]
    worker.events.extend(
        [
            WorkerEvent(
                event_type=WorkerEventType.STATE,
                payload={"state": "WAITING_SYNC"},
            ),
            WorkerEvent(
                event_type=WorkerEventType.SYNC,
                payload={
                    "status": "WAITING_SYNC",
                    "quality": "WAITING",
                    "trigger_count": 0,
                    "first_trigger_host_monotonic_ns": None,
                    "trigger_time_utc": None,
                },
            ),
        ]
    )
    _wait_until(app, lambda: window.sync_status_label.text() == "等待同步触发")
    assert window.overall_status == "等待同步"

    worker.events.extend(
        [
            WorkerEvent(
                event_type=WorkerEventType.SYNC,
                payload={
                    "status": "MISSING_TRIGGER",
                    "quality": "FAIL",
                    "trigger_count": 0,
                    "first_trigger_host_monotonic_ns": None,
                    "trigger_time_utc": None,
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.FAILED,
                message="sync trigger missing before controlled stop",
            ),
        ]
    )
    worker.finish(1)

    _wait_until(app, lambda: window.worker is None)
    assert window.overall_status == "失败"
    assert window.sync_status_label.text() == "缺少同步触发"
    assert window.sync_quality_label.text() == "FAIL"
    assert window.trigger_count_label.text() == "0"
    alerts = window.alerts_edit.toPlainText()
    assert alerts.count("未检测到合格同步触发") == 1
    assert "background:#f8d7da" in window.sync_status_label.styleSheet()
    assert "MISSING_TRIGGER" in window.timeline_last_event_label.text() or "FAILED" in (
        window.timeline_last_event_label.text()
    )
    window.close()


def test_four_channel_a_mode_preview_switches_without_mixing_waterfall(
    tmp_path: Path,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    first_channels = [
        [1.0, 4.0, 2.0],
        [2.0, 8.0, 3.0],
        [3.0, 12.0, 4.0],
        [4.0, 16.0, 5.0],
    ]
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="ultrasound",
            payload={
                "host_monotonic_ns": 1_000_000_000,
                "values": first_channels[0],
                "channels": first_channels,
                "channel_count": 4,
                "geometry": "a_line",
                "format_metrics": [
                    {
                        "dtype": "uint16",
                        "zero_fraction": 0.0,
                        "nonfinite_fraction": 0.0,
                        "full_scale_fraction": 0.0,
                        "full_scale_value": 65535,
                        "all_zero": False,
                    }
                    for _channel in first_channels
                ],
            },
        )
    )

    assert window.ultrasound_channel_combo.count() == 4
    assert list(window.ultrasound_curve.getData()[1]) == first_channels[0]
    assert window.ultrasound_waterfall_image.image.shape == (1, 3)
    assert "峰值深度索引 1" in window.ultrasound_peak_label.text()
    assert "零值 0.00%" in window.ultrasound_peak_label.text()
    assert "UNASSESSED" in window.ultrasound_peak_label.text()

    window.ultrasound_channel_combo.setCurrentIndex(2)
    assert list(window.ultrasound_curve.getData()[1]) == first_channels[2]
    assert window.ultrasound_waterfall_image.image.shape == (1, 3)
    assert len(window._ultrasound_trend_x) == 1

    second_channels = [[value + 1 for value in channel] for channel in first_channels]
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="ultrasound",
            payload={
                "host_monotonic_ns": 1_100_000_000,
                "values": second_channels[0],
                "channels": second_channels,
                "channel_count": 4,
                "geometry": "a_line",
            },
        )
    )
    assert list(window.ultrasound_curve.getData()[1]) == second_channels[2]
    assert window.ultrasound_waterfall_image.image.shape == (2, 3)
    assert len(window.ultrasound_peak_depth_curve.getData()[1]) == 2
    assert len(window.ultrasound_peak_strength_curve.getData()[1]) == 2
    window.close()


def test_all_zero_ultrasound_format_alert_is_debounced(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    event = WorkerEvent(
        event_type=WorkerEventType.PREVIEW,
        modality="ultrasound",
        payload={
            "host_monotonic_ns": 1_000_000_000,
            "values": [0.0, 0.0],
            "channels": [[0.0, 0.0]],
            "channel_count": 1,
            "geometry": "a_line",
            "format_metrics": [
                {
                    "dtype": "uint16",
                    "zero_fraction": 1.0,
                    "nonfinite_fraction": 0.0,
                    "full_scale_fraction": 0.0,
                    "full_scale_value": 65535,
                    "all_zero": True,
                }
            ],
        },
    )

    window._handle_worker_event(event)
    window._handle_worker_event(event)

    assert window.alerts_edit.toPlainText().count("当前帧全零") == 1
    assert "零值 100.00%" in window.ultrasound_peak_label.text()
    window.close()


def test_experiment_metadata_dialog_allows_blank_and_builds_structured_values() -> None:
    app = QApplication.instance() or QApplication(["test-experiment-metadata"])
    dialog = ExperimentMetadataDialog(TrialExperimentMetadata())

    assert dialog.build_metadata() == TrialExperimentMetadata()
    dialog.height_edit.setText("173.5")
    dialog.weight_edit.setText("66.2")
    dialog.age_edit.setText("25")
    dialog.muscle_edit.setText(" gastrocnemius medialis ")
    dialog.laterality_combo.setCurrentIndex(
        dialog.laterality_combo.findData("left")
    )
    dialog.channel_mapping_edits[0].setText("GM proximal")
    dialog.speed_edit.setText("0.8")
    dialog.slope_edit.setText("-5")
    dialog.notes_edit.setPlainText("  Stable signal.  ")

    metadata = dialog.build_metadata()
    assert metadata.subject.height_cm == 173.5
    assert metadata.subject.age_years == 25
    assert metadata.ultrasound_probe.muscle == "gastrocnemius medialis"
    assert metadata.ultrasound_probe.channel_mapping == (
        "GM proximal",
        None,
        None,
        None,
    )
    assert metadata.measured_condition.treadmill_speed_mps == 0.8
    assert metadata.measured_condition.slope_deg == -5
    assert metadata.trial_notes == "Stable signal."
    dialog.close()
    app.processEvents()


def test_collector_passes_optional_experiment_metadata_and_locks_editor(
    tmp_path: Path,
) -> None:
    app, window, created = _window_with_fake(tmp_path)
    assert "未填写" in window.experiment_metadata_summary.text()
    window.set_experiment_metadata(
        {
            "subject": {"height_cm": 171, "leg_length_cm": 91},
            "ultrasound_probe": {
                "muscle": "vastus lateralis",
                "laterality": "right",
                "longitudinal_position": "middle",
                "channel_mapping": ["VL-1", "VL-2", "VL-3", "VL-4"],
                "probe_reapplied": False,
            },
            "measured_condition": {"treadmill_speed_mps": 0.7},
            "trial_notes": "Baseline trial",
        }
    )
    assert "已填写" in window.experiment_metadata_summary.text()

    window.run_preflight()
    window.start_trial()
    worker = created[0]
    metadata = worker.request.experiment_metadata
    assert metadata.subject.height_cm == 171
    assert metadata.ultrasound_probe.channel_mapping[3] == "VL-4"
    assert metadata.ultrasound_probe.probe_reapplied is False
    assert metadata.measured_condition.treadmill_speed_mps == 0.7
    assert metadata.trial_notes == "Baseline trial"
    assert not window.experiment_metadata_button.isEnabled()

    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.COMPLETED,
            payload={
                "state": "FINALIZED",
                "manifest_path": str(tmp_path / "manifest.json"),
            },
        )
    )
    worker.finish(0)
    _wait_until(app, lambda: window.worker is None)
    assert window.experiment_metadata_button.isEnabled()
    assert window.experiment_metadata.subject == metadata.subject
    assert window.experiment_metadata.ultrasound_probe.muscle == "vastus lateralis"
    assert window.experiment_metadata.ultrasound_probe.probe_reapplied is None
    assert window.experiment_metadata.measured_condition == metadata.measured_condition
    assert window.experiment_metadata.trial_notes is None
    assert "一次性备注" in window.experiment_metadata_summary.text()
    window.close()


def test_experiment_metadata_is_scoped_by_project_and_subject(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    window.set_experiment_metadata(
        {
            "subject": {"height_cm": 171},
            "ultrasound_probe": {
                "muscle": "vastus lateralis",
                "probe_reapplied": True,
            },
            "measured_condition": {"treadmill_speed_mps": 0.8},
            "trial_notes": "T/001 only",
        }
    )

    window.subject_code_edit.setText("002")
    assert window.experiment_metadata == TrialExperimentMetadata()
    assert "已清空以避免串写" in window.experiment_metadata_summary.text()
    window.set_experiment_metadata({"subject": {"height_cm": 182}})

    window.subject_code_edit.setText("001")
    assert window.experiment_metadata.subject.height_cm == 171
    assert window.experiment_metadata.trial_notes == "T/001 only"
    assert "已恢复" in window.experiment_metadata_summary.text()

    window.project_combo.setCurrentIndex(0)
    assert window.project_combo.currentData()["project_code"] == "F"
    assert window.experiment_metadata == TrialExperimentMetadata()
    assert "已清空以避免串写" in window.experiment_metadata_summary.text()
    window.project_combo.setCurrentIndex(1)
    assert window.experiment_metadata.subject.height_cm == 171
    assert window.experiment_metadata.ultrasound_probe.muscle == "vastus lateralis"
    window.close()


def test_condition_switch_clears_only_condition_and_trial_scoped_metadata(
    tmp_path: Path,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    window.set_experiment_metadata(
        {
            "subject": {"height_cm": 171},
            "ultrasound_probe": {
                "muscle": "vastus lateralis",
                "fixation_method": "elastic wrap",
            },
            "measured_condition": {
                "treadmill_speed_mps": 0.8,
                "assist_level": 3,
            },
            "trial_notes": "condition-specific note",
        }
    )
    original_index = window.condition_combo.currentIndex()
    target_index = 0 if original_index != 0 else 1

    window.condition_combo.setCurrentIndex(target_index)

    metadata = window.experiment_metadata
    assert metadata.subject.height_cm == 171
    assert metadata.ultrasound_probe.muscle == "vastus lateralis"
    assert metadata.ultrasound_probe.fixation_method == "elastic wrap"
    assert metadata.measured_condition == MeasuredConditionMetadata()
    assert metadata.trial_notes is None
    assert "实测工况与 Trial 备注已清空" in window.experiment_metadata_summary.text()
    window.subject_code_edit.setText("002")
    window.subject_code_edit.setText("001")
    assert window.experiment_metadata.measured_condition == MeasuredConditionMetadata()
    assert window.experiment_metadata.trial_notes is None
    window.close()
