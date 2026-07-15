from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.apps.collector import CollectorWindow
from exo_collection.orchestration.models import TrialRunRequest


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
) -> tuple[QApplication, CollectorWindow, list[FakeCollectorWorker]]:
    app = QApplication.instance() or QApplication(["test-exo-collector"])
    created: list[FakeCollectorWorker] = []

    def factory(request: TrialRunRequest) -> FakeCollectorWorker:
        worker = FakeCollectorWorker(request)
        created.append(worker)
        return worker

    window = CollectorWindow(
        tmp_path,
        worker_factory=factory,
        poll_interval_ms=5,
    )
    return app, window, created


def test_collector_locks_condition_polls_events_and_finalizes(tmp_path: Path) -> None:
    app, window, created = _window_with_fake(tmp_path)
    window.project_name_edit.setText("Gait Study")
    window.subject_code_edit.setText("SUB-007")
    window.operator_edit.setText("Operator A")
    window.condition_combo.setCurrentIndex(1)
    window.repeat_spin.setValue(3)
    window.duration_spin.setValue(1.5)

    window.start_button.click()
    assert len(created) == 1
    worker = created[0]
    request = worker.request
    assert worker.started
    assert request.data_root == tmp_path.resolve()
    assert request.project_name == "Gait Study"
    assert request.subject_code == "SUB-007"
    assert request.operator == "Operator A"
    assert request.condition_code == "WALK_LEVEL"
    assert request.repeat_index == 3
    assert request.duration_s == 1.5
    assert window.configuration_locked
    assert not window.condition_combo.isEnabled()
    assert not window.repeat_spin.isEnabled()
    assert window.stop_button.isEnabled()
    assert window._poll_timer.isActive()

    worker.events.extend(
        [
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
                    "queue_depth": 2,
                    "queue_capacity": 64,
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
    _wait_until(app, lambda: window.state_label.text() == "Trial: RECORDING")

    imu_row = window._health_rows["imu"]
    assert window.health_table.item(imu_row, 1).text() == "DEGRADED"
    assert window.health_table.item(imu_row, 2).text() == "123"
    assert window.health_table.item(imu_row, 3).text() == "198.5 Hz"
    assert window.health_table.item(imu_row, 4).text() == "2/64"
    assert "preview delay" in window.alerts_edit.toPlainText()
    assert list(window.ultrasound_curve.getData()[1]) == [2.0, 4.0, 8.0, 4.0]
    assert list(window.imu_curve.getData()[1]) == [0.1, 0.2, 0.3]
    assert list(window.encoder_curve.getData()[1]) == [10.0, 11.0]

    window.stop_button.click()
    assert worker.stop_requests == 1
    assert window.state_label.text() == "Trial: STOPPING"
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

    assert window.state_label.text() == "Trial: FINALIZED"
    assert str(manifest_path) in window.manifest_label.text()
    assert not window.configuration_locked
    assert window.condition_combo.isEnabled()
    assert window.start_button.isEnabled()
    assert worker.closed
    assert worker.join_timeouts == [0]
    window.close()


def test_collector_shows_failed_worker_error_without_blocking_ui(tmp_path: Path) -> None:
    app, window, created = _window_with_fake(tmp_path)
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
    assert window.state_label.text() == "Trial: FAILED"
    assert "simulated disk full" in window.alerts_edit.toPlainText()
    assert window.start_button.isEnabled()
    assert worker.closed
    window.close()

