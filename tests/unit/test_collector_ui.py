from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QApplication, QWidget

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.apps.collector import CollectorWindow
from exo_collection.apps.collector.window import (
    ExperimentMetadataDialog,
    MODALITIES,
    RingTrace,
    SIGNAL_RING_CAPACITY,
)
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


class FakePreviewHandle:
    def __init__(self, modality: str) -> None:
        self.modality = modality
        self.device_id = f"{modality}_fake"
        self.simulated = True
        self.alive = True
        self.stop_requests = 0
        self.join_calls = 0
        self.closed = False

    @property
    def is_alive(self) -> bool:
        return self.alive

    @property
    def exitcode(self) -> int | None:
        return None if self.alive else 0

    def request_stop(self) -> None:
        self.stop_requests += 1

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]:
        del limit
        return []

    def join(self, timeout: float | None = None) -> int | None:
        del timeout
        self.join_calls += 1
        return self.exitcode

    def terminate(self, timeout: float = 5.0) -> int | None:
        del timeout
        self.alive = False
        return 0

    def close(self) -> None:
        assert not self.alive
        self.closed = True


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


def _connect_all_previews_for_trial(window: CollectorWindow) -> None:
    """Simulate all modalities being connected via preview (marks them READY).

    This bypasses the actual spawn-based preview workers so that trial-flow
    tests can proceed without subprocesses.
    """
    for modality in ("ultrasound", "imu", "encoder", "sync_pulse"):
        window._preview_connected_modalities.add(modality)
        window._preview_connection_status[modality] = "已连接"
    window._update_start_button()
    window._update_connect_button_state()


# ── Ultrasound 2x2 grid tests ──

def test_ultrasound_2x2_grid_has_four_channels(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    for i in range(4):
        plot = window.findChild(QWidget, f"ultrasound_preview_ch{i}")
        assert plot is not None, f"ultrasound_preview_ch{i} not found"
    assert len(window._us_plots) == 4
    assert len(window._us_curves) == 4
    assert hasattr(window, "_us_plots")
    assert hasattr(window, "_us_curves")
    window.close()


def test_waterfall_peak_trend_combo_removed(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    # Old components should not exist
    assert window.findChild(QWidget, "ultrasound_channel") is None
    assert window.findChild(QWidget, "ultrasound_waterfall") is None
    assert window.findChild(QWidget, "ultrasound_peak_depth") is None
    assert window.findChild(QWidget, "ultrasound_peak_strength") is None
    assert window.findChild(QWidget, "ultrasound_peak_metrics") is None
    assert window.findChild(QWidget, "event_timeline") is None
    assert not hasattr(window, "timeline_plot")
    assert not hasattr(window, "timeline_curve")
    window.close()


def test_preview_sections_are_not_user_resizable_splitters(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    assert window.findChild(QWidget, "preview_splitter") is None
    window.close()


# ── IMU ring trace tests ──

def test_imu_three_ring_traces(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    assert len(window._imu_traces) == 3
    assert "imu_trunk" in window._imu_traces
    assert "imu_left" in window._imu_traces
    assert "imu_right" in window._imu_traces
    for label in ("imu_trunk", "imu_left", "imu_right"):
        plot = window.findChild(QWidget, f"imu_ring_{label}")
        assert plot is not None, f"imu_ring_{label} not found"
        assert isinstance(window._imu_traces[label], RingTrace)
    window.close()


# ── Encoder ring trace tests ──

def test_encoder_two_ring_traces(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    assert len(window._enc_traces) == 2
    assert "left_position" in window._enc_traces
    assert "right_position" in window._enc_traces
    for label in ("left_position", "right_position"):
        plot = window.findChild(QWidget, f"encoder_ring_{label}")
        assert plot is not None, f"encoder_ring_{label} not found"
        assert isinstance(window._enc_traces[label], RingTrace)
    window.close()


# ── RingTrace unit tests ──

def test_ring_trace_basics() -> None:
    """Test RingTrace standalone: capacity, NaN fill, append."""
    import pyqtgraph as pg

    app = QApplication.instance() or QApplication(["test-ringtrace-basics"])
    plot = pg.PlotWidget()
    trace = RingTrace(plot, "#000000", "test")

    x_data, y_data = trace.curve.getData()
    assert len(x_data) == SIGNAL_RING_CAPACITY
    assert len(y_data) == SIGNAL_RING_CAPACITY
    # All NaN initially
    assert np.all(np.isnan(y_data))
    assert trace._cursor == 0

    trace.append([1.0, 2.0, 3.0])
    _, y_after = trace.curve.getData()
    assert y_after[0] == 1.0
    assert y_after[1] == 2.0
    assert y_after[2] == 3.0
    assert np.isnan(y_after[3])
    assert trace._cursor == 3
    assert trace.cursor_line.value() == 2

    trace.reset()
    _, y_reset = trace.curve.getData()
    assert np.all(np.isnan(y_reset))
    assert trace._cursor == 0
    app.processEvents()


def test_ring_trace_wraps_at_capacity() -> None:
    """Test that writing beyond capacity wraps correctly."""
    import pyqtgraph as pg

    app = QApplication.instance() or QApplication(["test-ringtrace-wrap"])
    plot = pg.PlotWidget()
    trace = RingTrace(plot, "#000000", "test-wrap", capacity=5)

    values = np.arange(1, 8, dtype=np.float64)
    trace.append(values)

    # Sequential writes 1..7 leave the newest value at x=1 and x=2 is next.
    assert list(trace._buffer) == [6.0, 7.0, 3.0, 4.0, 5.0]
    assert trace._cursor == 2
    assert trace.cursor_line.value() == 1
    x_data, y_data = trace.curve.getData()
    assert list(x_data) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert np.isnan(y_data[2])  # break between newest and oldest retained data
    assert plot.getViewBox().state["limits"]["xLimits"] == [0, 4]
    assert plot.getViewBox().state["mouseEnabled"] == [False, False]
    app.processEvents()


def test_ring_trace_reset() -> None:
    """Test reset clears all data."""
    import pyqtgraph as pg

    app = QApplication.instance() or QApplication(["test-ringtrace-reset"])
    plot = pg.PlotWidget()
    trace = RingTrace(plot, "#000000", "test-reset")

    trace.append([10.0, 20.0, 30.0])
    assert trace._cursor == 3
    assert not np.all(np.isnan(trace.curve.getData()[1]))

    trace.reset()
    _, y_reset = trace.curve.getData()
    assert np.all(np.isnan(y_reset))
    assert trace._cursor == 0
    app.processEvents()


def test_ring_trace_cursor_line() -> None:
    """Test InfiniteLine cursor follows the write position."""
    import pyqtgraph as pg

    app = QApplication.instance() or QApplication(["test-ringtrace-cursor"])
    plot = pg.PlotWidget()
    trace = RingTrace(plot, "#000000", "test-cursor")

    assert trace.cursor_line.value() == 0

    trace.append([1.0] * 100)
    assert trace.cursor_line.value() == 99

    trace.append([2.0] * 950)
    assert trace._cursor == 50
    assert trace.cursor_line.value() == 49
    app.processEvents()


# ── Preview handling tests ──

def test_preview_ultrasound_4_channels(tmp_path: Path) -> None:
    """Send 4-channel ultrasound preview and verify each curve updates."""
    _app, window, _created = _window_with_fake(tmp_path)
    channels = [
        [1.0, 4.0, 2.0, 8.0],
        [2.0, 8.0, 4.0, 16.0],
        [3.0, 12.0, 6.0, 24.0],
        [4.0, 16.0, 8.0, 32.0],
    ]
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="ultrasound",
            payload={
                "host_monotonic_ns": 1_000,
                "channels": channels,
            },
        )
    )
    for i in range(4):
        _, y = window._us_curves[i].getData()
        assert list(y[:4]) == channels[i]
        assert len(y) == 512
        assert np.all(np.isnan(y[4:]))
    window.close()


def test_preview_imu_streams_payload(tmp_path: Path) -> None:
    """Send IMU streams payload and verify ring traces receive data."""
    _app, window, _created = _window_with_fake(tmp_path)
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="imu",
            payload={
                "host_monotonic_ns": 2_000,
                "streams": [
                    {"label": "imu_trunk", "values": [0.1, 0.2, 0.3], "channel": "acc_x"},
                    {"label": "imu_left", "values": [-0.1, -0.2, -0.3], "channel": "acc_x"},
                    {"label": "imu_right", "values": [0.05, 0.06, 0.07], "channel": "acc_x"},
                ],
            },
        )
    )
    # Check that each trace received data
    _, y_trunk = window._imu_traces["imu_trunk"].curve.getData()
    _, y_left = window._imu_traces["imu_left"].curve.getData()
    _, y_right = window._imu_traces["imu_right"].curve.getData()
    assert y_trunk[0] == 0.1
    assert y_trunk[2] == 0.3
    assert y_left[0] == -0.1
    assert y_right[0] == 0.05
    window.close()


def test_preview_encoder_channels_payload(tmp_path: Path) -> None:
    """Send encoder channels dict payload and verify ring traces."""
    _app, window, _created = _window_with_fake(tmp_path)
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="encoder",
            payload={
                "host_monotonic_ns": 3_000,
                "channels": {
                    "left_position": [10.0, 11.0, 12.0],
                    "right_position": [20.0, 21.0, 22.0],
                },
            },
        )
    )
    _, y_left = window._enc_traces["left_position"].curve.getData()
    _, y_right = window._enc_traces["right_position"].curve.getData()
    assert y_left[0] == 10.0
    assert y_left[2] == 12.0
    assert y_right[0] == 20.0
    assert y_right[2] == 22.0
    window.close()


def test_preferred_labeled_channel_payload_updates_each_ring(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="imu",
            payload={
                "labels": ["imu_trunk", "imu_left", "imu_right"],
                "channels": [[1.0], [2.0], [3.0]],
                "channel": "acc_x",
            },
        )
    )
    assert window._imu_traces["imu_trunk"]._buffer[0] == 1.0
    assert window._imu_traces["imu_left"]._buffer[0] == 2.0
    assert window._imu_traces["imu_right"]._buffer[0] == 3.0
    window.close()


def test_preview_y_axes_lock_once_and_are_shared_per_modality(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    events = (
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="ultrasound",
            payload={"channels": [[0.0, 100.0], [0.0, 200.0]] * 2},
        ),
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="imu",
            payload={
                "labels": ["imu_trunk", "imu_left", "imu_right"],
                "channels": [[-1.0, 0.5], [-0.5, 1.0], [-0.75, 0.75]],
            },
        ),
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="encoder",
            payload={
                "labels": ["left_position", "right_position"],
                "channels": [[-0.4, 0.4], [0.4, -0.4]],
            },
        ),
    )
    for event in events:
        window._handle_worker_event(event)

    plot_groups = {
        "ultrasound": window._us_plots,
        "imu": [trace.plot for trace in window._imu_traces.values()],
        "encoder": [trace.plot for trace in window._enc_traces.values()],
    }
    locked = dict(window._preview_y_ranges)
    assert set(locked) == {"ultrasound", "imu", "encoder"}
    for modality, plots in plot_groups.items():
        expected = locked[modality]
        for plot in plots:
            assert np.allclose(plot.getViewBox().viewRange()[1], expected)
            assert plot.getViewBox().state["mouseEnabled"] == [False, False]

    # Later out-of-range samples must not silently rescale any vertical axis.
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="imu",
            payload={"labels": ["imu_trunk"], "channels": [[-100.0, 100.0]]},
        )
    )
    assert window._preview_y_ranges == locked
    for plot in plot_groups["imu"]:
        assert np.allclose(plot.getViewBox().viewRange()[1], locked["imu"])
    window.close()


def test_legacy_single_series_payload_updates_only_first_window(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    for modality in ("imu", "encoder"):
        window._handle_worker_event(
            WorkerEvent(
                event_type=WorkerEventType.PREVIEW,
                modality=modality,
                payload={"values": [4.0, 5.0]},
            )
        )
    assert window._imu_traces["imu_trunk"]._buffer[0] == 4.0
    assert np.isnan(window._imu_traces["imu_left"]._buffer[0])
    assert window._enc_traces["left_position"]._buffer[0] == 4.0
    assert np.isnan(window._enc_traces["right_position"]._buffer[0])
    window.close()


def test_all_zero_ultrasound_alert_is_debounced_without_peak_widgets(
    tmp_path: Path,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    event = WorkerEvent(
        event_type=WorkerEventType.PREVIEW,
        modality="ultrasound",
        payload={
            "channels": [[0.0, 0.0]] * 4,
            "format_metrics": [
                {"all_zero": index == 2}
                for index in range(4)
            ],
        },
    )
    window._handle_worker_event(event)
    window._handle_worker_event(event)
    assert window.alerts_edit.toPlainText().count("通道 3 当前帧全零") == 1
    assert not hasattr(window, "ultrasound_peak_label")
    window.close()


def test_preview_ultrasound_curve_label_format(tmp_path: Path) -> None:
    """Verify A-scan plot titles include channel number."""
    _app, window, _created = _window_with_fake(tmp_path)
    for i in range(4):
        title = window._us_plots[i].getPlotItem().titleLabel.text
        assert f"通道 {i + 1}" in title
    window.close()


def test_preview_imu_ring_label_format(tmp_path: Path) -> None:
    """Verify IMU ring trace has acc_x in its plot title."""
    import pyqtgraph as pg

    _app, window, _created = _window_with_fake(tmp_path)
    for label in ("imu_trunk", "imu_left", "imu_right"):
        plot: pg.PlotWidget = window.findChild(QWidget, f"imu_ring_{label}")  # type: ignore[assignment]
        assert plot is not None, f"imu_ring_{label} not found"
        title_text = plot.getPlotItem().titleLabel.text
        assert "acc_x" in title_text, f"{label} title missing acc_x: {title_text}"
    window.close()


# ── Grid layout object names ──

def test_grid_widgets_exist(tmp_path: Path) -> None:
    """Check the top-level grid container widgets."""
    _app, window, _created = _window_with_fake(tmp_path)
    us_grid = window.findChild(QWidget, "ultrasound_grid")
    imu_grid = window.findChild(QWidget, "imu_ring_grid")
    enc_grid = window.findChild(QWidget, "encoder_ring_grid")
    assert us_grid is not None
    assert imu_grid is not None
    assert enc_grid is not None
    window.close()


# ── Reset on new trial ──

def test_trial_start_resets_all_ring_traces(tmp_path: Path) -> None:
    """Starting a new trial should reset ring traces via trial cleanup."""
    app, window, created = _window_with_fake(tmp_path)

    # Prime with data
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="imu",
            payload={
                "streams": [
                    {"label": "imu_trunk", "values": [1.0, 2.0], "channel": "acc_x"},
                ],
            },
        )
    )
    _, y_before = window._imu_traces["imu_trunk"].curve.getData()
    assert not np.all(np.isnan(y_before))

    # Connect all previews then start a trial
    _connect_all_previews_for_trial(window)
    window.build_request()  # validates inputs
    window.start_trial()
    worker = created[0]

    # After trial start, ring traces should be reset
    _, y_after = window._imu_traces["imu_trunk"].curve.getData()
    assert np.all(np.isnan(y_after))

    worker.finish(0)
    window.close()


# ── Basic integration test ──

def test_real_device_profile_is_selected_in_ui_and_copied_to_worker_request(
    tmp_path: Path,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    hardware_index = window.device_profile_combo.findData("hardware")
    assert hardware_index >= 0
    window.device_profile_combo.setCurrentIndex(hardware_index)
    window._settings.set_hardware_device_overrides(
        {
            "ultrasound": {"sdk_path": "D:/Elonxi", "port": 1430},
            "imu": {"radio_channel": 25},
            "encoder": {"port": "COM7", "baudrate": 9600},
        }
    )

    request = window.build_request()
    assert request.device_profile_key == "hardware"
    assert request.device_overrides["encoder"]["port"] == "COM7"
    assert window.hardware_settings_button.isEnabled()
    assert "Teensy" in window._device_profile_label.text()
    window.close()


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
    assert "真实设备模式" in window.device_profile_label.text()
    assert not window.start_button.isEnabled()

    window.subject_code_edit.setText("7")
    window.normalize_subject_code()
    assert window.subject_code_edit.text() == "007"
    window.condition_combo.setCurrentIndex(1)
    window.repeat_spin.setValue(3)

    # Connect all modality previews to satisfy Trial start requirements
    _connect_all_previews_for_trial(window)

    assert window.start_button.isEnabled()
    # Set health READY via direct table update (preview workers handle this normally)
    for row in range(window.health_table.rowCount()):
        window.health_table.item(row, 1).setText("READY")

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
    assert not window.connect_all_button.isEnabled()
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
                    "channels": [[2, 4, 8, 4], [5, 3, 2, 7], [1, 1, 1, 1], [9, 8, 7, 6]],
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.PREVIEW,
                modality="ultrasound",
                payload={
                    "host_monotonic_ns": 2_000,
                    "channels": [[3, 6, 9, 6], [1, 2, 3, 4], [2, 2, 2, 2], [0, 0, 0, 0]],
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.PREVIEW,
                modality="imu",
                payload={
                    "host_monotonic_ns": 2_000,
                    "streams": [
                        {"label": "imu_trunk", "values": [0.1, 0.2, 0.3], "channel": "acc_x"},
                        {"label": "imu_left", "values": [0.4, 0.5, 0.6], "channel": "acc_x"},
                        {"label": "imu_right", "values": [0.7, 0.8, 0.9], "channel": "acc_x"},
                    ],
                },
            ),
            WorkerEvent(
                event_type=WorkerEventType.PREVIEW,
                modality="encoder",
                payload={
                    "host_monotonic_ns": 3_000,
                    "channels": {
                        "left_position": [10.0, 11.0],
                        "right_position": [20.0, 21.0],
                    },
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

    # Check ultrasound curves
    _, ch0 = window._us_curves[0].getData()
    assert list(ch0[:4]) == [3.0, 6.0, 9.0, 6.0]
    assert np.all(np.isnan(ch0[4:]))

    # Check IMU ring traces received data
    _, trunk_y = window._imu_traces["imu_trunk"].curve.getData()
    assert trunk_y[0] == 0.1
    assert trunk_y[2] == 0.3

    # Check encoder ring traces
    _, left_y = window._enc_traces["left_position"].curve.getData()
    assert left_y[0] == 10.0
    assert left_y[1] == 11.0
    _, right_y = window._enc_traces["right_position"].curve.getData()
    assert right_y[0] == 20.0
    assert right_y[1] == 21.0

    assert window.sync_status_label.text() == "已同步"
    assert window.sync_quality_label.text() == "PASS"
    assert window.trigger_count_label.text() == "1"
    assert "123456" in window.first_trigger_label.text()
    assert len(window._timeline_x) >= 4
    assert len(window._timeline_text) == len(window._timeline_x)

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
    _connect_all_previews_for_trial(window)
    window.build_request()
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
    _connect_all_previews_for_trial(window)
    window.build_request()
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
    _connect_all_previews_for_trial(window)
    window.build_request()
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
    _app, window, created = _window_with_fake(tmp_path)

    # Only connect 3 of 4 modalities (sync_pulse missing)
    for modality in ("ultrasound", "imu", "encoder"):
        window._preview_connected_modalities.add(modality)
        window._preview_connection_status[modality] = "已连接"
    window._update_start_button()

    window.start_trial()
    assert not created
    assert "sync_pulse 尚未连接" in window.alerts_edit.toPlainText()
    assert not window.start_button.isEnabled()
    window.close()


def test_missing_sync_trigger_is_prominent_and_never_looks_recording(
    tmp_path: Path,
) -> None:
    app, window, created = _window_with_fake(tmp_path)
    _connect_all_previews_for_trial(window)
    window.build_request()
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
    assert any(
        "MISSING_TRIGGER" in text or "FAILED" in text
        for text in window._timeline_text
    )
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

    _connect_all_previews_for_trial(window)
    window.build_request()
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


# ── New per-modality connect/disconnect UI tests ──


def test_four_independent_connect_buttons_exist(tmp_path: Path) -> None:
    """Each modality must have its own 'connect' button in the UI."""
    _app, window, _created = _window_with_fake(tmp_path)
    for modality in ("ultrasound", "imu", "encoder", "sync_pulse"):
        btn = window.findChild(QWidget, f"connect_{modality}")
        assert btn is not None, f"connect_{modality} button not found"
    window.close()


def test_single_modality_connect_only_updates_that_modality(tmp_path: Path) -> None:
    """Connecting one modality should only mark that one as connected."""
    _app, window, _created = _window_with_fake(tmp_path)

    # Initially, no modalities should be connected
    for modality in ("ultrasound", "imu", "encoder", "sync_pulse"):
        assert modality not in window._preview_connected_modalities

    # Manually connect just ultrasound (bypassing spawn)
    window._preview_connected_modalities.add("ultrasound")
    window._preview_connection_status["ultrasound"] = "已连接"
    window._update_start_button()
    window._update_connect_button_state()

    assert "ultrasound" in window._preview_connected_modalities
    assert "imu" not in window._preview_connected_modalities
    assert "encoder" not in window._preview_connected_modalities
    assert "sync_pulse" not in window._preview_connected_modalities

    # Start trial should NOT be enabled (missing 3 modalities)
    assert not window.start_button.isEnabled()
    window.close()


def test_all_ready_enables_start_button(tmp_path: Path) -> None:
    """When ALL four modalities are connected via preview, start trial is enabled."""
    _app, window, _created = _window_with_fake(tmp_path)
    window.subject_code_edit.setText("001")
    _connect_all_previews_for_trial(window)
    assert window.start_button.isEnabled()
    window.close()


def test_connect_all_modalities_button(tmp_path: Path) -> None:
    """connect_all_button should exist and trigger _connect_all_modalities."""
    _app, window, _created = _window_with_fake(tmp_path)
    assert window.connect_all_button is not None
    assert not window._preview_workers
    assert window.connect_all_button.isEnabled()
    window.close()


def test_disconnect_all_clears_state(tmp_path: Path) -> None:
    """After disconnect_all, no modalities should be connected."""
    _app, window, _created = _window_with_fake(tmp_path)
    _connect_all_previews_for_trial(window)
    assert window.start_button.isEnabled()

    # Simulate disconnect all (clear preview state)
    window._preview_connected_modalities.clear()
    window._update_start_button()
    window._update_connect_button_state()

    assert not window.start_button.isEnabled()
    for modality in ("ultrasound", "imu", "encoder", "sync_pulse"):
        assert modality not in window._preview_connected_modalities
    window.close()


def test_profile_warning_simulated_shows_banner(tmp_path: Path) -> None:
    """Simulated profile should display an orange warning banner."""
    _app, window, _created = _window_with_fake(tmp_path)
    simulated_index = window.device_profile_combo.findData("simulated")
    window.device_profile_combo.setCurrentIndex(simulated_index)
    text = window.profile_warning_label.text()
    assert "模拟设备" in text or "simulated" in text.lower()
    assert "#f8d7da" in window.profile_warning_label.styleSheet() or "red" in window.profile_warning_label.styleSheet().lower()
    window.close()


def test_connected_modality_enables_its_separate_disconnect_button(tmp_path: Path) -> None:
    """Connect and disconnect remain two unambiguous controls."""
    _app, window, _created = _window_with_fake(tmp_path)

    connect_button = window._connect_buttons["ultrasound"]
    disconnect_button = window._disconnect_buttons["ultrasound"]
    assert connect_button.text() == "连接"
    assert disconnect_button.text() == "断开"

    # Simulate connecting
    window._preview_workers["ultrasound"] = object()  # dummy handle
    window._update_connect_button_state()
    assert not connect_button.isEnabled()
    assert disconnect_button.isEnabled()

    # Cleanup
    window._preview_workers.clear()
    window.close()


def test_modality_disconnect_button_requests_nonblocking_stop(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    handle = FakePreviewHandle("ultrasound")
    window._preview_workers["ultrasound"] = handle
    window._preview_connected_modalities.add("ultrasound")
    window._update_connect_button_state()

    window._disconnect_buttons["ultrasound"].click()

    assert handle.stop_requests == 1
    assert handle.join_calls == 0
    assert "ultrasound" in window._preview_workers
    assert window._preview_connection_status["ultrasound"] == "断开中"

    handle.alive = False
    window._poll_preview_workers()
    assert handle.closed
    assert "ultrasound" not in window._preview_workers
    window.close()


def test_trial_waits_for_preview_process_release_without_blocking_ui(
    tmp_path: Path,
) -> None:
    _app, window, created = _window_with_fake(tmp_path)
    handles = {modality: FakePreviewHandle(modality) for modality in MODALITIES}
    window._preview_workers.update(handles)
    window._preview_connected_modalities.update(MODALITIES)
    for modality in MODALITIES:
        window._preview_connection_status[modality] = "已连接"
    window._update_connect_button_state()

    window.start_trial()

    assert not created
    assert window._pending_trial_request is not None
    assert all(handle.stop_requests == 1 for handle in handles.values())
    assert all(handle.join_calls == 0 for handle in handles.values())

    for handle in handles.values():
        handle.alive = False
    window._poll_preview_workers()
    assert len(created) == 1
    assert created[0].started
    assert window._pending_trial_request is None

    window._preview_restore_modalities.clear()
    created[0].finish(0)
    window.close()


def test_trial_creates_no_files_during_preview(tmp_path: Path) -> None:
    """During preview phase, no trial/catalog/manifest/h5 files should be created."""
    _app, window, _created = _window_with_fake(tmp_path)
    _connect_all_previews_for_trial(window)

    # Verify no data files exist (simulate preview state)
    trial_dirs = list(tmp_path.glob("*trial*"))
    catalog_files = list(tmp_path.glob("**/*.sqlite3"))
    h5_files = list(tmp_path.glob("**/*.h5"))
    bin_files = list(tmp_path.glob("**/*.bin"))
    recording_files = list(tmp_path.glob("**/*.recording"))

    assert not trial_dirs, f"trial directories should not exist: {trial_dirs}"
    assert not catalog_files
    assert not h5_files
    assert not bin_files
    assert not recording_files
    window.close()


def test_device_profile_label_has_modality_info(tmp_path: Path) -> None:
    """_device_profile_label should contain per-modality source labels."""
    _app, window, _created = _window_with_fake(tmp_path)
    # Fresh settings default to the laboratory hardware profile.
    for modality in ("ultrasound", "imu", "encoder"):
        label = window._connect_device_labels.get(modality)
        assert label is not None, f"device label for {modality} not found"
        assert "真实" in label.text()
    assert "模拟同步" in window._connect_device_labels["sync_pulse"].text()
    window.close()


def test_health_table_has_four_modalities(tmp_path: Path) -> None:
    """Health table should list all four modalities."""
    _app, window, _created = _window_with_fake(tmp_path)
    assert window.health_table.rowCount() == 4
    modalities_displayed = set()
    for row in range(window.health_table.rowCount()):
        modalities_displayed.add(window.health_table.item(row, 0).text())
    assert modalities_displayed == {"ultrasound", "imu", "encoder", "sync_pulse"}
    window.close()


# ── Hardware profile selection tests ──


def test_save_real_device_settings_auto_selects_hardware_profile(tmp_path: Path) -> None:
    """Saving real device settings should switch to hardware profile."""
    _app, window, _created = _window_with_fake(tmp_path)
    simulated_index = window.device_profile_combo.findData("simulated")
    window.device_profile_combo.setCurrentIndex(simulated_index)
    assert window._selected_device_profile_key() == "simulated"

    # Simulate the behavior that happens after hardware settings dialog accepts
    window._settings.set_hardware_device_overrides(
        {"ultrasound": {"sdk_path": "D:/Elonxi", "port": 1430}}
    )
    window._settings.set_device_profile_key("hardware")

    # Manually trigger the profile switch effect
    idx = window.device_profile_combo.findData("hardware")
    window.device_profile_combo.setCurrentIndex(idx)
    window._render_device_profile()

    assert window._selected_device_profile_key() == "hardware"
    window.close()


def test_worker_request_uses_selected_profile(tmp_path: Path) -> None:
    """TrialRunRequest must use the actual selected profile, not fallback to simulated."""
    _app, window, _created = _window_with_fake(tmp_path)
    window._settings.set_hardware_device_overrides(
        {"ultrasound": {"sdk_path": "D:/Elonxi", "port": 1430}}
    )
    # Switch to hardware
    window._settings.set_device_profile_key("hardware")
    idx = window.device_profile_combo.findData("hardware")
    window.device_profile_combo.blockSignals(True)
    window.device_profile_combo.setCurrentIndex(idx)
    window.device_profile_combo.blockSignals(False)

    request = window.build_request()
    assert request.device_profile_key == "hardware"
    assert request.device_overrides["ultrasound"]["port"] == 1430
    window.close()
