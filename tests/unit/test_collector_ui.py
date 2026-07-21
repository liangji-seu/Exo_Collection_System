from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QApplication, QDialog, QScrollArea, QSplitter, QWidget

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.acquisition.recording_stream import RecordingStreamEndpoint
from exo_collection.apps.collector import CollectorWindow
from exo_collection.apps.collector.device_settings import (
    DEVICE_SETTINGS_DIALOGS,
    EncoderDeviceSettingsDialog,
    ImuDeviceSettingsDialog,
    SyncPulseDeviceSettingsDialog,
    UltrasoundDeviceSettingsDialog,
    enumerate_serial_ports,
)
from exo_collection.apps.collector.window import (
    ExperimentMetadataDialog,
    HardwareDeviceSettingsDialog,
    MODALITIES,
    RingTrace,
    SIGNAL_RING_CAPACITY,
)
from exo_collection.configuration import (
    SharedAppSettings,
)
from exo_collection.orchestration.models import (
    MeasuredConditionMetadata,
    TrialExperimentMetadata,
    TrialRunRequest,
)


class FakeCollectorWorker:
    def __init__(
        self,
        request: TrialRunRequest,
        stream_endpoints: tuple[RecordingStreamEndpoint, ...] = (),
    ) -> None:
        self.request = request
        self.stream_endpoints = stream_endpoints
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


def test_ultrasound_dialog_restores_raw_ethernet_interface() -> None:
    app = QApplication.instance() or QApplication(["test-raw-interface"])
    dialog = UltrasoundDeviceSettingsDialog(
        {"interface_name": "\\Device\\NPF_TEST"}
    )

    assert (
        dialog.interface_combo.currentData()
        == "\\Device\\NPF_TEST"
    )

    dialog.close()
    app.processEvents()


def test_ultrasound_dialog_stops_active_interface_scan_before_exit() -> None:
    app = QApplication.instance() or QApplication(["test-stop-interface-scan"])
    dialog = UltrasoundDeviceSettingsDialog(
        {"interface_name": "\\Device\\NPF_TEST"}
    )

    class FakeScanWorker:
        def __init__(self) -> None:
            self.interrupted = False
            self.wait_timeout: int | None = None
            self.deleted = False

        def isRunning(self) -> bool:  # noqa: N802 - mirrors QThread
            return True

        def requestInterruption(self) -> None:  # noqa: N802 - Qt API
            self.interrupted = True

        def wait(self, timeout: int) -> bool:
            self.wait_timeout = timeout
            return True

        def deleteLater(self) -> None:  # noqa: N802 - Qt API
            self.deleted = True

    worker = FakeScanWorker()
    dialog._scan_worker = worker  # type: ignore[assignment]

    assert dialog._stop_scan_worker()
    assert worker.interrupted
    assert worker.wait_timeout == 2_500
    assert worker.deleted
    assert dialog._scan_worker is None
    dialog.close()
    app.processEvents()


def test_each_modality_dialog_restores_its_own_settings() -> None:
    app = QApplication.instance() or QApplication(["test-modality-dialogs"])
    imu = ImuDeviceSettingsDialog(
        {"radio_channel": 19, "sample_rate_hz": 100.0, "sensor_ids": ["A", "B", "C"]}
    )
    encoder = EncoderDeviceSettingsDialog(
        {"port": "COM7", "baudrate": 1_000_000, "vid": 0x16C0, "pid": 0x0483}
    )
    sync = SyncPulseDeviceSettingsDialog(
        {"sample_rate_hz": 2_000.0, "pulse_interval_s": 2.0}
    )

    assert imu.channel_spin.value() == 19
    assert imu.id_1_edit.text() == "A"
    assert imu.id_2_edit.text() == "B"
    assert imu.id_3_edit.text() == "C"
    assert encoder._selected_port() == "COM7"
    assert sync.rate_spin.value() == 2_000.0
    assert sync.interval_spin.value() == 2.0
    for dialog in (imu, encoder, sync):
        dialog.close()
    app.processEvents()


def test_imu_dialog_preserves_disabled_middle_slot() -> None:
    app = QApplication.instance() or QApplication(["test-imu-slot-settings"])
    dialog = ImuDeviceSettingsDialog(
        {
            "radio_channel": 25,
            "sample_rate_hz": 120.0,
            "sensor_ids": ["10B42610", "", "10B42620"],
        }
    )
    assert dialog.id_1_edit.text() == "10B42610"
    assert dialog.id_2_edit.text() == ""
    assert dialog.id_3_edit.text() == "10B42620"

    dialog.accept()
    assert dialog.validated_override["sensor_ids"] == (
        "10B42610",
        "",
        "10B42620",
    )
    dialog.close()
    app.processEvents()


def test_hardware_device_settings_dialog_preserves_empty_middle_slot() -> None:
    """Old HardwareDeviceSettingsDialog must not compress IMU1=A, IMU2=, IMU3=C
    into (A,C); the empty second slot is preserved as ""."""
    app = QApplication.instance() or QApplication(["test-hw-dialog-slots"])
    from unittest.mock import patch

    with (
        patch("exo_collection.apps.collector.window.build_adapters"),
        patch("exo_collection.apps.collector.window.load_device_profile"),
    ):
        dialog = HardwareDeviceSettingsDialog(
            {
                "imu": {
                    "sensor_ids": ["10B42610", "", "10B42620"],
                    "radio_channel": 25,
                    "sample_rate_hz": 120.0,
                },
            }
        )
        assert dialog.awinda_id_left.text() == "10B42610"
        assert dialog.awinda_id_mid.text() == ""
        assert dialog.awinda_id_right.text() == "10B42620"

        dialog.accept()
        sensor_ids = dialog.validated_overrides["imu"]["sensor_ids"]
        assert sensor_ids == ("10B42610", "", "10B42620"), (
            f"expected 3-slot tuple, got {sensor_ids!r}"
        )

    dialog.close()
    app.processEvents()


class FakePreviewHandle:
    def __init__(self, modality: str) -> None:
        self.modality = modality
        self.device_id = f"{modality}_fake"
        self.simulated = True
        self.alive = True
        self.stop_requests = 0
        self.join_calls = 0
        self.closed = False
        self.events: list[WorkerEvent] = []
        self.begin_recording_calls: list[str] = []
        self.end_recording_calls: list[str] = []
        self.discard_recording_backlog_calls = 0
        self.recording_backlog: list[object] = []
        self.begin_recording_error: Exception | None = None
        self.active_trial_uuid: str | None = None
        self._recording_endpoint = RecordingStreamEndpoint(
            queue=object(),
            device_id=self.device_id,
            modality=modality,
            descriptor={
                "device_id": self.device_id,
                "modality": modality,
                "display_name": f"Fake {modality}",
                "clock_domain": f"{modality}_clock",
                "event_kind": "frame_batch",
                "channels": [f"{modality}_channel"],
                "units": ["a.u."],
                "nominal_rate_hz": 100.0,
                "sample_shape": [1],
                "dtype": "<f8",
                "metadata": {"test_descriptor": modality},
            },
            configuration_snapshot={"test_config": modality},
        )

    @property
    def is_alive(self) -> bool:
        return self.alive

    @property
    def exitcode(self) -> int | None:
        return None if self.alive else 0

    def request_stop(self) -> None:
        self.stop_requests += 1

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]:
        result = self.events[:limit]
        del self.events[:limit]
        return result

    @property
    def recording_endpoint(self) -> RecordingStreamEndpoint:
        return self._recording_endpoint

    def begin_recording(self, trial_uuid: str) -> None:
        self.begin_recording_calls.append(trial_uuid)
        if self.begin_recording_error is not None:
            raise self.begin_recording_error
        self.active_trial_uuid = trial_uuid

    def end_recording(self, trial_uuid: str) -> None:
        self.end_recording_calls.append(trial_uuid)
        assert self.active_trial_uuid == trial_uuid
        self.active_trial_uuid = None

    def discard_recording_backlog(self) -> int:
        self.discard_recording_backlog_calls += 1
        discarded = len(self.recording_backlog)
        self.recording_backlog.clear()
        return discarded

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

    def factory(
        request: TrialRunRequest,
        endpoints: Mapping[str, RecordingStreamEndpoint],
    ) -> FakeCollectorWorker:
        worker = FakeCollectorWorker(request, tuple(endpoints.values()))
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


def _connect_all_previews_for_trial(
    window: CollectorWindow,
) -> dict[str, FakePreviewHandle]:
    """Simulate all modalities being connected via preview (marks them READY).

    This bypasses the actual spawn-based preview workers so that trial-flow
    tests can proceed without subprocesses.
    """
    handles: dict[str, FakePreviewHandle] = {}
    for modality in ("ultrasound", "imu", "encoder", "sync_pulse"):
        existing = window._preview_workers.get(modality)
        handle = (
            existing
            if isinstance(existing, FakePreviewHandle)
            else FakePreviewHandle(modality)
        )
        window._preview_workers[modality] = handle
        handles[modality] = handle
        window._preview_connected_modalities.add(modality)
        window._preview_connection_status[modality] = "已连接"
    window._update_start_button()
    window._update_connect_button_state()
    return handles


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


def test_1080p_layout_scrolls_controls_instead_of_crushing_them(
    tmp_path: Path,
) -> None:
    app, window, _created = _window_with_fake(tmp_path)
    window.resize(1920, 991)
    window.show()
    app.processEvents()

    body = window.findChild(QSplitter, "collector_body")
    controls_scroll = window.findChild(QScrollArea, "controls_scroll")
    controls_content = window.findChild(QWidget, "controls_content")
    assert body is not None
    assert controls_scroll is not None
    assert controls_content is not None

    # A maximized 1080p Windows desktop commonly has about 991 physical pixels
    # after taskbar and frame margins.  The main window minimum must remain well
    # below that, otherwise Windows rejects showMaximized() geometry.
    assert window.minimumSizeHint().height() < 700
    assert 610 <= controls_scroll.width() <= 650
    assert controls_content.height() >= controls_content.minimumSizeHint().height()
    assert controls_scroll.verticalScrollBar().maximum() == 0
    assert abs(window.project_combo.width() - window.subject_code_edit.width()) <= 2
    assert abs(window.condition_combo.width() - window.repeat_spin.width()) <= 2

    # The two toggle actions stay at their normal height and never overlap.
    action_buttons = [
        window.connect_all_button,
        window.start_button,
    ]
    for button in action_buttons:
        assert button.height() >= button.minimumSizeHint().height()
    for index, button in enumerate(action_buttons):
        for other in action_buttons[index + 1 :]:
            assert not button.geometry().intersects(other.geometry())

    assert body.sizes()[1] > body.sizes()[0]
    window.close()


def test_collector_theme_uses_direct_toggle_styles(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)

    assert window.connect_all_button.text() == "全部连接"
    assert window.start_button.text() == "开始写盘"
    assert "#0d6efd" in window.start_button.styleSheet()

    window._preview_workers["ultrasound"] = object()
    window._update_connect_button_state()
    assert window.connect_all_button.text() == "全部断开"
    assert "#f8d7da" in window.connect_all_button.styleSheet()
    window._preview_workers.clear()

    for button in window._connect_buttons.values():
        assert button.property("buttonRole") == "connect"
    for button in window._disconnect_buttons.values():
        assert button.property("buttonRole") == "disconnect"

    stylesheet = window.styleSheet()
    assert '#1d4ed8' in stylesheet
    assert '#15803d' in stylesheet
    assert '#b91c1c' in stylesheet
    window.close()


# ── IMU ring trace tests ──

def test_imu_three_sensor_plots_expose_nine_axis_traces(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    assert len(window._imu_traces) == 9
    for label in ("imu_trunk", "imu_left", "imu_right"):
        plot = window.findChild(QWidget, f"imu_ring_{label}")
        assert plot is not None, f"imu_ring_{label} not found"
        for axis in ("acc_x", "acc_y", "acc_z"):
            assert isinstance(window._imu_traces[f"{label}_{axis}"], RingTrace)
    window.close()


# ── Encoder ring trace tests ──

def test_encoder_two_sides_expose_position_velocity_and_torque(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    assert len(window._enc_traces) == 6
    left_plot = window.findChild(QWidget, "encoder_ring_left")
    right_plot = window.findChild(QWidget, "encoder_ring_right")
    assert left_plot is not None
    assert right_plot is not None
    for label in (
        "left_position",
        "left_velocity",
        "left_torque",
        "right_position",
        "right_velocity",
        "right_torque",
    ):
        assert isinstance(window._enc_traces[label], RingTrace)
        expected_plot = left_plot if label.startswith("left_") else right_plot
        assert window._enc_traces[label].plot is expected_plot
    assert np.allclose(window._preview_y_ranges["encoder"], (-13.0, 13.0))
    assert np.allclose(left_plot.getViewBox().viewRange()[1], (-13.0, 13.0))
    assert np.allclose(right_plot.getViewBox().viewRange()[1], (-13.0, 13.0))
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
        assert len(y) == 1000
        assert np.all(np.isnan(y[4:]))
    window.close()


def test_raw_ultrasound_packet_updates_only_its_tagged_channel(
    tmp_path: Path,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    before = [
        np.asarray(curve.getData()[1], dtype=np.float64).copy()
        for curve in window._us_curves
    ]

    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="ultrasound",
            payload={
                "host_monotonic_ns": 1_000,
                "channel_index": 2,
                "channels": [[-127.0, -10.0, 0.0, 128.0]],
            },
        )
    )

    after = [
        np.asarray(curve.getData()[1], dtype=np.float64)
        for curve in window._us_curves
    ]
    for index in (0, 1, 3):
        np.testing.assert_allclose(after[index], before[index], equal_nan=True)
    np.testing.assert_allclose(after[2][:4], [-127.0, -10.0, 0.0, 128.0])
    assert np.all(np.isnan(after[2][4:]))
    assert window._preview_y_ranges["ultrasound"] == (-128.0, 128.0)
    for plot in window._us_plots:
        assert np.allclose(plot.getViewBox().viewRange()[1], (-128.0, 128.0))
        assert plot.getViewBox().state["mouseEnabled"] == [False, False]
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
                    {"label": "imu_trunk_acc_x", "values": [0.1, 0.2, 0.3]},
                    {"label": "imu_left_acc_x", "values": [-0.1, -0.2, -0.3]},
                    {"label": "imu_right_acc_x", "values": [0.05, 0.06, 0.07]},
                ],
            },
        )
    )
    # Check that each trace received data
    _, y_trunk = window._imu_traces["imu_trunk_acc_x"].curve.getData()
    _, y_left = window._imu_traces["imu_left_acc_x"].curve.getData()
    _, y_right = window._imu_traces["imu_right_acc_x"].curve.getData()
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
                    "left_velocity": [1.0, 2.0, 3.0],
                    "left_torque": [0.1, 0.2, 0.3],
                    "right_position": [20.0, 21.0, 22.0],
                    "right_velocity": [4.0, 5.0, 6.0],
                    "right_torque": [0.4, 0.5, 0.6],
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
    _, y_left_velocity = window._enc_traces["left_velocity"].curve.getData()
    _, y_left_torque = window._enc_traces["left_torque"].curve.getData()
    _, y_right_velocity = window._enc_traces["right_velocity"].curve.getData()
    _, y_right_torque = window._enc_traces["right_torque"].curve.getData()
    assert y_left_velocity[2] == 3.0
    assert np.isclose(y_left_torque[2], 0.3)
    assert y_right_velocity[2] == 6.0
    assert np.isclose(y_right_torque[2], 0.6)
    window.close()


def test_preferred_labeled_channel_payload_updates_each_ring(tmp_path: Path) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="imu",
            payload={
                "labels": ["imu_trunk_acc_x", "imu_left_acc_x", "imu_right_acc_x"],
                "channels": [[1.0], [2.0], [3.0]],
                "channel": "acc_x",
            },
        )
    )
    assert window._imu_traces["imu_trunk_acc_x"]._buffer[0] == 1.0
    assert window._imu_traces["imu_left_acc_x"]._buffer[0] == 2.0
    assert window._imu_traces["imu_right_acc_x"]._buffer[0] == 3.0
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
    assert window._imu_traces["imu_trunk_acc_x"]._buffer[0] == 4.0
    assert np.isnan(window._imu_traces["imu_trunk_acc_y"]._buffer[0])
    assert window._enc_traces["left_position"]._buffer[0] == 4.0
    assert np.isnan(window._enc_traces["right_position"]._buffer[0])
    window.close()


def test_all_zero_ultrasound_alert_is_debounced_without_peak_widgets(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="exo_collection.collector.ui")
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
    assert caplog.text.count("通道 3 当前帧全零") == 1
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
    """Each sensor plot contains the configured three-axis traces."""
    import pyqtgraph as pg

    _app, window, _created = _window_with_fake(tmp_path)
    for label in ("imu_trunk", "imu_left", "imu_right"):
        plot: pg.PlotWidget = window.findChild(QWidget, f"imu_ring_{label}")  # type: ignore[assignment]
        assert plot is not None, f"imu_ring_{label} not found"
        title_text = plot.getPlotItem().titleLabel.text
        assert "acc_z" in title_text, f"{label} title missing axis: {title_text}"
        assert all(
            f"{label}_{axis}" in window._imu_traces
            for axis in ("acc_x", "acc_y", "acc_z")
        )
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


# ── Preserve continuous preview at Trial boundaries ──

def test_trial_start_preserves_all_live_preview_curves(tmp_path: Path) -> None:
    """Opening the recording gate must not create a visible signal gap."""
    _app, window, created = _window_with_fake(tmp_path)

    # Prime ultrasound, IMU and encoder displays before disk recording.
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="ultrasound",
            payload={"channel_index": 1, "values": [5.0] * 1000},
        )
    )
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="imu",
            payload={
                "streams": [
                        {"label": "imu_trunk_acc_x", "values": [1.0, 2.0]},
                ],
            },
        )
    )
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="encoder",
            payload={"channels": {"left_position": [3.0, 4.0]}},
        )
    )
    _x, ultrasound_before = window._us_curves[1].getData()
    _x, imu_before = window._imu_traces["imu_trunk_acc_x"].curve.getData()
    _x, encoder_before = window._enc_traces["left_position"].curve.getData()
    assert ultrasound_before[0] == 5.0
    assert imu_before[0] == 1.0
    assert encoder_before[0] == 3.0

    _connect_all_previews_for_trial(window)
    window.start_trial()
    worker = created[0]

    _x, ultrasound_after = window._us_curves[1].getData()
    _x, imu_after = window._imu_traces["imu_trunk_acc_x"].curve.getData()
    _x, encoder_after = window._enc_traces["left_position"].curve.getData()
    np.testing.assert_allclose(ultrasound_after, ultrasound_before, equal_nan=True)
    np.testing.assert_allclose(imu_after, imu_before, equal_nan=True)
    np.testing.assert_allclose(encoder_after, encoder_before, equal_nan=True)

    worker.finish(0)
    window.close()


# ── Basic integration test ──

def test_real_device_profile_is_selected_in_ui_and_copied_to_worker_request(
    tmp_path: Path,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    window._settings.set_device_profile_key("hardware")
    window._settings.set_hardware_device_overrides(
        {
            "ultrasound": {"interface_name": "\\Device\\NPF_TEST"},
            "imu": {"radio_channel": 25},
            "encoder": {"port": "COM7", "baudrate": 1_000_000},
        }
    )

    request = window.build_request()
    assert request.device_profile_key == "hardware"
    assert request.device_overrides["encoder"]["port"] == "COM7"
    assert not hasattr(window, "device_profile_combo")
    assert not hasattr(window, "hardware_settings_button")
    assert window._configure_buttons["encoder"].isEnabled()
    window._render_device_profile()
    assert "Teensy" in window._device_profile_label.text()
    window.close()


def test_collector_locks_condition_polls_events_and_finalizes(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="exo_collection.collector.ui")
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
    assert window.start_button.isEnabled()
    assert window.start_button.text() == "停止写盘"
    assert "#dc3545" in window.start_button.styleSheet()
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
                            {"label": "imu_trunk_acc_x", "values": [0.1, 0.2, 0.3]},
                            {"label": "imu_left_acc_x", "values": [0.4, 0.5, 0.6]},
                            {"label": "imu_right_acc_x", "values": [0.7, 0.8, 0.9]},
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
    _wait_until(app, lambda: "采集中" in window.overall_status)

    imu_row = window._health_rows["imu"]
    assert window.health_table.item(imu_row, 0).text() == "IMU"
    assert window.health_table.item(imu_row, 1).text() == "123"
    assert window.health_table.item(imu_row, 2).text() == "198.5 Hz"
    assert window.health_table.item(imu_row, 3).text() == "3"
    assert window.health_table.item(imu_row, 4).text() == "—"
    assert "设备状态：DEGRADED" in window.health_table.item(imu_row, 0).toolTip()
    assert "preview delay" in caplog.text

    # Check ultrasound curves
    _, ch0 = window._us_curves[0].getData()
    assert list(ch0[:4]) == [3.0, 6.0, 9.0, 6.0]
    assert np.all(np.isnan(ch0[4:]))

    # Check IMU ring traces received data
    _, trunk_y = window._imu_traces["imu_trunk_acc_x"].curve.getData()
    assert trunk_y[0] == 0.1
    assert trunk_y[2] == 0.3

    # Check encoder ring traces
    _, left_y = window._enc_traces["left_position"].curve.getData()
    assert left_y[0] == 10.0
    assert left_y[1] == 11.0
    _, right_y = window._enc_traces["right_position"].curve.getData()
    assert right_y[0] == 20.0
    assert right_y[1] == 21.0

    assert window.sync_status_label.text() == ""
    assert window.sync_status_label.property("indicatorState") == "green"
    assert window.sync_status_label.property("syncReceived") is True
    assert "合格触发：1" in window.sync_status_label.toolTip()
    assert "123456" in window.sync_status_label.toolTip()
    assert "质量：PASS" in window.sync_status_label.toolTip()
    assert len(window._timeline_x) >= 4
    assert len(window._timeline_text) == len(window._timeline_x)

    window.start_button.click()
    assert worker.stop_requests == 1
    assert window.overall_status == "保存中"
    assert not window.start_button.isEnabled()

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

    assert window.overall_status == "可采集"
    assert str(manifest_path) in caplog.text
    assert not hasattr(window, "manifest_label")
    assert not hasattr(window, "open_log_dir_button")
    assert not window.configuration_locked
    assert window.condition_combo.isEnabled()
    assert window.start_button.isEnabled()
    assert worker.closed
    assert worker.join_timeouts == [0]
    window.close()


def test_collector_shows_failed_worker_error_without_blocking_ui(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="exo_collection.collector.ui")
    app, window, created = _window_with_fake(tmp_path)
    handles = _connect_all_previews_for_trial(window)
    identities = {name: id(handle) for name, handle in handles.items()}
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
    assert "simulated disk full" in caplog.text
    assert window.start_button.isEnabled()
    assert worker.closed
    assert window._preview_connected_modalities == set(MODALITIES)
    assert {
        name: id(window._preview_workers[name]) for name in MODALITIES
    } == identities
    assert all(handle.stop_requests == 0 for handle in handles.values())
    assert all(len(handle.end_recording_calls) == 1 for handle in handles.values())
    window.close()


def test_collector_rejects_terminal_event_from_another_trial(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="exo_collection.collector.ui")
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
    assert "已拒绝不属于当前 Trial" in caplog.text
    assert "未发布 COMPLETED/FAILED" in caplog.text
    window.close()


def test_collector_forces_hung_controlled_stop_and_preserves_recovery_semantics(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="exo_collection.collector.ui")
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
    alerts = caplog.text
    assert "受控停止等待超时" in alerts
    assert ".recording" in alerts
    assert "FINALIZED" in alerts
    window.close()


def test_start_rejects_stale_ready_state_without_live_preview_handle(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="exo_collection.collector.ui")
    _app, window, created = _window_with_fake(tmp_path)

    # A stale UI READY flag is insufficient without the corresponding live
    # handle and its complete recording endpoint.
    for modality in ("ultrasound", "imu", "encoder"):
        window._preview_connected_modalities.add(modality)
        window._preview_connection_status[modality] = "已连接"
    window._update_start_button()

    window.start_trial()
    assert not created
    assert "preview is not READY" in caplog.text
    assert not window.configuration_locked
    window.close()


def test_optional_missing_sync_is_neutral_and_does_not_prevent_finalization(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="exo_collection.collector.ui")
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
    _wait_until(app, lambda: window.overall_status == "等待同步")
    assert window.overall_status == "等待同步"
    assert window.sync_status_label.property("indicatorState") == "yellow"
    assert window.sync_status_label.property("syncReceived") is False
    assert "等待同步信号" in window.sync_status_label.toolTip()
    assert "background-color:#FBBF24" in window.sync_status_label.styleSheet()

    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.SYNC,
            payload={
                "status": "NOT_RECEIVED",
                "quality": "OPTIONAL",
                "trigger_count": 0,
                "first_trigger_host_monotonic_ns": None,
                "trigger_time_utc": None,
            },
        )
    )
    _wait_until(
        app,
        lambda: window.sync_status_label.property("indicatorState") == "neutral",
    )
    assert window.overall_status == "等待同步"
    assert window.sync_status_label.text() == ""
    assert window.sync_status_label.property("syncReceived") is False
    assert "未收到同步信号（可选）" in window.sync_status_label.toolTip()
    assert "质量：OPTIONAL" in window.sync_status_label.toolTip()
    assert "background-color:#94A3B8" in window.sync_status_label.styleSheet()
    assert any("NOT_RECEIVED" in text for text in window._timeline_text)

    manifest_path = tmp_path / "trial-without-sync" / "manifest.json"
    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.COMPLETED,
            message="Trial package finalized without optional sync pulse",
            payload={
                "state": "FINALIZED",
                "manifest_path": str(manifest_path),
            },
        )
    )
    worker.finish(0)
    _wait_until(app, lambda: window.worker is None)
    assert window.overall_status == "可采集"
    assert worker.closed
    assert str(manifest_path) in caplog.text
    assert not any(
        record.levelno >= logging.WARNING and "同步" in record.getMessage()
        for record in caplog.records
    )
    assert "sync trigger missing" not in caplog.text
    assert "未检测到合格同步触发" not in caplog.text

    # Device/recording failures remain severe and are covered separately by
    # test_collector_shows_failed_worker_error_without_blocking_ui; only the
    # absence of the optional pulse is neutral here.
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

    # The current policy allows recording any explicit READY subset; the live
    # endpoint itself is validated again when start is clicked.
    assert window.start_button.isEnabled()
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


def test_trial_form_has_no_global_device_config_and_modalities_are_clickable(
    tmp_path: Path,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    assert not hasattr(window, "device_profile_combo")
    assert not hasattr(window, "hardware_settings_button")
    assert not hasattr(window, "profile_warning_label")
    assert set(window._configure_buttons) == set(MODALITIES)
    for modality, button in window._configure_buttons.items():
        assert button.objectName() == f"configure_{modality}"
        assert button.property("buttonRole") == "deviceConfig"
        assert "自动保存" in button.toolTip()
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


def test_trial_reuses_ready_preview_handles_without_disconnect_or_reconnect(
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

    assert len(created) == 1
    worker = created[0]
    assert worker.started
    assert len(worker.stream_endpoints) == len(MODALITIES)
    assert {endpoint.modality for endpoint in worker.stream_endpoints} == set(
        MODALITIES
    )
    assert all(handle.stop_requests == 0 for handle in handles.values())
    assert all(handle.join_calls == 0 for handle in handles.values())
    assert all(len(handle.begin_recording_calls) == 1 for handle in handles.values())
    assert all(
        endpoint.descriptor["metadata"]["test_descriptor"]
        == endpoint.modality
        for endpoint in worker.stream_endpoints
    )
    assert all(
        endpoint.configuration_snapshot["test_config"] == endpoint.modality
        for endpoint in worker.stream_endpoints
    )
    identities = {name: id(handle) for name, handle in handles.items()}

    window.request_controlled_stop()
    assert all(handle.stop_requests == 0 for handle in handles.values())
    assert all(len(handle.end_recording_calls) == 1 for handle in handles.values())

    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.COMPLETED,
            payload={"state": "FINALIZED"},
        )
    )
    worker.finish(0)
    window.poll_worker_events()

    assert window.worker is None
    assert window._preview_connected_modalities == set(MODALITIES)
    assert {
        name: id(window._preview_workers[name]) for name in MODALITIES
    } == identities
    assert all(handle.stop_requests == 0 for handle in handles.values())
    assert window._worker_state == "PREFLIGHT_READY"
    assert window.start_button.isEnabled()
    window.close()


def test_new_trial_discards_stale_recording_queue_items_before_attach(
    tmp_path: Path,
) -> None:
    _app, window, created = _window_with_fake(tmp_path)
    handles = _connect_all_previews_for_trial(window)
    for modality, handle in handles.items():
        handle.recording_backlog.extend(
            [f"stale-{modality}-boundary", f"stale-{modality}-raw"]
        )

    window.start_trial()

    assert len(created) == 1
    assert all(
        handle.discard_recording_backlog_calls == 1
        for handle in handles.values()
    )
    assert all(not handle.recording_backlog for handle in handles.values())
    assert all(
        len(handle.begin_recording_calls) == 1 for handle in handles.values()
    )
    created[0].finish(0)
    window.close()


def test_partial_begin_failure_rolls_back_only_successful_recording_gates(
    tmp_path: Path,
) -> None:
    _app, window, created = _window_with_fake(tmp_path)
    handles = _connect_all_previews_for_trial(window)
    handles["imu"].begin_recording_error = RuntimeError("IMU gate refused")

    window.start_trial()

    assert len(created) == 1
    worker = created[0]
    assert worker.started
    assert worker.stop_requests == 1
    assert worker.exitcode == -15
    assert worker.closed
    # Handles are attached in sorted modality order: encoder succeeds, IMU
    # fails, and the remaining two gates are never opened.
    assert len(handles["encoder"].begin_recording_calls) == 1
    assert len(handles["encoder"].end_recording_calls) == 1
    assert len(handles["imu"].begin_recording_calls) == 1
    assert handles["imu"].end_recording_calls == []
    assert handles["sync_pulse"].begin_recording_calls == []
    assert handles["ultrasound"].begin_recording_calls == []
    assert all(handle.stop_requests == 0 for handle in handles.values())
    assert window.worker is None
    assert not window.configuration_locked
    assert window.overall_status == "失败"
    assert window._preview_connected_modalities == set(MODALITIES)
    window.close()


def test_preview_events_continue_before_during_and_after_disk_recording(
    tmp_path: Path,
) -> None:
    _app, window, created = _window_with_fake(tmp_path)
    handles = _connect_all_previews_for_trial(window)
    ultrasound = handles["ultrasound"]
    original_identity = id(ultrasound)

    def publish_four_channel_frames(base_value: float) -> None:
        for channel_index in range(4):
            value = base_value + channel_index
            ultrasound.events.append(
                WorkerEvent(
                    event_type=WorkerEventType.PREVIEW,
                    modality="ultrasound",
                    payload={
                        "channel_index": channel_index,
                        "values": [value] * 1000,
                    },
                )
            )
        window._poll_preview_workers()
        for channel_index, curve in enumerate(window._us_curves):
            _x, y = curve.getData()
            assert y is not None
            assert y[0] == base_value + channel_index

    publish_four_channel_frames(10.0)  # connected preview, before any Trial

    window.start_trial()
    worker = created[0]
    publish_four_channel_frames(20.0)  # recording gate open

    window.request_controlled_stop()
    publish_four_channel_frames(30.0)  # gate closed, finalization still running

    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.COMPLETED,
            payload={"state": "FINALIZED"},
        )
    )
    worker.finish(0)
    window.poll_worker_events()
    publish_four_channel_frames(40.0)  # finalized, same worker remains READY

    assert id(window._preview_workers["ultrasound"]) == original_identity
    assert "ultrasound" in window._preview_connected_modalities
    assert ultrasound.stop_requests == 0
    assert ultrasound.begin_recording_calls == [str(worker.request.trial_uuid)]
    assert ultrasound.end_recording_calls == [str(worker.request.trial_uuid)]
    window.close()


def test_legacy_one_argument_worker_factory_remains_supported(
    tmp_path: Path,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    handle = FakePreviewHandle("ultrasound")
    calls: list[TrialRunRequest] = []

    def legacy_factory(request: TrialRunRequest) -> FakeCollectorWorker:
        calls.append(request)
        return FakeCollectorWorker(request)

    window._worker_factory = legacy_factory
    request = window.build_request()
    worker = window._create_recording_worker(
        request,
        {"ultrasound": handle.recording_endpoint},
    )

    assert calls == [request]
    assert worker.request is request
    window.close()


def test_close_finalizes_recording_before_shutting_down_preview_workers(
    tmp_path: Path,
) -> None:
    app, window, created = _window_with_fake(tmp_path)
    handles = _connect_all_previews_for_trial(window)
    window.start_trial()
    worker = created[0]

    window.close()

    assert window.worker is worker
    assert worker.stop_requests == 1
    assert all(len(handle.end_recording_calls) == 1 for handle in handles.values())
    assert all(handle.stop_requests == 0 for handle in handles.values())

    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.COMPLETED,
            payload={"state": "FINALIZED"},
        )
    )
    worker.finish(0)
    window.poll_worker_events()
    app.processEvents()
    assert window.worker is None
    assert all(handle.stop_requests == 1 for handle in handles.values())
    assert not window._preview_workers


def test_encoder_port_list_excludes_bluetooth_virtual_ports(
    monkeypatch: object,
) -> None:
    import serial.tools.list_ports

    monkeypatch.setattr(  # type: ignore[attr-defined]
        serial.tools.list_ports,
        "comports",
        lambda: [
            SimpleNamespace(
                device="COM7",
                description="蓝牙链接上的标准串行",
                hwid="BTHENUM\\fake",
            ),
            SimpleNamespace(
                device="COM12",
                description="USB Serial Device",
                hwid="USB VID:PID=16C0:0483",
            ),
        ],
    )

    assert enumerate_serial_ports() == [("COM12", "USB Serial Device")]


def test_recording_branch_fault_fails_trial_but_keeps_preview_alive(
    tmp_path: Path,
) -> None:
    _app, window, created = _window_with_fake(tmp_path)
    handles = _connect_all_previews_for_trial(window)
    identities = {name: id(handle) for name, handle in handles.items()}
    window.start_trial()
    worker = created[0]
    trial_uuid = str(worker.request.trial_uuid)
    ultrasound = handles["ultrasound"]

    ultrasound.events.extend(
        [
            WorkerEvent(
                event_type=WorkerEventType.FAILED,
                modality="ultrasound",
                trial_uuid=trial_uuid,
                payload={
                    "state": "FAULT",
                    "trial_uuid": trial_uuid,
                    "fault": "recording queue full for ultrasound",
                },
                message="recording queue full for ultrasound",
            ),
            WorkerEvent(
                event_type=WorkerEventType.PREVIEW,
                modality="ultrasound",
                payload={
                    "channel_index": 2,
                    "values": [77.0] * 1000,
                },
            ),
        ]
    )
    window._poll_preview_workers()

    assert window.overall_status == "失败"
    assert window.configuration_locked
    assert worker.stop_requests == 1
    assert all(len(handle.end_recording_calls) == 1 for handle in handles.values())
    assert all(handle.stop_requests == 0 for handle in handles.values())
    assert not window._preview_disconnect_deadlines
    assert window._preview_connected_modalities == set(MODALITIES)
    assert {
        name: id(window._preview_workers[name]) for name in MODALITIES
    } == identities
    _x, channel_three = window._us_curves[2].getData()
    assert channel_three is not None
    assert channel_three[0] == 77.0

    # A late COMPLETED event cannot erase the latched lossless recording fault.
    worker.events.append(
        WorkerEvent(
            event_type=WorkerEventType.COMPLETED,
            trial_uuid=trial_uuid,
            payload={"state": "FINALIZED", "trial_uuid": trial_uuid},
        )
    )
    worker.finish(0)
    window.poll_worker_events()

    assert window.worker is None
    assert window.overall_status == "失败"
    assert not window.configuration_locked
    assert window.start_button.isEnabled()
    assert window._preview_connected_modalities == set(MODALITIES)
    assert {
        name: id(window._preview_workers[name]) for name in MODALITIES
    } == identities
    window.close()


def test_true_preview_worker_failure_still_disconnects_that_modality(
    tmp_path: Path,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    handle = FakePreviewHandle("ultrasound")
    window._preview_workers["ultrasound"] = handle
    window._preview_connected_modalities.add("ultrasound")

    window._handle_preview_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.FAILED,
            modality="ultrasound",
            payload={"state": "FAILED", "traceback": "adapter crashed"},
            message="Preview worker failed",
        ),
        handle,
        "ultrasound",
    )

    assert handle.stop_requests == 1
    assert "ultrasound" not in window._preview_connected_modalities
    assert "ultrasound" in window._preview_disconnect_deadlines
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


def test_device_connection_rows_omit_source_device_id_column(tmp_path: Path) -> None:
    """Long hardware identifiers stay out of the compact connection grid."""
    _app, window, _created = _window_with_fake(tmp_path)
    for modality in ("ultrasound", "imu", "encoder"):
        assert window.findChild(QWidget, f"device_label_{modality}") is None
    assert not hasattr(window, "_connect_device_labels")

    window._set_preview_status(
        "ultrasound",
        "READY",
        r"ultrasound_raw_ethernet_very_long_device_identifier",
        False,
    )
    status = window._connect_status_labels["ultrasound"]
    assert status.text() == ""
    assert status.property("indicatorState") == "green"
    assert "状态：READY" in status.toolTip()
    assert "very_long_device_identifier" in status.toolTip()

    window._set_preview_status("ultrasound", "连接中", "device", False)
    assert status.property("indicatorState") == "yellow"
    window._set_preview_status("ultrasound", "错误", "device", False, error="boom")
    assert status.property("indicatorState") == "red"
    assert "详情：boom" in status.toolTip()
    window.close()


def test_health_table_has_four_modalities(tmp_path: Path) -> None:
    """Compact table uses Chinese labels and embeds the sync indicator."""
    _app, window, _created = _window_with_fake(tmp_path)
    assert window.health_table.rowCount() == 4
    assert window.health_table.columnCount() == 5
    assert [
        window.health_table.horizontalHeaderItem(column).text()
        for column in range(window.health_table.columnCount())
    ] == ["模态", "样本/帧", "实际速率", "丢包", "同步"]
    modalities_displayed = set()
    for row in range(window.health_table.rowCount()):
        modalities_displayed.add(window.health_table.item(row, 0).text())
    assert modalities_displayed == {"超声", "IMU", "电机编码器", "同步脉冲"}
    assert window.sync_status_label.property("indicatorState") == "yellow"
    assert "等待同步信号" in window.sync_status_label.toolTip()
    assert window.health_table.cellWidget(window._health_rows["sync_pulse"], 4) is not None
    assert not hasattr(window, "manifest_label")
    assert not hasattr(window, "open_log_dir_button")
    window.close()


# ── Hardware profile selection tests ──


def test_save_one_device_setting_merges_and_selects_hardware_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, window, _created = _window_with_fake(tmp_path)
    window._settings.set_device_profile_key("simulated")
    window._settings.set_hardware_device_overrides(
        {"encoder": {"port": "COM8", "baudrate": 1_000_000}}
    )
    assert window._selected_device_profile_key() == "simulated"

    class AcceptedUltrasoundDialog:
        def __init__(self, current: object, parent: object) -> None:
            del current, parent

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

        @property
        def validated_override(self) -> dict[str, object]:
            return {"interface_name": "\\Device\\NPF_TEST", "nominal_rate_hz": 20.0}

    monkeypatch.setitem(
        DEVICE_SETTINGS_DIALOGS,
        "ultrasound",
        AcceptedUltrasoundDialog,  # type: ignore[arg-type]
    )
    window._configure_buttons["ultrasound"].click()

    assert window._selected_device_profile_key() == "hardware"
    restored = window._settings.hardware_device_overrides
    assert restored["ultrasound"]["interface_name"] == "\\Device\\NPF_TEST"
    assert restored["encoder"]["port"] == "COM8"
    window.close()


def test_worker_request_uses_selected_profile(tmp_path: Path) -> None:
    """TrialRunRequest must use the actual selected profile, not fallback to simulated."""
    _app, window, _created = _window_with_fake(tmp_path)
    window._settings.set_hardware_device_overrides(
        {"ultrasound": {"interface_name": "\\Device\\NPF_TEST"}}
    )
    window._settings.set_device_profile_key("hardware")

    request = window.build_request()
    assert request.device_profile_key == "hardware"
    assert (
        request.device_overrides["ultrasound"]["interface_name"]
        == "\\Device\\NPF_TEST"
    )
    window.close()


# ── IMU preview: slot-based labels (two-device scenario) ──────


def test_preview_imu_two_device_labels_skip_empty_slot(tmp_path: Path) -> None:
    """With slots 1+3 enabled, their acc_x traces update independently."""
    _app, window, _created = _window_with_fake(tmp_path)
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="imu",
            payload={
                "host_monotonic_ns": 2_000,
                "labels": ["imu_trunk_acc_x", "imu_right_acc_x"],
                "channels": [[1.0], [2.0]],
                "channel": "acc_x",
            },
        )
    )
    # Only trunk and right traces should update
    assert window._imu_traces["imu_trunk_acc_x"]._buffer[0] == 1.0
    assert window._imu_traces["imu_right_acc_x"]._buffer[0] == 2.0
    # Left trace must stay untouched (still NaN)
    assert np.isnan(window._imu_traces["imu_left_acc_x"]._buffer[0])
    window.close()


def test_preview_imu_streams_two_device_labels_skip_empty_slot(
    tmp_path: Path,
) -> None:
    """Streams payload with slots 1+3: trunk/right updated, left stays NaN."""
    _app, window, _created = _window_with_fake(tmp_path)
    window._handle_worker_event(
        WorkerEvent(
            event_type=WorkerEventType.PREVIEW,
            modality="imu",
            payload={
                "host_monotonic_ns": 3_000,
                "streams": [
                    {"label": "imu_trunk_acc_x", "values": [0.5, 0.6]},
                    {"label": "imu_right_acc_x", "values": [0.7, 0.8]},
                ],
            },
        )
    )
    assert window._imu_traces["imu_trunk_acc_x"]._buffer[0] == 0.5
    assert window._imu_traces["imu_right_acc_x"]._buffer[0] == 0.7
    # Left trace must stay untouched
    assert np.isnan(window._imu_traces["imu_left_acc_x"]._buffer[0])
    window.close()
