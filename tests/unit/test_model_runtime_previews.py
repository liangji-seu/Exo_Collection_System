"""Online previews share Collector rendering without opening device ports."""

from types import SimpleNamespace
import time

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.apps.collector.signal_previews import (
    ENCODER_PREVIEW_LABELS,
    HoverDetailsPlotWidget,
    SignalPreviewMixin,
)
from exo_collection.apps.collector.window import CollectorWindow
from exo_collection.apps.model_runtime.motor_service import MotorFeedback
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    widget = ModelRuntimeWindow(settings, smoke_test=True)
    yield widget
    widget.close()
    app.processEvents()


def feed(window, modality, **payload):
    window._handle_worker_event(modality, WorkerEvent(
        event_type=WorkerEventType.PREVIEW, modality=modality, payload=payload,
    ))


def test_online_uses_collector_components_and_focus_keeps_stop_bar(window):
    assert ModelRuntimeWindow.feed_signal_preview is CollectorWindow.feed_signal_preview
    assert ModelRuntimeWindow.build_signal_preview_panels is SignalPreviewMixin.build_signal_preview_panels
    assert set(window._panels) == {"ultrasound", "imu", "encoder", "mocap", "emg"}
    assert len(window._us_plots) == 4
    assert len({id(trace.plot) for trace in window._imu_traces.values()}) == 3
    assert len({id(trace.plot) for trace in window._enc_traces.values()}) == 2
    assert isinstance(window._model_output_panel._plot, HoverDetailsPlotWidget)
    assert window._motor is None and not window._handles
    window._set_preview_focus_mode(True)
    assert window._controls_scroll.isHidden()
    assert not window._persistent_command_label.isHidden()
    window._set_preview_focus_mode(False)
    assert not window._controls_scroll.isHidden()


def test_usb_imu_sparse_packets_keep_three_sensor_traces_separate(window):
    feed(window, "imu", channels={
        "imu_left_leg_acc_x": [float("nan")],
        "imu_right_leg_acc_x": [1.0],
        "imu_pelvis_acc_x": [float("nan")],
    })
    feed(window, "imu", channels={"imu_pelvis_acc_x": [2.0, 3.0]})
    assert window._imu_traces["imu_left_leg_acc_x"]._count == 0
    assert window._imu_traces["imu_right_leg_acc_x"]._count == 1
    assert window._imu_traces["imu_pelvis_acc_x"]._count == 2
    assert window._imu_traces["imu_right_leg_acc_x"]._buffer[0] == 1.0
    assert window._preview_y_ranges["imu"] == (-10.0, 10.0)


def test_ultrasound_packet_overwrites_only_its_own_channel(window):
    feed(window, "ultrasound", channel_index=2, channels=[[-127.0, 0.0, 127.0]])
    for index, curve in enumerate(window._us_curves):
        values = curve.getData()[1]
        assert len(values) == 1000
        if index == 2:
            np.testing.assert_allclose(values[:3], [-127.0, 0.0, 127.0])
        else:
            assert np.isnan(values).all()
    assert window._preview_y_ranges["ultrasound"] == (-128.0, 128.0)


def test_emg_channels_and_marker_table_match_collector(window):
    feed(window, "emg", labels=["肌肉 A", "肌肉 B"], unit_serials=["123", "456"],
         channels=[[1.0, 2.0], [3.0, 4.0]])
    assert tuple(window._emg_traces) == ("肌肉 A (123)", "肌肉 B (456)")
    assert all(trace._count == 2 for trace in window._emg_traces.values())
    feed(window, "mocap", marker_names=["RASI", "LASI"],
         latest_xyz_mm=[[1.0, 2.0, 3.0], [4.0, float("nan"), 6.0]])
    assert window._mocap_table.rowCount() == 2
    assert window._mocap_table.item(0, 0).text() == "RASI"
    assert window._mocap_table.item(0, 3).text() == "3.000"
    assert window._mocap_table.item(1, 2).text() == "—"


def test_motor_service_feedback_updates_encoder_without_second_connection(window):
    values = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    feedback = MotorFeedback(time.perf_counter_ns(), 1, 0, 0, *values)
    window._motor = SimpleNamespace(
        feedback=feedback, connected=True, enabled=False, torque_limit_nm=0.0,
        fault=None, history=(), last=None,
    )
    try:
        window._poll_motor()
        window._poll_motor()
        for label, value in zip(ENCODER_PREVIEW_LABELS, values):
            assert window._enc_traces[label]._count == 1
            assert window._enc_traces[label]._buffer[0] == value
        assert not window._handles
    finally:
        window._motor = None
