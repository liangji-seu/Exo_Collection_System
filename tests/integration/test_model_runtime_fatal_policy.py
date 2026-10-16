"""Exercise real motor parsing through Qt fault cleanup using only fake I/O."""
import struct
from queue import Queue
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from exo_collection.apps.model_runtime.motor_service import SerialTorqueOutput
from exo_collection.apps.model_runtime.run_coordinator import OutputState
from exo_collection.apps.model_runtime.torque import calc_crc8
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


class Wire:
    def __init__(self, order):
        self.order, self.frames, self.closed = order, [], False

    def write(self, data):
        self.frames.append(bytes(data))
        self.order.append("wire_write")
        return len(data)

    def close(self):
        self.closed = True
        self.order.append("motor_close")


class Input:
    is_alive = True
    recording_active = False
    recording_endpoint = None

    def __init__(self, order, name):
        self.order, self.name = order, name

    def request_stop(self):
        self.is_alive = False
        self.order.append(self.name + "_stop")

    def join(self, **kwargs):
        pass

    def close(self):
        self.order.append(self.name + "_close")


@pytest.fixture
def window(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = SharedAppSettings(QSettings(str(tmp_path / "ui.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    widget = ModelRuntimeWindow(settings, smoke_test=True)
    widget._preview_timer.stop()
    widget._watchdog_timer.stop()
    yield widget
    widget.close()
    app.processEvents()


def status(state, error=0):
    payload = struct.pack("<HBBffffffI", 3, state, error, 0, 0, 0, 0, 0, 0, 1234)
    return b"\xcc" + payload + bytes([calc_crc8(payload), 0x55])


@pytest.mark.parametrize("error", [253, 254, 3])
def test_real_motor_fault_closes_all_before_popup_and_preserves_actual_reason(window, monkeypatch, error):
    order, popups = [], []
    wire = Wire(order)
    motor = SerialTorqueOutput("FAKE_ONLY", 18, serial_factory=lambda **kw: wire, background=False)
    motor.connect()
    motor.feed_feedback(status(0))
    motor.set_enabled(True)
    motor.feed_feedback(status(2))
    motor.send(2.5, 2.5)
    window._run.motor = window._run.output = motor
    window._run.output_state = OutputState.ENABLED
    window._run.reason = "底层已确认使能"
    for name in ("imu", "emg"):
        window._handles[name] = Input(order, name)
        window._hub.register_stream(name, Queue(), channels=("x",), sample_shape=(1,), rate_hz=100)
        window._registered.add(name)
    window._hub.register_stream("encoder", Queue(), channels=("x",), sample_shape=(1,), rate_hz=100)
    def popup(title, detail):
        assert wire.closed and window._motor is None
        assert not window._handles and not window._hub.modalities
        order.append("popup")
        popups.append((title, detail))
    monkeypatch.setattr(window, "_show_fatal_popup", popup)
    order.clear()
    boundary = len(wire.frames)
    motor.feed_feedback(status(2, error))
    window._poll_motor()
    assert len(popups) == 1 and f"error={error}" in popups[0][1]
    assert "底层已确认使能" not in popups[0][1]
    assert "诊断日志" in popups[0][1]
    assert order[0] == "wire_write" and order[-1] == "popup"
    assert "imu_close" in order and "emg_close" in order and "motor_close" in order
    for frame in wire.frames[boundary:]:
        if frame[0] == 0xAA:
            assert struct.unpack("<H3f", frame[1:15])[1:3] == (0, 0)
        else:
            assert frame[1] == 0xFD
    window._poll_motor()
    assert len(popups) == 1
    assert window._controls.motor_button.text() == "连接电机（保持失能）"


def test_recovery_deadline_closes_remaining_inputs_and_popup_once(window, monkeypatch):
    order = []
    window._handles["emg"] = Input(order, "emg")
    shown = []
    monkeypatch.setattr(window, "_show_fatal_popup", lambda *args: shown.append(args))
    assert window._request_input_recovery("imu", "read failed")
    modality, generation, phase, _ = window._recovery_pending
    window._recovery_pending = (modality, generation, phase, -1)
    window._on_recovery_timer()
    assert window._fatal_latched and not window._handles
    assert "emg_close" in order and len(shown) == 1
    window._on_recovery_timer()
    assert len(shown) == 1


def test_recovery_cancelled_if_motor_attached_before_timer(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_connect_modality", lambda *a, **kw: calls.append(a))
    assert window._request_input_recovery("imu", "read failed")
    window._run.motor = object()
    window._on_recovery_timer()
    assert not calls and window._recovery_pending is None
    window._run.motor = None


def test_fatal_popup_can_open_again_after_acknowledgement(window):
    window._smoke_test = False
    window._show_fatal_popup("第一次", "第一次故障")
    first = window._fault_dialog
    assert first.isVisible()
    first.accept()
    window._show_fatal_popup("第二次", "第二次故障")
    assert window._fault_dialog is not first and window._fault_dialog.isVisible()
    window._fault_dialog.accept()
    window._smoke_test = True


def test_connect_all_does_not_reopen_after_first_fatal_connection(window, monkeypatch):
    calls = []
    window._spec = SimpleNamespace(features=SimpleNamespace(inputs=[
        SimpleNamespace(modality="imu"), SimpleNamespace(modality="emg")]))
    monkeypatch.setattr(window, "_show_fatal_popup", lambda *args: None)
    def connect(modality, **kwargs):
        calls.append(modality)
        window._fatal_incident("连接失败", "bad device", source="test")
        return False
    monkeypatch.setattr(window, "_connect_modality", connect)
    window.connect_all()
    assert calls == ["imu"] and window._fatal_latched and not window._handles


def test_motor_port_open_failure_closes_inputs_without_reopening_encoder(window, monkeypatch):
    order, reopened, shown = [], [], []
    window._handles["imu"] = Input(order, "imu")
    window._encoder_input_requested = True
    window._profile_key, window._smoke_test = "hardware", False
    window._controls.port.addItem("FAKE")
    window._controls.port.setCurrentText("FAKE")
    class FailedMotor:
        def __init__(self, *args, **kwargs):
            pass
        def connect(self):
            raise OSError("port open failed")
        def close(self):
            order.append("motor_close")
    monkeypatch.setattr("exo_collection.apps.model_runtime.motor_service.SerialTorqueOutput", FailedMotor)
    monkeypatch.setattr(window, "_connect_modality", lambda *args, **kwargs: reopened.append(args))
    monkeypatch.setattr(window, "_show_fatal_popup", lambda *args: shown.append(args))
    window._toggle_motor()
    assert window._fatal_latched and not window._handles and not reopened
    assert "motor_close" in order and "imu_close" in order
    assert len(shown) == 1 and "port open failed" in shown[0][1]
    window._smoke_test = True
