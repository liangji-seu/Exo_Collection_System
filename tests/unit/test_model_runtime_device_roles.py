"""Input-only connection, explicit output and exclusive encoder ownership."""
import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QPushButton

from exo_collection.apps.model_runtime import window as module
from exo_collection.apps.model_runtime.run_coordinator import RunState
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.configuration import SharedAppSettings


class Handle:
    def __init__(self, factory, **kwargs):
        self.factory = factory
        self.is_alive = False
        self.recording_active = False
        self.recording_endpoint = None
    def start(self): self.is_alive = True
    def request_stop(self): self.is_alive = False
    def join(self, **kwargs): pass
    def terminate(self, **kwargs): self.is_alive = False
    def close(self): self.is_alive = False


class Motor(MockTorqueOutput):
    constructed = 0
    def __init__(self, port, **kwargs):
        assert "encoder" not in self.window._handles
        super().__init__(kwargs["torque_limit_nm"])
        type(self).constructed += 1
        self.connected = False
        self.fault = None
        self.feedback = SimpleNamespace(host_monotonic_ns=time.perf_counter_ns(),
            left_position_rad=0.1, left_velocity_rad_s=0.2, left_torque_nm=0.3,
            right_position_rad=0.4, right_velocity_rad_s=0.5, right_torque_nm=0.6,
            state=0, error=0)
    def connect(self): self.connected = True
    def close(self):
        self.connected = False
        super().close()


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("hardware")
    widget = module.ModelRuntimeWindow(settings, smoke_test=True)
    monkeypatch.setattr(module, "ModalityPreviewProcessHandle", Handle)
    Motor.constructed = 0
    Motor.window = widget
    monkeypatch.setattr("exo_collection.apps.model_runtime.motor_service.SerialTorqueOutput", Motor)
    yield widget
    widget.close()
    app.processEvents()


def test_simplified_controls_and_encoder_input_never_construct_motor(window):
    assert len(window._controls.findChildren(QDoubleSpinBox)) == 5
    for name in ("right_id", "pelvis_id", "sign", "direction_confirmed", "apply_button", "preflight_button"):
        assert not hasattr(window._controls, name)
    window._connect_modality("encoder")
    assert "encoder" in window._handles
    assert window._motor is None and Motor.constructed == 0
    assert window._encoder_input_requested
    assert window.findChild(QPushButton) is not None


def test_explicit_output_handoff_keeps_imu_run_and_unsubscribe_keeps_output(window):
    window.install_demo_spec()
    window._connect_modality("encoder")
    old_handle = window._handles["encoder"]
    # Isolate output ownership from the already-tested inference engine.
    window._run.run_state = RunState.RUNNING
    window._smoke_test = False
    window._controls.port.setCurrentText("FAKE_PORT")
    window._toggle_motor()
    assert Motor.constructed == 1 and window._motor.connected
    assert not old_handle.is_alive and "encoder" not in window._handles
    assert window._run.running
    assert not window._motor.enabled  # Generic/demo model cannot arm hardware.
    assert "encoder" in window._hub.modalities
    assert window._hub.rate_hz("encoder") == pytest.approx(1000 / 33)
    window._poll_motor()
    window._hub.drain()
    assert window._hub.latest("encoder")[0] == window._motor.feedback.host_monotonic_ns
    assert window._hub.latest("encoder")[1].tolist() == pytest.approx([.1, .2, .3, .4, .5, .6])
    motor = window._motor
    window._disconnect_modality("encoder")
    assert "encoder" not in window._hub.modalities
    assert window._motor is motor and motor.connected
    window._connect_modality("encoder")
    assert "encoder" in window._hub.modalities
    motor.set_limit(3)
    motor.set_enabled(True)
    motor.send(1, -2)
    window._model_output_panel.add_command(motor.last)
    assert "-2.000" in window._model_output_panel._values_label.text()
    window._disconnect_motor()
    assert window._motor is None and not motor.connected
    assert "-2.000" not in window._model_output_panel._values_label.text()
    assert not any(label.startswith("指令") for label in window._model_output_panel._traces)
    assert "encoder" in window._handles


def test_disconnect_all_does_not_reopen_encoder(window):
    window._connect_modality("encoder")
    window._smoke_test = False
    window._controls.port.setCurrentText("FAKE_PORT")
    window._toggle_motor()
    window.disconnect_all()
    assert not window._encoder_input_requested
    assert window._handles == {} and window._motor is None


def test_failed_output_connect_closes_all_without_restoring_encoder(window, monkeypatch):
    window._connect_modality("encoder")
    window._smoke_test = False
    window._controls.port.setCurrentText("FAKE_PORT")
    monkeypatch.setattr(Motor, "connect", lambda self: (_ for _ in ()).throw(OSError("fake connect failure")))
    monkeypatch.setattr(window, "_show_error", lambda *args: None)
    shown = []
    monkeypatch.setattr(window, "_show_fatal_popup", lambda *args: shown.append(args))
    window._toggle_motor()
    assert window._motor is None
    assert not window._handles and not window._encoder_input_requested
    assert window._fatal_latched and len(shown) == 1
