"""Operator workflow with a real model and spawned simulated input, no serial I/O."""
import time
from pathlib import Path
from queue import Empty

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


@pytest.mark.integration
@pytest.mark.parametrize("deployment", [None, "imu_right_pelvis_torque_100hz_rf94_20261016"])
def test_one_run_connects_warms_predicts_and_recovers_display_input_delay(tmp_path, monkeypatch, deployment):
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families() and Path("C:/Windows/Fonts/msyh.ttc").exists():
        font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
        app.setFont(QFont(QFontDatabase.applicationFontFamilies(font_id)[0], 9))
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    window = ModelRuntimeWindow(settings, smoke_test=True)
    errors = []
    monkeypatch.setattr(window, "_show_error", lambda *args: errors.append(args))
    serial_attempts = []

    def forbid_motor(*args, **kwargs):
        serial_attempts.append((args, kwargs))
        raise AssertionError("Run must never construct an output motor")

    monkeypatch.setattr("exo_collection.apps.model_runtime.motor_service.SerialTorqueOutput", forbid_motor)

    def pump_until(predicate, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            assert not errors, errors
            if predicate():
                return
            time.sleep(.003)
        assert predicate(), window.statusBar().currentMessage()

    class TemporarilyHiddenQueue:
        """The worker keeps producing while the subscriber temporarily sees Empty."""
        def __init__(self, queue):
            self.queue = queue
            self.until = time.perf_counter() + .18
            self.hidden_reads = 0

        def get_nowait(self):
            if time.perf_counter() < self.until:
                self.hidden_reads += 1
                raise Empty
            return self.queue.get_nowait()

        def qsize(self):
            return self.queue.qsize()

    try:
        window.resize(1740, 1000)
        window.show()
        window._controls.mass.setValue(80)
        window._controls.gain.setValue(.4)
        window._controls.limit.setValue(12.5)
        assert not window._handles
        if deployment is None:
            window._load_bundled_model()
        else:
            window.load_model_from_path(window._project_root() / "runtime_data/models" / deployment / "model_spec.json")
        # One user click is retained even while Torch is loading in the background.
        assert window._run_button.isEnabled()
        window._run_button.click()
        pump_until(lambda: window.controller_running and len(window._model_output_panel._normalized_y) >= 6)
        assert set(window._handles) == {"imu"}
        assert window._registered == {"imu"}
        handle = window._handles["imu"]
        controller = window._controller
        assert handle.is_alive and window._spec.model.backend == "torchscript"
        if deployment is not None:
            assert window._spec.features.control_rate_hz == 100
            assert window._spec.feature_window_samples == 94
        assert window._model_output_panel._traces["期望辅助右 (Nm)"]["ys"]
        assert window._spec.assistance.motor_distribution == "right_to_both_same_sign"
        assert window._model_output_panel._desired_left_nm == window._model_output_panel._desired_right_nm
        assert not window._model_output_panel._raw_left_curve.isVisible()
        assert window._motor is None and not controller.enabled and not serial_attempts

        # Exercise the whole GUI watchdog path during a real input queue delay,
        # not just the extractor in isolation. No old prediction is replayed.
        with window._hub._lock:
            stream = window._hub._streams["imu"]
            delayed = TemporarilyHiddenQueue(stream.queue)
            stream.queue = delayed
        pump_until(lambda: controller.waiting_for_inputs, timeout=2)
        assert window.controller_running and not controller.enabled
        previous_grid = controller.last_predict_ns
        pump_until(lambda: not controller.waiting_for_inputs
                   and controller.last_predict_ns > previous_grid, timeout=5)
        assert delayed.hidden_reads > 0
        assert window._controller is controller and window._handles["imu"] is handle
        assert controller.last_error is None and not serial_attempts

        # Applying operator values must preserve the deployment's bilateral
        # routing. This exercises the actual Stop/Run path after an edit.
        window._controls.gain.setValue(.2)
        assert not window.controller_running
        window._run_button.click()
        pump_until(lambda: window.controller_running and len(window._model_output_panel._normalized_y) >= 3)
        assert window._spec.assistance.motor_distribution == "right_to_both_same_sign"
        assert window._spec.assistance.gain == .2
        assert window._model_output_panel._desired_left_nm == window._model_output_panel._desired_right_nm
        assert window._motor is None and not serial_attempts

        # Stop cancels the run intent; continued worker input cannot restart it.
        window.emergency_stop()
        deadline = time.monotonic() + .35
        while time.monotonic() < deadline:
            app.processEvents()
            assert not window.controller_running and window._controller is None
            time.sleep(.003)
        assert handle.is_alive and not serial_attempts and not errors
    finally:
        window.close()
        app.processEvents()
