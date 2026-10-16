"""UI wiring and real TCN -> middle layer -> Mock acceptance, without hardware."""
import os
import time
from queue import Queue

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSettings, QThread
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings
from exo_collection.domain.events import SampleBatch


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase.families():
        from pathlib import Path
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        if font_path.exists():
            QFontDatabase.addApplicationFont(str(font_path))
            app.setFont(QFont("Microsoft YaHei", 9))
    return app


@pytest.fixture
def window(qt_app, tmp_path):
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    widget = ModelRuntimeWindow(settings, smoke_test=True)
    yield widget
    widget.close()
    qt_app.processEvents()


def pump(app, seconds=0.1):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def load_bundled(window):
    window._load_bundled_model()
    deadline = time.perf_counter() + 15
    while window._load_generation is not None and time.perf_counter() < deadline:
        pump(QApplication.instance(), 0.01)
    assert window._spec is not None, window.statusBar().currentMessage()


def test_load_without_devices_stays_unarmed_and_waits(window, qt_app):
    load_bundled(window)
    pump(qt_app)
    assert window._spec is not None
    assert window._spec.model.backend == "torchscript"
    assert not window.controller_running
    assert not window._controller.enabled
    assert not window._preflight(quiet=True)
    assert window._motor is None
    with pytest.raises(ValueError, match="模拟|smoke"):
        window._require_real_inputs()


def test_named_local_model_selector_auto_loads_and_has_no_file_picker(window, monkeypatch):
    from PySide6.QtWidgets import QPushButton
    requests = []
    monkeypatch.setattr(window, "load_model_from_path", requests.append)
    from exo_collection.apps.model_runtime.model_catalog import discover_local_models
    catalog = discover_local_models(window._project_root() / "runtime_data/models")
    assert window._model_choice.count() == 1 + len(catalog.entries)
    v1_index = window._model_choice.findText("右髋估计·双侧同号助力 · IMU-TCN v1")
    assert v1_index > 0
    assert not window._model_choice.model().item(0).isEnabled()
    assert not hasattr(window, "_choose_spec")
    assert not any("浏览" in button.text() for button in window.findChildren(QPushButton))
    window._model_choice.setCurrentIndex(v1_index)
    assert len(requests) == 1 and requests[0].name == "model_spec.json"
    window._show_model_choice(requests[0])
    assert "输入：右腿 + 骨盆 IMU" in window._model_choice.toolTip()
    assert "输出：右髋力矩 (Nm/kg)" in window._model_choice.toolTip()
    assert window._load_button.isHidden() and window._spec_path_edit.isHidden()


def test_programmatic_model_gets_friendly_read_only_display(window, monkeypatch, tmp_path):
    import json
    requests = []
    monkeypatch.setattr(window, "load_model_from_path", requests.append)
    custom = tmp_path / "opaque_spec.json"
    source = window._project_root() / "runtime_data/models/imu_right_pelvis_torque_v1_20261010/model_spec.json"
    custom.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "model_display.json").write_text(json.dumps({"name": "实验右髋估计模型"}), encoding="utf-8")
    window._show_model_choice(custom)
    assert window._model_choice.currentText() == "实验右髋估计模型"
    assert not window._model_choice.model().item(window._model_choice.currentIndex()).isEnabled()
    assert str(custom) in window._model_choice.toolTip()
    assert requests == []


def test_empty_catalog_has_useful_placeholder_and_no_picker(window, monkeypatch):
    from exo_collection.apps.model_runtime.model_catalog import ModelCatalog
    monkeypatch.setattr("exo_collection.apps.model_runtime.window.discover_local_models", lambda directory: ModelCatalog((), ()))
    empty = ModelRuntimeWindow(window._settings, smoke_test=True)
    try:
        assert empty._model_choice.count() == 1
        assert empty._model_choice.currentText() == "暂无可用模型"
        assert "部署目录" in empty._model_choice.toolTip()
        assert not empty._run_button.isEnabled()
    finally:
        empty.close()


def test_parameter_edit_disarms_and_requires_reapply(window, qt_app):
    load_bundled(window)
    controls = window._controls
    controls.mass.setValue(60)
    controls.gain.setValue(0.1)
    controls.limit.setValue(0.5)
    window._apply_controls()
    assert window._applied_signature == controls.signature()
    controls.limit.setValue(0.2)
    assert window._applied_signature is None
    assert window._controller is None
    window._hub.register_stream("imu", Queue(), channels=IMU_CHANNELS, sample_shape=(2, 12), rate_hz=100,
        descriptor={"channels": IMU_CHANNELS, "units": IMU_UNITS, "sample_shape": (2, 12),
                    "metadata": {"device_ids": ["imu_right_leg", "imu_pelvis"], "simulated": True}})
    window._run_prediction()
    assert not controls.enable.isChecked()
    assert window._startup.active
    assert window._controller is not None and not window._controller.enabled


def _batch(t_ns, value=0.5):
    data = np.zeros((1, 2, 12), dtype=np.float32)
    data[:, :, 0] = value
    data[:, :, 2] = 9.81
    return SampleBatch(
        device_id="imu", modality="imu", clock_domain="host", first_sample_index=0,
        sample_count=1, sequence_number=0, sample_rate_hz=100,
        host_monotonic_ns=t_ns, data=data,
    )


def test_real_tcn_mock_pipeline_and_stale_input_shutdown(window, qt_app, monkeypatch):
    window.resize(1740, 1000)
    window.show()
    window._body_splitter.setSizes([430, 1290])
    pump(qt_app, 0.1)
    handler_threads = []
    for name in ("_on_pipeline_sample", "_on_prediction", "_on_command", "_on_state", "_on_fault"):
        original = getattr(window, name)
        def capture(payload, handler=original, handler_name=name):
            handler_threads.append((handler_name, QThread.currentThread()))
            return handler(payload)
        monkeypatch.setattr(window, name, capture)
    load_bundled(window)
    def forbidden_motor(*args, **kwargs):
        raise AssertionError("Input-only Run must never construct a motor service")
    monkeypatch.setattr("exo_collection.apps.model_runtime.motor_service.SerialTorqueOutput", forbidden_motor)
    controls = window._controls
    controls.mass.setValue(60)
    controls.gain.setValue(0.2)
    controls.cutoff.setValue(0)
    controls.limit.setValue(0.1)
    queue = Queue()
    window._hub.register_stream(
        "imu", queue, channels=IMU_CHANNELS, sample_shape=(2, 12), rate_hz=100,
        descriptor={"channels": IMU_CHANNELS, "units": IMU_UNITS, "sample_shape": (2, 12),
                    "metadata": {"device_ids": ["imu_right_leg", "imu_pelvis"], "simulated": True}},
    )
    window._apply_controls()
    now = time.perf_counter_ns()
    for i in range(210):
        queue.put(_batch(now - (209 - i) * 10_000_000))
    deadline = time.perf_counter() + 3
    while time.perf_counter() < deadline:
        queue.put(_batch(time.perf_counter_ns()))
        pump(qt_app, 0.01)
        if window._controller.ready:
            break
    assert window._controller.ready, window._controller.preflight()
    window._run_prediction()
    assert window.controller_running, window.statusBar().currentMessage()
    assert not window._controller.enabled
    output = window._torque
    for index in range(300):
        queue.put(_batch(time.perf_counter_ns(), value=0.5 + 0.2 * np.sin(index / 18)))
        pump(qt_app, 0.01)
    assert window._motor is None
    assert len(window._model_output_panel._normalized_y) > 30
    assert window._model_output_panel._normalized_x[-1] - window._model_output_panel._normalized_x[0] > 2
    assert any(abs(v) > 0 for v in window._model_output_panel._normalized_y)
    assert any(abs(v) > 0 for v in window._model_output_panel._traces["期望辅助右 (Nm)"]["ys"])
    assert all(not cmd.enabled and cmd.right_nm == 0 for cmd in window.torque_history)
    assert "未下发" in window._model_output_panel._values_label.text()
    assert all(cmd.left_nm == 0 and abs(cmd.right_nm) <= 0.1 for cmd in window.torque_history)
    assert {name for name, _ in handler_threads} >= {
        "_on_pipeline_sample", "_on_prediction", "_on_command", "_on_state",
    }
    assert all(thread == qt_app.thread() for _, thread in handler_threads)
    screenshot = window._project_root() / "runtime_data/online_qa/simplified_online_ui.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(screenshot))
    pump(qt_app, 0.4)
    assert window.controller_running and window._controller.waiting_for_inputs
    assert not output.enabled and output.last.right_nm == 0
    for _ in range(240):
        queue.put(_batch(time.perf_counter_ns()))
        pump(qt_app, 0.01)
    assert window.controller_running and not window._controller.waiting_for_inputs
    assert not output.enabled and output.last.right_nm == 0


def test_queued_old_owner_is_discarded_after_new_run(window, qt_app, monkeypatch):
    delivered = []
    # This test isolates queued signal ownership from input readiness.
    monkeypatch.setattr(window, "_preflight", lambda **kwargs: True)
    monkeypatch.setattr(window, "_on_pipeline_sample", lambda payload:
                        delivered.append((payload, QThread.currentThread())))
    window.install_demo_spec()
    window._hub.register_stream("imu", Queue(), channels=("x",), sample_shape=(1,), rate_hz=100)
    window._run_prediction()
    old_owner = window._controller
    old_owner.pipeline_sample.emit({"owner": "old"})
    assert delivered == []  # even emissions from GUI are explicitly queued
    window.emergency_stop()
    window._run_prediction()
    assert window._controller is not old_owner
    window._controller.pipeline_sample.emit({"owner": "current"})
    pump(qt_app)
    assert [payload["owner"] for payload, _ in delivered] == ["current"]
    assert all(thread == qt_app.thread() for _, thread in delivered)


def test_model_reload_preserves_operator_mass(window):
    load_bundled(window)
    assert window._controls.mass.value() == 0
    window._controls.mass.setValue(67.5)
    load_bundled(window)
    assert window._controls.mass.value() == 67.5
    assert not window.controller_running


def test_real_tcn_zero_mass_runs_raw_then_re_run_converts(window, qt_app):
    load_bundled(window)
    controls = window._controls
    controls.gain.setValue(0.2)
    controls.cutoff.setValue(0)
    queue = Queue()
    window._hub.register_stream(
        "imu", queue, channels=IMU_CHANNELS, sample_shape=(2, 12), rate_hz=100,
        descriptor={"channels": IMU_CHANNELS, "units": IMU_UNITS, "sample_shape": (2, 12),
                    "metadata": {"device_ids": ["imu_right_leg", "imu_pelvis"], "simulated": True}},
    )

    def warm_and_run():
        # A fresh controller may warm from the shared stream before Run.
        # This test rewinds a synthetic 2.1-second source history each round;
        # discard the previous round rather than injecting non-monotonic data.
        window._hub.drain()
        window._hub.clear("imu")
        window._apply_controls()
        now = time.perf_counter_ns()
        for index in range(210):
            queue.put(_batch(now - (209 - index) * 10_000_000))
        deadline = time.perf_counter() + 3
        while time.perf_counter() < deadline:
            queue.put(_batch(time.perf_counter_ns()))
            pump(qt_app, 0.01)
            if window._controller.ready:
                break
        assert window._preflight(quiet=True), controls.preflight_label.text()
        window._run_prediction()
        assert window.controller_running, window.statusBar().currentMessage()
        for _ in range(20):
            queue.put(_batch(time.perf_counter_ns()))
            pump(qt_app, 0.01)

    assert controls.mass.value() == 0
    warm_and_run()
    panel = window._model_output_panel
    assert panel._raw_right is not None  # Core raw inference still runs.
    assert not panel._normalized_y  # Nm display waits for actual conversion.
    assert "待填写体重" in panel._values_label.text()
    assert "不可换算" in controls.preflight_label.text()
    assert "预测右 (Nm)" not in panel._traces
    assert "期望辅助右 (Nm)" not in panel._traces
    assert all(not command.enabled and command.right_nm == command.left_nm == 0
               for command in window.torque_history)
    controls.mass.setValue(60)
    assert not window.controller_running
    warm_and_run()
    assert panel._desired_right_nm == pytest.approx(panel._raw_right * 60 * 0.2)
    assert panel._normalized_y[-1] == pytest.approx(panel._raw_right * 60)
    assert panel._traces["期望辅助右 (Nm)"]["ys"]
    assert "待填写体重" not in panel._values_label.text()
    assert all(not command.enabled and command.right_nm == 0 for command in window.torque_history)


def test_generic_nm_model_needs_no_mass_and_uses_nm_labels(window, qt_app):
    window.install_demo_spec()
    window._controls.gain.setValue(0.5)
    window._controls.cutoff.setValue(0)
    queue = Queue()
    window._hub.register_stream("imu", queue, channels=("x",), sample_shape=(1,), rate_hz=100)
    now = time.perf_counter_ns()
    for index in range(30):
        queue.put(SampleBatch(
            device_id="imu", modality="imu", clock_domain="host", first_sample_index=index,
            sample_count=1, sequence_number=index, sample_rate_hz=100,
            host_monotonic_ns=now - (29 - index) * 10_000_000,
            data=np.asarray([[0.5]], dtype=np.float32),
        ))
    window._run_prediction()
    assert window.controller_running, window.statusBar().currentMessage()
    pump(qt_app, 0.08)
    panel = window._model_output_panel
    assert window._controls.mass.value() == 0
    assert window._controller.assistance_preflight()["ready"]
    assert panel._normalized_y
    assert "Nm/kg" not in panel._values_label.text()
    assert "Nm/kg" not in panel._normalized_plot.getPlotItem().getAxis("left").labelText
    assert "Nm/kg" not in panel._normalized_plot.getPlotItem().titleLabel.text
    assert "Nm" in panel._values_label.text()
    assert "40 帧" not in window._controls.preflight_label.text()
    assert all(not command.enabled and command.right_nm == 0 for command in window.torque_history)


def test_unconvertible_middle_layer_blocks_motor_arm(window, monkeypatch):
    load_bundled(window)
    monkeypatch.setattr(window, "_require_real_inputs", lambda: None)
    window._controls.limit.setValue(0.5)
    window._apply_controls()
    with pytest.raises(ValueError, match="不可换算"):
        window._validate_motor_arm()


def test_output_panel_handles_missing_nm_and_bilateral_units(window):
    panel = window._model_output_panel
    panel.add_pipeline({
        "time_ns": 1, "output_unit": "Nm/kg", "raw_estimate_left": 0.1,
        "raw_estimate_right": 0.2, "desired_left_nm": None, "desired_right_nm": None,
        "estimate_nm": None, "estimate_left_nm": None,
        "assistance_ready": False, "assistance_errors": ["待填写体重"],
    })
    panel.add_prediction(1, None, None)
    assert not panel._normalized_y
    assert not panel._traces
    panel.add_pipeline({
        "time_ns": 2, "output_unit": "Nm", "raw_estimate_left": 2,
        "raw_estimate_right": 4, "desired_left_nm": 1, "desired_right_nm": 2,
        "estimate_nm": 4, "estimate_left_nm": 2,
        "assistance_ready": True, "assistance_errors": [],
    })
    assert list(panel._normalized_y) == [4]
    assert list(panel._raw_left_y) == [2]
    assert panel._traces["期望辅助左 (Nm)"]["ys"][-1] == 1
    assert "Nm/kg" not in panel._values_label.text()
