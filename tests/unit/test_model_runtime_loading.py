"""Cancellation, replacement, failure, and bounded application diagnostics."""
import json
import threading
import time
from queue import Queue

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from exo_collection.apps.model_runtime import model_loader
from exo_collection.apps.model_runtime.backends import build_model_runtime
from exo_collection.apps.model_runtime.diagnostics import Diagnostics
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow

from exo_collection.configuration import SharedAppSettings
from exo_collection.domain.events import SampleBatch
import numpy as np


def wait(app, predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.003)
    assert predicate()


@pytest.fixture
def widget(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    window = ModelRuntimeWindow(settings, smoke_test=True)
    yield app, window
    window.close()
    app.processEvents()


def spec_file(tmp_path, name="model.json", amplitude=1):
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": "1.0.0", "model": {"backend": "demo", "demo": {"amplitude": amplitude}},
        "features": {"control_rate_hz": 20, "window_s": 0.2, "inputs": [{"modality": "imu"}]},
        "output": {"kind": "torque_left_right", "single_output_maps_to": "right_torque", "torque_limit_nm": 0},
        "safety": {"enable_required": True, "zero_on_error": True, "watchdog_s": 0.5}}), encoding="utf-8")
    return path


@pytest.mark.parametrize("action", ["stop", "close", "reload"])
def test_late_loader_result_cannot_restore_cancelled_model(widget, tmp_path, monkeypatch, action):
    app, window = widget
    entered, release = threading.Event(), threading.Event()
    calls = []
    def blocked(spec, **kwargs):
        calls.append(spec.model.demo["amplitude"])
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return build_model_runtime(spec, **kwargs)
    monkeypatch.setattr(model_loader, "build_model_runtime", blocked)
    window.load_model_from_path(spec_file(tmp_path))
    wait(app, entered.is_set)
    try:
        if action == "close":
            window.close()
        elif action == "stop":
            window.emergency_stop()
        else:
            window.load_model_from_path(spec_file(tmp_path, "superseded.json", 2))
            window.load_model_from_path(spec_file(tmp_path, "latest.json", 3))
        release.set()
        if action == "reload":
            wait(app, lambda: window._spec is not None)
            assert calls == [1, 3]
            assert window._spec.model.demo["amplitude"] == 3
            assert window._spec_path.name == "latest.json"
        else:
            wait(app, lambda: window._loader._pending is None)
            time.sleep(0.03)
            app.processEvents()
            assert window._spec is None and window._controller is None
            assert not window._startup.active  # Selected model may be retried by a fresh explicit Run.
        assert not window.controller_running and window._motor is None
    finally:
        release.set()


def test_failed_load_has_traceback_and_keeps_timer_alive(widget, tmp_path):
    app, window = widget
    path = tmp_path / "missing.json"
    window.load_model_from_path(path)
    wait(app, lambda: window._load_generation is None)
    assert window._controller is None and not window._startup.active
    assert window._run_button.isEnabled()  # Retry remains explicit; failure never starts inputs.
    window._poll_previews()
    rows = [json.loads(line) for line in window._diagnostics.path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "startup"
    failure = next(row for row in rows if row["kind"] == "model_load_failed")
    assert "FileNotFoundError" in failure["traceback"]
    assert str(window._diagnostics.path) in window._controls.log_label.toolTip()


def test_apply_parameters_reuses_loaded_weights(widget, tmp_path, monkeypatch):
    app, window = widget
    window.load_model_from_path(spec_file(tmp_path))
    wait(app, lambda: window._load_generation is None)
    loaded = window._runtime
    def forbidden(*args, **kwargs):
        raise AssertionError("Apply must not synchronously reload a model")
    monkeypatch.setattr("exo_collection.apps.model_runtime.window.build_model_runtime", forbidden)
    window._controls.gain.setValue(0.2)
    window._apply_controls()
    assert window._runtime is loaded and window._controller is not None


def test_preview_poll_is_bounded_and_errors_are_aggregated(widget):
    app, window = widget
    queue = Queue()
    window._hub.register_stream("imu", queue, sample_shape=(1,))
    for index in range(1000):
        queue.put(SampleBatch(device_id="imu", modality="imu", clock_domain="host", first_sample_index=index,
            sample_count=1, sequence_number=index, sample_rate_hz=100, host_monotonic_ns=index,
            data=np.array([[index]], dtype=np.float32)))
    window._poll_previews()
    assert window._hub.count("imu") == 256
    class Broken:
        def poll_events(self, **kwargs):
            raise ValueError("broken preview queue")
    window._handles["imu"] = Broken()
    try:
        for _ in range(30):
            window._poll_previews()
        rows = [json.loads(line) for line in window._diagnostics.path.read_text(encoding="utf-8").splitlines()]
        errors = [row for row in rows if row["kind"] == "worker_poll_failed"]
        assert len(errors) == 1 and "ValueError" in errors[0]["traceback"]
    finally:
        window._handles.clear()


def test_diagnostics_rotates_and_preserves_active_files(tmp_path):
    active = Diagnostics(tmp_path, max_bytes=1000, backups=2)
    for _ in range(20):
        old = Diagnostics(tmp_path, max_bytes=1000, backups=2)
        old.close()
    assert active.path.exists()
    for _ in range(30):
        active.record("bounded", message="x" * 150)
    assert len(list(tmp_path.glob(active.path.name + "*"))) <= 3
    assert len(list(tmp_path.glob("online_*.jsonl"))) <= 9
    active.close()


def test_diagnostics_invalid_directory_never_enters_safety_layer(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")
    diagnostic = Diagnostics(blocked)
    assert not diagnostic.available
    diagnostic.record("still_safe", exception=ValueError("test"))
    diagnostic.close()


def test_cancelled_runtime_unload_failure_does_not_kill_loader(widget, tmp_path, monkeypatch):
    app, window = widget
    entered, release = threading.Event(), threading.Event()
    calls = []
    def builder(spec, **kwargs):
        runtime = build_model_runtime(spec, **kwargs)
        calls.append(runtime)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            def bad_unload():
                raise RuntimeError("unload failed")
            runtime.unload = bad_unload
        return runtime
    monkeypatch.setattr(model_loader, "build_model_runtime", builder)
    window.load_model_from_path(spec_file(tmp_path))
    wait(app, entered.is_set)
    window.load_model_from_path(spec_file(tmp_path, "replacement.json", 2))
    release.set()
    wait(app, lambda: window._spec is not None)
    assert window._spec.model.demo["amplitude"] == 2
    rows = [json.loads(line) for line in window._diagnostics.path.read_text(encoding="utf-8").splitlines()]
    assert any(row["kind"] == "model_unload_failed" and "RuntimeError" in row["traceback"] for row in rows)


def test_preview_queue_failure_is_visible_without_changing_poll_contract(monkeypatch):
    from exo_collection.apps.collector.device_preview import ModalityPreviewProcessHandle
    handle = ModalityPreviewProcessHandle(None, device_id="test", modality="imu")
    try:
        def fail():
            raise OSError("queue read failed")
        monkeypatch.setattr(handle._event_queue, "get_nowait", fail)
        assert handle.poll_events() == []
        assert isinstance(handle.last_poll_error, OSError)
    finally:
        handle.close()


def test_diagnostics_writer_failure_is_reported_once_and_does_not_raise(tmp_path, monkeypatch):
    from exo_collection.apps.model_runtime import diagnostics
    messages = []
    monkeypatch.setattr(diagnostics, "_stderr", messages.append)
    diagnostic = Diagnostics(tmp_path)
    def fail(record):
        raise OSError("disk full")
    monkeypatch.setattr(diagnostic._handler, "emit", fail)
    for _ in range(10):
        diagnostic.record("heartbeat")
    assert not diagnostic.available
    assert len(messages) == 1 and "disk full" in messages[0]
    diagnostic.close()


def test_diagnostics_serialization_failure_preserves_event_identity(tmp_path):
    diagnostic = Diagnostics(tmp_path)
    diagnostic.record("bad_numeric", value=float("nan"))
    diagnostic.record("next_good", value=1)
    diagnostic.close()
    rows = [json.loads(line) for line in diagnostic.path.read_text(encoding="utf-8").splitlines()]
    assert any(row["kind"] == "bad_numeric" and "serialization_error" in row for row in rows)
    assert any(row["kind"] == "next_good" for row in rows)
