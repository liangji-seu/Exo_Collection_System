"""Explicit Run waits asynchronously for all declared inputs and is cancellable."""
import json
import threading
import time
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.apps.model_runtime import window as module, model_loader
from exo_collection.apps.model_runtime.backends import build_model_runtime
from exo_collection.apps.model_runtime.spec import ModelSpec
from exo_collection.apps.model_runtime.startup import StartupCoordinator, StartupPhase
from exo_collection.configuration import SharedAppSettings
from exo_collection.domain.events import SampleBatch


class Handle:
    created = []
    def __init__(self, factory, *, modality, **kwargs):
        self.modality = modality
        self.is_alive = False
        self.recording_active = False
        self.recording_endpoint = None
        self.created.append(self)
    def start(self): self.is_alive = True
    def poll_events(self, **kwargs): return []
    def begin_recording(self, trial): self.recording_active = True
    def end_recording(self, trial): self.recording_active = False
    def request_stop(self): self.is_alive = False
    def join(self, **kwargs): pass
    def terminate(self, **kwargs): self.is_alive = False
    def close(self): self.is_alive = False


@pytest.fixture
def widget(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    Handle.created = []
    monkeypatch.setattr(module, "ModalityPreviewProcessHandle", Handle)
    def forbidden(*args, **kwargs):
        raise AssertionError("One-click Run may connect INPUT only")
    monkeypatch.setattr("exo_collection.apps.model_runtime.motor_service.SerialTorqueOutput", forbidden)
    window = module.ModelRuntimeWindow(settings, smoke_test=True)
    errors = []
    monkeypatch.setattr(window, "_show_error", lambda *args: errors.append(args))
    yield app, window, errors
    window.close()
    app.processEvents()


def spec(window_s=.2):
    return ModelSpec.model_validate({"model": {"backend": "demo"},
        "features": {"control_rate_hz": 20, "window_s": window_s,
                     "inputs": [{"modality": "imu"}, {"modality": "encoder"}]},
        "output": {"torque_limit_nm": 0},
        "safety": {"watchdog_s": .5}})


def install(window, window_s=.2):
    model = spec(window_s)
    assert window._install_runtime(model, base_dir=None)
    window._model_template_spec = model
    window._controls.load_spec(model)


def endpoint(handle):
    queue = Queue()
    handle.recording_endpoint = SimpleNamespace(queue=queue, device_id=handle.modality,
        descriptor={"channels": ["x"], "sample_shape": (1,), "nominal_rate_hz": 100},
        configuration_snapshot={})
    return queue


def warm(queue, modality, seconds):
    count = int((seconds + .1) * 100) + 1
    queue.put(SampleBatch(device_id=modality, modality=modality, clock_domain="host",
        first_sample_index=0, sample_count=count, sequence_number=0, sample_rate_hz=100,
        host_monotonic_ns=time.perf_counter_ns() - (count - 1) * 10_000_000,
        data=np.ones((count, 1), dtype=np.float32)))


def test_intent_generation_deadlines_are_stable_and_cancelled_work_cannot_finish():
    now = [0.]
    state = StartupCoordinator(clock=lambda: now[0])
    generation = state.begin()
    assert state.begin() == generation
    state.wait(generation, StartupPhase.CONNECTING, timeout_s=3)
    now[0] = 2
    state.wait(generation, StartupPhase.CONNECTING, timeout_s=3)
    assert state.deadline == 3
    now[0] = 4
    assert state.expired
    state.cancel("stop")
    assert not state.finish(generation)
    replacement = state.begin()
    assert replacement != generation
    assert not state.wait(generation, StartupPhase.WARMING, timeout_s=2)
    assert state.finish(replacement)


@pytest.mark.parametrize("window_s", [.1, .4, 1.2])
def test_run_connects_missing_multimodal_inputs_once_then_waits_for_each_full_window(widget, window_s):
    app, window, errors = widget
    install(window, window_s)
    window._run_prediction()
    generation = window._startup.generation
    assert {handle.modality for handle in Handle.created} == {"imu", "encoder"}
    assert window._startup.phase is StartupPhase.CONNECTING
    window._run_prediction()
    assert len(Handle.created) == 2 and window._startup.generation == generation
    queues = {handle.modality: endpoint(handle) for handle in Handle.created}
    window._poll_previews()
    assert window._startup.phase is StartupPhase.WARMING
    assert not window.controller_running and not errors
    assert "预热" in window._model_status.text()
    for modality, queue in queues.items():
        warm(queue, modality, window_s)
    window._startup_probe_after = 0
    window._poll_previews()
    assert window.controller_running
    assert not window._startup.active and not errors
    assert window._motor is None
    assert all(handle.recording_active for handle in Handle.created)


def test_existing_input_stream_is_reused_without_reconnecting(widget):
    app, window, errors = widget
    install(window)
    window._hub.register_stream("imu", Queue(), channels=("x",), sample_shape=(1,), rate_hz=100)
    window._run_prediction()
    assert [handle.modality for handle in Handle.created] == ["encoder"]
    assert not errors


@pytest.mark.parametrize("action", ["stop", "disconnect", "parameter", "model", "close"])
def test_pending_run_is_cancelled_and_late_input_cannot_start_it(widget, action):
    app, window, errors = widget
    install(window)
    window._run_prediction()
    generation = window._startup.generation
    if action == "stop": window.emergency_stop()
    elif action == "disconnect": window._disconnect_modality("imu")
    elif action == "parameter": window._controls.mass.setValue(60)
    elif action == "model": window.install_demo_spec()
    else: window.close()
    assert not window._startup.active and not window._startup.finish(generation)
    window._advance_startup()
    assert not window.controller_running


def test_connection_failure_closes_connections_and_warmup_timeout_is_inline(widget, monkeypatch):
    app, window, errors = widget
    install(window)
    popups = []
    monkeypatch.setattr(window, "_show_fatal_popup", lambda *args: popups.append(args))
    monkeypatch.setattr(Handle, "start", lambda self: (_ for _ in ()).throw(ValueError("invalid profile")))
    window._run_prediction()
    assert not window._startup.active and window._fatal_latched
    assert not window._handles and len(popups) == 1
    assert "连接失败" in popups[0][0] and "invalid profile" in popups[0][1]
    monkeypatch.setattr(Handle, "start", lambda self: setattr(self, "is_alive", True))
    now = [0.]
    window._startup = StartupCoordinator(clock=lambda: now[0])
    window._run_prediction()
    now[0] = 31
    window._advance_startup()
    assert not window._startup.active
    assert "超时" in window._fault_label.text() and not errors


@pytest.mark.parametrize("ready_state", ["READY", "RECORDING"])
def test_transient_worker_failure_reconnects_once_without_restarting_prediction(widget, ready_state):
    app, window, errors = widget
    install(window)
    window._run_prediction()
    window._handle_worker_event("imu", WorkerEvent(event_type=WorkerEventType.FAILED,
        modality="imu", message="USB disconnected", payload={}))
    assert not window._startup.active and window._startup.phase is StartupPhase.CANCELLED
    assert window._recovery_pending[0] == "imu" and not window._fatal_latched
    assert "imu" not in window._handles and not errors
    window._on_recovery_timer()
    assert "imu" in window._handles and window._recovery_pending[2] == "wait"
    window._handle_worker_event("imu", WorkerEvent(event_type=WorkerEventType.STATE,
        modality="imu", payload={"state": ready_state}))
    window._on_recovery_timer()
    assert window._recovery_pending is None and not window.controller_running


def test_run_intent_survives_async_model_loading_without_early_connections(widget, tmp_path, monkeypatch):
    app, window, errors = widget
    entered, release = threading.Event(), threading.Event()
    original = model_loader.build_model_runtime
    def blocked(model, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(model, **kwargs)
    monkeypatch.setattr(model_loader, "build_model_runtime", blocked)
    path = tmp_path / "model.json"
    path.write_text(json.dumps(spec().model_dump()), encoding="utf-8")
    window.load_model_from_path(path)
    try:
        assert window._run_button.isEnabled()
        window._run_button.click()
        generation = window._startup.generation
        window._run_prediction()
        assert window._startup.phase is StartupPhase.LOADING
        assert Handle.created == [] and window._startup.generation == generation
        release.set()
        end = time.monotonic() + 5
        while window._load_generation is not None and time.monotonic() < end:
            app.processEvents()
            time.sleep(.002)
        assert window._spec is not None and window._startup.active
        assert {handle.modality for handle in Handle.created} == {"imu", "encoder"}
        assert not window.controller_running and not errors
    finally:
        release.set()
