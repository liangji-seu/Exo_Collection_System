"""Loading a real deployment must not suspend an already active input stream."""
import json
import threading
import time

import pytest
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

from exo_collection.apps.collector import device_preview
from exo_collection.apps.model_runtime import model_loader
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings


def pump_until(app, predicate, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.003)
    assert predicate(), "condition did not become true"


@pytest.mark.integration
def test_live_imu_keeps_advancing_during_cold_tcn_load_and_queue_pressure(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    window = ModelRuntimeWindow(settings, smoke_test=True)
    # Deliberately small queues fail quickly if the GUI suspends draining.
    monkeypatch.setattr(device_preview, "DEFAULT_PREVIEW_QUEUE_SIZE", 4)
    original_handle = device_preview.ModalityPreviewProcessHandle
    def small_handle(*args, **kwargs):
        kwargs["recording_queue_size"] = 8
        return original_handle(*args, **kwargs)
    monkeypatch.setattr("exo_collection.apps.model_runtime.window.ModalityPreviewProcessHandle", small_handle)
    build = model_loader.build_model_runtime
    built, release = threading.Event(), threading.Event()
    def slow_real_build(*args, **kwargs):
        runtime = build(*args, **kwargs)
        built.set()
        assert release.wait(10), "test did not release model loader"
        return runtime
    monkeypatch.setattr(model_loader, "build_model_runtime", slow_real_build)
    beats = []
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: beats.append(time.monotonic()))
    timer.start()
    try:
        window._connect_modality("imu")
        pump_until(app, lambda: window._hub.count("imu") > 100 and window._preview_counts.get("imu", 0) > 3)
        handle = window._handles["imu"]
        pid = handle._process.pid
        raw_before = window._hub.latest("imu")[0]
        preview_before = window._preview_counts["imu"]
        cursors_before = tuple(trace._cursor for trace in window._imu_traces.values())
        before_beat = len(beats)
        start = time.monotonic()
        window._load_bundled_model()
        assert time.monotonic() - start < 0.25
        assert window._run_button.isEnabled()  # Run can latch an intent while loading.
        pump_until(app, built.is_set)
        # Keep a slow loader blocked while verifying new preview and raw items,
        # rather than inferring liveness from the green connection button.
        pump_until(app, lambda: window._hub.latest("imu")[0] > raw_before + 1_000_000_000
                   and window._preview_counts["imu"] > preview_before + 5
                   and tuple(t._cursor for t in window._imu_traces.values()) != cursors_before)
        assert len(beats) > before_beat + 10
        assert handle.is_alive and handle._process.pid == pid
        assert handle.last_poll_error is None
        assert window._hub.boundary("imu") != "FAULT"
        window._record_progress(time.monotonic())
        release.set()
        pump_until(app, lambda: window._load_generation is None)
        assert window._spec.model.backend == "torchscript"
        assert window._runtime is not None and window._run_button.isEnabled()
        assert not window.controller_running and not window._controller.enabled
        assert window._motor is None
        loaded_raw = window._hub.latest("imu")[0]
        loaded_preview = window._preview_counts["imu"]
        pump_until(app, lambda: window._hub.latest("imu")[0] > loaded_raw
                   and window._preview_counts["imu"] > loaded_preview + 2)
        assert max(b - a for a, b in zip(beats, beats[1:])) < 1.5
        records = [json.loads(line) for line in window._diagnostics.path.read_text(encoding="utf-8").splitlines()]
        # The 1 Hz diagnostic timer can emit a startup heartbeat before the
        # worker has delivered raw samples. Validate the latest observed
        # heartbeat after the explicit progress snapshot above.
        heartbeat = next(row for row in reversed(records) if row["kind"] == "heartbeat")
        assert heartbeat["streams"]["imu"]["last_raw_ns"] > raw_before
        assert any(row["kind"] == "model_ready" for row in records)
    finally:
        release.set()
        timer.stop()
        window.close()
        app.processEvents()
