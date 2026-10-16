"""Real deployed TCN with a full sparse USB-shaped ring and a live Qt window."""
import json
import threading
import time
from queue import Empty, Queue

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS
from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.apps.model_runtime.diagnostics import Diagnostics
from exo_collection.configuration import SharedAppSettings
from exo_collection.domain.events import SampleBatch


@pytest.mark.parametrize("deployment", [None, "imu_right_pelvis_torque_100hz_rf94_20261016"])
def test_full_ring_three_independent_100hz_imus_keeps_real_tcn_running(tmp_path, monkeypatch, deployment):
    app = QApplication.instance() or QApplication([])
    settings = SharedAppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    settings.set_device_profile_key("simulated")
    monkeypatch.setattr("exo_collection.apps.model_runtime.window.Diagnostics",
                        lambda directory: Diagnostics(tmp_path / "diagnostics"))
    window = ModelRuntimeWindow(settings, smoke_test=True)
    window.resize(1740, 1000)
    window.show()
    class TransientEmptyQueue(Queue):
        hide_once = False
        hidden_reads = 0
        def get_nowait(self):
            if self.hide_once and not self.empty():
                self.hide_once = False
                self.hidden_reads += 1
                raise Empty  # multiprocessing feeder visibility is momentarily empty.
            return super().get_nowait()
    queue = TransientEmptyQueue()
    ids = ("imu_left_leg", "imu_right_leg", "imu_pelvis")
    window._hub.register_stream("imu", queue, channels=IMU_CHANNELS,
        sample_shape=(3, 12), rate_hz=100, descriptor={
            "channels": IMU_CHANNELS, "units": IMU_UNITS, "sample_shape": (3, 12),
            "metadata": {"device_ids": ids, "simulated": True}})
    stopped = threading.Event()
    producer_times = []

    def publish(at_ns, index):
        for slot in range(3):
            data = np.full((1, 3, 12), np.nan, dtype=np.float32)
            data[0, slot] = 0.1 * np.sin(index / 30 + slot)
            data[0, slot, 2] = 9.81
            queue.put(SampleBatch(device_id="imu_usb", modality="imu", clock_domain="host",
                first_sample_index=index, sample_count=1, sequence_number=index,
                sample_rate_hz=100, host_monotonic_ns=at_ns, data=data))

    def produce():
        index = 1400
        next_sample = time.perf_counter()
        while not stopped.is_set():
            remaining = next_sample - time.perf_counter()
            if remaining > 0:
                # Python's high-resolution sleep avoids Windows Event.wait's
                # ~15.6ms quantization of the nominal 10ms source interval.
                time.sleep(min(.001, remaining))
                continue
            at_ns = time.perf_counter_ns()
            publish(at_ns, index)
            producer_times.append(at_ns)
            index += 1
            next_sample += .01

    def pump_until(predicate, seconds=15):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents()
            if predicate():
                return
            time.sleep(.002)
        assert predicate(), window.statusBar().currentMessage()

    producer = threading.Thread(target=produce, daemon=True)
    try:
        if deployment is None:
            window._load_bundled_model()
        else:
            window.load_model_from_path(window._project_root() / "runtime_data/models" / deployment / "model_spec.json")
        pump_until(lambda: window._load_generation is None)
        assert window._spec is not None
        end_ns = time.perf_counter_ns()
        for index in range(1400):
            publish(end_ns - (1399 - index) * 10_000_000, index)
        window._hub.drain()
        producer.start()
        window._controls.mass.setValue(60)
        window._controls.gain.setValue(.2)
        window._controls.cutoff.setValue(3)
        window._controls.limit.setValue(.2)
        pump_until(lambda: window._hub.latest_for_device("imu", ids[1]) is not None)
        # Let the normal GUI pump refresh after the initial synthetic history.
        until = time.monotonic() + 2.2
        while time.monotonic() < until:
            app.processEvents()
            time.sleep(.002)
        window._run_button.click()
        assert window.controller_running, window.statusBar().currentMessage()
        controller = window._controller
        ticks = []
        controller.pipeline_sample.connect(lambda payload: ticks.append(payload["time_ns"]))
        injected = False
        end = time.monotonic() + 6
        while time.monotonic() < end:
            app.processEvents()
            if len(ticks) >= 3 and not injected:
                queue.hide_once = True
                injected = True
            assert window.controller_running, (controller.last_error, window.statusBar().currentMessage())
            time.sleep(.002)
        assert queue.hidden_reads == 1
        assert window._hub.count("imu") == 4096
        assert len(ticks) >= (450 if deployment is not None else 70)
        assert ticks[-1] - ticks[0] > 4_000_000_000
        assert window._model_output_panel._normalized_y
        assert window._model_output_panel._traces["期望辅助右 (Nm)"]["ys"]
        assert window._motor is None and not controller.enabled
        # A genuinely missing selected source must still stop and stay stopped.
        output = window._torque
        # Explicitly arm only the in-memory mock to verify that the fault edge
        # actually removes an enabled/nonzero output. No serial object exists.
        controller.set_enabled(True)
        output.send(.1, .1)
        assert output.enabled and output.last.right_nm != 0
        missing_since_ns = time.perf_counter_ns()
        stopped.set()
        producer.join(1)
        pump_until(lambda: not window.controller_running and window._controller is None, seconds=2)
        assert "stale" in controller.last_error
        assert "新鲜" in window.statusBar().currentMessage()
        assert not output.enabled
        assert output.last.left_nm == output.last.right_nm == 0
        assert controller._stop_event.is_set()
        details = controller.last_fault_details
        fault_delay_s = (details["fault_at_ns"] - missing_since_ns) / 1e9
        assert fault_delay_s <= window._spec.safety.watchdog_s
        assert details["selected_sources"]["right_device_id"] == ids[1]
        assert details["selected_sources"]["max_sample_age_ms"] == 50
        assert "drain_ms" in details["timing"] and "extract_ms" in details["timing"]
        assert window._workspace._stream_states["model_output"] == "error"
        end = time.monotonic() + .3
        while time.monotonic() < end:
            publish(time.perf_counter_ns(), 3000)
            app.processEvents()
            assert not window.controller_running and not output.enabled
            time.sleep(.01)
        source_rate = (len(producer_times) - 1) * 1e9 / (producer_times[-1] - producer_times[0])
        assert 90 <= source_rate <= 110
        evidence = {
            "scenario": "real TorchScript + GUI + saturated 4096 ring + three sparse 100Hz streams",
            "model_deployment": deployment or "imu_right_pelvis_torque_v1_20261010",
            "model_rate_hz": window._spec.features.control_rate_hz,
            "model_window_frames": window._spec.feature_window_samples,
            "injection": "one get_nowait Empty despite queued fresh input after 3 received predictions",
            "max_sample_age_ms": 50,
            "old_implementation": {"result": "failed", "reason": "selected IMU stream is stale",
                "injection_after_received_predictions": 3,
                "source": "pre-fix execution of this same integration scenario"},
            "fixed_implementation": {"result": "passed", "received_predictions": len(ticks),
                "measured_source_hz_per_device": source_rate,
                "measured_prediction_hz": (len(ticks)-1)*1e9/(ticks[-1]-ticks[0]),
                "plotted_span_s": (ticks[-1] - ticks[0]) / 1e9, "transient_empty_reads": queue.hidden_reads,
                "true_source_stop_delay_s": fault_delay_s, "fault": details,
                "fresh_input_after_fault_does_not_restart": True},
            "real_usb_hardware_tested": False,
        }
        artifact = window._project_root() / ("runtime_data/online_qa/rf94_100hz_adaptation/gui_saturated_streams.json"
            if deployment is not None else "runtime_data/online_qa/imu_runtime_queue_visibility_reproduction.json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        stopped.set()
        if producer.is_alive():
            producer.join(1)
        window.close()
        app.processEvents()
