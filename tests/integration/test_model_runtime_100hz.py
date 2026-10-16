"""Real RF94 backend and real scheduler with synthetic USB-shaped input only."""
import json
from pathlib import Path
from queue import Queue
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS
from exo_collection.apps.model_runtime.backends import build_model_runtime
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.spec import load_model_spec, ModelSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.domain.events import SampleBatch


@pytest.mark.integration
def test_rf94_real_scheduler_fresh_windows_and_stale_stop(tmp_path):
    app = QApplication.instance() or QApplication([])
    root = Path(__file__).resolve().parents[2]
    deployment = root / "runtime_data/models/imu_right_pelvis_torque_100hz_rf94_20261016"
    values = load_model_spec(deployment / "model_spec.json").model_dump()
    values["imu_tcn"].update(right_device_id="right", pelvis_device_id="pelvis")
    values["assistance"].update(mass_kg=80, gain=.2)
    spec = ModelSpec.model_validate(values)
    runtime = build_model_runtime(spec, base_dir=deployment)
    queue = Queue()
    hub = SubscriptionHub(buffer_capacity=4096)
    hub.register_stream("imu", queue, channels=IMU_CHANNELS, sample_shape=(3, 12),
        rate_hz=100, descriptor={"channels": IMU_CHANNELS, "units": IMU_UNITS,
        "sample_shape": (3, 12), "metadata": {"device_ids": ["left", "right", "pelvis"]}})
    predictions, reports, produced = [], [], []
    output = MockTorqueOutput(torque_limit_nm=18)
    controller = InferenceController(hub, spec, runtime, output,
        recording_callback=predictions.append,
        diagnostic_callback=lambda kind, fields: reports.append((kind, fields)))
    stopped = threading.Event()

    def publish(at_ns, index):
        for slot in range(3):
            data = np.full((1, 3, 12), np.nan, dtype=np.float32)
            data[0, slot] = .1 * np.sin(index / 30 + slot)
            data[0, slot, 2] = 9.81
            queue.put(SampleBatch(device_id="imu", modality="imu", clock_domain="host",
                first_sample_index=index, sample_count=1, sequence_number=index,
                sample_rate_hz=100, host_monotonic_ns=at_ns, data=data))

    def produce():
        index, deadline = 1400, time.perf_counter()
        while not stopped.is_set():
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(min(.001, remaining))
                continue
            at_ns = time.perf_counter_ns()
            publish(at_ns, index)
            produced.append(at_ns)
            index += 1
            deadline += .01

    producer = threading.Thread(target=produce, daemon=True)
    try:
        end_ns = time.perf_counter_ns()
        for index in range(1400):
            publish(end_ns - (1399-index)*10_000_000, index)
        hub.drain()
        producer.start()
        # Building the saturated history takes wall time. Wait for a complete
        # live window instead of hiding that genuine gap in synthetic times.
        warmup_deadline = time.monotonic() + 3
        while not controller.preflight()["ready"] and time.monotonic() < warmup_deadline:
            hub.drain()
            app.processEvents()
            time.sleep(.002)
        assert controller.preflight()["ready"]
        controller.set_enabled(True)  # MockTorqueOutput only; no serial object exists.
        controller.start()
        until = time.monotonic() + 6
        while time.monotonic() < until:
            app.processEvents()
            assert controller.last_error is None
            time.sleep(.002)
        assert len(predictions) > 450  # Detect Windows ~15.6 ms wait throttling.
        grids = np.array([p["source_grid_time_ns"] for p in predictions], dtype=np.int64)
        assert np.all(np.diff(grids) > 0)
        assert np.all(np.diff(grids) % 10_000_000 == 0)
        assert all(tuple(p["model_features_shape"]) == (12, 94) for p in predictions)
        assert all(p["desired_left_nm"] == p["desired_right_nm"] for p in predictions)
        assert all(abs(c.left_nm) <= 18 and c.left_nm == c.right_nm for c in output.history)
        actual_hz = (len(predictions)-1)*1e9/(grids[-1]-grids[0])
        timing = [fields for kind, fields in reports if kind == "inference_timing"]
        assert timing and timing[-1]["configured_rate_hz"] == 100
        stopped.set()
        producer.join(1)
        timeout = time.monotonic() + 1
        while controller.last_error is None and time.monotonic() < timeout:
            app.processEvents()
            time.sleep(.002)
        assert "stale" in (controller.last_error or "")
        assert not output.enabled and output.last.left_nm == output.last.right_nm == 0
        evidence = {"model": deployment.name, "hardware_accessed": False,
            "scenario": "real backend + real scheduler + saturated ring + three sparse 100 Hz inputs",
            "predictions": len(predictions), "effective_prediction_hz": actual_hz,
            "grid_interval_ms_p50_p95": (np.percentile(np.diff(grids), [50,95])/1e6).tolist(),
            "source_hz": (len(produced)-1)*1e9/(produced[-1]-produced[0]),
            "last_rate_report": timing[-1], "stale_stop": controller.last_error,
            "all_features_shape": [12,94], "motor_output": "in-memory mock only"}
        path = root / "runtime_data/online_qa/rf94_100hz_adaptation/runtime_scheduler.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        stopped.set()
        if producer.is_alive():
            producer.join(1)
        assert controller.stop_and_wait(3000)
        runtime.unload()
