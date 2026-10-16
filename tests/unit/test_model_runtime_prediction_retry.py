"""Deterministic replay of the 2026-10-15 source-grid expiry incident."""
import threading
from queue import Queue

import numpy as np
import pytest

from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.spec import ModelSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.domain.events import SampleBatch
from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS


# Exact source timestamps from the saved raw IMU file around the fault.
GRID = 1484370100000000
FAULT = 1484370191827600


def setup_runtime(mode="fresh"):
    clock = [FAULT]
    queue, hub = Queue(), SubscriptionHub(buffer_capacity=4096)
    channels = IMU_CHANNELS
    hub.register_stream("imu", queue, channels=channels, sample_shape=(2, 12), rate_hz=100,
        descriptor={"channels": channels, "units": IMU_UNITS, "sample_shape": (2, 12),
                    "metadata": {"device_ids": ["R", "P"]}})
    index = [0]
    def sample(at, slot):
        data = np.full((1, 2, 12), np.nan, dtype=np.float32)
        data[0, slot] = slot + 1
        queue.put(SampleBatch(device_id="imu", modality="imu", clock_domain="host",
            host_monotonic_ns=at, first_sample_index=index[0], sample_count=1,
            sequence_number=index[0], sample_rate_hz=100, data=data))
        index[0] += 1
    for at in range(GRID - 2_100_000_000, GRID - 20_000_000 + 1, 10_000_000):
        sample(at, 0)
        sample(at, 1)
    for at in (1484370099979900, 1484370100230400, 1484370115115500,
               1484370130697600, 1484370146679000, 1484370163357500):
        sample(at, 0)
    for at in (1484370084377000, 1484370100071600, 1484370115277500,
               1484370130647300, 1484370146830700):
        sample(at, 1)
    hub.drain()
    spec = ModelSpec.model_validate({
        "schema_version": "1.0.0", "model": {"backend": "torchscript", "weights_path": "unused.ts", "input_layout": "btc"},
        "features": {"control_rate_hz": 20, "window_s": 2, "inputs": [{"modality": "imu"}]},
        "imu_tcn": {"enabled": True, "right_device_id": "R", "pelvis_device_id": "P", "max_sample_age_s": .05},
        "output": {"kind": "torque_left_right", "unit": "Nm/kg", "single_output_maps_to": "right_torque", "torque_limit_nm": 5},
        "assistance": {"mass_kg": 1, "gain": 1, "lowpass_hz": 0, "delay_s": 0,
                       "motor_distribution": "right_to_both_same_sign", "right_sign": 1},
    })
    entered, release = threading.Event(), threading.Event()
    output = MockTorqueOutput(5, enable_required=True)
    class Runtime:
        calls = 0
        def predict(self, features):
            self.calls += 1
            if self.calls == 1 and mode != "no_new_grid":
                # New packets were waiting behind the ordinary pump budget.
                sample(1484370179370000, 0)
                sample(1484370179117500, 1)
            if self.calls == 2:
                if mode == "slow":
                    clock[0] += 100_000_000
                elif mode == "inflight_stop":
                    entered.set()
                    assert release.wait(2)
                elif mode == "motor_fault":
                    output.fault = "fake feedback watchdog fault"
                elif mode == "nonfinite":
                    return np.array([np.nan], dtype=np.float32)
            return np.array([self.calls], dtype=np.float32)
    runtime = Runtime()
    diagnostics, recorded, pipelines = [], [], []
    controller = InferenceController(hub, spec, runtime, output, clock_ns=lambda: clock[0],
        recording_callback=recorded.append,
        diagnostic_callback=lambda kind, fields: diagnostics.append((kind, fields)))
    controller.pipeline_sample.connect(pipelines.append)
    controller.set_enabled(True)
    return controller, runtime, output, recorded, pipelines, diagnostics, clock, entered, release


def test_incident_expired_result_is_discarded_and_only_fresh_retry_is_sent_and_recorded():
    controller, runtime, output, recorded, pipelines, diagnostics, *_ = setup_runtime()
    controller._tick()
    assert runtime.calls == 2 and controller.last_error is None
    assert controller.enabled and output.enabled
    assert output.last.left_nm == output.last.right_nm == 2
    assert len(recorded) == len(pipelines) == 1
    assert recorded[0]["raw_output"].tolist() == [2]
    assert recorded[0]["source_grid_time_ns"] == GRID + 50_000_000
    discarded = next(fields for kind, fields in diagnostics if kind == "prediction_discarded")
    assert discarded["source_age_ms"] == pytest.approx(107.4506)
    assert discarded["send_age_limit_ms"] == 100
    assert discarded["current_source_limit_ms"] == 50
    assert discarded["grid_end_ns"] == GRID
    assert [kind for kind, _ in diagnostics].count("prediction_retry") == 1
    accepted = next(fields for kind, fields in diagnostics if kind == "prediction_retry_accepted")
    assert accepted["source_age_ms"] == pytest.approx(45.1486)


@pytest.mark.parametrize("mode,calls", [("no_new_grid", 1), ("slow", 2), ("nonfinite", 2), ("motor_fault", 2)])
def test_failed_retry_is_bounded_and_zeroes_both_sides(mode, calls):
    controller, runtime, output, recorded, pipelines, diagnostics, *_ = setup_runtime(mode)
    controller._tick()
    assert runtime.calls == calls and controller.last_error
    assert controller._stop_event.is_set() and not controller.enabled and not output.enabled
    assert output.last.left_nm == output.last.right_nm == 0
    assert not recorded and not pipelines
    assert sum(kind == "prediction_retry" for kind, _ in diagnostics) <= 1
    controller._tick()
    assert runtime.calls == calls


def test_emergency_stop_during_retry_cannot_send_or_record_late_nonzero():
    controller, runtime, output, recorded, pipelines, _, _, entered, release = setup_runtime("inflight_stop")
    thread = threading.Thread(target=controller._tick)
    thread.start()
    assert entered.wait(2)
    controller.request_stop()
    boundary = len(output.history)
    release.set()
    thread.join(2)
    assert not thread.is_alive() and runtime.calls == 2
    assert not controller.enabled and not output.enabled
    assert not recorded and not pipelines
    assert all(command.left_nm == command.right_nm == 0 for command in output.history[boundary:])


def test_true_stale_selected_input_still_stops_before_prediction():
    controller, runtime, output, recorded, _, _, clock, _, _ = setup_runtime()
    clock[0] += 60_000_000
    controller._tick()
    assert runtime.calls == 0 and "stale" in controller.last_error
    assert not controller.enabled and not output.enabled and not recorded
    assert output.last.left_nm == output.last.right_nm == 0
