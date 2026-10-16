"""Replay the observed IPC backlog and input gaps without devices or real time."""
from queue import Queue

import numpy as np
import pytest

from exo_collection.apps.model_runtime import inference, subscription
from exo_collection.apps.model_runtime.imu_tcn import ImuTcnFeatureExtractor
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.spec import ModelSpec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.domain.events import SampleBatch


INCIDENT_CHECK_NS = 1_475_683_219_163_400
RIGHT_LATEST_NS = 1_475_683_139_469_300
PELVIS_LATEST_NS = 1_475_683_139_133_700


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class DelayedQueue(Queue):
    delay_once_ns = 0
    delay_every_ns = 0

    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def get_nowait(self):
        item = super().get_nowait()
        self.clock.now += self.delay_once_ns + self.delay_every_ns
        self.delay_once_ns = 0
        return item


class Runtime:
    def __init__(self):
        self.windows = []

    def predict(self, features):
        self.windows.append(features.copy())
        return np.asarray([.1], dtype=np.float32)


def batch(t_ns, slot, *, invalid=False):
    values = np.full((1, 3, 12), np.nan, dtype=np.float32)
    values[0, slot] = 1
    if invalid:
        values[0, slot, :6] = np.nan
    return SampleBatch(device_id="imu_usb", modality="imu", clock_domain="host",
                       first_sample_index=0, sequence_number=0, sample_count=1,
                       sample_rate_hz=100, host_monotonic_ns=t_ns, data=values)


def setup(monkeypatch, *, allow=False, pending=False):
    clock = Clock(INCIDENT_CHECK_NS - 4_088_000 if pending else INCIDENT_CHECK_NS)
    monkeypatch.setattr(subscription, "perf_counter_ns", clock)
    monkeypatch.setattr(inference, "perf_counter_ns", clock)
    hub = SubscriptionHub(buffer_capacity=4096)
    queue = DelayedQueue(clock)
    channels = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "mx", "my", "mz", "r", "p", "y")
    hub.register_stream("imu", queue, channels=channels, sample_shape=(3, 12), rate_hz=100,
        descriptor={"channels": channels, "sample_shape": (3, 12),
                    "units": ("m/s2",) * 3 + ("rad/s",) * 3 + ("",) * 6,
                    "metadata": {"device_ids": ["left", "right", "pelvis"]}})
    latest = [INCIDENT_CHECK_NS - 64_855_000, RIGHT_LATEST_NS, PELVIS_LATEST_NS]
    if not pending:
        latest = [clock.now - 1_000_000] * 3
    history = [(end - (219-index) * 10_000_000, slot)
               for slot, end in enumerate(latest) for index in range(220)]
    for t_ns, slot in sorted(history):
        queue.put(batch(t_ns, slot))
    hub.drain()
    if pending:
        # Observed fault: first IPC read costs 4.088 ms, leaving 22 queued
        # packets while selected-source cached ages are 79.694/80.030 ms.
        for index in range(7):
            for slot in range(3):
                queue.put(batch(INCIDENT_CHECK_NS - 62_000_000 + index * 10_000_000, slot))
        queue.put(batch(INCIDENT_CHECK_NS - 1_000_000, 0))
        queue.put(batch(INCIDENT_CHECK_NS - 700_000, 1))
        queue.delay_once_ns = 4_088_000
    spec = ModelSpec.model_validate({
        "schema_version": "1.0.0", "model": {"backend": "torchscript", "weights_path": "unused.ts", "input_layout": "btc"},
        "features": {"control_rate_hz": 20, "window_s": 2, "inputs": [{"modality": "imu"}]},
        "output": {"unit": "Nm/kg", "single_output_maps_to": "right_torque"},
        "imu_tcn": {"enabled": True, "right_device_id": "right", "pelvis_device_id": "pelvis"},
        "assistance": {"mass_kg": 60, "gain": .2, "lowpass_hz": 0},
    })
    runtime, output, diagnostics = Runtime(), MockTorqueOutput(5, enable_required=True), []
    controller = InferenceController(hub, spec, runtime, output, clock_ns=clock,
        allow_input_wait_recovery=allow, diagnostic_callback=lambda kind, fields: diagnostics.append((kind, fields)))
    return clock, hub, queue, controller, runtime, output, diagnostics


def test_incident_slow_first_get_catches_up_before_stale_verdict(monkeypatch):
    clock, hub, queue, controller, *_ = setup(monkeypatch, pending=True)
    assert hub.drain(max_per_stream=64, max_duration_per_stream_s=.002) == 1
    assert queue.qsize() == 22
    assert controller._extractor.extract(hub, controller._spec.features).size == 0
    assert controller._extractor.last_diagnostics["right_age_ms"] == pytest.approx(79.6941)
    assert controller._extractor.last_reason == "selected IMU stream is stale"

    clock, hub, queue, controller, runtime, output, _ = setup(monkeypatch, pending=True)
    controller._tick()
    assert len(runtime.windows) == 1
    assert controller._timing["catch_up_items"] == 22
    assert queue.empty() and controller.last_error is None
    assert not controller._stop_event.is_set()
    assert controller._extractor.last_diagnostics["pelvis_age_ms"] < 50
    assert not output.enabled and output.last.right_nm == 0
    # Wall time alone cannot produce another prediction on unchanged sources.
    clock.now += 20_000_000
    controller._tick()
    assert len(runtime.windows) == 1


def test_catch_up_has_per_stream_bounds_and_does_not_starve_other_required_sources(monkeypatch):
    clock = Clock(0)
    monkeypatch.setattr(subscription, "perf_counter_ns", clock)
    hub, queues = SubscriptionHub(), {}
    for modality in ("imu", "emg", "unused"):
        queue = queues[modality] = DelayedQueue(clock)
        queue.delay_every_ns = 4_000_000
        hub.register_stream(modality, queue, channels=("x",), sample_shape=(1,), rate_hz=100)
        for index in range(100):
            queue.put(SampleBatch(device_id=modality, modality=modality, clock_domain="host",
                first_sample_index=index, sequence_number=index, sample_count=1,
                sample_rate_hz=100, host_monotonic_ns=index, data=np.ones((1, 1))))
    consumed = hub.drain(max_per_stream=256, max_duration_per_stream_s=.012,
                         priority_modalities=("imu", "emg"), only_modalities=("imu", "emg"))
    assert consumed == 6 and hub.count("imu") == hub.count("emg") == 3
    assert hub.count("unused") == 0 and queues["unused"].qsize() == 100
    assert clock.now == 24_000_000


def test_display_only_gap_waits_and_recovers_only_after_new_valid_history(monkeypatch):
    clock, hub, queue, controller, runtime, output, events = setup(monkeypatch, allow=True)
    controller._tick()
    consumed_grid = controller._extractor.last_grid_end_ns
    clock.now += 120_000_000
    controller._tick()
    assert controller.waiting_for_inputs and "stale" in controller.input_wait_reason
    assert not controller._stop_event.is_set() and controller.last_error is None
    assert controller._extractor.last_grid_end_ns == consumed_grid
    assert not output.enabled and output.last.right_nm == 0
    controller._tick()
    assert len(runtime.windows) == 1
    # Publish real sparse arrivals after a 120 ms gap. Historical invalid grid
    # points must age out naturally; no padding or stale-frame replay is used.
    for index in range(230):
        clock.now += 10_000_000
        for slot in range(3):
            queue.put(batch(clock.now, slot))
        controller._tick()
        if index == 3:
            assert len(runtime.windows) == 1 and controller.waiting_for_inputs
    assert len(runtime.windows) > 1 and not controller.waiting_for_inputs
    assert controller.last_predict_ns > consumed_grid
    assert [kind for kind, _ in events].count("input_resumed") == 1
    assert not output.enabled and all(not command.enabled for command in output.history)
    before = len(runtime.windows)
    controller._tick()
    assert len(runtime.windows) == before


@pytest.mark.parametrize("edge", ["arm", "attach"])
def test_explicit_output_intent_keeps_stale_fault_latched(monkeypatch, edge):
    clock, hub, queue, controller, runtime, output, events = setup(monkeypatch, allow=True)
    controller._tick()
    if edge == "arm":
        controller.set_enabled(True)
    else:
        output = MockTorqueOutput(5, enable_required=True)
        controller.set_torque_output(output)
    clock.now += 100_000_000
    controller._tick()
    assert controller._stop_event.is_set() and "stale" in controller.last_error
    assert not controller.waiting_for_inputs and not output.enabled
    for slot in range(3):
        queue.put(batch(clock.now, slot))
    before = len(runtime.windows)
    controller._tick()
    assert len(runtime.windows) == before
    with pytest.raises(ValueError, match="stopped"):
        controller.set_enabled(True)


def test_nonfinite_contract_failure_is_not_recoverable_input_wait(monkeypatch):
    clock, hub, queue, controller, runtime, output, events = setup(monkeypatch, allow=True)
    # Put a malformed finite-indexed device sample exactly on the next grid.
    clock.now = (clock.now // 50_000_000 + 1) * 50_000_000
    queue.put(batch(clock.now, 1, invalid=True))
    queue.put(batch(clock.now, 2))
    controller._tick()
    assert controller._stop_event.is_set()
    assert not controller.waiting_for_inputs and not runtime.windows


def test_model_failure_is_not_recoverable_input_wait(monkeypatch):
    clock, hub, queue, controller, runtime, output, events = setup(monkeypatch, allow=True)
    def fail(features):
        raise RuntimeError("model failure")
    runtime.predict = fail
    controller._tick()
    assert controller._stop_event.is_set() and controller.last_error == "model failure"
    assert not controller.waiting_for_inputs
