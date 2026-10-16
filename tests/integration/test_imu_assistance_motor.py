"""Real exported TCN through assistance and the serial protocol, using fake I/O."""
from pathlib import Path
from queue import Queue
import struct
import threading
import time

import numpy as np
import pytest

from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS
from exo_collection.apps.model_runtime.backends import build_model_runtime
from exo_collection.apps.model_runtime.imu_tcn import ImuTcnFeatureExtractor
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.motor_service import SerialTorqueOutput
from exo_collection.apps.model_runtime.spec import load_model_spec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import calc_crc8
from exo_collection.apps.model_runtime.torque import MockTorqueOutput
from exo_collection.domain.events import SampleBatch


class Wire:
    def __init__(self):
        self.frames = []
    def write(self, data):
        self.frames.append(bytes(data))
        return len(data)
    def close(self):
        pass


def status(state):
    payload = struct.pack("<HBBffffffI", 10, state, 0, 0, 0, 0, 0, 0, 0, 1234)
    return b"\xcc" + payload + bytes([calc_crc8(payload), 0x55])


@pytest.mark.parametrize("right_sign", [1, -1])
@pytest.mark.parametrize("final_edge", ["inflight_estop", "stale", "stop"])
def test_tcn_serial_limit_and_inflight_estop(right_sign, final_edge):
    bundle = Path(__file__).resolve().parents[2] / "runtime_data/models/imu_right_pelvis_torque_v1_20261010"
    spec = load_model_spec(bundle / "model_spec.json")
    spec.imu_tcn.right_device_id, spec.imu_tcn.pelvis_device_id = "R", "P"
    spec.assistance.mass_kg, spec.assistance.gain = 60, 1
    spec.assistance.lowpass_hz = 0
    spec.assistance.right_sign = right_sign
    assert spec.assistance.motor_distribution == "right_to_both_same_sign"
    spec.output.torque_limit_nm = 0.1
    runtime = build_model_runtime(spec, base_dir=bundle)
    now = [time.perf_counter_ns()]
    queue, hub = Queue(), SubscriptionHub(buffer_capacity=1024)
    hub.register_stream("imu", queue, channels=IMU_CHANNELS, sample_shape=(2, 12), rate_hz=100,
        descriptor={"channels": IMU_CHANNELS, "units": IMU_UNITS, "sample_shape": (2, 12),
                    "metadata": {"device_ids": ["R", "P"], "simulated": False}})
    # Interleaved USB packets: the other device slot is NaN, never a made-up sample.
    def sample(t, slot):
        data = np.full((1, 2, 12), np.nan, dtype=np.float32)
        data[0, slot] = 0
        data[0, slot, 2] = 9.81
        queue.put(SampleBatch(device_id="imu", modality="imu", clock_domain="host",
            first_sample_index=0, sample_count=1, sequence_number=0,
            sample_rate_hz=100, host_monotonic_ns=t, data=data))
    for i in range(210):
        t = now[0] - (209-i)*10_000_000
        sample(t, 0)
        sample(t+1000, 1)
    hub.drain()
    wire = Wire()
    motor = SerialTorqueOutput("FAKE", 0.1, serial_factory=lambda **kwargs: wire,
                              clock=lambda: now[0], background=False)
    motor.connect()
    motor.feed_feedback(status(0))
    extractor = ImuTcnFeatureExtractor(spec.imu_tcn, clock_ns=lambda: now[0])
    controller = InferenceController(hub, spec, runtime, motor, extractor=extractor, clock_ns=lambda: now[0])
    samples = []
    controller.pipeline_sample.connect(samples.append)
    try:
        controller.set_enabled(True)
        assert not motor.enabled  # The enable command is still awaiting confirmation.
        now[0] += 1_000_000
        motor.feed_feedback(status(2))
        assert motor.enabled
        controller._tick()
        sent = [struct.unpack("<H3f", frame[1:15]) for frame in wire.frames if frame[0] == 0xAA]
        assert any(abs(right) > 0 for _, _, right, _ in sent)
        assert all(left == right and abs(left) <= 0.1 and abs(right) <= 0.1 and limit <= 0.1
                   for _, left, right, limit in sent)
        assert samples[-1]["desired_left_nm"] == samples[-1]["desired_right_nm"]
        assert samples[-1]["raw_estimate_left"] == samples[-1]["estimate_left_nm"] == 0
        assert samples[-1]["desired_right_nm"] == pytest.approx(samples[-1]["raw_estimate_right"] * 60 * right_sign)
        assert abs(samples[-1]["desired_right_nm"]) > .1
        assert samples[-1]["motor_distribution"] == "right_to_both_same_sign"
        assert sent[-1][1] == sent[-1][2] == np.copysign(
            sent[-1][3], samples[-1]["desired_right_nm"])

        if final_edge in {"stale", "stop"}:
            if final_edge == "stale":
                now[0] += 60_000_000
                controller._tick()
                assert "stale" in controller.last_error
            else:
                controller.request_stop()
            assert not controller.enabled and not motor.enabled
            assert motor.last.left_nm == motor.last.right_nm == 0
            boundary = len(wire.frames)
            now[0] += 10_000_000
            sample(now[0], 0)
            sample(now[0], 1)
            controller._tick()
            assert not motor.enabled
            assert not any(struct.unpack("<H3f", frame[1:15])[1:3] != (0, 0)
                           for frame in wire.frames[boundary:] if frame[0] == 0xAA)
            with pytest.raises(ValueError, match="stopped"):
                controller.set_enabled(True)
            return

        entered, release = threading.Event(), threading.Event()
        class BlockingRuntime:
            def predict(self, features):
                entered.set()
                assert release.wait(2)
                return np.array([1.0], dtype=np.float32)
        controller._runtime = BlockingRuntime()
        # Maintain the advertised 100 Hz input between model ticks. Jumping
        # straight by 50 ms leaves a 51 ms source gap after enable, so depending
        # on grid phase the freshness gate can reject before predict() enters.
        for _ in range(5):
            now[0] += 10_000_000
            sample(now[0], 0)
            sample(now[0], 1)
        hub.drain()
        worker = threading.Thread(target=controller._tick)
        worker.start()
        assert entered.wait(2)
        controller.set_enabled(False)
        boundary = len(wire.frames)
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        for frame in wire.frames[boundary:]:
            if frame[0] == 0xAA:
                _, left, right, _ = struct.unpack("<H3f", frame[1:15])
                assert left == right == 0
        assert not motor.enabled
    finally:
        controller.set_enabled(False)
        motor.close()


def test_real_tcn_predicts_without_mass_while_assistance_remains_unavailable():
    bundle = Path(__file__).resolve().parents[2] / "runtime_data/models/imu_right_pelvis_torque_v1_20261010"
    spec = load_model_spec(bundle / "model_spec.json")
    spec.imu_tcn.right_device_id, spec.imu_tcn.pelvis_device_id = "R", "P"
    spec.assistance.mass_kg = 0
    runtime = build_model_runtime(spec, base_dir=bundle)
    now = [time.perf_counter_ns()]
    queue, hub = Queue(), SubscriptionHub(buffer_capacity=1024)
    hub.register_stream("imu", queue, channels=IMU_CHANNELS, sample_shape=(2, 12), rate_hz=100,
        descriptor={"channels": IMU_CHANNELS, "units": IMU_UNITS, "sample_shape": (2, 12),
                    "metadata": {"device_ids": ["R", "P"], "simulated": False}})
    for i in range(210):
        for slot in (0, 1):
            data = np.full((1, 2, 12), np.nan, dtype=np.float32)
            data[0, slot] = 0
            data[0, slot, 2] = 9.81
            queue.put(SampleBatch(device_id="imu", modality="imu", clock_domain="host",
                first_sample_index=0, sample_count=1, sequence_number=i,
                sample_rate_hz=100, host_monotonic_ns=now[0]-(209-i)*10_000_000, data=data))
    hub.drain()
    extractor = ImuTcnFeatureExtractor(spec.imu_tcn, clock_ns=lambda: now[0])
    sink = MockTorqueOutput(5, enable_required=False)
    controller = InferenceController(hub, spec, runtime, sink, extractor=extractor, clock_ns=lambda: now[0])
    samples, predictions = [], []
    controller.pipeline_sample.connect(samples.append)
    controller.prediction.connect(predictions.append)
    assert controller.preflight()["ready"], controller.preflight()
    assert not controller.assistance_preflight()["ready"]
    with pytest.raises(ValueError, match="体重"):
        controller.set_enabled(True)
    controller._tick()
    assert controller.last_error is None
    assert len(samples) == len(predictions) == 1
    assert np.isfinite(samples[0]["estimate_nm_per_kg"])
    assert samples[0]["estimate_nm"] is None
    assert samples[0]["desired_right_nm"] is None
    assert samples[0]["estimate_left_nm"] == 0
    assert samples[0]["desired_left_nm"] is None
    assert samples[0]["motor_distribution"] == "right_to_both_same_sign"
    assert predictions[0][1] == 0 and predictions[0][2] is None
    assert sink.last.left_nm == sink.last.right_nm == 0
    assert not sink.enabled
