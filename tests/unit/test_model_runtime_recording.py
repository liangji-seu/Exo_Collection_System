"""Recording acceptance tests; all motor traffic goes to an in-memory serial."""
import json
import re
import struct
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from queue import Queue
from queue import Full
from uuid import uuid4

import numpy as np
import pytest

from exo_collection.acquisition.recording_stream import RecordedRawEvent, RecordingBoundary, RecordingBoundaryKind
from exo_collection.apps.model_runtime import recording
from exo_collection.apps.model_runtime.recording import OnlineRecording, read_online_recording
from exo_collection.apps.model_runtime.motor_service import SerialTorqueOutput
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import HEAD_STATUS, FRAME_TAIL, calc_crc8
from exo_collection.domain.events import SampleBatch, FrameBatch
from exo_collection.domain.states import TrialState


class Serial:
    def write(self, data):
        return len(data)
    def close(self):
        pass


def feedback(seq=1, state=2, error=0):
    payload = struct.pack("<HBBffffffI", seq, state, error, 1, 2, 3, -1, -2, -3, 1234)
    return bytes([HEAD_STATUS]) + payload + bytes([calc_crc8(payload), FRAME_TAIL])


def batch(at, *, modality="imu", index=0, count=4, value=7, rate=1000):
    return SampleBatch(device_id=modality + "0", modality=modality, clock_domain="device",
        host_monotonic_ns=at, first_sample_index=index, sample_count=count,
        sequence_number=index, device_timestamp=123.5, sample_rate_hz=rate,
        data=np.full((count, 2), value, dtype=np.int16))


def edge(recorder, enabled, at):
    recorder.motor_event("enabled" if enabled else "disabled", {
        "host_monotonic_ns": at, "reason": "test_disable"})


def test_confirmed_real_motor_enable_is_only_recording_boundary(tmp_path):
    # Deliberately differ from the writer's wall clock: consuming the queue
    # later must not replace the reception UTC derived from this start anchor.
    start_utc = datetime(2020, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
    rec = OnlineRecording(tmp_path, now=lambda: start_utc)
    clock = [1_000_000_000]
    motor = SerialTorqueOutput("FAKE", 5, background=False, clock=lambda: clock[0],
        serial_factory=lambda **kwargs: Serial(), history_capacity=2, recording_sink=rec.motor_event)
    motor.connect()
    motor.feed_feedback(feedback(state=0))
    assert not rec.active and not list(tmp_path.iterdir())
    motor.set_enabled(True)
    assert not rec.active
    motor.feed_feedback(feedback(seq=2))
    assert rec.active
    for seq in range(25):
        clock[0] += 1_000_000
        motor.send(100, -100)
        motor.feed_feedback(feedback(seq=seq + 3))
    clock[0] += 1_000_000
    motor.set_enabled(False)
    assert not rec.active
    motor.send(1, 1)
    motor.feed_feedback(feedback(seq=99))
    motor.close()
    assert rec.close()
    assert len(rec.completed) == 1 and rec.error is None
    result = read_online_recording(rec.completed[0])
    commands = [e for e in result["events"] if e["kind"] == "sent_command"]
    frames = [e for e in result["events"] if e["kind"] == "feedback"]
    assert len(commands) == 26  # 25 sends and explicit safe zero before disable
    assert len(frames) == 26  # confirmation plus 25 subsequent frames
    assert commands[0]["payload"]["left_nm"] == 5
    assert commands[0]["payload"]["right_nm"] == -5
    assert frames[0]["payload"]["device_time"] == 1234
    assert bytes.fromhex(frames[0]["payload"]["raw_frame"]["hex"]) == feedback(seq=2)
    assert result["session"]["start_host_monotonic_ns"] == 1_000_000_000
    encoder = result["streams"]["raw/encoder.h5"]
    assert encoder["data"].shape == (26, 6) and encoder["data"].dtype == np.float32
    np.testing.assert_array_equal(encoder["data"], np.tile([1, 2, 3, -1, -2, -3], (26, 1)))
    assert encoder["host_monotonic_ns"].tolist() == [e["host_monotonic_ns"] for e in frames]
    assert encoder["device_time"].tolist() == [1234.] * 26
    assert encoder["source_sequence"].tolist() == [e["payload"]["seq"] for e in frames]
    assert result["session"]["streams"] == {}  # Feedback needs no post-disable watermark.
    import h5py
    utc_anchor_ns = int(start_utc.timestamp()) * 1_000_000_000 + start_utc.microsecond * 1000
    with h5py.File(rec.completed[0] / "raw/encoder.h5", "r") as file:
        expected_utc = [utc_anchor_ns + e["host_monotonic_ns"] - 1_000_000_000 for e in frames]
        assert file["samples/host_utc_ns"][:].tolist() == expected_utc
        device_metadata = json.loads(file["metadata/device"][()])
        assert "session UTC/monotonic" in device_metadata["metadata"]["host_utc_semantics"]


def test_delayed_raw_tail_crossing_batches_replay_and_session_isolation(tmp_path):
    fixed = datetime(2026, 10, 15, 1, 2, 3, tzinfo=timezone.utc)
    rec = OnlineRecording(tmp_path, now=lambda: fixed)
    rec.configure({"model_sha256": "abc", "motor_distribution": "right_to_both_same_sign"}, {"imu": {}})
    edge(rec, True, 102_000_000)
    rec.raw("imu", batch(100_000_000), {"channels": ["a", "b"], "units": ["x", "x"]})
    rec.prediction({"host_monotonic_ns": 103_000_000, "source_grid_time_ns": 101_000_000,
        "raw_output": np.array([1.2]), "estimate_nm": 60., "desired_left_nm": 3., "desired_right_nm": 3.,
        "model_features": np.ones((12, 40)), "model_features_dtype": "float32", "model_features_shape": [12, 40]})
    edge(rec, False, 106_000_000)
    edge(rec, True, 110_000_000)
    # A late batch for the old interval must be routed only to its old session.
    rec.raw("imu", batch(104_000_000, index=4))
    rec.raw("imu", batch(110_000_000, index=8, value=9))
    edge(rec, False, 112_000_000)
    rec.raw("imu", batch(114_000_000, index=12))
    assert rec.close()
    assert len(rec.completed) == 2 and rec.error is None
    first, second = [read_online_recording(path) for path in rec.completed]
    a, b = first["streams"]["raw/imu.h5"], second["streams"]["raw/imu.h5"]
    assert a["host_monotonic_ns"].tolist() == [102_000_000, 103_000_000, 104_000_000, 105_000_000]
    assert a["sample_index"].tolist() == [2, 3, 4, 5]
    assert a["source_sequence"].tolist() == [0, 0, 4, 4]
    assert a["data"].dtype == np.int16
    assert b["host_monotonic_ns"].tolist() == [110_000_000, 111_000_000]
    assert np.all(b["data"] == 9)
    prediction = next(e for e in first["events"] if e["kind"] == "prediction")
    assert len(prediction["payload"]["model_features"]) == 12
    assert first["timeline"] == sorted(first["timeline"], key=lambda e: e["host_monotonic_ns"])
    assert all(re.fullmatch(r"\d{8}_\d{6}_\d{6}", path.name) for path in rec.completed)
    assert rec.completed[0].name != rec.completed[1].name


def test_hub_pre_ring_tap_preserves_all_modalities_and_binary_dtype(tmp_path):
    rec = OnlineRecording(tmp_path)
    rec.configure(streams={"ultrasound": {}, "emg": {}, "mocap": {}})
    hub = SubscriptionHub(buffer_capacity=1, raw_tap=rec.raw)
    edge(rec, True, 102_000_000)
    for modality in ("ultrasound", "emg", "mocap"):
        queue = Queue()
        hub.register_stream(modality, queue, sample_shape=(2,), rate_hz=1000)
        if modality == "ultrasound":
            event = FrameBatch(device_id="us0", modality=modality, clock_domain="device",
                host_monotonic_ns=100_000_000, first_frame_index=0, frame_count=8,
                sequence_number=0, frame_rate_hz=1000, device_timestamp=31,
                data=np.arange(16, dtype=np.int16).reshape(8, 2))
        else:
            event = batch(100_000_000, modality=modality, count=8)
        queue.put(RecordedRawEvent(str(uuid4()), modality, event.device_id, event))
    hub.drain()
    edge(rec, False, 106_000_000)
    for modality in hub.modalities:
        rec.raw(modality, batch(108_000_000, modality=modality, index=20))
    assert rec.close() and rec.error is None
    result = read_online_recording(rec.completed[0])
    assert set(result["streams"]) == {"raw/ultrasound.bin", "raw/emg.bin", "raw/mocap.h5"}
    assert result["streams"]["raw/ultrasound.bin"]["data"].tolist() == [[4, 5], [6, 7], [8, 9], [10, 11]]
    assert all(len(stream["data"]) == 4 for stream in result["streams"].values())
    assert hub.count("emg") == 1  # Full eight-row batch was saved despite ring eviction.


@pytest.mark.parametrize("failure", ["timeout", "source_fault"])
def test_unverified_or_faulted_tail_is_aborted(tmp_path, failure):
    rec = OnlineRecording(tmp_path, tail_timeout_s=.01)
    rec.configure(streams={"imu": {}})
    edge(rec, True, 100)
    if failure == "source_fault":
        rec.raw("imu", RecordingBoundary(RecordingBoundaryKind.FAULT, str(uuid4()), "imu", "imu0", message="overflow"))
    edge(rec, False, 200)
    assert rec.close()
    result = read_online_recording(rec.completed[0])
    assert result["manifest"].state == TrialState.ABORTED
    assert not result["session"]["complete"] and rec.error


def test_disk_failure_keeps_partial_evidence_and_thread_exits(tmp_path, monkeypatch):
    class BrokenWriter:
        def __init__(self, path, **kwargs):
            path.write_bytes(b"partial evidence")
            raise OSError("disk full")
    monkeypatch.setattr(recording, "Hdf5SignalWriter", BrokenWriter)
    rec = OnlineRecording(tmp_path)
    rec.configure(streams={"imu": {}})
    edge(rec, True, 100)
    rec.raw("imu", batch(110))
    edge(rec, False, 200)
    rec.raw("imu", batch(300, index=8))
    assert rec.close()
    assert not rec._thread.is_alive() and rec.error
    assert not rec.completed
    assert list(tmp_path.rglob("imu.h5.partial"))
    assert list(tmp_path.rglob("recording_failure.json"))


def test_queue_overflow_is_nonblocking_and_cannot_publish_success(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    real = recording._SessionWriter
    class SlowWriter(real):
        def event(self, *args):
            entered.set()
            release.wait(1)
            return super().event(*args)
    monkeypatch.setattr(recording, "_SessionWriter", SlowWriter)
    rec = OnlineRecording(tmp_path, queue_capacity=2)
    edge(rec, True, 100)
    assert entered.wait(1)
    start = time.monotonic()
    for n in range(100):
        rec.prediction({"host_monotonic_ns": 110 + n})
    edge(rec, False, 300)
    assert time.monotonic() - start < .2
    release.set()
    assert rec.close()
    assert rec.error and "overflow" in rec.error
    result = read_online_recording(rec.completed[0])
    assert result["manifest"].state == TrialState.ABORTED


def test_start_queue_failure_rolls_back_active_and_retains_evidence(tmp_path, monkeypatch):
    rec = OnlineRecording(tmp_path, queue_capacity=1)
    def full(_item):
        raise Full
    monkeypatch.setattr(rec._queue, "put_nowait", full)
    edge(rec, True, 100)
    assert not rec.active
    assert rec.close()
    assert not rec.completed and "overflow" in rec.error
    assert list(tmp_path.rglob("recording_failure.json"))


def test_start_constructor_failure_stops_active_and_retains_evidence(tmp_path, monkeypatch):
    class BrokenSession:
        def __init__(self, *_args):
            raise OSError("session disk unavailable")
    monkeypatch.setattr(recording, "_SessionWriter", BrokenSession)
    rec = OnlineRecording(tmp_path)
    edge(rec, True, 100)
    assert rec.close()
    assert not rec.active and not rec._thread.is_alive()
    assert not rec.completed and "unavailable" in rec.error
    assert list(tmp_path.rglob("recording_failure.json"))


def test_cross_stop_last_batch_without_later_watermark_is_conservatively_aborted(tmp_path):
    rec = OnlineRecording(tmp_path)
    rec.configure(streams={"imu": {}})
    edge(rec, True, 102_000_000)
    rec.raw("imu", batch(100_000_000, count=8))
    edge(rec, False, 106_000_000)
    assert rec.close()
    result = read_online_recording(rec.completed[0])
    assert len(result["streams"]["raw/imu.h5"]["data"]) == 4
    assert result["manifest"].state == TrialState.ABORTED
    assert "raw tail not verified" in rec.error


def test_raw_ethernet_four_single_channel_packets_preserve_channel_and_trailer(tmp_path):
    rec = OnlineRecording(tmp_path)
    rec.configure(streams={"ultrasound": {}})
    edge(rec, True, 100_000_000)
    descriptor = {"metadata": {"protocol": "raw_ethernet_uint8", "simulated": False,
        "transport": "raw_ethernet_scapy_npcap"}, "sample_shape": [1000],
        "channels": ["ch_1", "ch_2", "ch_3", "ch_4"]}
    for channel in range(4):
        rec.raw("ultrasound", FrameBatch(device_id="raw_ethernet", modality="ultrasound",
            clock_domain="ultrasound_raw_ethernet_clock", host_monotonic_ns=101_000_000 + channel * 100_000,
            first_frame_index=channel, frame_count=1, sequence_number=channel,
            frame_rate_hz=1000, data=np.full((1, 1000), channel, dtype=np.uint8),
            channel=channel, tail_flags=channel % 2), descriptor)
    edge(rec, False, 105_000_000)
    rec.raw("ultrasound", batch(110_000_000, modality="ultrasound"))
    assert rec.close() and rec.error is None
    stream = read_online_recording(rec.completed[0])["streams"]["raw/ultrasound.bin"]
    assert stream["data"].shape == (4, 1000) and stream["data"].dtype == np.uint8
    assert stream["channel"].tolist() == [0, 1, 2, 3]
    assert stream["has_trailer"].tolist() == [False, True, False, True]
    assert stream["flags"].tolist() == [0, 5, 2, 7]


def test_real_tcn_recorded_actual_features_reproduce_prediction_with_fake_motor(tmp_path):
    from exo_collection.adapters.imu.simulated import IMU_CHANNELS, IMU_UNITS
    from exo_collection.apps.model_runtime.backends import build_model_runtime
    from exo_collection.apps.model_runtime.imu_tcn import ImuTcnFeatureExtractor
    from exo_collection.apps.model_runtime.inference import InferenceController
    from exo_collection.apps.model_runtime.spec import load_model_spec

    bundle = Path(__file__).resolve().parents[2] / "runtime_data/models/imu_right_pelvis_torque_v1_20261010"
    spec = load_model_spec(bundle / "model_spec.json")
    spec.imu_tcn.right_device_id, spec.imu_tcn.pelvis_device_id = "R", "P"
    spec.assistance.mass_kg, spec.assistance.gain = 60, .5
    spec.assistance.lowpass_hz, spec.assistance.delay_s, spec.assistance.right_sign = 0, 0, -1
    spec.output.torque_limit_nm = .1
    runtime = build_model_runtime(spec, base_dir=bundle)
    rec = OnlineRecording(tmp_path)
    rec.configure({"spec": spec.model_dump(mode="json")}, {"imu": {}})
    queue, hub = Queue(), SubscriptionHub(buffer_capacity=1024, raw_tap=rec.raw)
    hub.register_stream("imu", queue, channels=IMU_CHANNELS, sample_shape=(2, 12), rate_hz=100,
        descriptor={"channels": IMU_CHANNELS, "units": IMU_UNITS, "sample_shape": (2, 12),
                    "metadata": {"device_ids": ["R", "P"], "simulated": False}})
    now, index = [time.perf_counter_ns()], [0]
    def sample(at, slot):
        data = np.full((1, 2, 12), np.nan, dtype=np.float32)
        data[0, slot] = 0
        data[0, slot, 2] = 9.81
        queue.put(SampleBatch(device_id="imu", modality="imu", clock_domain="host",
            first_sample_index=index[0], sample_count=1, sequence_number=index[0],
            sample_rate_hz=100, host_monotonic_ns=at, data=data))
        index[0] += 1
    for i in range(210):
        sample(now[0] - (209-i) * 10_000_000, 0)
        sample(now[0] - (209-i) * 10_000_000 + 1000, 1)
    hub.drain()
    motor = SerialTorqueOutput("FAKE", .1, background=False, clock=lambda: now[0],
        serial_factory=lambda **kwargs: Serial(), recording_sink=rec.motor_event)
    motor.connect()
    motor.feed_feedback(feedback(state=0))
    controller = InferenceController(hub, spec, runtime, motor,
        extractor=ImuTcnFeatureExtractor(spec.imu_tcn, clock_ns=lambda: now[0]),
        clock_ns=lambda: now[0], recording_callback=rec.prediction)
    try:
        controller.set_enabled(True)
        now[0] += 1_000_000
        motor.feed_feedback(feedback(seq=2))
        now[0] += 50_000_000
        sample(now[0], 0)
        sample(now[0] + 1000, 1)
        controller._tick()
        assert controller.last_error is None and motor.enabled
        now[0] += 1_000_000
        controller.request_stop()
        sample(now[0] + 10_000_000, 0)
        sample(now[0] + 10_001_000, 1)
        hub.drain()
    finally:
        motor.close()
        rec.close()
    assert rec.error is None, rec.error
    result = read_online_recording(rec.completed[0])
    prediction = next(entry["payload"] for entry in result["events"] if entry["kind"] == "prediction")
    features = np.asarray(prediction["model_features"], dtype=prediction["model_features_dtype"])
    assert features.shape == (12, 40)
    np.testing.assert_allclose(runtime.predict(features), prediction["raw_output"], rtol=1e-6)
    assert prediction["desired_left_nm"] == prediction["desired_right_nm"]
    assert prediction["desired_right_nm"] == pytest.approx(prediction["raw_estimate_right"] * 60 * .5 * -1)
    sent = [entry["payload"] for entry in result["events"] if entry["kind"] == "sent_command"]
    assert any(command["left_nm"] == command["right_nm"] != 0 for command in sent)
    assert all(abs(command["left_nm"]) <= .1 and abs(command["right_nm"]) <= .1 for command in sent)
