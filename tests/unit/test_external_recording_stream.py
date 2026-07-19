from __future__ import annotations

from dataclasses import asdict
import json
from queue import Queue
from threading import Timer
from time import perf_counter_ns, time_ns
from uuid import uuid4

import numpy as np
import pytest

from exo_collection.acquisition.recording_stream import (
    RecordedRawEvent,
    RecordingBoundary,
    RecordingBoundaryKind,
    RecordingStreamEndpoint,
    RecordingStreamProducer,
)
from exo_collection.acquisition.stream_proxy import (
    RecordingStreamFault,
    RecordingStreamProtocolError,
    StreamProxyAdapter,
)
from exo_collection.adapters.base import StartToken, TrialContext
from exo_collection.domain.events import (
    EdgeType,
    FrameBatch,
    SampleBatch,
    SyncPulseEvent,
)
from exo_collection.orchestration.models import TrialRunRequest
from exo_collection.readers.binary_block import BlockBinaryReader


def _descriptor() -> dict[str, object]:
    return {
        "device_id": "raw_us",
        "modality": "ultrasound",
        "display_name": "Raw Ethernet ultrasound",
        "clock_domain": "raw_us_clock",
        "event_kind": "frame_batch",
        "channels": ["ch_1", "ch_2", "ch_3", "ch_4"],
        "units": ["a.u.", "a.u.", "a.u.", "a.u."],
        "nominal_rate_hz": 20.0,
        "sample_shape": [1000],
        "dtype": np.dtype(np.uint8).str,
        "metadata": {"protocol": "raw_ethernet_uint8", "simulated": False},
    }


def _endpoint(queue: Queue[object]) -> RecordingStreamEndpoint:
    return RecordingStreamEndpoint(
        queue=queue,
        device_id="raw_us",
        modality="ultrasound",
        descriptor=_descriptor(),
        configuration_snapshot={"interface_name": "npcap-test"},
    )


def _producer(queue: Queue[object]) -> RecordingStreamProducer:
    return RecordingStreamProducer(
        queue,
        device_id="raw_us",
        modality="ultrasound",
        descriptor=_descriptor(),
        configuration_snapshot={"interface_name": "npcap-test"},
    )


def _frame(channel: int, sequence: int) -> FrameBatch:
    wire = np.arange(1000, dtype=np.uint16).astype(np.uint8)[None, :]
    wire[0, 0] = 0
    wire[0, 1] = channel + 1
    wire[0, -1] = 0xFF
    return FrameBatch(
        device_id="raw_us",
        modality="ultrasound",
        clock_domain="raw_us_clock",
        first_frame_index=sequence,
        frame_count=1,
        sequence_number=sequence,
        frame_rate_hz=20.0,
        host_monotonic_ns=1_000_000_000 + sequence,
        host_utc_ns=2_000_000_000 + sequence,
        data=wire,
        channel=channel,
        tail_flags=1,
    )


def _prepared_proxy(
    queue: Queue[object], trial_uuid: str
) -> StreamProxyAdapter:
    proxy = StreamProxyAdapter(_endpoint(queue), start_boundary_timeout_s=0.2)
    proxy.connect()
    prepared = proxy.prepare(TrialContext(trial_uuid=trial_uuid))
    assert prepared.trial_uuid == trial_uuid
    return proxy


def test_proxy_consumes_ordered_four_channel_frames_verbatim_until_end() -> None:
    queue: Queue[object] = Queue(maxsize=16)
    trial_uuid = str(uuid4())
    producer = _producer(queue)
    frames = [_frame(channel, channel) for channel in range(4)]
    producer.begin(trial_uuid)
    for frame in frames:
        assert producer.forward(frame)
    producer.end(trial_uuid)

    proxy = _prepared_proxy(queue, trial_uuid)
    proxy.start(StartToken())
    observed = []
    while not proxy.stream_ended:
        event = proxy.get_event(timeout=0.1)
        if event is not None:
            observed.append(event)

    assert [event.channel for event in observed] == [0, 1, 2, 3]
    for actual, expected in zip(observed, frames, strict=True):
        assert actual.tail_flags == 1
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual.data.shape == (1, 1000)
    report = proxy.stop()
    assert report.batches_emitted == 4
    assert report.samples_emitted == 4
    assert report.fault is None
    assert proxy.configuration_snapshot() == {"interface_name": "npcap-test"}
    assert asdict(proxy.descriptor())["sample_shape"] == (1000,)
    proxy.close()


def test_proxy_rejects_wrong_trial_raw_wrapper() -> None:
    queue: Queue[object] = Queue(maxsize=8)
    trial_uuid = str(uuid4())
    producer = _producer(queue)
    producer.begin(trial_uuid)
    queue.put_nowait(
        RecordedRawEvent(
            trial_uuid=str(uuid4()),
            modality="ultrasound",
            device_id="raw_us",
            event=_frame(0, 0),
        )
    )
    proxy = _prepared_proxy(queue, trial_uuid)
    proxy.start()

    with pytest.raises(RecordingStreamProtocolError, match="trial mismatch"):
        proxy.get_event(timeout=0.1)


def test_proxy_raises_adapter_fault_and_never_reports_clean_end() -> None:
    queue: Queue[object] = Queue(maxsize=8)
    trial_uuid = str(uuid4())
    producer = _producer(queue)
    producer.begin(trial_uuid)
    producer.abort("capture process crashed")
    proxy = _prepared_proxy(queue, trial_uuid)
    proxy.start()

    with pytest.raises(RecordingStreamFault, match="capture process crashed"):
        proxy.get_event(timeout=0.1)
    assert not proxy.stream_ended
    assert proxy.health().status.value == "UNHEALTHY"


def test_proxy_requires_matching_start_metadata() -> None:
    queue: Queue[object] = Queue(maxsize=4)
    trial_uuid = str(uuid4())
    queue.put_nowait(
        RecordingBoundary(
            kind=RecordingBoundaryKind.START,
            trial_uuid=trial_uuid,
            modality="ultrasound",
            device_id="raw_us",
            descriptor={**_descriptor(), "dtype": "<u2"},
            configuration_snapshot={"interface_name": "npcap-test"},
        )
    )
    proxy = _prepared_proxy(queue, trial_uuid)

    with pytest.raises(RecordingStreamProtocolError, match="descriptor differs"):
        proxy.start()


def test_proxy_discards_previous_trial_residue_before_current_start() -> None:
    queue: Queue[object] = Queue(maxsize=16)
    old_trial_uuid = str(uuid4())
    current_trial_uuid = str(uuid4())
    old_producer = _producer(queue)
    old_producer.begin(old_trial_uuid)
    old_producer.forward(_frame(0, 0))
    old_producer.end(old_trial_uuid)
    current_producer = _producer(queue)
    current_producer.begin(current_trial_uuid)
    current_frame = _frame(1, 1)
    current_producer.forward(current_frame)
    current_producer.end(current_trial_uuid)

    proxy = _prepared_proxy(queue, current_trial_uuid)
    proxy.start()
    observed = proxy.get_event(timeout=0.1)
    assert isinstance(observed, FrameBatch)
    assert observed.channel == 1
    assert proxy.get_event(timeout=0.1) is None
    assert proxy.stream_ended
    proxy.stop()


def _sync_descriptor() -> dict[str, object]:
    return {
        "device_id": "sync_stream",
        "modality": "sync_pulse",
        "display_name": "Sync pulse",
        "clock_domain": "sync_clock",
        "event_kind": "sample_batch",
        "channels": ["voltage"],
        "units": ["V"],
        "nominal_rate_hz": 1000.0,
        "sample_shape": [1],
        "dtype": np.dtype(np.float64).str,
        "metadata": {"simulated": True, "source": "external-test"},
    }


def _sync_endpoint(queue: Queue[object]) -> RecordingStreamEndpoint:
    return RecordingStreamEndpoint(
        queue=queue,
        device_id="sync_stream",
        modality="sync_pulse",
        descriptor=_sync_descriptor(),
        configuration_snapshot={"threshold": 2.5},
    )


def _signal_descriptor(modality: str) -> dict[str, object]:
    if modality == "imu":
        channels = (
            "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
            "mag_x", "mag_y", "mag_z", "roll", "pitch", "yaw",
        )
        units = (
            "m/s2", "m/s2", "m/s2", "rad/s", "rad/s", "rad/s",
            "a.u.", "a.u.", "a.u.", "deg", "deg", "deg",
        )
        sample_shape = [3, 12]
        device_id = "imu_stream"
        clock_domain = "imu_clock"
        rate = 200.0
    else:
        channels = (
            "left_position", "left_velocity", "left_torque",
            "right_position", "right_velocity", "right_torque",
        )
        units = ("rad", "rad/s", "N*m", "rad", "rad/s", "N*m")
        sample_shape = [6]
        device_id = "encoder_stream"
        clock_domain = "encoder_clock"
        rate = 200.0
    return {
        "device_id": device_id,
        "modality": modality,
        "display_name": f"External {modality}",
        "clock_domain": clock_domain,
        "event_kind": "sample_batch",
        "channels": list(channels),
        "units": list(units),
        "nominal_rate_hz": rate,
        "sample_shape": sample_shape,
        "dtype": np.dtype(np.float32).str,
        "metadata": {"simulated": False, "source": "external-test"},
    }


def _signal_endpoint(
    queue: Queue[object], modality: str
) -> RecordingStreamEndpoint:
    descriptor = _signal_descriptor(modality)
    return RecordingStreamEndpoint(
        queue=queue,
        device_id=str(descriptor["device_id"]),
        modality=modality,
        descriptor=descriptor,
        configuration_snapshot={"source": "persistent-preview"},
    )


def test_run_trial_external_streams_never_build_device_adapters_and_write_raw(
    tmp_path, monkeypatch
) -> None:
    import exo_collection.orchestration.simulated as orchestration

    trial_uuid = uuid4()
    host_ns = perf_counter_ns()
    utc_ns = time_ns()
    ultrasound_queue: Queue[object] = Queue(maxsize=32)
    sync_queue: Queue[object] = Queue(maxsize=16)
    ultrasound_producer = _producer(ultrasound_queue)
    sync_producer = RecordingStreamProducer(
        sync_queue,
        device_id="sync_stream",
        modality="sync_pulse",
        descriptor=_sync_descriptor(),
        configuration_snapshot={"threshold": 2.5},
    )
    ultrasound_producer.begin(trial_uuid)
    sync_producer.begin(trial_uuid)
    expected_frames = []
    for channel in range(4):
        frame = _frame(channel, channel).model_copy(
            update={
                "host_monotonic_ns": host_ns + channel * 1_000,
                "host_utc_ns": utc_ns + channel * 1_000,
            }
        )
        expected_frames.append(frame.data[0].tobytes())
        ultrasound_producer.forward(frame)
    sync_producer.forward(
        SampleBatch(
            device_id="sync_stream",
            modality="sync_pulse",
            clock_domain="sync_clock",
            first_sample_index=0,
            sample_count=8,
            sequence_number=0,
            sample_rate_hz=1000.0,
            host_monotonic_ns=host_ns,
            host_utc_ns=utc_ns,
            data=np.asarray([[0.0], [0.0], [5.0], [5.0], [0.0], [0.0], [0.0], [0.0]]),
        )
    )
    sync_producer.forward(
        SyncPulseEvent(
            device_id="sync_stream",
            modality="sync_pulse",
            clock_domain="sync_clock",
            pulse_id="pulse-1",
            source_device="sync_stream",
            edge_type=EdgeType.RISING,
            sample_index=2,
            amplitude=5.0,
            detection_threshold=2.5,
            confidence=1.0,
            detector_version="test",
            host_monotonic_ns=host_ns + 2_000_000,
            host_utc_ns=utc_ns + 2_000_000,
        )
    )
    def forbidden_adapter_build(*_args, **_kwargs):
        raise AssertionError("external stream mode must not build device adapters")

    monkeypatch.setattr(orchestration, "_make_adapters", forbidden_adapter_build)
    request = TrialRunRequest(
        data_root=tmp_path,
        device_profile_key="hardware",
        trial_uuid=trial_uuid,
        enabled_modalities=frozenset({"ultrasound", "sync_pulse"}),
        duration_s=0.001,
        sync_wait_timeout_s=0.001,
    )
    published = []
    end_timer: list[Timer] = []

    def publish(event) -> None:
        published.append(event)
        if event.payload.get("state") == "RECORDING" and not end_timer:
            timer = Timer(
                0.08,
                lambda: (
                    ultrasound_producer.end(trial_uuid),
                    sync_producer.end(trial_uuid),
                ),
            )
            timer.start()
            end_timer.append(timer)

    class AlreadyRequestedStop:
        @staticmethod
        def is_set() -> bool:
            return True

    try:
        result = orchestration.run_trial(
            request,
            stop_requested=AlreadyRequestedStop(),
            publish=publish,
            recording_streams={
                "ultrasound": _endpoint(ultrasound_queue),
                "sync_pulse": _sync_endpoint(sync_queue),
            },
            # If duration/sync/local-stop incorrectly initiated END waiting,
            # this would fail well before the UI-owned timer closes the gates.
            recording_stream_end_timeout_s=0.02,
        )
    finally:
        for timer in end_timer:
            timer.cancel()

    assert result.state == "FINALIZED"
    assert result.modality_counts["ultrasound"] == 4
    assert result.modality_counts["sync_pulse"] == 8
    assert not any(event.event_type.value == "preview" for event in published)
    with BlockBinaryReader(result.trial_directory / "raw/ultrasound.bin") as reader:
        records = list(reader)
    assert [record.data[0].tobytes() for record in records] == expected_frames
    assert [record.header.flags for record in records] == [4, 5, 6, 7]
    assert not (result.trial_directory / "raw/imu.h5").exists()
    assert not (result.trial_directory / "raw/encoder.h5").exists()
    assert not (result.trial_directory / "reports/imu_encoder_preview.png").exists()
    manifest = json.loads((result.trial_directory / "manifest.json").read_text("utf-8"))
    assert {item["modality"] for item in manifest["modalities"]} == {
        "ultrasound",
        "sync_pulse",
    }
    artifact_paths = {item["relative_path"] for item in manifest["artifacts"]}
    assert "raw/imu.h5" not in artifact_paths
    assert "raw/encoder.h5" not in artifact_paths
    assert "reports/imu_encoder_preview.png" not in artifact_paths
