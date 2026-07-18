"""Unit tests for the single-modality preview worker and lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
import pickle
from pathlib import Path
from queue import Queue
import time
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.acquisition.preview import build_preview_event
from exo_collection.adapters.base import (
    AdapterState,
    ModalityAdapter,
    ModalityDescriptor,
    PreparedInfo,
    SimulationConfig,
    StartToken,
    StopReport,
    TrialContext,
)
from exo_collection.apps.collector.device_preview import (
    InProcessPreviewRunner,
    ModalityPreviewProcessHandle,
    ProfileModalityAdapterFactory,
    _build_preview_event,
    _preview_is_due,
    _preview_rate_limit_key,
)
from exo_collection.domain.events import (
    DeviceStatusEvent,
    FrameBatch,
    HealthSnapshot,
    HealthStatus,
    DeviceStatus,
    SampleBatch,
    SyncPulseEvent,
    EdgeType,
)


# ── Fake adapters ──


class FakeSampleBatchAdapter:
    """Fake adapter that emits SampleBatch from an internal queue."""

    def __init__(self, descriptor: ModalityDescriptor | None = None) -> None:
        self._descriptor = descriptor or ModalityDescriptor(
            device_id="fake_imu", modality="imu", display_name="Fake IMU",
            clock_domain="host", event_kind="sample",
            nominal_rate_hz=200.0, channels=("acc_x",), units=("m/s^2",),
            sample_shape=(1,), dtype="float64", metadata={},
        )
        self._state = AdapterState.DISCONNECTED
        self._events: list[Any] = []
        self._connected = False
        self._prepared = False
        self._started = False
        self._stopped = False
        self._closed = False
        self._sample_counter = 0

    @property
    def state(self) -> AdapterState:
        return self._state

    def descriptor(self) -> ModalityDescriptor:
        return self._descriptor

    def configuration_snapshot(self) -> dict[str, Any]:
        return {}

    def connect(self, config: Any = None) -> None:
        self._state = AdapterState.CONNECTED
        self._connected = True

    def prepare(self, trial: TrialContext) -> PreparedInfo:
        self._state = AdapterState.PREPARED
        self._prepared = True
        return PreparedInfo(
            device_id=self._descriptor.device_id,
            modality=self._descriptor.modality,
            trial_uuid=str(trial.trial_uuid),
            clock_domain=self._descriptor.clock_domain,
            nominal_rate_hz=self._descriptor.nominal_rate_hz,
            channels=self._descriptor.channels,
            units=self._descriptor.units,
            queue_capacity=64,
        )

    def start(self, start_token: StartToken) -> None:
        self._state = AdapterState.RUNNING
        self._started = True

    def stop(self) -> StopReport:
        self._state = AdapterState.STOPPED
        self._stopped = True
        return StopReport(
            device_id=self._descriptor.device_id, modality=self._descriptor.modality,
            batches_emitted=self._sample_counter, samples_emitted=self._sample_counter,
            injected_dropped_batches=0, raw_queue_overflows=0,
            first_data_monotonic_ns=None, last_data_monotonic_ns=None, fault=None,
        )

    def health(self) -> HealthSnapshot:
        return HealthSnapshot(
            device_id=self._descriptor.device_id, modality=self._descriptor.modality,
            status=HealthStatus.HEALTHY, device_status=DeviceStatus.RECORDING,
            connected=True, ready=True, sampling=self._started,
            queue_depth=0, queue_capacity=64, last_data_host_monotonic_ns=None,
            actual_sample_rate_hz=self._descriptor.nominal_rate_hz if self._started else 0.0,
            nominal_sample_rate_hz=self._descriptor.nominal_rate_hz,
            dropped_packets=0, message="ok", metrics={"samples_emitted": self._sample_counter},
        )

    def close(self) -> None:
        self._state = AdapterState.CLOSED
        self._closed = True

    def get_event(self, timeout: float | None = None) -> Any | None:
        if self._events:
            return self._events.pop(0)
        self._sample_counter += 1
        data = np.zeros((4, 3, 12), dtype=np.float64)
        data[:, 0, 0] = float(self._sample_counter)
        data[:, 1, 0] = float(self._sample_counter) + 1.0
        data[:, 2, 0] = float(self._sample_counter) + 2.0
        return SampleBatch(
            device_id=self._descriptor.device_id,
            modality=self._descriptor.modality,
            clock_domain=self._descriptor.clock_domain,
            data=data,
            sample_rate_hz=self._descriptor.nominal_rate_hz,
            host_monotonic_ns=int(self._sample_counter * 1e9 / self._descriptor.nominal_rate_hz),
            sequence_number=self._sample_counter,
            first_sample_index=(self._sample_counter - 1) * data.shape[0],
            sample_count=data.shape[0],
        )

    poll_event = get_event


class FakeUltrasoundAdapter:
    """Fake adapter that emits FrameBatch for 4-channel ultrasound."""

    def __init__(self) -> None:
        self._descriptor = ModalityDescriptor(
            device_id="fake_us", modality="ultrasound", display_name="Fake US",
            clock_domain="host", event_kind="frame",
            nominal_rate_hz=25.0, channels=("ch1", "ch2", "ch3", "ch4"),
            units=("adc","adc","adc","adc"), sample_shape=(1000,), dtype="float32", metadata={},
        )
        self._state = AdapterState.DISCONNECTED
        self._started = False
        self._frame_counter = 0

    @property
    def state(self) -> AdapterState:
        return self._state

    def descriptor(self) -> ModalityDescriptor:
        return self._descriptor

    def configuration_snapshot(self) -> dict[str, Any]:
        return {}

    def connect(self, config: Any = None) -> None:
        self._state = AdapterState.CONNECTED

    def prepare(self, trial: TrialContext) -> PreparedInfo:
        self._state = AdapterState.PREPARED
        return PreparedInfo(
            device_id=self._descriptor.device_id, modality=self._descriptor.modality,
            trial_uuid=str(trial.trial_uuid), clock_domain=self._descriptor.clock_domain,
            nominal_rate_hz=self._descriptor.nominal_rate_hz,
            channels=self._descriptor.channels, units=self._descriptor.units,
            queue_capacity=32,
        )

    def start(self, start_token: StartToken) -> None:
        self._state = AdapterState.RUNNING
        self._started = True

    def stop(self) -> StopReport:
        self._state = AdapterState.STOPPED
        return StopReport(
            device_id=self._descriptor.device_id, modality=self._descriptor.modality,
            batches_emitted=self._frame_counter, samples_emitted=self._frame_counter * 4 * 1000,
            injected_dropped_batches=0, raw_queue_overflows=0,
            first_data_monotonic_ns=None, last_data_monotonic_ns=None, fault=None,
        )

    def health(self) -> HealthSnapshot:
        return HealthSnapshot(
            device_id=self._descriptor.device_id, modality=self._descriptor.modality,
            status=HealthStatus.HEALTHY, device_status=DeviceStatus.RECORDING,
            connected=True, ready=True, sampling=self._started,
            queue_depth=0, queue_capacity=32, last_data_host_monotonic_ns=None,
            actual_sample_rate_hz=25.0, nominal_sample_rate_hz=25.0,
            dropped_packets=0, message="ok",
            metrics={"samples_emitted": self._frame_counter * 4 * 1000},
        )

    def close(self) -> None:
        self._state = AdapterState.CLOSED

    def get_event(self, timeout: float | None = None) -> Any | None:
        self._frame_counter += 1
        data = np.empty((1, 4, 1000), dtype=np.float32)
        data[0, :, :] = np.arange(1000, dtype=np.float32) * 0.01
        return FrameBatch(
            device_id=self._descriptor.device_id,
            modality=self._descriptor.modality,
            clock_domain=self._descriptor.clock_domain,
            data=data,
            frame_rate_hz=25.0,
            host_monotonic_ns=int(self._frame_counter * 1e9 / 25.0),
            sequence_number=self._frame_counter,
            first_frame_index=self._frame_counter - 1,
            frame_count=1,
        )

    poll_event = get_event


class FailingConnectAdapter:
    """Adapter that raises on connect."""

    def __init__(self) -> None:
        self._descriptor = ModalityDescriptor(
            device_id="fail_connect", modality="imu", display_name="Fail IMU",
            clock_domain="host", event_kind="sample",
            nominal_rate_hz=200.0, channels=("acc_x",), units=("m/s^2",),
            sample_shape=(1,), dtype="float64", metadata={},
        )
        self._state = AdapterState.DISCONNECTED

    @property
    def state(self) -> AdapterState:
        return self._state

    def descriptor(self) -> ModalityDescriptor:
        return self._descriptor

    def configuration_snapshot(self) -> dict[str, Any]:
        return {}

    def connect(self, config: Any = None) -> None:
        raise RuntimeError("connection refused")

    def prepare(self, trial: TrialContext) -> PreparedInfo:
        raise RuntimeError("not connected")

    def start(self, start_token: StartToken) -> None:
        raise RuntimeError("not connected")

    def stop(self) -> StopReport:
        raise RuntimeError("not connected")

    def health(self) -> HealthSnapshot:
        return HealthSnapshot(
            device_id=self._descriptor.device_id, modality=self._descriptor.modality,
            status=HealthStatus.UNHEALTHY, device_status=DeviceStatus.DISCONNECTED,
            connected=False, ready=False, sampling=False,
            queue_depth=0, queue_capacity=0, last_data_host_monotonic_ns=None,
            actual_sample_rate_hz=0.0, nominal_sample_rate_hz=200.0,
            dropped_packets=0, message="not connected",
            metrics={},
        )

    def close(self) -> None:
        self._state = AdapterState.CLOSED

    def get_event(self, timeout: float | None = None) -> Any | None:
        return None

    poll_event = get_event


# ── InProcessPreviewRunner tests ──


def test_preview_runner_lifecycle_basic() -> None:
    """Full lifecycle: start -> poll events -> stop -> dispose."""
    runner = InProcessPreviewRunner(
        adapter_factory=FakeSampleBatchAdapter,
        device_id="fake_imu",
        modality="imu",
        simulated=True,
    )
    runner.start()
    assert runner.is_alive

    events = runner.poll_events(limit=10)
    assert any(e.event_type == WorkerEventType.STATE and
               e.payload.get("state") == "READY"
               for e in events)
    assert any(e.event_type == WorkerEventType.PREVIEW for e in events), (
        f"got events: {[(e.event_type.value, e.modality) for e in events]}"
    )

    runner.request_stop()
    runner.join(timeout=1.0)
    assert not runner.is_alive
    runner.close()


def test_preview_runner_no_files_created(tmp_path: Path) -> None:
    """Preview runner must NEVER create catalog/trial/manifest/h5/bin files."""
    runner = InProcessPreviewRunner(
        adapter_factory=FakeSampleBatchAdapter,
        device_id="fake_imu",
        modality="imu",
        simulated=True,
    )
    runner.start()
    runner.poll_events(limit=50)
    runner.request_stop()
    runner.join(timeout=1.0)
    runner.close()

    # Check no data files were created
    for pattern in ("*.sqlite3", "*.h5", "*.bin", "*.recording", "manifest.*"):
        found = list(tmp_path.glob(pattern))
        assert not found, f"preview runner should not create {pattern}: {found}"


def test_preview_runner_failing_connect_is_handled() -> None:
    """A failing connect should be caught and reported."""
    runner = InProcessPreviewRunner(
        adapter_factory=FailingConnectAdapter,
        device_id="fail_connect",
        modality="imu",
        simulated=True,
    )
    # start() calls connect() which raises
    with pytest.raises(RuntimeError, match="connection refused"):
        runner.start()
    # The adapter should be released
    runner.close()


def test_preview_runner_stop_before_start() -> None:
    """Stopping a runner that hasn't started is a no-op."""
    runner = InProcessPreviewRunner(
        adapter_factory=FakeSampleBatchAdapter,
        device_id="fake_imu",
        modality="imu",
        simulated=True,
    )
    runner.request_stop()
    runner.join(timeout=0.5)
    assert not runner.is_alive
    runner.close()


def test_preview_runner_multiple_start_raises() -> None:
    """Starting a runner twice should be prevented."""
    runner = InProcessPreviewRunner(
        adapter_factory=FakeSampleBatchAdapter,
        device_id="fake_imu",
        modality="imu",
        simulated=True,
    )
    runner.start()
    with pytest.raises(RuntimeError):
        runner.start()
    runner.request_stop()
    runner.join()
    runner.close()


def test_preview_runner_disconnects_cleanly() -> None:
    """After stop/join, properties reflect disconnected state."""
    runner = InProcessPreviewRunner(
        adapter_factory=FakeSampleBatchAdapter,
        device_id="fake_imu",
        modality="imu",
        simulated=True,
    )
    runner.start()
    runner.request_stop()
    ec = runner.join()
    assert ec == 0
    assert not runner.is_alive
    assert runner.exitcode == 0
    runner.close()


# ── _build_preview_event tests ──


def test_build_preview_event_sample_batch_imu() -> None:
    desc = ModalityDescriptor(
        device_id="test_imu", modality="imu", display_name="Test IMU",
        clock_domain="host", event_kind="sample",
        nominal_rate_hz=200.0, channels=("acc_x",), units=("m/s^2",),
        sample_shape=(1,), dtype="float64", metadata={},
    )
    batch = SampleBatch(
        device_id="test_imu", modality="imu", clock_domain="host",
        data=np.arange(72, dtype=np.float64).reshape(2, 3, 12),
        sample_rate_hz=200.0, host_monotonic_ns=1000, sequence_number=1,
        first_sample_index=0, sample_count=2,
    )
    event = _build_preview_event(batch, "imu", "test_imu", desc, True)
    assert event is not None
    assert event.event_type == WorkerEventType.PREVIEW
    assert event.modality == "imu"
    channels = event.payload.get("channels", [])
    assert isinstance(channels, list)
    assert len(channels) == 3


def test_build_preview_event_imu_slot_1_3_labels_not_mapped_to_left() -> None:
    """Slot 1+3 config produces labels (imu_trunk, imu_right), never imu_left."""
    desc = ModalityDescriptor(
        device_id="test_imu", modality="imu", display_name="Test IMU",
        clock_domain="host", event_kind="sample",
        nominal_rate_hz=200.0, channels=("acc_x",), units=("m/s^2",),
        sample_shape=(2,), dtype="float64",
        metadata={"preview_labels": ["imu_trunk", "imu_right"]},
    )
    batch = SampleBatch(
        device_id="test_imu", modality="imu", clock_domain="host",
        data=np.arange(48, dtype=np.float64).reshape(2, 2, 12),
        sample_rate_hz=200.0, host_monotonic_ns=1000, sequence_number=1,
        first_sample_index=0, sample_count=2,
    )
    event = _build_preview_event(batch, "imu", "test_imu", desc, True)
    assert event is not None
    assert event.event_type == WorkerEventType.PREVIEW
    labels = event.payload.get("labels", [])
    assert labels == ["imu_trunk", "imu_right"]


def test_build_preview_event_ultrasound() -> None:
    desc = ModalityDescriptor(
        device_id="test_us", modality="ultrasound", display_name="Test US",
        clock_domain="host", event_kind="frame",
        nominal_rate_hz=25.0, channels=("ch1", "ch2", "ch3", "ch4"),
        units=("adc","adc","adc","adc"), sample_shape=(512,), dtype="float32", metadata={},
    )
    data = np.random.rand(1, 4, 512).astype(np.float32)
    batch = FrameBatch(
        device_id="test_us", modality="ultrasound", clock_domain="host",
        data=data, frame_rate_hz=25.0, host_monotonic_ns=1000, sequence_number=1,
        first_frame_index=0, frame_count=1,
    )
    event = _build_preview_event(batch, "ultrasound", "test_us", desc, True)
    assert event is not None
    assert event.event_type == WorkerEventType.PREVIEW
    channels = event.payload.get("channels", [])
    assert len(channels) == 4


def test_build_preview_event_raw_ultrasound_uses_only_in_frame_adc_bytes() -> None:
    """Decode the wire signature without mutating the complete raw frame."""

    desc = ModalityDescriptor(
        device_id="raw_us", modality="ultrasound", display_name="Raw US",
        clock_domain="host", event_kind="frame",
        nominal_rate_hz=25.0, channels=("ch1", "ch2", "ch3", "ch4"),
        units=("adc", "adc", "adc", "adc"), sample_shape=(1000,),
        dtype="uint8", metadata={},
    )
    adc = ((np.arange(997, dtype=np.uint16) * 7 + 11) % 250).astype(np.uint8)
    complete_frame = np.concatenate(
        (
            np.array([0x00, 0x03], dtype=np.uint8),
            adc,
            np.array([0xFF], dtype=np.uint8),
        )
    )
    original = complete_frame.copy()
    batch = FrameBatch(
        device_id="raw_us", modality="ultrasound", clock_domain="host",
        data=complete_frame[None, :], frame_rate_hz=25.0,
        host_monotonic_ns=1000, sequence_number=1,
        first_frame_index=0, frame_count=1, channel=2, tail_flags=1,
    )

    event = _build_preview_event(batch, "ultrasound", "raw_us", desc, False)

    assert event is not None
    assert event.payload["channel_index"] == 2
    assert event.payload["shape"] == [997]
    assert event.payload["preview_sample_count"] == 512
    centered_adc = (adc.astype(np.int16) - 127).astype(np.float32)
    indices = np.linspace(0, adc.size - 1, 512, dtype=np.int64)
    assert event.payload["values"] == pytest.approx(centered_adc[indices].tolist())
    assert np.array_equal(complete_frame, original)
    assert np.array_equal(np.asarray(batch.data), original[None, :])


def test_build_preview_event_encoder() -> None:
    desc = ModalityDescriptor(
        device_id="test_enc", modality="encoder", display_name="Test Enc",
        clock_domain="host", event_kind="sample",
        nominal_rate_hz=960.0, channels=("left", "right"), units=("deg", "deg"),
        sample_shape=(6,), dtype="float64", metadata={},
    )
    batch = SampleBatch(
        device_id="test_enc", modality="encoder", clock_domain="host",
        data=np.array([[10.5, 1.0, 2.0, 20.3, 3.0, 4.0]], dtype=np.float64),
        sample_rate_hz=960.0, host_monotonic_ns=1000, sequence_number=1,
        first_sample_index=0, sample_count=1,
    )
    event = _build_preview_event(batch, "encoder", "test_enc", desc, True)
    assert event is not None
    assert event.payload["labels"] == ["left_position", "right_position"]
    assert event.payload["channels"] == [[10.5], [20.3]]


def test_build_preview_public_api_respects_preview_labels_from_extra_payload() -> None:
    """build_preview_event (public) with extra_payload preview_labels
    for shape (1,2,12) IMU batch → labels are exactly (imu_trunk, imu_right),
    channels=2, and imu_left never appears."""
    batch = SampleBatch(
        device_id="test_imu", modality="imu", clock_domain="host",
        data=np.arange(24, dtype=np.float64).reshape(1, 2, 12),
        sample_rate_hz=200.0, host_monotonic_ns=1000, sequence_number=1,
        first_sample_index=0, sample_count=1,
    )
    event = build_preview_event(
        batch,
        extra_payload={"preview_labels": ["imu_trunk", "imu_right"]},
    )
    assert event is not None
    assert event.event_type == WorkerEventType.PREVIEW
    assert event.modality == "imu"
    labels = event.payload.get("labels", [])
    assert labels == ["imu_trunk", "imu_right"]
    assert "imu_left" not in labels
    channels = event.payload.get("channels", [])
    assert len(channels) == 2
    assert event.payload.get("channel_count") == 2


def test_build_preview_event_none_for_unknown_type() -> None:
    desc = ModalityDescriptor(
        device_id="test", modality="imu", display_name="Test",
        clock_domain="host", event_kind="sample",
        nominal_rate_hz=100.0, channels=("x",), units=("u",),
        sample_shape=(1,), dtype="float64", metadata={},
    )
    # Pass something that isn't a SampleBatch or FrameBatch
    event = _build_preview_event("not_a_batch", "imu", "test", desc, True)
    assert event is None


# ── Per-channel preview rate limiting ──


def _raw_ultrasound_preview_event(channel_index: int) -> WorkerEvent:
    return WorkerEvent(
        event_type=WorkerEventType.PREVIEW,
        modality="ultrasound",
        payload={"channel_index": channel_index, "values": [1.0]},
    )


def test_preview_rate_limit_allows_four_ultrasound_channels_at_same_time() -> None:
    """Interleaved ch0..ch3 packets are four independent UI streams."""

    last_sent: dict[tuple[str, int | None], float] = {}
    events = [_raw_ultrasound_preview_event(channel) for channel in range(4)]

    assert [_preview_rate_limit_key(event) for event in events] == [
        ("ultrasound", 0),
        ("ultrasound", 1),
        ("ultrasound", 2),
        ("ultrasound", 3),
    ]
    assert all(
        _preview_is_due(
            event,
            now=10.0,
            last_sent_by_stream=last_sent,
            interval_s=0.1,
        )
        for event in events
    )
    assert len(last_sent) == 4


def test_preview_rate_limit_throttles_same_ultrasound_channel() -> None:
    last_sent: dict[tuple[str, int | None], float] = {}
    event = _raw_ultrasound_preview_event(1)

    assert _preview_is_due(
        event, now=20.0, last_sent_by_stream=last_sent, interval_s=0.1
    )
    assert not _preview_is_due(
        event, now=20.05, last_sent_by_stream=last_sent, interval_s=0.1
    )
    assert last_sent == {("ultrasound", 1): 20.0}


def test_preview_rate_limit_different_ultrasound_channels_do_not_interfere() -> None:
    last_sent: dict[tuple[str, int | None], float] = {}

    assert _preview_is_due(
        _raw_ultrasound_preview_event(0),
        now=30.0,
        last_sent_by_stream=last_sent,
        interval_s=0.1,
    )
    assert _preview_is_due(
        _raw_ultrasound_preview_event(3),
        now=30.01,
        last_sent_by_stream=last_sent,
        interval_s=0.1,
    )
    assert not _preview_is_due(
        _raw_ultrasound_preview_event(0),
        now=30.01,
        last_sent_by_stream=last_sent,
        interval_s=0.1,
    )


# ── Queue pressure / downsample tests ──


def test_inprocess_runner_poll_respects_limit() -> None:
    """poll_events(limit=N) returns at most N events."""
    runner = InProcessPreviewRunner(
        adapter_factory=FakeSampleBatchAdapter,
        device_id="fake_imu",
        modality="imu",
        simulated=True,
    )
    runner.start()
    # Poll multiple times to accumulate events
    all_events = []
    for _ in range(20):
        all_events.extend(runner.poll_events(limit=5))
    assert len(all_events) >= 3  # CONNECTING, READY, at least 1 PREVIEW
    runner.request_stop()
    runner.join()
    runner.close()


def test_fake_ultrasound_adapter_works() -> None:
    """Verify the FakeUltrasoundAdapter can be used in a runner."""
    runner = InProcessPreviewRunner(
        adapter_factory=FakeUltrasoundAdapter,
        device_id="fake_us",
        modality="ultrasound",
        simulated=True,
    )
    runner.start()
    events = runner.poll_events(limit=10)
    preview_events = [e for e in events if e.event_type == WorkerEventType.PREVIEW]
    assert len(preview_events) >= 1
    runner.request_stop()
    runner.join()
    runner.close()


# ── Modality properties ──


def test_preview_runner_reports_modality_device_id() -> None:
    runner = InProcessPreviewRunner(
        adapter_factory=FakeSampleBatchAdapter,
        device_id="explicit_id",
        modality="encoder",
        simulated=False,
    )
    assert runner.modality == "encoder"
    assert runner.device_id == "explicit_id"
    assert runner.simulated is False
    runner.close()


def test_adapter_ready_status_does_not_mark_preview_ready_before_raw_data() -> None:
    """Only a real raw batch may make a preview connection READY."""

    def factory() -> FakeSampleBatchAdapter:
        adapter = FakeSampleBatchAdapter()
        adapter._events.append(
            DeviceStatusEvent(
                device_id="fake_imu",
                modality="imu",
                clock_domain="host",
                status=DeviceStatus.READY,
                previous_status=DeviceStatus.PREPARING,
            )
        )
        return adapter

    runner = InProcessPreviewRunner(
        adapter_factory=factory,
        device_id="fake_imu",
        modality="imu",
        simulated=True,
    )
    runner.start()
    first_poll = runner.poll_events(limit=10)
    assert not any(
        event.event_type is WorkerEventType.STATE
        and event.payload.get("state") == "READY"
        for event in first_poll
    )

    second_poll = runner.poll_events(limit=10)
    assert any(
        event.event_type is WorkerEventType.STATE
        and event.payload.get("state") == "READY"
        and event.payload.get("observed_raw_data") is True
        for event in second_poll
    )
    runner.request_stop()
    runner.join()
    runner.close()


def test_profile_modality_factory_is_spawn_pickle_safe() -> None:
    factory = ProfileModalityAdapterFactory(
        profile_key="simulated",
        modality="imu",
    )
    restored = pickle.loads(pickle.dumps(factory))
    adapter = restored()
    assert adapter.descriptor().modality == "imu"
    assert adapter.descriptor().device_id == "imu_sim"
    adapter.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows spawn contract")
def test_spawn_preview_emits_ready_and_preview_without_writing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    handle = ModalityPreviewProcessHandle(
        FakeSampleBatchAdapter,
        device_id="fake_imu",
        modality="imu",
        simulated=True,
        health_poll_interval_s=0.05,
    )
    try:
        handle.start()
        deadline = time.monotonic() + 10.0
        observed: list[WorkerEvent] = []
        while time.monotonic() < deadline:
            observed.extend(handle.poll_events(limit=100))
            ready = any(
                event.event_type is WorkerEventType.STATE
                and event.payload.get("state") == "READY"
                for event in observed
            )
            preview = any(
                event.event_type is WorkerEventType.PREVIEW for event in observed
            )
            if ready and preview:
                break
            assert handle.is_alive, observed
            time.sleep(0.02)
        assert ready and preview
        handle.request_stop()
        assert handle.join(timeout=5.0) == 0
    finally:
        if handle.is_alive:
            handle.terminate(timeout=1.0)
        handle.close()

    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert after == before
