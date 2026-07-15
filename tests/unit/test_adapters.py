from __future__ import annotations

import time
from uuid import uuid4

import numpy as np
import pytest

from exo_collection.adapters import (
    AdapterLifecycleError,
    ModalityAdapter,
    SimulatedEncoderAdapter,
    SimulatedEncoderConfig,
    SimulatedImuAdapter,
    SimulatedImuConfig,
    SimulatedSyncPulseAdapter,
    SimulatedSyncPulseConfig,
    SimulatedUltrasoundAdapter,
    SimulatedUltrasoundConfig,
    StartToken,
    TrialContext,
)
from exo_collection.domain.events import (
    DeviceStatus,
    DeviceStatusEvent,
    FrameBatch,
    HealthStatus,
    SampleBatch,
    SyncPulseEvent,
)


def _trial() -> TrialContext:
    return TrialContext(trial_uuid=uuid4(), session_uuid=uuid4())


@pytest.mark.parametrize(
    ("adapter", "event_type", "shape"),
    [
        (
            SimulatedUltrasoundAdapter(
                SimulatedUltrasoundConfig(queue_capacity=16, frames_per_batch=2)
            ),
            FrameBatch,
            (2, 4, 1000),
        ),
        (
            SimulatedImuAdapter(
                SimulatedImuConfig(
                    queue_capacity=16,
                    samples_per_batch=5,
                    device_ids=("a", "b", "c"),
                )
            ),
            SampleBatch,
            (5, 3, 12),
        ),
        (
            SimulatedEncoderAdapter(
                SimulatedEncoderConfig(queue_capacity=16, samples_per_batch=5)
            ),
            SampleBatch,
            (5, 6),
        ),
        (
            SimulatedSyncPulseAdapter(
                SimulatedSyncPulseConfig(queue_capacity=32, samples_per_batch=20)
            ),
            SampleBatch,
            (20, 1),
        ),
    ],
)
def test_simulated_adapter_lifecycle_and_shapes(adapter, event_type, shape) -> None:
    assert isinstance(adapter, ModalityAdapter)
    with pytest.raises(AdapterLifecycleError):
        adapter.start(StartToken())

    adapter.connect()
    connected = adapter.get_control_event(timeout=0.1)
    assert isinstance(connected, DeviceStatusEvent)
    assert connected.status is DeviceStatus.CONNECTED
    prepared = adapter.prepare(_trial())
    assert prepared.queue_capacity == adapter.health().queue_capacity
    adapter.start(StartToken())
    event = adapter.get_event(timeout=1.0)
    assert isinstance(event, event_type)
    assert event.data.shape == shape
    assert event.data.flags.c_contiguous
    assert event.host_monotonic_ns >= 0
    assert event.device_timestamp is not None
    assert adapter.health().sampling

    report = adapter.stop()
    assert report.samples_emitted >= shape[0]
    assert report.fault is None
    adapter.close()
    assert adapter.health().device_status is DeviceStatus.CLOSED


def test_ultrasound_supports_64_by_64_at_30_hz() -> None:
    adapter = SimulatedUltrasoundAdapter(
        SimulatedUltrasoundConfig(frame_shape=(64, 64), frame_rate_hz=30.0)
    )
    assert adapter.descriptor().sample_shape == (64, 64)
    assert adapter.descriptor().nominal_rate_hz == 30.0
    adapter.connect()
    adapter.prepare(_trial())
    adapter.start()
    event = adapter.get_event(timeout=1.0)
    adapter.stop()
    adapter.close()
    assert isinstance(event, FrameBatch)
    assert event.data.shape == (1, 64, 64)


@pytest.mark.parametrize(
    ("factory", "expected_shape"),
    [
        (
            lambda: SimulatedUltrasoundAdapter(
                SimulatedUltrasoundConfig(seed=41, frame_shape=(8, 8), frame_rate_hz=20)
            ),
            (1, 8, 8),
        ),
        (
            lambda: SimulatedImuAdapter(
                SimulatedImuConfig(seed=41, device_ids=("one",), samples_per_batch=4)
            ),
            (4, 1, 12),
        ),
        (
            lambda: SimulatedEncoderAdapter(
                SimulatedEncoderConfig(seed=41, samples_per_batch=4)
            ),
            (4, 6),
        ),
        (
            lambda: SimulatedSyncPulseAdapter(
                SimulatedSyncPulseConfig(seed=41, samples_per_batch=4)
            ),
            (4, 1),
        ),
    ],
)
def test_simulated_values_are_reproducible(factory, expected_shape) -> None:
    batches = []
    for _ in range(2):
        adapter = factory()
        adapter.connect()
        adapter.prepare(_trial())
        adapter.start()
        event = adapter.get_event(timeout=1.0)
        adapter.stop()
        adapter.close()
        assert event.data.shape == expected_shape
        batches.append(event.data.copy())
    np.testing.assert_array_equal(batches[0], batches[1])


def test_injected_drop_creates_an_observable_index_and_sequence_gap() -> None:
    adapter = SimulatedImuAdapter(
        SimulatedImuConfig(
            sample_rate_hz=200,
            samples_per_batch=10,
            drop_every_n_batches=2,
            queue_capacity=8,
        )
    )
    adapter.connect()
    adapter.prepare(_trial())
    adapter.start()
    first = adapter.get_event(timeout=1.0)
    second = adapter.get_event(timeout=1.0)
    adapter.stop()
    assert first.sequence_number == 0
    assert second.sequence_number == 2
    assert second.first_sample_index == first.first_sample_index + 20
    assert adapter.health().dropped_packets == 10
    adapter.close()


def test_raw_queue_overflow_is_fatal_and_never_silent() -> None:
    adapter = SimulatedEncoderAdapter(
        SimulatedEncoderConfig(
            sample_rate_hz=1000,
            samples_per_batch=1,
            queue_capacity=1,
        )
    )
    adapter.connect()
    adapter.prepare(_trial())
    adapter.start()
    deadline = time.monotonic() + 1.0
    while adapter.health().status is not HealthStatus.UNHEALTHY and time.monotonic() < deadline:
        time.sleep(0.005)
    health = adapter.health()
    assert health.status is HealthStatus.UNHEALTHY
    assert health.device_status is DeviceStatus.FAULT
    assert health.metrics["raw_queue_overflows"] == 1
    report = adapter.stop()
    assert report.raw_queue_overflows == 1
    assert "overflow" in report.fault
    adapter.close()


def test_injected_disconnect_becomes_fault() -> None:
    adapter = SimulatedImuAdapter(
        SimulatedImuConfig(
            sample_rate_hz=200,
            samples_per_batch=2,
            disconnect_after_batches=1,
        )
    )
    adapter.connect()
    adapter.prepare(_trial())
    adapter.start()
    assert isinstance(adapter.get_event(timeout=1.0), SampleBatch)
    deadline = time.monotonic() + 1.0
    while adapter.health().device_status is not DeviceStatus.FAULT and time.monotonic() < deadline:
        time.sleep(0.005)
    assert adapter.health().device_status is DeviceStatus.FAULT
    assert "disconnect" in adapter.stop().fault
    adapter.close()


def test_sync_adapter_emits_raw_waveform_and_detected_edges() -> None:
    adapter = SimulatedSyncPulseAdapter(
        SimulatedSyncPulseConfig(
            sample_rate_hz=1000,
            samples_per_batch=10,
            first_pulse_s=0.0,
            pulse_width_s=0.01,
            pulse_interval_s=0.1,
            noise_std_voltage=0.0,
            min_pulse_width_ns=2_000_000,
            debounce_ns=1_000_000,
            queue_capacity=32,
        )
    )
    adapter.connect()
    adapter.prepare(_trial())
    adapter.start()
    seen = []
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not any(
        isinstance(event, SyncPulseEvent) and event.edge_type.value == "falling"
        for event in seen
    ):
        event = adapter.get_event(timeout=0.1)
        if event is not None:
            seen.append(event)
    adapter.stop()
    adapter.close()
    raw = [event for event in seen if isinstance(event, SampleBatch)]
    edges = [event for event in seen if isinstance(event, SyncPulseEvent)]
    assert raw and raw[0].data.shape == (10, 1)
    assert [event.edge_type.value for event in edges[:2]] == ["rising", "falling"]
    assert edges[0].pulse_id == edges[1].pulse_id
    assert edges[1].pulse_width_ns == 10_000_000
