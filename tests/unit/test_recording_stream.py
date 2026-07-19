"""Regression tests for the loss-intolerant preview recording stream."""

from __future__ import annotations

from queue import Queue
from uuid import uuid4

import numpy as np
import pytest

from exo_collection.acquisition.recording_stream import (
    RecordedRawEvent,
    RecordingBoundary,
    RecordingBoundaryKind,
    RecordingCommand,
    RecordingCommandKind,
    RecordingStreamOverflow,
    RecordingStreamProducer,
)
from exo_collection.domain.events import FrameBatch


def _descriptor() -> dict[str, object]:
    return {
        "device_id": "raw_us",
        "modality": "ultrasound",
        "display_name": "Raw ultrasound",
        "clock_domain": "host",
        "event_kind": "frame",
        "channels": ["ch1", "ch2", "ch3", "ch4"],
        "units": ["adc", "adc", "adc", "adc"],
        "nominal_rate_hz": 25.0,
        "sample_shape": [1000],
        "dtype": "uint8",
        "metadata": {"transport": "raw_ethernet"},
    }


def _producer(queue: Queue[object]) -> RecordingStreamProducer:
    return RecordingStreamProducer(
        queue,
        device_id="raw_us",
        modality="ultrasound",
        descriptor=_descriptor(),
        configuration_snapshot={"interface": "ethernet-1"},
    )


def _frame(channel: int, sequence: int) -> FrameBatch:
    wire = np.full((1, 1000), channel + 1, dtype=np.uint8)
    wire[0, 0] = 0
    wire[0, 1] = channel + 1
    wire[0, -1] = 0xFF
    return FrameBatch(
        device_id="raw_us",
        modality="ultrasound",
        clock_domain="host",
        data=wire,
        frame_rate_hz=25.0,
        host_monotonic_ns=sequence + 1,
        sequence_number=sequence,
        first_frame_index=sequence,
        frame_count=1,
        channel=channel,
        tail_flags=1,
    )


def test_non_recording_events_are_not_enqueued() -> None:
    queue: Queue[object] = Queue(maxsize=8)
    producer = _producer(queue)

    assert producer.forward(_frame(0, 0)) is False
    assert queue.empty()


def test_four_channels_are_wrapped_between_ordered_trial_boundaries() -> None:
    queue: Queue[object] = Queue(maxsize=8)
    producer = _producer(queue)
    trial_uuid = str(uuid4())
    frames = [_frame(channel, channel) for channel in range(4)]

    producer.begin(trial_uuid)
    assert all(producer.forward(frame) for frame in frames)
    producer.end(trial_uuid)

    messages = [queue.get_nowait() for _ in range(6)]
    assert isinstance(messages[0], RecordingBoundary)
    assert messages[0].kind is RecordingBoundaryKind.START
    assert messages[0].trial_uuid == trial_uuid
    assert messages[0].descriptor == _descriptor()
    assert messages[0].configuration_snapshot == {"interface": "ethernet-1"}
    raw_messages = messages[1:5]
    assert all(isinstance(message, RecordedRawEvent) for message in raw_messages)
    assert [message.event.channel for message in raw_messages] == [0, 1, 2, 3]
    assert [message.event for message in raw_messages] == frames
    assert {message.trial_uuid for message in raw_messages} == {trial_uuid}
    assert isinstance(messages[-1], RecordingBoundary)
    assert messages[-1].kind is RecordingBoundaryKind.END
    assert messages[-1].trial_uuid == trial_uuid


def test_sequential_trials_keep_uuid_provenance_isolated() -> None:
    queue: Queue[object] = Queue(maxsize=12)
    producer = _producer(queue)
    first_uuid = str(uuid4())
    second_uuid = str(uuid4())

    producer.begin(first_uuid)
    producer.forward(_frame(0, 0))
    producer.end(first_uuid)
    assert producer.forward(_frame(1, 1)) is False
    producer.begin(second_uuid)
    producer.forward(_frame(2, 2))
    producer.end(second_uuid)

    messages = [queue.get_nowait() for _ in range(6)]
    assert [message.trial_uuid for message in messages] == [
        first_uuid,
        first_uuid,
        first_uuid,
        second_uuid,
        second_uuid,
        second_uuid,
    ]
    assert isinstance(messages[1], RecordedRawEvent)
    assert messages[1].event.channel == 0
    assert isinstance(messages[4], RecordedRawEvent)
    assert messages[4].event.channel == 2


def test_full_queue_raises_and_never_discards_existing_raw_data() -> None:
    queue: Queue[object] = Queue(maxsize=2)
    producer = _producer(queue)
    trial_uuid = str(uuid4())
    first_frame = _frame(0, 0)

    producer.begin(trial_uuid)
    producer.forward(first_frame)
    with pytest.raises(RecordingStreamOverflow, match="recording queue full"):
        producer.forward(_frame(1, 1))

    # Aborting cannot insert FAULT while full, and must never evict START/raw.
    assert producer.abort("overflow") is None
    first = queue.get_nowait()
    second = queue.get_nowait()
    assert isinstance(first, RecordingBoundary)
    assert first.kind is RecordingBoundaryKind.START
    assert isinstance(second, RecordedRawEvent)
    assert second.event is first_frame
    assert producer.recording is False


def test_recording_commands_require_canonical_trial_uuid() -> None:
    trial_uuid = uuid4()
    command = RecordingCommand(
        RecordingCommandKind.START_RECORDING,
        str(trial_uuid).upper(),
    )
    assert command.trial_uuid == str(trial_uuid)
    with pytest.raises(ValueError):
        RecordingCommand(RecordingCommandKind.STOP_RECORDING, "not-a-uuid")
    shutdown = RecordingCommand(RecordingCommandKind.SHUTDOWN)
    assert shutdown.trial_uuid is None
