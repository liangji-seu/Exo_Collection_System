from __future__ import annotations

from collections import deque
import struct
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from exo_collection.adapters.base import AdapterState, RawQueueOverflowError, TrialContext
from exo_collection.adapters.encoder.teensy_serial import (
    FRAME_TAIL,
    HEAD_STATUS,
    STATUS_FORMAT,
    STATUS_SIZE,
    MotorStatusStreamParser,
    TeensySerialEncoderAdapter,
    calc_crc8,
    find_teensy_port,
    parse_status_frame,
)


def make_frame(sequence: int, *, hardware_time_ms: int = 1234) -> bytes:
    frame = bytearray(
        struct.pack(
            STATUS_FORMAT,
            HEAD_STATUS,
            sequence,
            2,
            0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            hardware_time_ms,
            0,
            FRAME_TAIL,
        )
    )
    frame[-2] = calc_crc8(bytes(frame[1 : STATUS_SIZE - 2]))
    return bytes(frame)


class FakeSerial:
    def __init__(self, chunks: list[bytes] | None = None, **_: object) -> None:
        self.chunks = deque(chunks or [])
        self.is_open = True
        self.writes: list[bytes] = []

    @property
    def in_waiting(self) -> int:
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, size: int = 1) -> bytes:
        if self.chunks:
            chunk = self.chunks.popleft()
            if len(chunk) > size:
                self.chunks.appendleft(chunk[size:])
                return chunk[:size]
            return chunk
        time.sleep(0.002)
        return b""

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.is_open = False


def context() -> TrialContext:
    return TrialContext(trial_uuid=uuid4(), session_uuid=uuid4())


def wait_for(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached")


def test_status_frame_is_exactly_35_bytes_and_round_trips() -> None:
    raw = make_frame(7, hardware_time_ms=9876)
    assert STATUS_SIZE == 35
    assert len(raw) == 35
    parsed = parse_status_frame(raw)
    assert parsed is not None
    assert parsed.sequence == 7
    assert parsed.hardware_time_ms == 9876
    assert parsed.left_position == pytest.approx(1.0)


def test_stream_parser_handles_noise_fragmentation_bad_crc_and_multiple_frames() -> None:
    first, second = make_frame(1), make_frame(2)
    bad = bytearray(make_frame(99))
    bad[-2] ^= 0xFF
    parser = MotorStatusStreamParser()
    output = []
    stream = b"noise" + first + bytes(bad) + second
    for cut in (stream[:9], stream[9:31], stream[31:72], stream[72:]):
        output.extend(parser.feed(cut))
    assert [frame.sequence for frame in output] == [1, 2]
    assert parser.crc_or_format_errors >= 1
    assert parser.discarded_bytes >= len(b"noise")


def test_find_port_matches_vid_pid() -> None:
    ports = [
        SimpleNamespace(device="COM1", vid=1, pid=2),
        SimpleNamespace(device="COM7", vid=0x16C0, pid=0x0483),
    ]
    assert find_teensy_port(ports) == "COM7"


def test_adapter_is_read_only_and_emits_correct_batch() -> None:
    serial_port = FakeSerial([make_frame(10, hardware_time_ms=50) + make_frame(11, hardware_time_ms=60)])
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2, "nominal_rate_hz": 100},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() == 1)
    event = adapter.get_event(timeout=0.1)
    assert event.data.shape == (2, 6)
    assert event.data.dtype == np.float32
    assert event.device_timestamp == 50
    assert event.first_sample_index == 0
    assert serial_port.writes == []
    report = adapter.stop()
    adapter.close()
    assert report.samples_emitted == 2
    assert serial_port.writes == []
    assert not serial_port.is_open


def test_stop_flushes_remaining_valid_frames() -> None:
    serial_port = FakeSerial([make_frame(1), make_frame(2)])
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 20},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter._samples_emitted == 0 and len(adapter._pending) == 2)
    report = adapter.stop()
    event = adapter.get_event(timeout=0.1)
    adapter.close()
    assert report.samples_emitted == 2
    assert event.sample_count == 2


def test_raw_queue_overflow_is_fatal() -> None:
    serial_port = FakeSerial()
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 1, "queue_capacity": 1},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    adapter._accept_frame(parse_status_frame(make_frame(1)), 10)  # type: ignore[arg-type]
    with pytest.raises(RawQueueOverflowError):
        adapter._accept_frame(parse_status_frame(make_frame(2)), 20)  # type: ignore[arg-type]
    assert adapter.state is AdapterState.FAULTED
    assert adapter.health().status.value == "UNHEALTHY"
    adapter.close()


def test_sequence_gap_is_reported() -> None:
    serial_port = FakeSerial([make_frame(1) + make_frame(4)])
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() == 1)
    assert adapter.health().sequence_gaps == 2
    adapter.stop()
    adapter.close()
