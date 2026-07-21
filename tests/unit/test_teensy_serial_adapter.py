"""Tests for the 35‑byte single‑header Teensy encoder protocol adapter."""

from __future__ import annotations

from collections import deque
import struct
import time
from types import SimpleNamespace
from uuid import uuid4

import h5py
import numpy as np
import pytest

from exo_collection.adapters.base import (
    AdapterError,
    AdapterState,
    RawQueueOverflowError,
    TrialContext,
)
from exo_collection.adapters.encoder.teensy_serial import (
    FRAME_TAIL,
    HEAD_STATUS1,
    STATUS_FORMAT,
    STATUS_SIZE,
    _GAP_THRESHOLD_US,
    _TICK_US,
    BAUD_DEFAULT,
    MotorStatusStreamParser,
    TeensySerialEncoderAdapter,
    calc_crc8,
    find_teensy_port,
    parse_status_frame,
)
from exo_collection.writers.hdf5_signal import Hdf5SignalWriter
from exo_collection.acquisition.preview import build_preview_event


# ------------------------------------------------------------------
# Fixture helpers
# ------------------------------------------------------------------


class FakeSerial:
    def __init__(
        self,
        chunks: list[bytes] | None = None,
        **kwargs: object,
    ) -> None:
        self.chunks = deque(chunks or [])
        self.is_open = True
        self.writes: list[bytes] = []
        # Remember factory kwargs for test assertions.
        self._init_kwargs = kwargs

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


def wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached within timeout")


def make_frame(
    sequence: int = 0,
    *,
    teensy_time_us: int = 5000,
    state: int = 2,
    error: int = 0,
    left_position: float = 1.0,
    left_velocity: float = 2.0,
    left_torque: float = 3.0,
    right_position: float = 4.0,
    right_velocity: float = 5.0,
    right_torque: float = 6.0,
) -> bytearray:
    """Build a valid 35‑byte StatusFrame (raw bytearray)."""
    frame = bytearray(
        struct.pack(
            STATUS_FORMAT,
            HEAD_STATUS1,
            sequence,
            state,
            error,
            left_position,
            left_velocity,
            left_torque,
            right_position,
            right_velocity,
            right_torque,
            teensy_time_us,
            0,  # placeholder CRC
            FRAME_TAIL,
        )
    )
    # CRC covers bytes [1:33]
    frame[33] = calc_crc8(bytes(frame[1:33]))
    return frame


def make_raw_frame(**overrides: int | float) -> bytes:
    """Convenience – return immutable bytes."""
    return bytes(make_frame(**overrides))  # type: ignore[arg-type]


# ------------------------------------------------------------------
# 1. Protocol golden fixture
# ------------------------------------------------------------------


def test_status_size_is_35_bytes() -> None:
    assert STATUS_SIZE == 35
    assert struct.calcsize(STATUS_FORMAT) == 35


def test_golden_fixture_all_offsets() -> None:
    raw = make_raw_frame(
        sequence=0x0102,
        teensy_time_us=0xDEAD_BEEF,
        state=2,
        error=0,
    )
    assert len(raw) == 35

    # Single header.
    assert raw[0] == 0xCC
    # seq LE.
    assert raw[1] == 0x02
    assert raw[2] == 0x01
    # state, error.
    assert raw[3] == 2
    assert raw[4] == 0
    # Left position (1.0f).
    pos_bytes = struct.pack("<f", 1.0)
    assert raw[5:9] == pos_bytes
    # Right position (4.0f).
    rpos_bytes = struct.pack("<f", 4.0)
    assert raw[17:21] == rpos_bytes
    # teensy_time_us LE.
    time_bytes = struct.pack("<I", 0xDEAD_BEEF)
    assert raw[29:33] == time_bytes
    # CRC at offset 33.
    expected_crc = calc_crc8(raw[1:33])
    assert raw[33] == expected_crc
    # Tail.
    assert raw[34] == 0x55


def test_crc_covers_bytes_1_to_33_exclusive() -> None:
    raw = make_raw_frame()
    # CRC byte is at offset 33, tail at 34.
    calc = calc_crc8(raw[1:33])
    assert raw[33] == calc


def test_parsed_frame_fields() -> None:
    raw = make_raw_frame(
        sequence=7,
        state=2,
        error=0,
        left_position=0.5,
        right_position=-0.25,
        teensy_time_us=123456,
    )
    parsed = parse_status_frame(raw)
    assert parsed is not None
    assert parsed.sequence == 7
    assert parsed.state == 2
    assert parsed.error == 0
    assert parsed.left_position == pytest.approx(0.5)
    assert parsed.right_position == pytest.approx(-0.25)
    assert parsed.teensy_time_us == 123456


def test_motor_status_frame_field_names_use_teensy_time_us() -> None:
    raw = make_raw_frame(teensy_time_us=9999)
    parsed = parse_status_frame(raw)
    assert parsed is not None
    assert hasattr(parsed, "teensy_time_us")
    assert not hasattr(parsed, "hardware_time_ms")


# ------------------------------------------------------------------
# 2. Rejection tests
# ------------------------------------------------------------------


def test_reject_bad_crc() -> None:
    raw = make_raw_frame()
    bad = bytearray(raw)
    bad[33] ^= 0xFF
    assert parse_status_frame(bytes(bad)) is None


def test_reject_bad_tail() -> None:
    raw = make_raw_frame()
    bad = bytearray(raw)
    bad[34] = 0x00
    assert parse_status_frame(bytes(bad)) is None


def test_reject_wrong_length() -> None:
    raw = make_raw_frame()
    assert parse_status_frame(raw[:34]) is None
    assert parse_status_frame(raw + b"\x00") is None
    assert parse_status_frame(b"") is None


def test_reject_missing_head1() -> None:
    raw = make_raw_frame()
    bad = bytearray(raw)
    bad[0] = 0x00
    assert parse_status_frame(bytes(bad)) is None


# ------------------------------------------------------------------
# 3. Stream parser: noise, fragmentation, multi-frame, re-sync
# ------------------------------------------------------------------


def test_stream_parser_handles_noise_before_valid_frame() -> None:
    noise = b"random\xCCpre\xAAamble"
    valid = make_raw_frame()
    parser = MotorStatusStreamParser()
    output = parser.feed(noise + valid)
    assert len(output) == 1
    assert parser.discarded_bytes == len(noise)
    assert parser.crc_or_format_errors >= 0


def test_stream_parser_arbitrary_fragmentation() -> None:
    first, second = make_raw_frame(), make_raw_frame(teensy_time_us=10000)
    stream = first + second
    parser = MotorStatusStreamParser()
    output: list = []
    for cut in (stream[:7], stream[7:25], stream[25:45], stream[45:72]):
        output.extend(parser.feed(cut))
    assert len(output) == 2
    assert output[0].sequence == 0
    assert output[1].teensy_time_us == 10000


def test_stream_parser_consecutive_frames() -> None:
    frames = (
        make_raw_frame(teensy_time_us=0),
        make_raw_frame(teensy_time_us=5000),
        make_raw_frame(teensy_time_us=10000),
        make_raw_frame(teensy_time_us=15000),
    )
    parser = MotorStatusStreamParser()
    output = parser.feed(b"".join(frames))
    assert len(output) == 4
    for i, f in enumerate(output):
        assert f.teensy_time_us == i * 5000


def test_stream_parser_bad_frame_then_re_sync() -> None:
    good1 = make_raw_frame(teensy_time_us=5000)
    bad = bytearray(make_raw_frame(teensy_time_us=99999))
    bad[33] ^= 0xFF  # corrupt CRC
    good2 = make_raw_frame(teensy_time_us=15000)
    parser = MotorStatusStreamParser()
    stream = b"".join([bytes(f) for f in (good1, bad, good2)])
    output = parser.feed(stream)
    assert len(output) == 2
    assert output[0].teensy_time_us == 5000
    assert output[1].teensy_time_us == 15000
    assert parser.crc_or_format_errors >= 1


def test_stream_parser_cc_in_payload_no_false_sync() -> None:
    """A payload byte equal to the single 0xCC head must not split a frame."""
    frame = make_raw_frame(
        left_position=struct.unpack("<f", b"\xCC\xAA\x00\x00")[0],
        teensy_time_us=5000,
    )
    # Append a second valid frame that starts with a real CC.
    second = make_raw_frame(teensy_time_us=10000)
    parser = MotorStatusStreamParser()
    output = parser.feed(bytes(frame) + bytes(second))
    assert len(output) == 2
    assert output[0].teensy_time_us == 5000
    assert output[1].teensy_time_us == 10000


def test_stream_parser_false_cc_candidate_is_discarded_and_resynchronised() -> None:
    """A false single-byte head is rejected after a complete bad candidate."""
    parser = MotorStatusStreamParser()
    output = parser.feed(b"\xCC\xFF" + make_raw_frame(teensy_time_us=10_000))
    assert len(output) == 1
    assert output[0].teensy_time_us == 10_000
    assert parser.crc_or_format_errors == 1
    # Both the false 0xCC and the following junk byte are discarded.
    assert parser.discarded_bytes == 2


def test_stream_parser_partial_frame_awaits_more_data() -> None:
    raw = make_raw_frame()
    parser = MotorStatusStreamParser()
    output = parser.feed(raw[:20])
    assert len(output) == 0
    assert parser.buffered_bytes == 20
    output = parser.feed(raw[20:])
    assert len(output) == 1


def test_stream_parser_reset_clears_state() -> None:
    raw = make_raw_frame()
    parser = MotorStatusStreamParser()
    parser.feed(raw[:20])
    parser.reset()
    assert parser.buffered_bytes == 0
    assert parser.crc_or_format_errors == 0
    assert parser.discarded_bytes == 0


# ------------------------------------------------------------------
# 4. Serial factory params
# ------------------------------------------------------------------


def test_serial_factory_default_params() -> None:
    serial_port = FakeSerial(
        [make_raw_frame(), make_raw_frame(teensy_time_us=5050)],
    )
    assert BAUD_DEFAULT == 1_000_000

    def factory(**kwargs: object) -> FakeSerial:
        assert kwargs.get("baudrate") == 1_000_000
        assert kwargs.get("timeout") == 0.05
        return serial_port

    adapter = TeensySerialEncoderAdapter(
        {"port": "COM99", "batch_size": 2},
        serial_factory=factory,
    )
    adapter.connect()
    assert adapter.configuration_snapshot()["resolved_port"] == "COM99"
    adapter.close()


# ------------------------------------------------------------------
# 5. VID / PID auto-discovery
# ------------------------------------------------------------------


def test_find_port_matches_vid_pid() -> None:
    ports = [
        SimpleNamespace(device="COM1", vid=1, pid=2),
        SimpleNamespace(device="COM7", vid=0x16C0, pid=0x0483),
    ]
    assert find_teensy_port(ports) == "COM7"


def test_find_port_no_match() -> None:
    ports = [
        SimpleNamespace(device="COM1", vid=0x1111, pid=0x2222),
        SimpleNamespace(device="COM2", vid=0x3333, pid=0x4444),
    ]
    assert find_teensy_port(ports) is None


def test_find_port_explicit_vid_pid() -> None:
    ports = [
        SimpleNamespace(device="COM7", vid=0xABCD, pid=0x1234),
    ]
    assert find_teensy_port(ports, vid=0xABCD, pid=0x1234) == "COM7"
    assert find_teensy_port(ports) is None


def test_find_port_rejects_bluetooth_even_if_vid_pid_are_misreported() -> None:
    ports = [
        SimpleNamespace(
            device="COM7",
            vid=0x16C0,
            pid=0x0483,
            description="蓝牙链接上的标准串行",
            hwid="BTHENUM\\fake",
        ),
        SimpleNamespace(
            device="COM12",
            vid=0x16C0,
            pid=0x0483,
            description="USB Serial Device",
            hwid="USB VID:PID=16C0:0483",
        ),
    ]
    assert find_teensy_port(ports) == "COM12"


def test_connect_with_explicit_com_port() -> None:
    serial_port = FakeSerial([make_raw_frame()])
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM42", "batch_size": 1},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    snap = adapter.configuration_snapshot()
    assert snap["resolved_port"] == "COM42"
    adapter.close()


def test_connect_fails_when_no_port_available() -> None:
    adapter = TeensySerialEncoderAdapter(
        {"port": None, "vid": 0xDEAD, "pid": 0xBEEF},
        port_lister=lambda: [
            SimpleNamespace(device="COM1", vid=0x1111, pid=0x2222),
        ],
    )
    with pytest.raises(Exception):
        adapter.connect()


# ------------------------------------------------------------------
# 6. Time-based gap detection (no firmware seq gaps)
# ------------------------------------------------------------------


def test_two_frames_seq_zero_no_false_gap() -> None:
    """Both frames have seq=0 (command echo); time diff is normal → no gap."""
    serial_port = FakeSerial(
        [
            make_raw_frame(sequence=0, teensy_time_us=5000),
            make_raw_frame(sequence=0, teensy_time_us=10000),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() >= 1)
    health = adapter.health()
    # No timestamp gaps — normal 5000 us diff.
    assert health.sequence_gaps == 0
    assert health.dropped_packets == 0
    adapter.stop()
    adapter.close()


def test_normal_5000us_tick_no_gap() -> None:
    serial_port = FakeSerial(
        [
            make_raw_frame(teensy_time_us=5000),
            make_raw_frame(teensy_time_us=10000),
            make_raw_frame(teensy_time_us=15000),
            make_raw_frame(teensy_time_us=20000),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() >= 2)
    health = adapter.health()
    assert health.sequence_gaps == 0
    adapter.stop()
    adapter.close()


def test_time_discontinuity_detected_and_batch_split() -> None:
    """A jump beyond _GAP_THRESHOLD_US must trigger gap event and batch split."""
    serial_port = FakeSerial(
        [
            # Three normal frames, then a jump.
            make_raw_frame(teensy_time_us=5000),
            make_raw_frame(teensy_time_us=10000),
            make_raw_frame(teensy_time_us=30000),  # gap: 20000 us > 10000
            make_raw_frame(teensy_time_us=35000),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    # Expect a batch flushes when the gap is detected; first 2 frames form
    # one batch, then gap-flush, then the remaining 2 form another batch.
    wait_for(lambda: adapter.raw_queue.qsize() >= 2)
    health = adapter.health()
    assert health.sequence_gaps == 1
    # Verify the gap event is in health metrics.
    metrics = health.metrics
    assert metrics.get("timestamp_gap_events") is not None
    assert metrics.get("gap_threshold_us") == 7_500
    adapter.stop()
    adapter.close()


def test_uint32_wrap_is_not_reported_as_gap() -> None:
    """uint32 wraparound (e.g. 0xFFFF_FFF0 → 0x0000_000A) must be handled."""
    serial_port = FakeSerial(
        [
            make_raw_frame(teensy_time_us=0xFFFF_FFF0),
            make_raw_frame(teensy_time_us=0x0000_000A),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() >= 1)
    health = adapter.health()
    # uint32 wrap should NOT be flagged as a gap.
    assert health.sequence_gaps == 0
    adapter.stop()
    adapter.close()


def test_gap_threshold_has_half_period_jitter_margin() -> None:
    assert _GAP_THRESHOLD_US == 1.5 * _TICK_US
    assert _TICK_US == 5_000
    assert _GAP_THRESHOLD_US == 7_500


# ------------------------------------------------------------------
# 7. Data column layout and preview mapping
# ------------------------------------------------------------------


def test_data_six_columns_left_right_mapping() -> None:
    serial_port = FakeSerial(
        [
            make_raw_frame(
                left_position=0.1,
                left_velocity=0.2,
                left_torque=0.3,
                right_position=0.4,
                right_velocity=0.5,
                right_torque=0.6,
                teensy_time_us=5000,
            ),
            make_raw_frame(
                left_position=1.1,
                left_velocity=1.2,
                left_torque=1.3,
                right_position=1.4,
                right_velocity=1.5,
                right_torque=1.6,
                teensy_time_us=10000,
            ),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() == 1)
    event = adapter.get_event(timeout=0.1)
    assert event.data.shape == (2, 6)
    assert event.data.dtype == np.float32
    # Row 0: (lp, lv, lt, rp, rv, rt)
    np.testing.assert_allclose(
        event.data[0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    np.testing.assert_allclose(
        event.data[1], [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    )
    adapter.stop()
    adapter.close()


def test_preview_exposes_all_six_encoder_channels() -> None:
    serial_port = FakeSerial(
        [
            make_raw_frame(
                left_position=0.1,
                left_velocity=0.2,
                left_torque=0.3,
                right_position=0.4,
                right_velocity=0.5,
                right_torque=0.6,
                teensy_time_us=5000,
            ),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 1},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() == 1)
    event = adapter.get_event(timeout=0.1)
    preview = build_preview_event(event, trial_uuid=uuid4())
    payload = preview.payload
    np.testing.assert_allclose(
        np.asarray(payload["channels"]),
        np.asarray([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]),
    )
    assert payload["labels"] == [
        "left_position",
        "left_velocity",
        "left_torque",
        "right_position",
        "right_velocity",
        "right_torque",
    ]
    assert payload["channel"] == "position_velocity_torque"
    assert payload["channel_count"] == 6
    adapter.stop()
    adapter.close()


def test_config_default_200hz_and_units() -> None:
    adapter = TeensySerialEncoderAdapter()
    assert adapter._config.nominal_rate_hz == 200.0
    desc = adapter.descriptor()
    assert desc.nominal_rate_hz == 200.0
    assert desc.units == ("rad", "rad/s", "N*m", "rad", "rad/s", "N*m")
    assert desc.metadata["hardware_tick_hz"] == 1_000_000
    assert desc.metadata["device_timestamp_unit"] == "us"


# ------------------------------------------------------------------
# 8. State / error / seq in health and configuration snapshot
# ------------------------------------------------------------------


def test_state_error_seq_in_health_snapshot() -> None:
    serial_port = FakeSerial(
        [
            make_raw_frame(sequence=42, state=7, error=1, teensy_time_us=5000),
            make_raw_frame(sequence=43, state=7, error=0, teensy_time_us=10000),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() == 1)
    health = adapter.health()
    metrics = health.metrics
    assert metrics["last_fw_state"] == 7
    assert metrics["last_fw_state_name"] == "unknown"
    assert metrics["last_fw_error_code"] == 0
    assert metrics["last_fw_error_name"] == "normal"
    assert metrics["last_fw_sequence"] == 43
    assert metrics["non_zero_error_count"] == 1

    snap = adapter.configuration_snapshot()
    assert snap["last_fw_state"] == 7
    assert snap["last_fw_state_name"] == "unknown"
    assert snap["last_fw_error_code"] == 0
    assert snap["last_fw_error_name"] == "normal"
    assert snap["last_fw_sequence"] == 43
    assert snap["non_zero_error_count"] == 1
    adapter.stop()
    adapter.close()


@pytest.mark.parametrize(
    ("state", "error", "state_name", "error_name"),
    [
        (0, 0x00, "disabled", "normal"),
        (2, 0xFD, "enabled", "one_or_more_motor_feedback_timeouts"),
        (0, 0xFE, "disabled", "host_control_frame_timeout_auto_disabled"),
        (2, 0x12, "enabled", "AK80_V3_motor_fault_code"),
    ],
)
def test_firmware_state_and_error_codes_are_named_in_health(
    state: int,
    error: int,
    state_name: str,
    error_name: str,
) -> None:
    serial_port = FakeSerial(
        [make_raw_frame(state=state, error=error, teensy_time_us=5_000)]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM12", "batch_size": 1},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() == 1)
    metrics = adapter.health().metrics
    assert metrics["last_fw_state_name"] == state_name
    assert metrics["last_fw_error_name"] == error_name
    adapter.stop()
    adapter.close()


# ------------------------------------------------------------------
# 9. Lifecycle: stop, close, partial flush, queue overflow, zero write
# ------------------------------------------------------------------


def test_adapter_is_read_only_and_emits_correct_batch() -> None:
    serial_port = FakeSerial(
        [
            make_raw_frame(teensy_time_us=5000) + make_raw_frame(teensy_time_us=10000),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2, "nominal_rate_hz": 200},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() == 1)
    event = adapter.get_event(timeout=0.1)
    assert event.data.shape == (2, 6)
    assert event.data.dtype == np.float32
    # device_timestamp is teensy_time_us of the first frame.
    assert event.device_timestamp == 5000
    assert event.first_sample_index == 0
    assert event.sample_rate_hz == 200
    assert serial_port.writes == []
    report = adapter.stop()
    adapter.close()
    assert report.samples_emitted == 2
    assert serial_port.writes == []
    assert not serial_port.is_open


def test_stop_flushes_remaining_valid_frames() -> None:
    serial_port = FakeSerial(
        [make_raw_frame(teensy_time_us=5000), make_raw_frame(teensy_time_us=10000)],
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 20},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(
        lambda: adapter._samples_emitted == 0 and len(adapter._pending) == 2
    )
    report = adapter.stop()
    assert report.samples_emitted == 2
    event = adapter.get_event(timeout=0.1)
    assert event.sample_count == 2
    adapter.close()


def test_raw_queue_overflow_is_fatal() -> None:
    serial_port = FakeSerial()
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 1, "queue_capacity": 1},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    frame = parse_status_frame(make_raw_frame(teensy_time_us=5000))
    assert frame is not None
    adapter._accept_frame(frame, 10)
    with pytest.raises(RawQueueOverflowError):
        adapter._accept_frame(frame, 20)
    assert adapter.state is AdapterState.FAULTED
    assert adapter.health().status.value == "UNHEALTHY"
    adapter.close()


def test_zero_serial_writes_across_lifecycle() -> None:
    serial_port = FakeSerial(
        [
            make_raw_frame(teensy_time_us=5000),
            make_raw_frame(teensy_time_us=10000),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 1},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    assert serial_port.writes == []
    adapter.prepare(context())
    assert serial_port.writes == []
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() >= 1)
    assert serial_port.writes == []
    adapter.stop()
    assert serial_port.writes == []
    adapter.close()
    assert serial_port.writes == []


def test_stop_and_close_with_empty_pending() -> None:
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 20},
        serial_factory=lambda **kwargs: FakeSerial([]),
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    time.sleep(0.05)
    report = adapter.stop()
    assert report.samples_emitted == 0
    adapter.close()


# ------------------------------------------------------------------
# 10. HDF5 integration: 200 Hz, device_time via us ticks, gap handling
# ------------------------------------------------------------------


def test_hdf5_appends_device_time_using_hardware_tick_hz(tmp_path) -> None:
    """With hardware_tick_hz=1_000_000 and 200 Hz sample rate, the device_time
    step must be 5_000 ticks per sample."""
    adapter = TeensySerialEncoderAdapter()
    desc = adapter.descriptor()

    writer = Hdf5SignalWriter(
        path=tmp_path / "teensy_integration.h5",
        channels=desc.channels,
        units=desc.units,
        device_metadata=dict(desc.metadata),
        nominal_rate_hz=desc.nominal_rate_hz,
        chunk_rows=16,
        overwrite=True,
    )
    # Simulate one batch of 3 frames at 5000 us spacing.
    batch_data = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
         [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
         [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]],
        dtype=np.float32,
    )
    writer.append(
        batch_data,
        sample_index=0,
        device_time=5000,  # teensy_time_us of first frame
        host_monotonic_ns=100_000_000,
        sample_rate_hz=200.0,
    )
    # _normalise_device_times uses hardware_tick_hz / sample_rate_hz = 5000
    # So device_time should be [5000, 10000, 15000]
    with h5py.File(writer.path, "r") as f:
        dt = f["samples/device_time"][:]
        np.testing.assert_allclose(dt, [5000.0, 10000.0, 15000.0])
        assert f.attrs["nominal_rate_hz"] == 200.0
    writer.close()


def test_hdf5_batch_append_preserves_device_timestamp(tmp_path) -> None:
    """append_batch reconstructs device_time vector from device_timestamp."""
    adapter = TeensySerialEncoderAdapter()
    desc = adapter.descriptor()

    writer = Hdf5SignalWriter(
        path=tmp_path / "batch_append.h5",
        channels=desc.channels,
        units=desc.units,
        device_metadata=dict(desc.metadata),
        nominal_rate_hz=desc.nominal_rate_hz,
        chunk_rows=16,
        overwrite=True,
    )
    from exo_collection.domain.events import SampleBatch

    data = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
         [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]],
        dtype=np.float32,
    )
    batch = SampleBatch(
        device_id="test",
        modality="encoder",
        clock_domain="test_clock",
        first_sample_index=0,
        sample_count=2,
        sequence_number=0,
        device_timestamp=5000,
        sample_rate_hz=200.0,
        data=data,
    )
    writer.append_batch(batch)
    writer.close()

    with h5py.File(writer.path, "r") as f:
        dt = f["samples/device_time"][:]
        # step = hardware_tick_hz / sample_rate_hz = 1_000_000 / 200 = 5000
        np.testing.assert_allclose(dt, [5000.0, 10000.0])


def test_hdf5_discontinuity_on_time_gap(tmp_path) -> None:
    """Verify device_time fidelity across batches separated by a time gap.

    When the adapter detects a tick gap (e.g. 10000 → 35000 us) it flushes
    the current batch and starts a new one.  The HDF5 writer stores the
    device_timestamp per batch and expands it to a per-sample device_time
    vector using hardware_tick_hz / sample_rate_hz.

    The sample_index stays continuous (no frame counter is used), so the
    writer does NOT create a sample_index_gap discontinuity.  The time gap
    is visible as a jump in the stored device_time values.
    """

    adapter = TeensySerialEncoderAdapter()
    desc = adapter.descriptor()

    writer = Hdf5SignalWriter(
        path=tmp_path / "discontinuity.h5",
        channels=desc.channels,
        units=desc.units,
        device_metadata=dict(desc.metadata),
        nominal_rate_hz=desc.nominal_rate_hz,
        chunk_rows=16,
        overwrite=True,
    )

    from exo_collection.domain.events import SampleBatch

    # Batch 0: frames at times 5000, 10000
    batch0 = SampleBatch(
        device_id="test", modality="encoder", clock_domain="test_clock",
        first_sample_index=0, sample_count=2, sequence_number=0,
        device_timestamp=5000, sample_rate_hz=200.0,
        data=np.float32([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                         [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]]),
    )
    # Batch 1: frames at times 35000, 40000 (adapter flushed at gap 10000→35000).
    # sample_index continues from where batch 0 left off — no forged gap.
    batch1 = SampleBatch(
        device_id="test", modality="encoder", clock_domain="test_clock",
        first_sample_index=2, sample_count=2, sequence_number=1,
        device_timestamp=35000, sample_rate_hz=200.0,
        data=np.float32([[3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
                         [4.1, 4.2, 4.3, 4.4, 4.5, 4.6]]),
    )
    writer.append_batch(batch0)
    writer.append_batch(batch1)
    writer.close()

    with h5py.File(writer.path, "r") as f:
        dt = f["samples/device_time"][:]
        # device_time faithfully records the per-sample us timestamps.
        np.testing.assert_allclose(dt, [5000.0, 10000.0, 35000.0, 40000.0])
        # Batch split is visible as separate device_timestamps.
        assert len(f["samples/data"]) == 4
        # sample_index is continuous (2 = 0+2), so no sample_index_gap written.
        disc = f["events/discontinuities"][:]
        kinds = [str(d[2]) for d in disc]
        assert "sample_index_gap" not in kinds


# ------------------------------------------------------------------
# 11. Descriptor metadata sanity
# ------------------------------------------------------------------


def test_descriptor_metadata_fields() -> None:
    adapter = TeensySerialEncoderAdapter()
    desc = adapter.descriptor()
    meta = desc.metadata
    assert meta["protocol"] == "teensy_ak80_v3_status_v3"
    assert meta["status_size_bytes"] == 35
    assert meta["device_timestamp_field"] == "teensy_time_us"
    assert meta["device_timestamp_unit"] == "us"
    assert meta["hardware_tick_hz"] == 1_000_000
    assert meta["tick_period_us"] == 5_000
    assert meta["gap_threshold_us"] == 7_500
    assert meta["gap_threshold_periods"] == 1.5
    assert meta["crc8_range"] == "bytes[1:33]"
    assert meta["read_only"] is True
    assert meta["simulated"] is False
    assert meta["fw_seq_description"] == "command-echo (not frame counter)"
    assert meta["teensy_status_rate_hz"] == 200.0
    assert meta["motor_can_feedback_rate_hz"] == 50.0
    assert meta["torque_semantics"] == "estimated_from_iq"
    assert meta["torque_coefficient_nm_per_a"] == pytest.approx(0.5701)
    assert meta["state_codes"] == {"0": "disabled", "2": "enabled"}
    assert meta["error_codes"]["0xFD"] == "one_or_more_motor_feedback_timeouts"
    assert meta["error_codes"]["0xFE"] == "host_control_frame_timeout_auto_disabled"


# ------------------------------------------------------------------
# 12. Gap threshold — one missing frame (10_000 us) IS a gap
# ------------------------------------------------------------------


def test_one_missing_frame_10_000us_is_detected_as_gap() -> None:
    """A two-period interval is above the 1.5-period jitter margin."""
    serial_port = FakeSerial(
        [
            make_raw_frame(teensy_time_us=5000),
            make_raw_frame(teensy_time_us=10000),
            make_raw_frame(teensy_time_us=20000),  # gap: 10_000 us == threshold
            make_raw_frame(teensy_time_us=25000),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 2},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() >= 2)
    health = adapter.health()
    assert health.sequence_gaps == 1
    adapter.stop()
    adapter.close()


# ------------------------------------------------------------------
# 13. Unwrapped device_timestamp survives uint32 wrap across batches
# ------------------------------------------------------------------


def test_device_timestamp_unwrapped_across_uint32_wrap() -> None:
    """When teensy_time_us wraps around 2^32 between batches, the
    emitted device_timestamp must continue to increase monotonically
    instead of dropping back to a small value."""
    # Frame just before wrap, and frame just after wrap — for two batches.
    before_wrap = make_raw_frame(teensy_time_us=0xFFFF_FFF0)  # 4294967280
    after_wrap = make_raw_frame(teensy_time_us=0x0000_000A)    # 10

    serial_port = FakeSerial([before_wrap, after_wrap])
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 1},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() >= 2)

    # First batch: unwrapped = 0xFFFF_FFF0 = 4294967280
    event_1 = adapter.get_event(timeout=0.1)
    assert event_1.device_timestamp == 0xFFFF_FFF0

    # Second batch: unwrapped = 1 * 2^32 + 10 = 4294967306
    event_2 = adapter.get_event(timeout=0.1)
    assert event_2.device_timestamp == (1 << 32) + 10
    # Must be strictly larger than the first batch timestamp.
    assert event_2.device_timestamp > event_1.device_timestamp

    adapter.stop()
    adapter.close()


def test_hdf5_device_time_no_regression_at_uint32_wrap(tmp_path) -> None:
    """HDF5 device_time vector must not regress from ~4.29e9 to a small
    value when two batches straddle a uint32 wrap boundary.

    The current schema stores the first expanded device timestamp for each
    batch; Hdf5SignalWriter derives later rows from the nominal sample rate.
    """
    adapter = TeensySerialEncoderAdapter()
    desc = adapter.descriptor()

    writer = Hdf5SignalWriter(
        path=tmp_path / "wrap_integration.h5",
        channels=desc.channels,
        units=desc.units,
        device_metadata=dict(desc.metadata),
        nominal_rate_hz=desc.nominal_rate_hz,
        chunk_rows=16,
        overwrite=True,
    )
    from exo_collection.domain.events import SampleBatch

    data_row = np.float32([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

    # Batch before wrap: teensy_time_us = 0xFFFF_FFF0 (unwrapped same)
    batch_before = SampleBatch(
        device_id="test", modality="encoder", clock_domain="test_clock",
        first_sample_index=0, sample_count=1, sequence_number=0,
        device_timestamp=0xFFFF_FFF0, sample_rate_hz=200.0,
        data=data_row,
    )
    # Batch after wrap: unwrapped = 2^32 + 10 = 4294967306
    batch_after = SampleBatch(
        device_id="test", modality="encoder", clock_domain="test_clock",
        first_sample_index=1, sample_count=1, sequence_number=1,
        device_timestamp=(1 << 32) + 10, sample_rate_hz=200.0,
        data=data_row,
    )
    writer.append_batch(batch_before)
    writer.append_batch(batch_after)
    writer.close()

    with h5py.File(writer.path, "r") as f:
        dt = f["samples/device_time"][:]
        # device_time must be monotonic: [4294967280, 4294967306]
        np.testing.assert_allclose(dt, [4294967280.0, 4294967306.0])
        assert dt[1] > dt[0]


# ------------------------------------------------------------------
# 14. Dead state verification
# ------------------------------------------------------------------


def test_write_call_count_is_removed() -> None:
    """_write_call_count must not exist on the adapter instance."""
    adapter = TeensySerialEncoderAdapter()
    assert not hasattr(adapter, "_write_call_count")


@pytest.mark.parametrize(
    ("previous", "reset_value"),
    [(10_000, 100), (0xF000_0000, 100)],
)
def test_non_wrap_clock_reset_faults_without_regressed_event(
    previous: int, reset_value: int
) -> None:
    serial_port = FakeSerial(
        [
            make_raw_frame(teensy_time_us=5_000),
            make_raw_frame(teensy_time_us=previous),
            make_raw_frame(teensy_time_us=reset_value),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 1},
        serial_factory=lambda **kwargs: serial_port,
    )
    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.state is AdapterState.FAULTED)

    with pytest.raises(AdapterError, match="clock moved backward"):
        adapter.raise_if_faulted()
    events = []
    while (event := adapter.get_event(timeout=0.01)) is not None:
        events.append(event)
    assert [event.device_timestamp for event in events] == [5_000, previous]

    adapter.stop()
    adapter.close()


def test_gap_threshold_scales_with_configured_rate() -> None:
    serial_port = FakeSerial(
        [
            make_raw_frame(teensy_time_us=10_000),
            make_raw_frame(teensy_time_us=20_000),
            make_raw_frame(teensy_time_us=30_000),
        ]
    )
    adapter = TeensySerialEncoderAdapter(
        {"port": "COM7", "batch_size": 1, "nominal_rate_hz": 100.0},
        serial_factory=lambda **kwargs: serial_port,
    )
    assert adapter.descriptor().metadata["tick_period_us"] == 10_000
    assert adapter.descriptor().metadata["gap_threshold_us"] == 15_000

    adapter.connect()
    adapter.prepare(context())
    adapter.start()
    wait_for(lambda: adapter.raw_queue.qsize() == 3)
    assert adapter.health().sequence_gaps == 0
    adapter.stop()
    adapter.close()
