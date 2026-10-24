"""Deterministic fake-serial tests for the feedback-gated motor service."""

from __future__ import annotations

import math
import struct

import pytest

from exo_collection.apps.model_runtime.motor_service import (
    STATUS_STRUCT,
    MotorFeedback,
    SerialTorqueOutput,
)
from exo_collection.apps.model_runtime.torque import (
    FRAME_TAIL,
    HEAD_STATUS,
    build_control_frame,
    calc_crc8,
)


class Clock:
    def __init__(self) -> None:
        self.ns = 1_000_000_000

    def __call__(self) -> int:
        return self.ns

    def advance(self, seconds: float) -> None:
        self.ns += int(seconds * 1_000_000_000)


class FakeSerial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.rx = bytearray()
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self.rx)

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def read(self, count: int) -> bytes:
        out = bytes(self.rx[:count])
        del self.rx[:count]
        return out

    def close(self) -> None:
        self.closed = True


def status_frame(*, seq: int = 4, state: int = 2, error: int = 0) -> bytes:
    payload = struct.pack(
        "<HBBffffffI",
        seq,
        state,
        error,
        1.0,
        2.0,
        3.0,
        -1.0,
        -2.0,
        -3.0,
        1234,
    )
    return bytes([HEAD_STATUS]) + payload + bytes([calc_crc8(payload), FRAME_TAIL])


def make_service(clock: Clock, serial: FakeSerial, limit: float = 5.0) -> SerialTorqueOutput:
    service = SerialTorqueOutput(
        "FAKE",
        limit,
        serial_factory=lambda *args, **kwargs: serial,
        clock=clock,
        background=False,
    )
    service.connect()
    return service


def test_connect_does_not_enable_and_parser_handles_split_sticky_frames() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    frame = status_frame()
    assert service.connected and not service.enabled
    assert service.feed_feedback(frame[:7]) == 0
    assert service.feed_feedback(frame[7:] + frame) == 2
    assert isinstance(service.feedback, MotorFeedback)
    assert service.feedback.right_position_rad == pytest.approx(-1.0)


def test_enable_requires_feedback_then_clamps_every_transmitted_command() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    service.feed_feedback(status_frame())
    service.set_enabled(True)
    assert not service.enabled
    service.feed_feedback(status_frame(seq=5, state=2))
    assert service.enabled
    service.send(8.0, -8.0)
    seq, left, right, limit = struct.unpack("<H3f", serial.writes[-1][1:15])
    assert (left, right, limit) == pytest.approx((5.0, -5.0, 5.0))
    assert seq >= 0


def test_background_sender_repeats_the_atomic_gate_context_for_each_target():
    import time

    serial = FakeSerial()
    service = SerialTorqueOutput("FAKE", 5,
        serial_factory=lambda *args, **kwargs: serial, background=True)
    try:
        service.connect()
        service.feed_feedback(status_frame(state=0))
        service.set_enabled(True)
        service.feed_feedback(status_frame(seq=5, state=2))
        assert service.enabled

        first = {"mode": "ao", "sequence": 10, "output_allowed": True,
                 "post_left_nm": 1.25, "post_right_nm": 1.25}
        service.send_with_gate_context(1.25, 1.25, first)
        first["sequence"] = 99
        deadline = time.monotonic() + .1
        while time.monotonic() < deadline and sum(
                command.gate_context is not None and command.gate_context.get("sequence") == 10
                for command in service.history) < 2:
            time.sleep(.002)

        second = {"mode": "ao", "sequence": 11, "output_allowed": False,
                  "post_left_nm": 0.0, "post_right_nm": 0.0}
        service.send_with_gate_context(0.0, 0.0, second)
        deadline = time.monotonic() + .1
        while time.monotonic() < deadline and not any(
                command.gate_context is not None and command.gate_context.get("sequence") == 11
                for command in service.history):
            time.sleep(.002)

        repeated = [command for command in service.history
                    if command.gate_context is not None]
        assert sum(command.gate_context["sequence"] == 10 for command in repeated) >= 2
        assert any(command.gate_context["sequence"] == 11 for command in repeated)
        for command in repeated:
            context = command.gate_context
            assert context["post_left_nm"] == command.left_nm
            assert context["post_right_nm"] == command.right_nm
            assert context["output_allowed"] is (context["sequence"] == 10)
    finally:
        service.close()


def test_state_zero_pending_then_new_state_two_confirms_enable() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    service.feed_feedback(status_frame(state=0))
    service.set_enabled(True)
    assert not service.enabled
    service.poll()
    assert not service.enabled and service.fault is None
    service.feed_feedback(status_frame(seq=6, state=2))
    assert service.enabled


def test_enable_confirmation_timeout_is_fault() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    service.feed_feedback(status_frame(state=0))
    service.set_enabled(True)
    clock.advance(0.26)
    service.poll()
    assert service.fault == "enable confirmation timeout"


def test_crc_error_resynchronizes_and_bad_feedback_does_not_enable() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    bad = bytearray(status_frame(seq=1))
    bad[-2] ^= 0xFF
    assert service.feed_feedback(bytes(bad) + status_frame(seq=2)) == 1
    assert service.feedback is not None and service.feedback.seq == 2
    service.set_enabled(True)
    service.feed_feedback(status_frame(seq=3, state=2))
    assert service.enabled
    diagnostics = service.diagnostics
    assert diagnostics["parser_rejected_count"] == 1
    assert diagnostics["parser_crc_or_format_errors"] == 1
    assert diagnostics["parser_discarded_bytes"] >= 1


def test_repeated_unhealthy_feedback_latches_once_and_keeps_first_snapshot() -> None:
    clock, serial = Clock(), FakeSerial()
    events: list[tuple[str, object]] = []
    service = SerialTorqueOutput(
        "FAKE",
        5.0,
        serial_factory=lambda *args, **kwargs: serial,
        clock=clock,
        background=False,
        recording_sink=lambda kind, payload: events.append((kind, payload)),
    )
    service.connect()
    service.feed_feedback(status_frame(seq=1, state=0))
    service.set_enabled(True)
    service.feed_feedback(status_frame(seq=2, state=2))
    service.send(1.0, -1.0)

    bad = status_frame(seq=3, state=2, error=253)
    writes_before_fault = len(serial.writes)
    assert service.feed_feedback(bad) == 1
    writes_after_first_fault = len(serial.writes)
    assert writes_after_first_fault - writes_before_fault == 2
    assert service.fault == "unhealthy motor feedback state=2 error=253"
    assert not service.enabled

    # Repeated copies are accepted for audit continuity but cannot repeat the
    # zero/disable sequence or the recording stop event.
    assert service.feed_feedback(bad * 4) == 4
    assert len(serial.writes) == writes_after_first_fault
    assert sum(kind == "disabled" for kind, _ in events) == 1
    fault_events = [payload for kind, payload in events if kind == "fault"]
    assert len(fault_events) == 1
    assert [kind for kind, _ in events].index("fault") < [kind for kind, _ in events].index("disabled")
    assert fault_events[0]["feedback"]["error"] == 253

    snapshot = service.fault_snapshot
    assert snapshot is not None
    assert snapshot["feedback"]["state"] == 2
    assert snapshot["feedback"]["error"] == 253
    assert snapshot["feedback"]["raw_frame"] == bad.hex()
    assert snapshot["previous_feedback"]["error"] == 0
    snapshot["feedback"]["state"] = 99
    assert service.fault_snapshot["feedback"]["state"] == 2

    # A late healthy frame updates telemetry but never clears the latched
    # safety fault or re-arms output.
    healthy = status_frame(seq=20, state=2, error=0)
    assert service.feed_feedback(healthy) == 1
    assert service.fault == "unhealthy motor feedback state=2 error=253"
    assert not service.enabled
    assert len(serial.writes) == writes_after_first_fault
    diagnostics = service.diagnostics
    assert diagnostics["last_feedback_error"] == 0
    assert diagnostics["fault_snapshot"]["feedback"]["error"] == 253
    assert diagnostics["last_feedback_device_time_us"] == 1234
    assert diagnostics["device_feedback_age_us"] is None

    # Idempotent fault latching must not weaken an explicit disarm/close path.
    writes_before_explicit_disarm = len(serial.writes)
    service.set_enabled(False)
    assert len(serial.writes) == writes_before_explicit_disarm + 2
    assert serial.writes[-1][0] == 0xBB and serial.writes[-1][1] == 0xFD
    service.close()
    assert serial.closed


def test_partial_feedback_is_counted_once_and_recovers_after_corruption() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    frame = status_frame(seq=11)
    assert service.feed_feedback(frame[:7]) == 0
    diagnostics = service.diagnostics
    assert diagnostics["parser_partial_frames"] == 1
    assert diagnostics["parser_buffered_bytes"] == 7
    assert service.feed_feedback(frame[7:]) == 1
    assert service.diagnostics["parser_partial_frames"] == 1

    corrupt = bytearray(status_frame(seq=12))
    corrupt[-2] ^= 0xFF
    assert service.feed_feedback(bytes(corrupt) + status_frame(seq=13)) == 1
    diagnostics = service.diagnostics
    assert diagnostics["parser_rejected_count"] == 1
    assert diagnostics["parser_partial_frames"] == 1
    assert service.feedback is not None and service.feedback.seq == 13


def test_nan_latches_fault_and_disable_blocks_cached_old_command() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    service.feed_feedback(status_frame())
    service.set_enabled(True)
    service.feed_feedback(status_frame(seq=5, state=2))
    service.send(float("nan"), 1.0)
    assert service.fault == "non-finite torque command"
    assert not service.enabled
    count = len(serial.writes)
    service.send(4.0, 4.0)
    assert len(serial.writes) == count


def test_nonfinite_feedback_faults_and_can_be_cleared() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    service.feed_feedback(status_frame())
    service.set_enabled(True)
    bad = status_frame(seq=5)
    bad = bytearray(bad)
    # left position is the first float after head/seq/state/error.
    bad[5:9] = struct.pack("<f", float("nan"))
    bad[33] = calc_crc8(bytes(bad[1:33]))
    service.feed_feedback(bytes(bad))
    assert service.fault == "non-finite motor feedback"
    assert service.feedback is not None and service.feedback.seq == 4
    snapshot = service.fault_snapshot
    assert snapshot is not None
    assert snapshot["feedback"]["raw_frame"] == bytes(bad).hex()
    assert snapshot["feedback"]["state"] == 2
    assert snapshot["feedback"]["error"] == 0
    assert snapshot["feedback"]["device_time"] == 1234
    service.clear_fault()
    assert service.fault is None and not service.enabled
    assert serial.writes[-1][0] == 0xBB and serial.writes[-1][1] == 0xFD


def test_feedback_expiry_disarms_confirmed_output() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    service.feed_feedback(status_frame())
    service.set_enabled(True)
    service.feed_feedback(status_frame(seq=5, state=2))
    clock.advance(0.20)
    service.send(1.0, 1.0)
    clock.advance(0.06)
    service.poll()
    assert service.fault == "motor feedback expired or unhealthy"


def test_limit_drop_and_zero_send_safe_disable() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    service.feed_feedback(status_frame())
    service.set_enabled(True)
    service.feed_feedback(status_frame(seq=5, state=2))
    service.send(4.0, -4.0)
    service.set_limit(2.0)
    _, left, right, limit = struct.unpack("<H3f", serial.writes[-1][1:15])
    assert (left, right, limit) == pytest.approx((2.0, -2.0, 2.0))
    service.set_limit(0.0)
    assert service.torque_limit_nm == 0.0
    assert not service.enabled
    assert serial.writes[-1][0] == 0xBB and serial.writes[-1][1] == 0xFD


def test_command_expiry_faults_and_invalid_limit_is_rejected() -> None:
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    service.feed_feedback(status_frame())
    service.set_enabled(True)
    service.feed_feedback(status_frame(seq=5, state=2))
    service.send(0.0, 0.0)
    clock.advance(0.16)
    service.poll()
    assert service.fault == "torque command expired"
    with pytest.raises(ValueError):
        service.set_limit(math.inf)
    with pytest.raises(ValueError):
        service.set_limit(18.01)


def test_write_shortfall_latches_fault() -> None:
    class ShortSerial(FakeSerial):
        def write(self, data: bytes) -> int:
            self.writes.append(bytes(data))
            return len(data) - 1

    clock, serial = Clock(), ShortSerial()
    service = make_service(clock, serial)
    assert service.fault is not None
    assert "short serial write" in service.fault
    assert service.history == ()


@pytest.mark.parametrize("limit", [12.5, 12.6, 18])
def test_user_protocol_limit_clamps_every_fake_serial_frame_without_rounding_up(limit):
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial, limit=limit)
    try:
        assert not service.enabled
        service.feed_feedback(status_frame(state=0))
        service.set_enabled(True)
        service.feed_feedback(status_frame(seq=5, state=2))
        service.send(100, -100)
        clock.advance(.005)
        service.poll()
        for frame in serial.writes:
            if frame[0] != 0xAA:
                continue
            _, left, right, wire_limit = struct.unpack("<H3f", frame[1:15])
            assert 0 <= wire_limit <= limit
            assert abs(left) <= wire_limit and abs(right) <= wire_limit
        assert service.last.left_nm == limit and service.last.right_nm == -limit
        service.set_limit(9)
        _, left, right, wire_limit = struct.unpack("<H3f", serial.writes[-1][1:15])
        assert (left, right, wire_limit) == (9, -9, 9)
    finally:
        service.close()


@pytest.mark.parametrize("invalid", [-1, 18.01, float("nan"), float("inf"), float("-inf")])
def test_invalid_protocol_threshold_never_connects_and_disarms_existing_fake_output(invalid):
    with pytest.raises(ValueError):
        SerialTorqueOutput("NO_HARDWARE", invalid, background=False)
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial, limit=18)
    try:
        service.feed_feedback(status_frame())
        service.set_enabled(True)
        service.feed_feedback(status_frame(seq=5, state=2))
        assert service.enabled
        with pytest.raises(ValueError):
            service.set_limit(invalid)
        assert not service.enabled and service.fault
    finally:
        service.close()


def test_fault_blackbox_freezes_bounded_trace_and_copies_nested_evidence():
    clock, serial = Clock(), FakeSerial()
    service = SerialTorqueOutput("FAKE", 5, serial_factory=lambda **kw: serial,
        clock=clock, background=False, diagnostic_trace_capacity=8)
    service.connect()
    for seq in range(20):
        clock.advance(.005)
        service.feed_feedback(status_frame(seq=seq))
        service.zero()
    clock.advance(.012)
    terminal = status_frame(seq=21, error=253)
    service.feed_feedback(terminal)
    frozen = service.fault_snapshot
    trace = frozen["transport"]["trace"]
    assert len(trace) == 8
    assert trace[-1]["raw_frame_hex"] == terminal.hex()
    assert trace[-1]["kind"] == "rx_status"
    assert frozen["transport"]["counts"]["rx_valid_frames"] == 21
    assert frozen["transport"]["max_gap_ns"]["rx_valid"] == 12_000_000
    assert {row.get("command") for row in trace if row["kind"] == "tx"} == {"zero"}
    assert frozen["transport"]["threads"]["reader"]["started"] is False
    for _ in range(100):
        service.feed_feedback(terminal)
    assert service.fault_snapshot == frozen
    frozen["transport"]["trace"][-1]["error"] = 0
    assert service.fault_snapshot["transport"]["trace"][-1]["error"] == 253


@pytest.mark.parametrize("sink_raises", [False, True])
def test_fault_sink_cannot_delay_physical_zero_disable(sink_raises):
    clock, serial = Clock(), FakeSerial()
    seen = []
    service = SerialTorqueOutput("FAKE", 5, serial_factory=lambda **kw: serial,
        clock=clock, background=False)
    service.connect()
    service.feed_feedback(status_frame())
    service.set_enabled(True)
    service.feed_feedback(status_frame(seq=5))
    before = len(serial.writes)
    def sink(kind, value):
        seen.append((kind, len(serial.writes)))
        assert len(serial.writes) == before + 2
        if sink_raises:
            raise OSError("audit sink unavailable")
    service._recording_sink = sink
    service.feed_feedback(status_frame(seq=6, error=253))
    assert len(serial.writes) == before + 2
    assert seen and all(count == before + 2 for _, count in seen)
    assert [kind for kind, _ in seen][:2] == ["feedback", "fault"]


@pytest.mark.parametrize("operation", ["read", "write"])
def test_io_exception_preserves_system_fields_and_failed_frame(operation):
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    error = OSError(5, "device transport unavailable")
    error.winerror = 1167
    def failed(*args):
        raise error
    setattr(serial, operation, failed)
    if operation == "read":
        service.poll()
    else:
        service.zero()
    evidence = service.fault_snapshot["transport"]
    assert evidence["last_io_error"]["operation"] == operation
    assert evidence["last_io_error"]["type"] == "builtins.OSError"
    assert evidence["last_io_error"]["errno"] == 5
    assert evidence["last_io_error"]["winerror"] == 1167
    assert "unavailable" in evidence["last_io_error"]["repr"]
    if operation == "write":
        assert any(row.get("result") == "failed" and row.get("raw_frame_hex")
                   for row in evidence["trace"])


def test_fault_captures_thread_liveness_and_last_heartbeat():
    from types import SimpleNamespace
    clock, serial = Clock(), FakeSerial()
    service = make_service(clock, serial)
    service._reader_thread = SimpleNamespace(is_alive=lambda: True)
    service._sender_thread = SimpleNamespace(is_alive=lambda: False)
    service.poll()
    service.feed_feedback(status_frame(error=253))
    threads = service.fault_snapshot["transport"]["threads"]
    assert threads["reader"]["alive"] is True
    assert threads["sender"]["alive"] is False
    assert threads["reader"]["last_heartbeat_ns"] == clock.ns
    assert threads["sender"]["last_heartbeat_ns"] == clock.ns


class CloseFailureSerial(FakeSerial):
    def __init__(self, failures=1):
        super().__init__()
        self.failures = failures
        self.close_attempts = 0
        self.is_open = True
    def close(self):
        self.close_attempts += 1
        if self.failures:
            self.failures -= 1
            error = OSError(5, "serial close failed")
            error.winerror = 1167
            raise error
        self.is_open = False
        super().close()


def test_serial_close_failure_disarms_retains_handle_and_explicit_retry_releases():
    clock, serial = Clock(), CloseFailureSerial()
    service = make_service(clock, serial)
    before = len(serial.writes)
    with pytest.raises(OSError, match="serial close failed"):
        service.close()
    assert not service.connected and not service.enabled
    assert len(serial.writes) == before + 2  # zero and disable before close
    assert service._serial is serial and serial.is_open
    evidence = service.close_snapshot
    assert evidence["serial_close_succeeded"] is False
    assert evidence["serial_is_open_after"] is True
    assert evidence["error"]["errno"] == 5 and evidence["error"]["winerror"] == 1167
    service.send(5, 5)
    service.zero()
    with pytest.raises(RuntimeError, match="closed"):
        service.connect()
    assert len(serial.writes) == before + 2
    service.close()
    assert service._serial is None and serial.closed and not serial.is_open
    assert service.close_snapshot["attempt"] == 2
    assert service.close_snapshot["serial_close_succeeded"] is True
    service.close()
    assert serial.close_attempts == 2 and len(serial.writes) == before + 2


def test_thread_join_exception_does_not_skip_serial_close():
    from types import SimpleNamespace
    clock, serial = Clock(), CloseFailureSerial(failures=0)
    service = make_service(clock, serial)
    def join(**kwargs):
        raise RuntimeError("join failed")
    service._reader_thread = SimpleNamespace(join=join, is_alive=lambda: True)
    service.close()
    assert serial.closed and service._serial is None
    evidence = service.close_snapshot
    assert evidence["serial_close_succeeded"] is True
    assert evidence["join_errors"][0]["thread"] == "reader"
    assert "join failed" in evidence["join_errors"][0]["repr"]
    assert evidence["threads"]["reader"]["alive_after_serial_close"] is True


def test_normal_serial_close_is_idempotent_and_diagnostic_copy_is_detached():
    clock, serial = Clock(), CloseFailureSerial(failures=0)
    service = make_service(clock, serial)
    service.close()
    before = len(serial.writes)
    evidence = service.close_snapshot
    evidence["threads"]["reader"]["alive"] = "mutated"
    service.close()
    assert serial.close_attempts == 1 and len(serial.writes) == before
    assert service.close_snapshot["threads"]["reader"]["alive"] is False
    assert service.diagnostics["transport"]["close"]["serial_close_succeeded"] is True
