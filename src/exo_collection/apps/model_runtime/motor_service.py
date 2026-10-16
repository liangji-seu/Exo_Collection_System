"""Thread-safe serial torque service for the ExoCode motor controller.

The service owns one serial object for both control writes and feedback reads.
It deliberately does not open the port in ``__init__`` and never enables the
motors merely because a port was connected.  ``background=False`` together
with :meth:`poll` gives tests a deterministic, no-thread fake-serial path.
"""

from __future__ import annotations

import math
from copy import deepcopy
import struct
import threading
from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns, sleep
from typing import Callable, Any
from exo_collection.domain.motor_protocol import TORQUE_MAX_NM

from .torque import (
    DISABLE_CMD,
    ENABLE_CMD,
    FRAME_TAIL,
    HEAD_STATUS,
    SEND_HZ,
    TorqueCommand,
    TorqueOutput,
    build_control_frame,
    build_enable_frame,
    clamp_torque,
    calc_crc8,
)


SERIAL_BAUD = 1_000_000
STATUS_FORMAT = "<BHBBffffffIBB"
STATUS_STRUCT = struct.Struct(STATUS_FORMAT)
STATUS_SIZE = STATUS_STRUCT.size
_CONTROL_PERIOD_NS = int(1_000_000_000 / SEND_HZ)
_DISABLED_STATE = 0
_NORMAL_STATE = 2  # ExoCode.ino publishes state=2 for a healthy control loop.
_SAFE_ENABLE_STATES = {_DISABLED_STATE, _NORMAL_STATE}


@dataclass(frozen=True, slots=True)
class MotorFeedback:
    """One CRC-verified status frame and its host receive timestamp."""

    host_monotonic_ns: int
    seq: int
    state: int
    error: int
    left_position_rad: float
    left_velocity_rad_s: float
    left_torque_nm: float
    right_position_rad: float
    right_velocity_rad_s: float
    right_torque_nm: float
    device_time: int | None = None
    raw_frame: bytes = b""


def _copy_feedback_snapshot(feedback: MotorFeedback | None) -> dict[str, Any] | None:
    """Return a small, UI-safe projection of one feedback frame.

    ``MotorFeedback`` itself remains the public rich value.  The projection is
    intentionally limited to one frame so diagnostics cannot grow with the
    length of a run.
    """
    if feedback is None:
        return None
    raw_hex = bytes(feedback.raw_frame).hex()
    return {
        # Keep the diagnostics projection JSON-safe while retaining the full
        # frame for an offline protocol audit.
        "raw_frame": raw_hex,
        "raw_frame_hex": raw_hex,
        "seq": feedback.seq,
        "state": feedback.state,
        "error": feedback.error,
        "host_monotonic_ns": feedback.host_monotonic_ns,
        "device_time": feedback.device_time,
    }


def _copy_fault_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    """Detach the two bounded nested feedback projections from a snapshot."""
    if snapshot is None:
        return None
    return deepcopy(snapshot)


def _finite_limit(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("torque limit must be a finite non-negative number") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("torque limit must be a finite non-negative number")
    if value < 0:
        raise ValueError("torque limit must be non-negative")
    if value > TORQUE_MAX_NM:
        raise ValueError(f"torque limit cannot exceed {TORQUE_MAX_NM:g} Nm")
    return value


class SerialTorqueOutput(TorqueOutput):
    """A bilateral 200 Hz torque output with feedback-gated enabling.

    ``serial_factory`` receives ``(port, 1_000_000, timeout=..., write_timeout=...)``
    and is intended to be a pyserial-compatible factory. The factory is
    injected in tests; no hardware is accessed by construction.

    ExoCode status ``0`` (disabled) and ``2`` (enabled) with zero error are
    safe preconditions. After an enable frame, output remains zero and pending
    until a newer status ``2`` frame confirms the request.
    """

    def __init__(
        self,
        port: str,
        torque_limit_nm: float = 0.0,
        command_timeout_s: float = 0.15,
        feedback_timeout_s: float = 0.25,
        serial_factory: Callable[..., Any] | None = None,
        *,
        clock: Callable[[], int] | None = None,
        background: bool = True,
        history_capacity: int = 4096,
        diagnostic_trace_capacity: int = 1024,
        recording_sink=None,
    ) -> None:
        self._recording_sink = recording_sink
        self._recording_active = False
        self._deferred_record_events = None
        self.port = str(port)
        self._limit = _finite_limit(torque_limit_nm)
        self.command_timeout_ns = self._duration_ns(command_timeout_s, "command_timeout_s")
        self.feedback_timeout_ns = self._duration_ns(feedback_timeout_s, "feedback_timeout_s")
        self._serial_factory = serial_factory
        self._clock = clock or perf_counter_ns
        self._background = bool(background)

        self._serial: Any | None = None
        self._connected = False
        self._closed = False
        self._enabled = False
        self._enable_requested = False
        self._fault: str | None = None
        self._fault_snapshot: dict[str, Any] | None = None
        self._feedback: MotorFeedback | None = None
        self._previous_feedback: MotorFeedback | None = None
        self._feedback_generation = 0
        self._enable_request_generation: int | None = None
        self._enable_deadline_ns: int | None = None
        self._rx_buffer = bytearray()
        self._parser_rejected_count = 0
        self._parser_crc_or_format_errors = 0
        self._parser_discarded_bytes = 0
        self._parser_partial_frames = 0
        self._parser_partial_pending = False
        self._desired_left = 0.0
        self._desired_right = 0.0
        self._last_command_ns: int | None = None
        self._seq = 0
        self._history: deque[TorqueCommand] = deque(maxlen=int(history_capacity))
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._close_snapshot = None
        self._stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._sender_thread: threading.Thread | None = None
        self._io_trace: deque[dict[str, Any]] = deque(maxlen=max(1, min(4096, int(diagnostic_trace_capacity))))
        self._io_counts = {"rx_bytes": 0, "rx_valid_frames": 0, "tx_attempts": 0,
                           "tx_frames": 0, "tx_bytes": 0, "read_errors": 0, "write_errors": 0}
        self._io_last_ns = {"rx_valid": None, "tx": None, "reader_heartbeat": None,
                            "sender_heartbeat": None}
        self._io_max_gap_ns = {"rx_valid": 0, "tx": 0}
        self._last_io_error = None

    @staticmethod
    def _duration_ns(value: float, name: str) -> int:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite and positive") from exc
        if value != value or value in (float("inf"), float("-inf")) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return max(1, int(value * 1_000_000_000))

    def _now_ns(self) -> int:
        return int(self._clock())

    def _record_event_locked(self, kind, payload):
        # The sink contract is enqueue-only; never let audit failure prevent a
        # zero/disable write. OnlineRecording exposes any overflow to the UI.
        if self._deferred_record_events is not None:
            self._deferred_record_events.append((kind, payload))
            return
        if self._recording_sink is not None:
            try:
                self._recording_sink(kind, payload)
            except Exception:
                pass

    def _record_stop_locked(self, reason):
        if self._recording_active:
            self._record_event_locked("disabled", {
                "host_monotonic_ns": self._now_ns(), "reason": reason})
            self._recording_active = False

    def _io_mark_locked(self, channel: str, now: int) -> None:
        previous = self._io_last_ns[channel]
        if previous is not None:
            self._io_max_gap_ns[channel] = max(self._io_max_gap_ns[channel], max(0, now - previous))
        self._io_last_ns[channel] = now

    def _io_error_locked(self, operation: str, exc: Exception) -> dict[str, Any]:
        evidence = {"operation": operation, "host_monotonic_ns": self._now_ns(),
                    "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "repr": repr(exc), "errno": getattr(exc, "errno", None),
                    "winerror": getattr(exc, "winerror", None)}
        self._io_counts[operation + "_errors"] += 1
        self._last_io_error = evidence
        self._io_trace.append({"kind": "io_error", **evidence})
        return evidence

    def _transport_snapshot_locked(self, *, include_trace: bool = False) -> dict[str, Any]:
        now = self._now_ns()
        threads = {}
        for name, thread in (("reader", self._reader_thread), ("sender", self._sender_thread)):
            last = self._io_last_ns[name + "_heartbeat"]
            threads[name] = {"started": thread is not None,
                             "alive": thread is not None and thread.is_alive(),
                             "last_heartbeat_ns": last,
                             "heartbeat_age_ns": None if last is None else max(0, now - last)}
        result = {"port": self.port, "background": self._background,
                  "connected": self._connected, "stop_requested": self._stop.is_set(),
                  "close": deepcopy(self._close_snapshot),
                  "counts": dict(self._io_counts), "last_ns": dict(self._io_last_ns),
                  "max_gap_ns": dict(self._io_max_gap_ns), "threads": threads,
                  "last_io_error": deepcopy(self._last_io_error),
                  "trace_capacity": self._io_trace.maxlen, "trace_size": len(self._io_trace)}
        if include_trace:
            result["trace"] = deepcopy(list(self._io_trace))
        return result

    # -- public state -----------------------------------------------------

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def torque_limit_nm(self) -> float:
        with self._lock:
            return self._limit

    @property
    def feedback(self) -> MotorFeedback | None:
        with self._lock:
            return self._feedback

    @property
    def fault(self) -> str | None:
        with self._lock:
            return self._fault

    @property
    def fault_snapshot(self) -> dict[str, Any] | None:
        """Return the bounded snapshot captured at the first latched fault."""
        with self._lock:
            return _copy_fault_snapshot(self._fault_snapshot)

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return bounded transport/fault diagnostics for the UI and audit log.

        ``host_feedback_age_ns`` is measured from receipt on this host.  The
        motor's ``device_time_us`` is reported separately; no device-age value
        is synthesized because the two clocks are not synchronized.
        """
        with self._lock:
            now = self._now_ns()
            feedback = self._feedback
            command_age = (
                None
                if self._last_command_ns is None
                else max(0, now - self._last_command_ns)
            )
            host_age = (
                None
                if feedback is None
                else max(0, now - feedback.host_monotonic_ns)
            )
            return {
                "transport": self._transport_snapshot_locked(),
                "fault": self._fault,
                "fault_snapshot": _copy_fault_snapshot(self._fault_snapshot),
                "last_feedback": _copy_feedback_snapshot(feedback),
                "previous_feedback": _copy_feedback_snapshot(self._previous_feedback),
                "last_feedback_raw_frame": (
                    None if feedback is None else bytes(feedback.raw_frame)
                ),
                "previous_feedback_raw_frame": (
                    None
                    if self._previous_feedback is None
                    else bytes(self._previous_feedback.raw_frame)
                ),
                "last_feedback_state": None if feedback is None else feedback.state,
                "last_feedback_error": None if feedback is None else feedback.error,
                "last_feedback_host_monotonic_ns": (
                    None if feedback is None else feedback.host_monotonic_ns
                ),
                "last_feedback_device_time_us": (
                    None if feedback is None else feedback.device_time
                ),
                "host_feedback_age_ns": host_age,
                "device_feedback_age_us": None,
                "command_age_ns": command_age,
                "parser_rejected_count": self._parser_rejected_count,
                "parser_crc_or_format_errors": self._parser_crc_or_format_errors,
                "parser_discarded_bytes": self._parser_discarded_bytes,
                "parser_partial_frames": self._parser_partial_frames,
                "parser_rejected_bytes": self._parser_discarded_bytes,
                "parser_buffered_bytes": len(self._rx_buffer),
            }

    @property
    def history(self) -> tuple[TorqueCommand, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def last(self) -> TorqueCommand | None:
        with self._lock:
            return self._history[-1] if self._history else None

    # -- connection and lifecycle ---------------------------------------

    def connect(self) -> None:
        """Open the port, force zero/disable, and start I/O loops.

        The disable frame is deliberate: a controller may have retained an
        enabled state from an earlier host process, so connection itself first
        establishes a known disarmed state and still never sends enable.
        """
        with self._lock:
            if self._connected:
                return
            if self._closed:
                raise RuntimeError("motor service is closed")
            factory = self._serial_factory
            if factory is None:
                try:
                    import serial  # type: ignore
                except ImportError as exc:  # pragma: no cover - environment-specific
                    raise RuntimeError("pyserial is required for serial output") from exc
                factory = serial.Serial
            try:
                ser = factory(
                    port=self.port,
                    baudrate=SERIAL_BAUD,
                    timeout=0.01,
                    write_timeout=0.05,
                )
            except Exception:
                self._serial = None
                raise
            self._serial = ser
            self._connected = True
            self._stop.clear()
            self._write_control_locked(0.0, 0.0, record=True)
            self._write_frame_locked(build_enable_frame(DISABLE_CMD))
            if self._background:
                self._reader_thread = threading.Thread(
                    target=self._read_loop, name="motor-feedback", daemon=True
                )
                self._sender_thread = threading.Thread(
                    target=self._send_loop, name="motor-control", daemon=True
                )
                self._reader_thread.start()
                self._sender_thread.start()

    @property
    def close_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return deepcopy(self._close_snapshot)

    def close(self) -> None:
        """Disarm once; retain a failed serial handle for explicit close retry."""
        with self._close_lock:
            with self._lock:
                if self._closed and self._serial is None:
                    return
                first_close = not self._closed
                self._closed = True
                self._stop.set()
                self._enable_requested = False
                self._enable_request_generation = None
                self._enable_deadline_ns = None
                self._enabled = False
                self._desired_left = self._desired_right = 0.0
                self._last_command_ns = None
                if first_close and self._connected:
                    self._write_control_locked(0.0, 0.0, record=True)
                    self._write_frame_locked(build_enable_frame(DISABLE_CMD))
                self._record_stop_locked("disconnected")
                ser = self._serial
                # A failed close keeps ownership but must never keep control.
                self._connected = False
                attempt = 1 if self._close_snapshot is None else self._close_snapshot["attempt"] + 1
            def error_fields(exc):
                return {"type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "repr": repr(exc), "errno": getattr(exc, "errno", None),
                        "winerror": getattr(exc, "winerror", None)}
            def serial_open():
                try:
                    value = getattr(ser, "is_open", None)
                    return value if isinstance(value, bool) else None
                except Exception:
                    return None
            result = {"attempt": attempt, "started_monotonic_ns": self._now_ns(),
                      "serial_is_open_before": serial_open(), "serial_close_succeeded": False,
                      "error": None, "join_errors": [], "threads": {}}
            # Join failures cannot skip the underlying OS handle release.
            for name, thread in (("reader", self._reader_thread), ("sender", self._sender_thread)):
                if thread is not None and thread is not threading.current_thread():
                    try:
                        thread.join(timeout=0.5)
                    except Exception as exc:
                        result["join_errors"].append({"thread": name, **error_fields(exc)})
                try:
                    alive = thread is not None and thread.is_alive()
                except Exception:
                    alive = None
                result["threads"][name] = {"alive": alive}
            failure = None
            try:
                if ser is not None:
                    ser.close()
                    if serial_open() is True:
                        raise OSError("serial close returned with port still open")
                result["serial_close_succeeded"] = True
            except Exception as exc:
                failure = exc
                result["error"] = error_fields(exc)
            for name, thread in (("reader", self._reader_thread), ("sender", self._sender_thread)):
                try:
                    result["threads"][name]["alive_after_serial_close"] = thread is not None and thread.is_alive()
                except Exception:
                    result["threads"][name]["alive_after_serial_close"] = None
            result["serial_is_open_after"] = serial_open()
            result["completed_monotonic_ns"] = self._now_ns()
            with self._lock:
                self._close_snapshot = result
                if failure is None:
                    self._serial = None
                self._io_trace.append({"kind": "close", **deepcopy(result)})
            if failure is not None:
                raise failure

    disconnect = close

    # -- command API -----------------------------------------------------

    def set_limit(self, value: float) -> None:
        """Apply a finite protocol-bounded limit immediately, including output."""
        try:
            value = _finite_limit(value)
        except ValueError:
            with self._lock:
                self._latch_fault_locked("invalid torque limit")
            raise
        with self._lock:
            self._limit = value
            self._desired_left = clamp_torque(self._desired_left, value)
            self._desired_right = clamp_torque(self._desired_right, value)
            if value <= 0 and (self._enabled or self._enable_requested):
                self._enabled = False
                self._enable_requested = False
                self._enable_request_generation = None
                self._enable_deadline_ns = None
                self._last_command_ns = None
            if self._connected:
                self._write_control_locked(
                    self._desired_left if self._enabled else 0.0,
                    self._desired_right if self._enabled else 0.0,
                    record=True,
                )
                if value <= 0:
                    self._write_frame_locked(build_enable_frame(DISABLE_CMD))
                    self._record_stop_locked("zero_torque_limit")

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            if not enabled:
                # The state transition and clearing of desired values are one
                # lock-protected operation, so an old inference result cannot
                # restore non-zero torque after an emergency stop.
                self._enable_requested = False
                self._enabled = False
                self._enable_request_generation = None
                self._enable_deadline_ns = None
                self._desired_left = self._desired_right = 0.0
                self._last_command_ns = None
                if self._connected:
                    self._write_control_locked(0.0, 0.0, record=True)
                    self._write_frame_locked(build_enable_frame(DISABLE_CMD))
                self._record_stop_locked("disabled")
                return

            if self._fault:
                raise RuntimeError(f"cannot enable while fault is latched: {self._fault}")
            if self._limit <= 0:
                self._latch_fault_locked("cannot enable with zero torque limit")
                raise RuntimeError("cannot enable with zero torque limit")
            if not self._connected:
                raise RuntimeError("cannot enable while serial port is disconnected")
            now = self._now_ns()
            if not self._feedback_is_fresh_safe_locked(now):
                self._latch_fault_locked("cannot enable without fresh safe feedback")
                raise RuntimeError("cannot enable without fresh safe feedback")

            # Arm at zero even when send() was used for a disabled preview.
            # The next explicit command then starts from a known safe output.
            self._desired_left = self._desired_right = 0.0
            self._last_command_ns = now
            self._enable_requested = True
            self._enable_request_generation = self._feedback_generation
            self._enable_deadline_ns = now + self.feedback_timeout_ns
            self._write_control_locked(0.0, 0.0, record=True)
            self._write_frame_locked(build_enable_frame(ENABLE_CMD))
            # The current frame was received before the request and cannot
            # confirm it. Only a later state=2 frame may set enabled=True.
            self._enabled = False

    def send(self, left_nm: float, right_nm: float) -> None:
        """Cache and transmit one command, hard-clamped immediately before write."""
        if not self._finite_command(left_nm) or not self._finite_command(right_nm):
            with self._lock:
                self._latch_fault_locked("non-finite torque command")
            return
        with self._lock:
            if self._fault or not self._connected:
                self._desired_left = self._desired_right = 0.0
                return
            self._desired_left = clamp_torque(left_nm, self._limit)
            self._desired_right = clamp_torque(right_nm, self._limit)
            self._last_command_ns = self._now_ns()
            if not self._enabled:
                left_nm = right_nm = 0.0
            self._write_control_locked(left_nm, right_nm, record=True)

    @staticmethod
    def _finite_command(value: float) -> bool:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        return value == value and value not in (float("inf"), float("-inf"))

    def zero(self) -> None:
        with self._lock:
            self._desired_left = self._desired_right = 0.0
            if self._connected:
                self._last_command_ns = self._now_ns()
                self._write_control_locked(0.0, 0.0, record=True)

    def clear_fault(self) -> None:
        with self._lock:
            had_fault = self._fault is not None
            self._fault = None
            self._fault_snapshot = None
            if self._connected and (had_fault or self._enabled or self._enable_requested):
                self._enabled = False
                self._enable_requested = False
                self._write_control_locked(0.0, 0.0, record=True)
                self._write_frame_locked(build_enable_frame(DISABLE_CMD))
            self._enable_request_generation = None
            self._enable_deadline_ns = None
            self._enabled = False
            self._enable_requested = False
            self._desired_left = self._desired_right = 0.0
            self._last_command_ns = None
            self._record_stop_locked("fault_cleared")

    # -- fake-serial / parser API ---------------------------------------

    def poll(self) -> None:
        """Process currently available input and perform one control tick."""
        with self._lock:
            if not self._connected or self._serial is None:
                return
            ser = self._serial
            self._io_last_ns["reader_heartbeat"] = self._now_ns()
            try:
                count = int(getattr(ser, "in_waiting", 0) or 0)
                chunk = ser.read(count or 1) if count or hasattr(ser, "read") else b""
            except Exception as exc:
                self._io_error_locked("read", exc)
                self._latch_fault_locked(f"serial read failed: {exc}")
                return
            if chunk:
                self._feed_bytes_locked(bytes(chunk))
            self._send_tick_locked()

    def feed_feedback(self, data: bytes) -> int:
        """Feed arbitrary bytes to the parser; returns accepted frame count."""
        with self._lock:
            return self._feed_bytes_locked(bytes(data))

    # -- I/O internals ----------------------------------------------------

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if not self._connected or self._serial is None:
                    break
                ser = self._serial
                self._io_last_ns["reader_heartbeat"] = self._now_ns()
            try:
                chunk = ser.read(128)
            except Exception as exc:
                with self._lock:
                    if not self._stop.is_set():
                        self._io_error_locked("read", exc)
                        self._latch_fault_locked(f"serial read failed: {exc}")
                break
            if chunk:
                with self._lock:
                    self._feed_bytes_locked(bytes(chunk))

    def _send_loop(self) -> None:
        next_tick = self._now_ns()
        while not self._stop.is_set():
            now = self._now_ns()
            if now >= next_tick:
                with self._lock:
                    self._send_tick_locked()
                next_tick += _CONTROL_PERIOD_NS
                if next_tick < now - _CONTROL_PERIOD_NS:
                    next_tick = now + _CONTROL_PERIOD_NS
            sleep(0.001)

    def _send_tick_locked(self) -> None:
        self._io_last_ns["sender_heartbeat"] = self._now_ns()
        if not self._connected or self._serial is None or self._fault:
            return
        # A connection alone must not start a control stream. Explicit zero()
        # or send() calls still transmit a zero frame while disarmed.
        if not self._enabled and not self._enable_requested:
            return
        now = self._now_ns()
        if self._enabled:
            if self._last_command_ns is None or now - self._last_command_ns > self.command_timeout_ns:
                self._latch_fault_locked("torque command expired")
                return
            if not self._feedback_is_fresh_normal_locked(now):
                self._latch_fault_locked("motor feedback expired or unhealthy")
                return
        elif self._enable_requested:
            if self._enable_deadline_ns is not None and now > self._enable_deadline_ns:
                self._latch_fault_locked("enable confirmation timeout")
                return
            if not self._feedback_is_fresh_safe_locked(now):
                self._latch_fault_locked("motor feedback expired or unhealthy")
                return
        left = self._desired_left if self._enabled else 0.0
        right = self._desired_right if self._enabled else 0.0
        self._write_control_locked(left, right, record=True)

    def _write_control_locked(self, left: float, right: float, *, record: bool) -> bool:
        frame = build_control_frame(self._seq, left, right, self._limit)
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFFFF
        ok = self._write_frame_locked(frame)
        if record and ok:
            command = TorqueCommand(
                    seq=seq,
                    left_nm=clamp_torque(left, self._limit),
                    right_nm=clamp_torque(right, self._limit),
                    torque_limit_nm=self._limit,
                    enabled=self._enabled,
                    host_monotonic_ns=self._now_ns(),
                    frame=frame,
                )
            self._history.append(command)
            self._record_event_locked("sent_command", command)
        return ok

    def _write_frame_locked(self, frame: bytes) -> bool:
        ser = self._serial
        if not self._connected or ser is None:
            return False
        now = self._now_ns()
        kind = "enable" if frame == build_enable_frame(ENABLE_CMD) else (
            "disable" if frame == build_enable_frame(DISABLE_CMD) else "torque")
        entry = {"kind": "tx", "command": kind, "host_monotonic_ns": now,
                 "raw_frame_hex": frame.hex(), "result": "pending"}
        if kind == "torque":
            seq, left, right, limit = struct.unpack("<H3f", frame[1:15])
            entry.update(seq=seq, left_nm=left, right_nm=right, torque_limit_nm=limit,
                         command="zero" if left == right == 0 else "torque")
        self._io_trace.append(entry)
        self._io_counts["tx_attempts"] += 1
        try:
            with self._write_lock:
                result = ser.write(frame)
            if result is not None and int(result) != len(frame):
                raise IOError(f"short serial write: {result}/{len(frame)} bytes")
            entry["result"] = "written"
            entry["completed_monotonic_ns"] = self._now_ns()
            self._io_counts["tx_frames"] += 1
            self._io_counts["tx_bytes"] += len(frame)
            self._io_mark_locked("tx", now)
            return True
        except Exception as exc:
            entry["result"] = "failed"
            entry["error"] = self._io_error_locked("write", exc)
            if not self._closed and self._fault is None:
                self._latch_fault_locked(f"serial write failed: {exc}")
            return False

    def _latch_fault_locked(
        self,
        reason: str,
        *,
        fault_feedback: MotorFeedback | None = None,
    ) -> None:
        # A bad status frame can arrive at the feedback rate.  Safety shutdown
        # is a first-edge action; repeating it for every copy of the same fault
        # would create an unbounded serial write storm and duplicate audit
        # events.  Explicit close()/set_enabled(False) still write their own
        # zero/disable frames below their public APIs.
        if self._fault is not None:
            return
        self._fault = str(reason)
        now = self._now_ns()
        current_feedback = fault_feedback if fault_feedback is not None else self._feedback
        self._fault_snapshot = {
            "transport": self._transport_snapshot_locked(include_trace=True),
            "reason": self._fault,
            "host_monotonic_ns": now,
            "feedback": _copy_feedback_snapshot(current_feedback),
            "previous_feedback": _copy_feedback_snapshot(self._previous_feedback),
            "state": None if current_feedback is None else current_feedback.state,
            "error": None if current_feedback is None else current_feedback.error,
            "device_time_us": None if current_feedback is None else current_feedback.device_time,
            "command_age_ns": (
                None
                if self._last_command_ns is None
                else max(0, now - self._last_command_ns)
            ),
            "parser_rejected_count": self._parser_rejected_count,
            "parser_crc_or_format_errors": self._parser_crc_or_format_errors,
            "parser_discarded_bytes": self._parser_discarded_bytes,
            "parser_partial_frames": self._parser_partial_frames,
        }
        # Capture the fault edge before the shutdown events.  This gives a
        # recording sink one bounded, inspectable fault record even though the
        # subsequent zero/disable transition closes the active session.
        # Buffer audit events until physical zero/disable completes. A slow or
        # failing sink must never delay the first safety writes.
        self._deferred_record_events = []
        if self._recording_active and fault_feedback is not None and "non-finite" not in reason:
            self._record_event_locked("feedback", fault_feedback)
        self._record_event_locked("fault", _copy_fault_snapshot(self._fault_snapshot))
        self._enable_requested = False
        self._enabled = False
        self._desired_left = self._desired_right = 0.0
        self._last_command_ns = None
        if self._connected and self._serial is not None:
            # Best-effort shutdown. _write_frame_locked is re-entrant-safe and
            # still serializes with any concurrent writer.
            self._write_control_locked(0.0, 0.0, record=True)
            self._write_frame_locked(build_enable_frame(DISABLE_CMD))
        self._record_stop_locked("fault: " + self._fault)
        pending = self._deferred_record_events
        self._deferred_record_events = None
        for kind, payload in pending:
            self._record_event_locked(kind, payload)

    def _feedback_is_fresh_normal_locked(self, now: int) -> bool:
        fb = self._feedback
        return bool(
            fb is not None
            and now - fb.host_monotonic_ns <= self.feedback_timeout_ns
            and fb.state == _NORMAL_STATE
            and fb.error == 0
        )

    def _feedback_is_fresh_safe_locked(self, now: int) -> bool:
        fb = self._feedback
        return bool(
            fb is not None
            and now - fb.host_monotonic_ns <= self.feedback_timeout_ns
            and fb.state in _SAFE_ENABLE_STATES
            and fb.error == 0
        )

    def _feed_bytes_locked(self, data: bytes) -> int:
        self._io_counts["rx_bytes"] += len(data)
        self._rx_buffer.extend(data)
        accepted = 0
        while True:
            idx = self._rx_buffer.find(bytes([HEAD_STATUS]))
            if idx < 0:
                # Retain a possible one-byte prefix only; status frames have a
                # single-byte header so all other junk can be discarded.
                if self._rx_buffer:
                    self._parser_discarded_bytes += len(self._rx_buffer)
                self._rx_buffer.clear()
                self._parser_partial_pending = False
                break
            if idx:
                self._parser_discarded_bytes += idx
                del self._rx_buffer[:idx]
            if len(self._rx_buffer) < STATUS_SIZE:
                if not self._parser_partial_pending:
                    self._parser_partial_frames += 1
                    self._parser_partial_pending = True
                break
            self._parser_partial_pending = False
            candidate = bytes(self._rx_buffer[:STATUS_SIZE])
            if candidate[-1] != FRAME_TAIL or calc_crc8(candidate[1:-2]) != candidate[-2]:
                # Drop only the candidate's first byte, then search again. This
                # handles CRC errors, half frames, and a valid frame embedded
                # after a corrupt candidate.
                self._parser_rejected_count += 1
                self._parser_crc_or_format_errors += 1
                self._io_trace.append({"kind": "rx_status_rejected", "host_monotonic_ns": self._now_ns(),
                                       "raw_frame_hex": candidate.hex()})
                self._parser_discarded_bytes += 1
                del self._rx_buffer[:1]
                continue
            try:
                (
                    _head,
                    seq,
                    state,
                    error,
                    lp,
                    lv,
                    lt,
                    rp,
                    rv,
                    rt,
                    _device_time,
                    _crc,
                    _tail,
                ) = STATUS_STRUCT.unpack(candidate)
            except struct.error:
                self._parser_rejected_count += 1
                self._parser_crc_or_format_errors += 1
                self._parser_discarded_bytes += 1
                del self._rx_buffer[:1]
                continue
            del self._rx_buffer[:STATUS_SIZE]
            values = (lp, lv, lt, rp, rv, rt)
            now = self._now_ns()
            self._io_counts["rx_valid_frames"] += 1
            self._io_mark_locked("rx_valid", now)
            self._io_trace.append({"kind": "rx_status", "host_monotonic_ns": now,
                "seq": seq, "state": state, "error": error,
                "raw_frame_hex": candidate.hex(), "device_time_us": _device_time,
                "values": [value if math.isfinite(value) else repr(value) for value in values]})
            if not all(math.isfinite(value) for value in values):
                # Preserve the offending CRC-valid frame in the fault
                # snapshot, but never publish its non-finite values as usable
                # feedback or let them reach control decisions.
                offending_feedback = MotorFeedback(
                    host_monotonic_ns=now,
                    seq=seq,
                    state=state,
                    error=error,
                    left_position_rad=lp,
                    left_velocity_rad_s=lv,
                    left_torque_nm=lt,
                    right_position_rad=rp,
                    right_velocity_rad_s=rv,
                    right_torque_nm=rt,
                    device_time=_device_time,
                    raw_frame=candidate,
                )
                self._latch_fault_locked(
                    "non-finite motor feedback",
                    fault_feedback=offending_feedback,
                )
                continue
            self._feedback_generation += 1
            self._previous_feedback = self._feedback
            self._feedback = MotorFeedback(
                host_monotonic_ns=now,
                seq=seq,
                state=state,
                error=error,
                left_position_rad=lp,
                left_velocity_rad_s=lv,
                left_torque_nm=lt,
                right_position_rad=rp,
                right_velocity_rad_s=rv,
                right_torque_nm=rt,
                device_time=_device_time, raw_frame=candidate,
            )
            accepted += 1
            # Include the unhealthy terminal frame before closing its session.
            if self._recording_active and error == 0 and not (self._enabled and state != _NORMAL_STATE):
                self._record_event_locked("feedback", self._feedback)
            if error != 0:
                self._latch_fault_locked(
                    f"unhealthy motor feedback state={state} error={error}",
                    fault_feedback=self._feedback,
                )
            elif self._enable_requested:
                # state=0 is expected while the enable command propagates. A
                # state=2 frame only confirms this request when it arrived
                # after the request generation, never from pre-request state.
                if (
                    state == _NORMAL_STATE
                    and self._enable_request_generation is not None
                    and self._feedback_generation > self._enable_request_generation
                ):
                    self._enabled = True
                    self._recording_active = True
                    self._record_event_locked("enabled", {"host_monotonic_ns": now, "port": self.port})
                    self._record_event_locked("feedback", self._feedback)
                    self._enable_requested = False
                    self._enable_request_generation = None
                    self._enable_deadline_ns = None
            elif self._enabled and state != _NORMAL_STATE:
                self._latch_fault_locked(
                    f"unhealthy motor feedback state={state} error={error}",
                    fault_feedback=self._feedback,
                )
        return accepted


__all__ = [
    "MotorFeedback",
    "SERIAL_BAUD",
    "STATUS_FORMAT",
    "STATUS_SIZE",
    "STATUS_STRUCT",
    "SerialTorqueOutput",
]
