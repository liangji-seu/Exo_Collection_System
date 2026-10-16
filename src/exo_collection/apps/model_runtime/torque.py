"""Torque output interface for the online-testing runtime.

First phase ships a **Mock** output (``MockTorqueOutput``): it builds the exact
motor-control wire frames from the old system and records every command in a
bounded history, but never opens a serial port.  Swapping in a real Teensy
writer later means implementing the same ``TorqueOutput`` protocol; the frame
builders and safety policy (clamp, enable-required, zero-on-error) stay shared.

Protocol bytes are ported verbatim from ``old_system/motor_controller.py``; the
CRC-8 is byte-identical to ``adapters/encoder/teensy_serial.calc_crc8``
(polynomial 0x07, init 0x00) and is re-implemented here so this module stays a
pure, dependency-light protocol layer.
"""

from __future__ import annotations

import struct
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterator
from exo_collection.domain.motor_protocol import TORQUE_MAX_NM

# Frame constants (old_system/motor_controller.py + teensy_serial.py).
HEAD_CONTROL = 0xAA
HEAD_ENABLE = 0xBB
HEAD_STATUS = 0xCC
FRAME_TAIL = 0x55

ENABLE_CMD = 0xFC
DISABLE_CMD = 0xFD

CRC8_POLY = 0x07
SEND_HZ = 200

_CONTROL_PAYLOAD = struct.Struct("<H3f")


def _finite_torque_limit(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("torque limit must be a finite non-negative number") from exc
    if not 0 <= number <= TORQUE_MAX_NM:
        raise ValueError(f"torque limit must be finite and within 0..{TORQUE_MAX_NM:g} Nm")
    return number


def calc_crc8(data: bytes) -> int:
    """CRC-8 (poly 0x07, init 0x00) — identical to teensy_serial.calc_crc8."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (
                ((crc << 1) ^ CRC8_POLY) & 0xFF
                if crc & 0x80
                else (crc << 1) & 0xFF
            )
    return crc


def build_control_frame(seq: int, l_tq: float, r_tq: float, limit: float) -> bytes:
    """One bilateral torque command frame: ``<0xAA> <H 3f> <crc8> <0x55>``.

    ``seq`` is a uint16 command counter; ``limit`` is the torque ceiling echoed
    to the firmware (NaN-safe clamp helper is separate).
    """
    # Keep the protocol builder itself as a safety boundary.  Callers may
    # bypass a backend and build a frame directly, so both the echoed limit
    # and the two command fields are bounded here as well.
    frame_limit = _finite_torque_limit(limit)
    # A decimal such as 0.1 rounds upward when encoded as float32.  Keep the
    # wire limit conservative so the value decoded by firmware never exceeds
    # the operator's requested ceiling.
    wire_limit = struct.unpack("<f", struct.pack("<f", frame_limit))[0]
    if wire_limit > frame_limit:
        # Step one representable float32 downward by manipulating its bits;
        # repeatedly stepping a Python float64 would require millions of
        # iterations for ordinary decimals such as 0.1.
        wire_bits = struct.unpack("<I", struct.pack("<f", wire_limit))[0]
        wire_limit = struct.unpack("<f", struct.pack("<I", wire_bits - 1))[0]
    frame_limit = wire_limit
    payload = _CONTROL_PAYLOAD.pack(
        int(seq) & 0xFFFF,
        clamp_torque(l_tq, frame_limit),
        clamp_torque(r_tq, frame_limit),
        frame_limit,
    )
    crc = calc_crc8(payload)
    return struct.pack("B", HEAD_CONTROL) + payload + struct.pack("BB", crc, FRAME_TAIL)


def build_enable_frame(cmd: int) -> bytes:
    """Enable/disable frame: ``<0xBB> <cmd> <crc8> <0x55>`` (cmd 0xFC/0xFD)."""
    payload = bytes([int(cmd) & 0xFF])
    crc = calc_crc8(payload)
    return struct.pack("B", HEAD_ENABLE) + payload + struct.pack("BB", crc, FRAME_TAIL)


def enable_frame() -> bytes:
    return build_enable_frame(ENABLE_CMD)


def disable_frame() -> bytes:
    return build_enable_frame(DISABLE_CMD)


def clamp_torque(value: float, limit_nm: float) -> float:
    """Clamp a torque value into ``[-limit, +limit]``, coercing non-finite to 0."""
    if value is None or (isinstance(value, float) and value != value):
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not float("-inf") < value < float("inf"):
        return 0.0
    limit = max(0.0, float(limit_nm))
    return max(-limit, min(limit, value))


@dataclass(frozen=True, slots=True)
class TorqueCommand:
    """One recorded torque command (Mock backend) or one intended send."""

    seq: int
    left_nm: float
    right_nm: float
    torque_limit_nm: float
    enabled: bool
    host_monotonic_ns: int
    frame: bytes


class TorqueOutput(ABC):
    """Sink for bilateral motor torque produced by the model runtime."""

    @abstractmethod
    def set_enabled(self, enabled: bool) -> None:
        """Arm/disarm torque output.  Disarming must zero the motors."""

    @abstractmethod
    def send(self, left_nm: float, right_nm: float) -> None:
        """Command one control cycle of bilateral torque."""

    @abstractmethod
    def set_limit(self, value: float) -> None:
        """Change the hard torque limit; implementations apply it immediately."""

    @abstractmethod
    def zero(self) -> None:
        """Immediate zero-torque command (used on fault / emergency stop)."""

    @property
    @abstractmethod
    def enabled(self) -> bool: ...

    @property
    @abstractmethod
    def torque_limit_nm(self) -> float: ...

    @abstractmethod
    def close(self) -> None: ...


class MockTorqueOutput(TorqueOutput):
    """Records torque commands without touching hardware.

    Still enforces the full safety policy so the runtime path — clamp,
    enable-required, zero-on-error — is exercised end to end in tests and in the
    GUI before any real motor board is attached.
    """

    def __init__(
        self,
        torque_limit_nm: float = 5.0,
        *,
        enable_required: bool = True,
        zero_on_error: bool = True,
        history_capacity: int = 4096,
        record_frames: bool = True,
    ) -> None:
        self._limit = _finite_torque_limit(torque_limit_nm)
        self._enable_required = bool(enable_required)
        self._zero_on_error = bool(zero_on_error)
        self._record_frames = bool(record_frames)
        self._enabled = not self._enable_required
        self._seq = 0
        self._history: deque[TorqueCommand] = deque(maxlen=history_capacity)
        self._lock = threading.RLock()

    # -- protocol ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def torque_limit_nm(self) -> float:
        with self._lock:
            return self._limit

    def set_limit(self, value: float) -> None:
        """Set the limit, rejecting invalid values and applying it immediately."""
        value = _finite_torque_limit(value)
        with self._lock:
            self._limit = value

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            was_enabled = self._enabled
            self._enabled = bool(enabled)
            if was_enabled and not self._enabled:
                # Disarming immediately commands zero torque.
                self._record(0.0, 0.0)

    def send(self, left_nm: float, right_nm: float) -> None:
        with self._lock:
            if not self._enabled:
                left_nm = right_nm = 0.0
            left_nm = clamp_torque(left_nm, self._limit)
            right_nm = clamp_torque(right_nm, self._limit)
            self._record(left_nm, right_nm)

    def zero(self) -> None:
        with self._lock:
            self._record(0.0, 0.0)

    def close(self) -> None:
        with self._lock:
            if self._enabled:
                self._record(0.0, 0.0)
            self._enabled = False

    # -- history -----------------------------------------------------------

    def _record(self, left_nm: float, right_nm: float) -> None:
        seq = self._seq
        self._seq += 1
        frame = (
            build_control_frame(seq, left_nm, right_nm, self._limit)
            if self._record_frames
            else b""
        )
        self._history.append(
            TorqueCommand(
                seq=seq,
                left_nm=left_nm,
                right_nm=right_nm,
                torque_limit_nm=self._limit,
                enabled=self._enabled,
                host_monotonic_ns=perf_counter_ns(),
                frame=frame,
            )
        )

    @property
    def history(self) -> tuple[TorqueCommand, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def last(self) -> TorqueCommand | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def __iter__(self) -> Iterator[TorqueCommand]:
        return iter(self.history)


__all__ = [
    "CRC8_POLY",
    "DISABLE_CMD",
    "ENABLE_CMD",
    "FRAME_TAIL",
    "HEAD_CONTROL",
    "HEAD_ENABLE",
    "HEAD_STATUS",
    "SEND_HZ",
    "TORQUE_MAX_NM",
    "MockTorqueOutput",
    "TorqueCommand",
    "TorqueOutput",
    "build_control_frame",
    "build_enable_frame",
    "calc_crc8",
    "clamp_torque",
    "disable_frame",
    "enable_frame",
]
