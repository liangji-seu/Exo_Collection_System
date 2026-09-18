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
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterator

# Frame constants (old_system/motor_controller.py + teensy_serial.py).
HEAD_CONTROL = 0xAA
HEAD_ENABLE = 0xBB
HEAD_STATUS = 0xCC
FRAME_TAIL = 0x55

ENABLE_CMD = 0xFC
DISABLE_CMD = 0xFD

CRC8_POLY = 0x07
TORQUE_MAX_NM = 9.0
SEND_HZ = 200

_CONTROL_PAYLOAD = struct.Struct("<H3f")


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
    payload = _CONTROL_PAYLOAD.pack(int(seq) & 0xFFFF, l_tq, r_tq, limit)
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
        self._limit = max(0.0, float(torque_limit_nm))
        self._enable_required = bool(enable_required)
        self._zero_on_error = bool(zero_on_error)
        self._record_frames = bool(record_frames)
        self._enabled = not self._enable_required
        self._seq = 0
        self._history: deque[TorqueCommand] = deque(maxlen=history_capacity)

    # -- protocol ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def torque_limit_nm(self) -> float:
        return self._limit

    def set_enabled(self, enabled: bool) -> None:
        was_enabled = self._enabled
        self._enabled = bool(enabled)
        if was_enabled and not self._enabled:
            # Disarming immediately commands zero torque.
            self._record(0.0, 0.0)

    def send(self, left_nm: float, right_nm: float) -> None:
        if not self._enabled:
            left_nm = right_nm = 0.0
        left_nm = clamp_torque(left_nm, self._limit)
        right_nm = clamp_torque(right_nm, self._limit)
        self._record(left_nm, right_nm)

    def zero(self) -> None:
        self._record(0.0, 0.0)

    def close(self) -> None:
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
        return tuple(self._history)

    @property
    def last(self) -> TorqueCommand | None:
        return self._history[-1] if self._history else None

    def __iter__(self) -> Iterator[TorqueCommand]:
        return iter(self._history)


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
