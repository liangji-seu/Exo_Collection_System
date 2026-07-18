"""Read-only Teensy motor/encoder status adapter.

Firmware ``StatusFrame`` (C struct, ExoCode.ino)::

    Offset  Size  Field          Type
    ──────────────────────────────────────
      0      1    head1          0xCC
      1      1    head2          0xAA
      2      2    seq            uint16 LE  (command echo; NOT a frame counter)
      4      1    state          uint8
      5      1    error_code     uint8
      6      4    left_pos       float32 LE  (rad)
     10      4    left_vel       float32 LE  (rad/s)
     14      4    left_torque    float32 LE  (N*m)
     18      4    right_pos      float32 LE  (rad)
     22      4    right_vel      float32 LE  (rad/s)
     26      4    right_torque   float32 LE  (N*m)
     30      4    teensy_time_us uint32 LE  (1 MHz, wraps ~71.6 min)
     34      1    crc8           CRC‑8 poly=0x07 init=0 over bytes [2:34]
     35      1    tail           0x55
    ──────────────────────────────────────
    Total   36 bytes

Firmware sends at 200 Hz (5 ms cycle).  The adapter is strictly read‑only:
it never calls ``serial.write()`` during any lifecycle phase.

Time‑based gap detection uses the unwrapped ``teensy_time_us`` field.
Threshold: ``2 * (1_000_000 us / 200 Hz) = 10_000 us``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import struct
from threading import Event, Lock, Thread, current_thread
from time import perf_counter_ns, time_ns
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np

from exo_collection.adapters.base import AdapterError, ModalityDescriptor
from exo_collection.adapters.encoder.simulated import ENCODER_CHANNELS, ENCODER_UNITS
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.domain.events import SampleBatch


HEAD_STATUS1 = 0xCC
HEAD_STATUS2 = 0xAA
FRAME_TAIL = 0x55
STATUS_FORMAT = "<BBHBBffffffIBB"
STATUS_STRUCT = struct.Struct(STATUS_FORMAT)
STATUS_SIZE = STATUS_STRUCT.size  # 36 bytes
BAUD_DEFAULT = 1_000_000
TEENSY_VID = 0x16C0
TEENSY_PID = 0x0483
CRC8_POLY = 0x07

# Nominal firmware cycle in microseconds: 1e6 / 200 = 5_000 us.
_TICK_US = 5_000
_GAP_THRESHOLD_US = 2 * _TICK_US  # 10_000 us (one missed frame tolerates jitter)

# Compatibility names retained for callers of the earlier draft.
HEAD_STATUS = HEAD_STATUS1
FRAME_HEADER = HEAD_STATUS1
FRAME_FOOTER = FRAME_TAIL
PAYLOAD_STRUCT = STATUS_STRUCT
PAYLOAD_SIZE = STATUS_SIZE

__all__ = [
    "BAUD_DEFAULT",
    "CRC8_POLY",
    "FRAME_FOOTER",
    "FRAME_HEADER",
    "FRAME_TAIL",
    "HEAD_STATUS",
    "HEAD_STATUS1",
    "HEAD_STATUS2",
    "MotorStatusFrame",
    "MotorStatusStreamParser",
    "PAYLOAD_SIZE",
    "PAYLOAD_STRUCT",
    "STATUS_FORMAT",
    "STATUS_SIZE",
    "STATUS_STRUCT",
    "TEENSY_PID",
    "TEENSY_VID",
    "TeensyEncoderConfig",
    "TeensySerialEncoderAdapter",
    "_crc8",
    "_match_teensy_port",
    "calc_crc8",
    "find_teensy_port",
    "parse_status_frame",
]


def calc_crc8(data: bytes) -> int:
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


_crc8 = calc_crc8


@dataclass(frozen=True, slots=True)
class MotorStatusFrame:
    sequence: int          # firmware command-echo seq (NOT a frame counter)
    state: int
    error: int
    left_position: float
    left_velocity: float
    left_torque: float
    right_position: float
    right_velocity: float
    right_torque: float
    teensy_time_us: int    # uint32, 1 MHz, wraps ~71.6 minutes


def parse_status_frame(data: bytes) -> MotorStatusFrame | None:
    """Validate and decode one exact firmware status frame."""

    if len(data) != STATUS_SIZE:
        return None
    if data[0] != HEAD_STATUS1 or data[1] != HEAD_STATUS2:
        return None
    if data[-1] != FRAME_TAIL:
        return None
    # CRC covers bytes [2:34] — seq through teensy_time_us (inclusive).
    if calc_crc8(data[2:STATUS_SIZE - 2]) != data[-2]:
        return None
    try:
        (
            _head1,
            _head2,
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
            _crc,
            _tail,
        ) = STATUS_STRUCT.unpack(data)
    except struct.error:
        return None
    return MotorStatusFrame(
        sequence=sequence,
        state=state,
        error=error,
        left_position=left_position,
        left_velocity=left_velocity,
        left_torque=left_torque,
        right_position=right_position,
        right_velocity=right_velocity,
        right_torque=right_torque,
        teensy_time_us=teensy_time_us,
    )


class MotorStatusStreamParser:
    """Incremental parser that re-synchronises on ``0xCC 0xAA``.

    Handles noise, fragmentation, consecutive frames, and payload bytes that
    happen to equal ``0xCC`` or ``0xAA``.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.crc_or_format_errors: int = 0
        self.discarded_bytes: int = 0

    def reset(self) -> None:
        self._buffer.clear()
        self.crc_or_format_errors = 0
        self.discarded_bytes = 0

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[MotorStatusFrame]:
        self._buffer.extend(chunk)
        frames: list[MotorStatusFrame] = []
        while True:
            head1_idx = self._buffer.find(HEAD_STATUS1)
            if head1_idx < 0:
                self.discarded_bytes += len(self._buffer)
                self._buffer.clear()
                break
            # Discard leading junk before HEAD_STATUS1.
            if head1_idx:
                self.discarded_bytes += head1_idx
                del self._buffer[:head1_idx]
            # Need at least 2 bytes to check HEAD_STATUS2.
            if len(self._buffer) < 2:
                break
            if self._buffer[1] != HEAD_STATUS2:
                # False start: skip this HEAD_STATUS1 and resume search.
                self.crc_or_format_errors += 1
                self.discarded_bytes += 1
                del self._buffer[0]
                continue
            # Need a full frame.
            if len(self._buffer) < STATUS_SIZE:
                break
            candidate = bytes(self._buffer[:STATUS_SIZE])
            parsed = parse_status_frame(candidate)
            if parsed is None:
                # Frame candidate failed CRC/tail — skip HEAD_STATUS1, re‑sync.
                self.crc_or_format_errors += 1
                self.discarded_bytes += 1
                del self._buffer[0]
                continue
            # Valid frame consumed.
            del self._buffer[:STATUS_SIZE]
            frames.append(parsed)
        return frames

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)


class SerialPort(Protocol):
    is_open: bool
    in_waiting: int

    def read(self, size: int = 1) -> bytes: ...

    def write(self, data: bytes) -> int: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TeensyEncoderConfig:
    device_id: str = "encoder_teensy"
    clock_domain: str = "encoder_teensy_clock"
    port: str | None = None
    baudrate: int = BAUD_DEFAULT
    vid: int = TEENSY_VID
    pid: int = TEENSY_PID
    nominal_rate_hz: float = 200.0
    batch_size: int = 20
    queue_capacity: int = 256
    read_size: int = 128
    read_timeout_s: float = 0.05

    def __post_init__(self) -> None:
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if self.baudrate <= 0 or self.nominal_rate_hz <= 0:
            raise ValueError("baudrate and nominal_rate_hz must be positive")
        if self.batch_size <= 0 or self.queue_capacity <= 0 or self.read_size <= 0:
            raise ValueError("batch_size, queue_capacity and read_size must be positive")
        if self.read_timeout_s <= 0:
            raise ValueError("read_timeout_s must be positive")


def _coerce_config(
    value: TeensyEncoderConfig | Mapping[str, Any] | None,
) -> TeensyEncoderConfig:
    if value is None:
        return TeensyEncoderConfig()
    if isinstance(value, TeensyEncoderConfig):
        return value
    raw = dict(value)
    parameters = raw.pop("parameters", None)
    if isinstance(parameters, Mapping):
        raw.update(parameters)
    if "id" in raw and "device_id" not in raw:
        raw["device_id"] = raw.pop("id")
    if "baud" in raw and "baudrate" not in raw:
        raw["baudrate"] = raw.pop("baud")
    allowed = TeensyEncoderConfig.__dataclass_fields__
    return TeensyEncoderConfig(
        **{key: item for key, item in raw.items() if key in allowed}
    )


def find_teensy_port(
    ports: Iterable[Any], *, vid: int = TEENSY_VID, pid: int = TEENSY_PID
) -> str | None:
    for port in ports:
        if getattr(port, "vid", None) == vid and getattr(port, "pid", None) == pid:
            return str(port.device)
    return None


def _match_teensy_port(
    vid: int = TEENSY_VID, pid: int = TEENSY_PID,
) -> str | None:
    try:
        import serial.tools.list_ports
    except ImportError as exc:
        raise AdapterError(
            "pyserial not installed; install hardware dependencies before "
            "connecting to Teensy."
        ) from exc
    return find_teensy_port(
        serial.tools.list_ports.comports(), vid=vid, pid=pid,
    )


@dataclass
class _FrameHealth:
    """Mutable tracker scoped to a trial for state/error/time auditing."""
    last_state: int | None = None
    last_error_code: int | None = None
    last_fw_sequence: int | None = None
    non_zero_error_count: int = 0
    last_teensy_time_us: int | None = None
    timestamp_gap_events: list[dict[str, Any]] = field(default_factory=list)

    # Accumulates elapsed us across uint32 wrap for the unwrapped "time ruler".
    _wraps: int = 0

    def unwrap_time(self, raw: int) -> int:
        """Return unwrapped device time (int) accounting for uint32 wraps."""
        if self.last_teensy_time_us is None:
            return int(raw)
        delta = int(raw) - self.last_teensy_time_us
        if delta < -(2**31):
            # Forward wrap: raw wrapped around after ~71.6 min.
            self._wraps += 1
        return self._wraps * 2**32 + int(raw)

    def detect_gap(self, raw: int) -> int | None:
        """Return elapsed us if a time discontinuity is detected, else None."""
        if self.last_teensy_time_us is None:
            return None
        prev = self.last_teensy_time_us
        # Handle uint32 arithmetic: forward diff in [0, 2^32).
        diff = (int(raw) - prev) & 0xFFFF_FFFF
        # If the difference looks like a backward wrap (large forward jump),
        # clamp to the expected one-tick delta.
        if diff > 2**31:
            return None
        threshold = _GAP_THRESHOLD_US
        if diff > threshold:
            return int(diff)
        return None


class TeensySerialEncoderAdapter(QueuedHardwareAdapter):
    """Read motor feedback only; this class never calls ``serial.write``.

    The adapter tracks firmware state/error/seq for audit in health snapshots
    and configuration snapshots.  Time‑based gap detection (via unwrapped
    ``teensy_time_us``) is used instead of the command‑echo ``seq`` field.
    """

    def __init__(
        self,
        config: TeensyEncoderConfig | Mapping[str, Any] | None = None,
        *,
        serial_factory: Callable[..., SerialPort] | None = None,
        port_lister: Callable[[], Iterable[Any]] | None = None,
    ) -> None:
        self._config = _coerce_config(config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._serial_factory = serial_factory
        self._port_lister = port_lister
        self._serial: SerialPort | None = None
        self._resolved_port: str | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._parser = MotorStatusStreamParser()
        self._pending: list[tuple[MotorStatusFrame, int]] = []
        self._pending_lock = Lock()
        self._sample_index = 0
        self._batch_sequence = 0
        self._frame_health = _FrameHealth()
        self._first_data_received = Event()
        self._write_call_count = 0

    # ------------------------------------------------------------------
    # Modality descriptor
    # ------------------------------------------------------------------

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="encoder",
            display_name="Teensy bilateral motor encoder feedback",
            clock_domain=cfg.clock_domain,
            event_kind="sample_batch",
            channels=ENCODER_CHANNELS,
            units=ENCODER_UNITS,
            nominal_rate_hz=cfg.nominal_rate_hz,
            sample_shape=(len(ENCODER_CHANNELS),),
            dtype=np.dtype(np.float32).str,
            metadata={
                "simulated": False,
                "protocol": "teensy_status_v2",
                "status_format": STATUS_FORMAT,
                "status_size_bytes": STATUS_SIZE,
                "crc8_polynomial": "0x07",
                "crc8_range": "bytes[2:34]",
                "vid": cfg.vid,
                "pid": cfg.pid,
                "baudrate": cfg.baudrate,
                "read_only": True,
                "device_timestamp_field": "teensy_time_us",
                "device_timestamp_unit": "us",
                "hardware_tick_hz": 1_000_000,
                "tick_period_us": _TICK_US,
                "gap_threshold_us": _GAP_THRESHOLD_US,
                "fw_seq_description": "command-echo (not frame counter)",
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return {
            **asdict(self._config),
            "resolved_port": self._resolved_port,
            "last_fw_state": self._frame_health.last_state,
            "last_fw_error_code": self._frame_health.last_error_code,
            "last_fw_sequence": self._frame_health.last_fw_sequence,
            "non_zero_error_count": self._frame_health.non_zero_error_count,
        }

    # ------------------------------------------------------------------
    # Hardware lifecycle
    # ------------------------------------------------------------------

    def _connect_hardware(self) -> None:
        cfg = self._config
        if self._port_lister is not None:
            detected = find_teensy_port(
                self._port_lister(), vid=cfg.vid, pid=cfg.pid,
            )
        else:
            detected = (
                _match_teensy_port(cfg.vid, cfg.pid) if not cfg.port else None
            )
        port = cfg.port or detected
        if not port:
            raise AdapterError(
                f"Teensy serial port not found "
                f"(VID=0x{cfg.vid:04X}, PID=0x{cfg.pid:04X})."
            )
        if self._serial_factory is None:
            try:
                import serial
            except ImportError as exc:
                raise AdapterError(
                    "pyserial not installed; cannot connect to Teensy."
                ) from exc
            self._serial_factory = serial.Serial
        try:
            self._serial = self._serial_factory(
                port=port,
                baudrate=cfg.baudrate,
                timeout=cfg.read_timeout_s,
            )
        except BaseException as exc:
            raise AdapterError(
                f"Cannot open Teensy serial port {port}: {exc}"
            ) from exc
        self._resolved_port = port
        self._first_data_received.clear()

    def _reset_trial_state(self) -> None:
        self._stop_event.clear()
        self._parser.reset()
        self._pending.clear()
        self._sample_index = 0
        self._batch_sequence = 0
        self._frame_health = _FrameHealth()
        self._first_data_received.clear()

    def _start_hardware(self) -> None:
        if self._serial is None or not getattr(self._serial, "is_open", True):
            raise AdapterError("Teensy serial port is not open")
        self._first_data_received.clear()
        self._stop_event.clear()
        self._thread = Thread(
            target=self._read_guarded,
            name=f"teensy-read-{self._config.device_id}",
            daemon=True,
        )
        self._thread.start()

    def _stop_hardware(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=max(1.0, self._config.read_timeout_s * 4))
            if thread.is_alive():
                raise AdapterError(
                    "Teensy serial read thread did not stop within timeout"
                )
        self._thread = None
        self._emit_pending(force=True)

    def _close_hardware(self) -> None:
        self._stop_event.set()
        serial_port, self._serial = self._serial, None
        if serial_port is not None:
            serial_port.close()

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _read_guarded(self) -> None:
        try:
            assert self._serial is not None
            while not self._stop_event.is_set():
                size = max(
                    1,
                    min(
                        self._config.read_size,
                        int(getattr(self._serial, "in_waiting", 0)) or 1,
                    ),
                )
                chunk = self._serial.read(size)
                if not chunk:
                    continue
                received_ns = perf_counter_ns()
                for frame in self._parser.feed(chunk):
                    self._accept_frame(frame, received_ns)
        except BaseException as exc:
            self._set_fault(exc)
            self._stop_event.set()

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _accept_frame(
        self, frame: MotorStatusFrame, received_ns: int,
    ) -> None:
        health = self._frame_health

        # --- State / error tracking (for health audit) ---
        if health.last_state is None:
            health.last_state = frame.state
        else:
            health.last_state = frame.state
        health.last_error_code = frame.error
        health.last_fw_sequence = frame.sequence
        if frame.error != 0:
            health.non_zero_error_count += 1

        # --- Time-based gap detection ---
        gap_us = health.detect_gap(frame.teensy_time_us)

        if gap_us is not None and health.last_teensy_time_us is not None:
            # Time discontinuity — flush current batch and start a new one.
            unwrapped_current = health.unwrap_time(frame.teensy_time_us)
            unwrapped_prev = health.unwrap_time(health.last_teensy_time_us)
            health.timestamp_gap_events.append({
                "previous_teensy_time_us": health.last_teensy_time_us,
                "current_teensy_time_us": frame.teensy_time_us,
                "delta_us": gap_us,
                "gap_threshold_us": _GAP_THRESHOLD_US,
                "sample_index": self._sample_index,
                "host_monotonic_ns": received_ns,
            })
            # Emit whatever we have so the gap isn't smeared across batches.
            self._emit_pending(force=True)
            # The emitted batch already consumed the pending frames; the current
            # frame is still outside pending, so it will be appended normally.

        health.last_teensy_time_us = frame.teensy_time_us

        with self._pending_lock:
            self._pending.append((frame, received_ns))
            should_emit = len(self._pending) >= self._config.batch_size
        if should_emit:
            self._emit_pending(force=False)

        # Signal that we've received the first valid frame (READY).
        self._first_data_received.set()

    # ------------------------------------------------------------------
    # Batch emission
    # ------------------------------------------------------------------

    def _emit_pending(self, *, force: bool) -> None:
        with self._pending_lock:
            if not self._pending or (
                not force and len(self._pending) < self._config.batch_size
            ):
                return
            count = len(self._pending) if force else self._config.batch_size
            selected = self._pending[:count]
            del self._pending[:count]
        frames = [item[0] for item in selected]
        first_host_ns = selected[0][1]

        data = np.ascontiguousarray(
            np.asarray(
                [
                    (
                        frame.left_position,
                        frame.left_velocity,
                        frame.left_torque,
                        frame.right_position,
                        frame.right_velocity,
                        frame.right_torque,
                    )
                    for frame in frames
                ],
                dtype=np.float32,
            )
        )
        event = SampleBatch(
            session_uuid=(
                str(self._trial.session_uuid)
                if self._trial is not None
                and self._trial.session_uuid is not None
                else None
            ),
            trial_uuid=(
                str(self._trial.trial_uuid)
                if self._trial is not None
                else None
            ),
            device_id=self._config.device_id,
            modality="encoder",
            clock_domain=self._config.clock_domain,
            host_monotonic_ns=first_host_ns,
            host_utc_ns=time_ns(),
            first_sample_index=self._sample_index,
            sample_count=len(frames),
            sequence_number=self._batch_sequence,
            device_timestamp=frames[0].teensy_time_us,
            sample_rate_hz=self._config.nominal_rate_hz,
            data=data,
        )
        self._publish_raw(
            event, item_count=len(frames), host_monotonic_ns=first_host_ns,
        )
        self._sample_index += len(frames)
        self._batch_sequence += 1

    # ------------------------------------------------------------------
    # Health / audit
    # ------------------------------------------------------------------

    def _dropped_packets(self) -> int:
        return 0  # no seq‑based packet counting

    def _sequence_gaps(self) -> int:
        return len(self._frame_health.timestamp_gap_events)

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        health = self._frame_health
        return {
            "crc_or_format_errors": self._parser.crc_or_format_errors,
            "discarded_serial_bytes": self._parser.discarded_bytes,
            "buffered_serial_bytes": self._parser.buffered_bytes,
            "resolved_port": self._resolved_port,
            "read_only": True,
            "first_data_received": self._first_data_received.is_set(),
            "last_fw_state": health.last_state,
            "last_fw_error_code": health.last_error_code,
            "last_fw_sequence": health.last_fw_sequence,
            "non_zero_error_count": health.non_zero_error_count,
            "timestamp_gap_events": json.dumps(health.timestamp_gap_events)
            if health.timestamp_gap_events
            else None,
            "gap_threshold_us": _GAP_THRESHOLD_US,
            "tick_period_us": _TICK_US,
        }
