"""Read-only Teensy motor/encoder status adapter.

The firmware status frame is exactly 35 bytes.  The leading byte, CRC byte,
and trailing byte are already part of ``STATUS_FORMAT``; they are not an
additional envelope around that struct.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import struct
from threading import Event, Lock, Thread, current_thread
from time import perf_counter_ns, time_ns
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np

from exo_collection.adapters.base import AdapterError, ModalityDescriptor
from exo_collection.adapters.encoder.simulated import ENCODER_CHANNELS, ENCODER_UNITS
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.domain.events import SampleBatch


HEAD_STATUS = 0xCC
FRAME_TAIL = 0x55
STATUS_FORMAT = "<BHBBffffffIBB"
STATUS_STRUCT = struct.Struct(STATUS_FORMAT)
STATUS_SIZE = STATUS_STRUCT.size  # 35 bytes, including head, CRC and tail
BAUD_DEFAULT = 9600
TEENSY_VID = 0x16C0
TEENSY_PID = 0x0483

# Compatibility names retained for callers of the earlier draft.
FRAME_HEADER = HEAD_STATUS
FRAME_FOOTER = FRAME_TAIL
PAYLOAD_STRUCT = STATUS_STRUCT
PAYLOAD_SIZE = STATUS_SIZE
CRC8_POLY = 0x07


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
    sequence: int
    state: int
    error: int
    left_position: float
    left_velocity: float
    left_torque: float
    right_position: float
    right_velocity: float
    right_torque: float
    hardware_time_ms: int


def parse_status_frame(data: bytes) -> MotorStatusFrame | None:
    """Validate and decode one exact firmware status frame."""

    if len(data) != STATUS_SIZE or data[0] != HEAD_STATUS or data[-1] != FRAME_TAIL:
        return None
    if calc_crc8(data[1 : STATUS_SIZE - 2]) != data[-2]:
        return None
    try:
        (
            _head,
            sequence,
            state,
            error,
            left_position,
            left_velocity,
            left_torque,
            right_position,
            right_velocity,
            right_torque,
            hardware_time_ms,
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
        hardware_time_ms=hardware_time_ms,
    )


class MotorStatusStreamParser:
    """Incremental parser that re-synchronizes one byte after a bad candidate."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.crc_or_format_errors = 0
        self.discarded_bytes = 0

    def reset(self) -> None:
        self._buffer.clear()
        self.crc_or_format_errors = 0
        self.discarded_bytes = 0

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[MotorStatusFrame]:
        self._buffer.extend(chunk)
        frames: list[MotorStatusFrame] = []
        while True:
            index = self._buffer.find(bytes((HEAD_STATUS,)))
            if index < 0:
                self.discarded_bytes += len(self._buffer)
                self._buffer.clear()
                break
            if index:
                self.discarded_bytes += index
                del self._buffer[:index]
            if len(self._buffer) < STATUS_SIZE:
                break
            candidate = bytes(self._buffer[:STATUS_SIZE])
            parsed = parse_status_frame(candidate)
            if parsed is None:
                self.crc_or_format_errors += 1
                del self._buffer[0]
                continue
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

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TeensyEncoderConfig:
    device_id: str = "encoder_teensy"
    clock_domain: str = "encoder_teensy_clock"
    port: str | None = None
    baudrate: int = BAUD_DEFAULT
    vid: int = TEENSY_VID
    pid: int = TEENSY_PID
    nominal_rate_hz: float = 100.0
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


def _coerce_config(value: TeensyEncoderConfig | Mapping[str, Any] | None) -> TeensyEncoderConfig:
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
    return TeensyEncoderConfig(**{key: item for key, item in raw.items() if key in allowed})


def find_teensy_port(
    ports: Iterable[Any], *, vid: int = TEENSY_VID, pid: int = TEENSY_PID
) -> str | None:
    for port in ports:
        if getattr(port, "vid", None) == vid and getattr(port, "pid", None) == pid:
            return str(port.device)
    return None


def _match_teensy_port(vid: int = TEENSY_VID, pid: int = TEENSY_PID) -> str | None:
    try:
        import serial.tools.list_ports
    except ImportError as exc:
        raise AdapterError(
            "未安装 pyserial；请安装硬件依赖后再连接 Teensy。"
        ) from exc
    return find_teensy_port(serial.tools.list_ports.comports(), vid=vid, pid=pid)


class TeensySerialEncoderAdapter(QueuedHardwareAdapter):
    """Read motor feedback only; this class never calls ``serial.write``."""

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
        self._last_device_sequence: int | None = None
        self._sequence_gap_count = 0

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
                "protocol": "teensy_status_v1",
                "status_format": STATUS_FORMAT,
                "status_size_bytes": STATUS_SIZE,
                "crc8_polynomial": "0x07",
                "vid": cfg.vid,
                "pid": cfg.pid,
                "baudrate": cfg.baudrate,
                "read_only": True,
                "device_timestamp_unit": "ms",
                "device_sequence_modulus": 2**16,
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return {**asdict(self._config), "resolved_port": self._resolved_port}

    def _connect_hardware(self) -> None:
        cfg = self._config
        if self._port_lister is not None:
            detected = find_teensy_port(self._port_lister(), vid=cfg.vid, pid=cfg.pid)
        else:
            detected = _match_teensy_port(cfg.vid, cfg.pid) if not cfg.port else None
        port = cfg.port or detected
        if not port:
            raise AdapterError(
                f"未找到 Teensy 串口（VID=0x{cfg.vid:04X}, PID=0x{cfg.pid:04X}）。"
            )
        if self._serial_factory is None:
            try:
                import serial
            except ImportError as exc:
                raise AdapterError("未安装 pyserial；无法连接 Teensy。") from exc
            self._serial_factory = serial.Serial
        try:
            self._serial = self._serial_factory(
                port=port,
                baudrate=cfg.baudrate,
                timeout=cfg.read_timeout_s,
            )
        except BaseException as exc:
            raise AdapterError(f"无法打开 Teensy 串口 {port}: {exc}") from exc
        self._resolved_port = port

    def _reset_trial_state(self) -> None:
        self._stop_event.clear()
        self._parser.reset()
        self._pending.clear()
        self._sample_index = 0
        self._batch_sequence = 0
        self._last_device_sequence = None
        self._sequence_gap_count = 0

    def _start_hardware(self) -> None:
        if self._serial is None or not getattr(self._serial, "is_open", True):
            raise AdapterError("Teensy 串口未打开")
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
                raise AdapterError("Teensy 串口读取线程未在超时内停止")
        self._thread = None
        self._emit_pending(force=True)

    def _close_hardware(self) -> None:
        self._stop_event.set()
        serial_port, self._serial = self._serial, None
        if serial_port is not None:
            serial_port.close()

    def _read_guarded(self) -> None:
        try:
            assert self._serial is not None
            while not self._stop_event.is_set():
                size = max(1, min(self._config.read_size, int(getattr(self._serial, "in_waiting", 0)) or 1))
                chunk = self._serial.read(size)
                if not chunk:
                    continue
                received_ns = perf_counter_ns()
                for frame in self._parser.feed(chunk):
                    self._accept_frame(frame, received_ns)
        except BaseException as exc:
            self._set_fault(exc)
            self._stop_event.set()

    def _accept_frame(self, frame: MotorStatusFrame, received_ns: int) -> None:
        previous = self._last_device_sequence
        if previous is not None:
            missing = (frame.sequence - previous - 1) % (2**16)
            if missing:
                self._sequence_gap_count += missing
        self._last_device_sequence = frame.sequence
        with self._pending_lock:
            self._pending.append((frame, received_ns))
            should_emit = len(self._pending) >= self._config.batch_size
        if should_emit:
            self._emit_pending(force=False)

    def _emit_pending(self, *, force: bool) -> None:
        with self._pending_lock:
            if not self._pending or (not force and len(self._pending) < self._config.batch_size):
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
                if self._trial is not None and self._trial.session_uuid is not None
                else None
            ),
            trial_uuid=str(self._trial.trial_uuid) if self._trial is not None else None,
            device_id=self._config.device_id,
            modality="encoder",
            clock_domain=self._config.clock_domain,
            host_monotonic_ns=first_host_ns,
            host_utc_ns=time_ns(),
            first_sample_index=self._sample_index,
            sample_count=len(frames),
            sequence_number=self._batch_sequence,
            device_timestamp=frames[0].hardware_time_ms,
            sample_rate_hz=self._config.nominal_rate_hz,
            data=data,
        )
        self._publish_raw(event, item_count=len(frames), host_monotonic_ns=first_host_ns)
        self._sample_index += len(frames)
        self._batch_sequence += 1

    def _dropped_packets(self) -> int:
        return self._sequence_gap_count

    def _sequence_gaps(self) -> int:
        return self._sequence_gap_count

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        return {
            "crc_or_format_errors": self._parser.crc_or_format_errors,
            "discarded_serial_bytes": self._parser.discarded_bytes,
            "buffered_serial_bytes": self._parser.buffered_bytes,
            "resolved_port": self._resolved_port,
            "read_only": True,
        }


__all__ = [
    "BAUD_DEFAULT",
    "CRC8_POLY",
    "FRAME_FOOTER",
    "FRAME_HEADER",
    "FRAME_TAIL",
    "HEAD_STATUS",
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
