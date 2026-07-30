"""TCP client for the h/p/cosmos gaitway-3D streaming server.

Protocol source: TM-ICD-0004-ARS, issue A, revision 5 (2021-03-16).
The gaitway application owns the TCP server and accepts one client on port
49500. Commands are ASCII/CRLF; responses and samples are little-endian
length-prefixed binary packets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import socket
from struct import Struct, unpack_from
from threading import Event, Lock, Thread, current_thread
from time import perf_counter_ns, time_ns
from typing import Any, Mapping

import numpy as np

from exo_collection.adapters.base import AdapterError, ModalityDescriptor
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.domain.events import SampleBatch


GAITWAY_DEFAULT_PORT = 49_500
PACKET_SETTINGS = 0x0000
PACKET_TYPE_I = 0x0001
PACKET_ACK = 0x0006
PACKET_NAK = 0x0015
TYPE_I_HEADER_SIZE = 16
TYPE_I_SAMPLE_SIZE = 36
_TYPE_I_HEADER = Struct("<HHI8x")
_TYPE_I_SAMPLE = Struct("<8fHH")
_VALID_SAMPLE_RATES = frozenset({100, 200, 250, 400, 500, 1000, 2000})

FORCE_PLATE_CHANNELS = (
    "fx",
    "fy",
    "fz",
    "cop_x",
    "cop_y",
    "tz",
    "treadmill_speed",
    "treadmill_elevation",
    "heart_rate",
    "digital_inputs",
)
FORCE_PLATE_UNITS = (
    "N",
    "N",
    "N",
    "m",
    "m",
    "N*m",
    "m/s",
    "%",
    "bpm",
    "bitmask",
)


class GaitwayPacketError(AdapterError):
    """Malformed or unexpected gaitway protocol packet."""


class GaitwayCommandRejected(AdapterError):
    """The gaitway server returned a NAK packet."""


class GaitwayPacketFramer:
    """Incrementally frame TCP bytes using the packet's leading U16 size."""

    def __init__(self, *, maximum_packet_size: int = 65_535) -> None:
        self.maximum_packet_size = int(maximum_packet_size)
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes) -> list[bytes]:
        if data:
            self._buffer.extend(data)
        packets: list[bytes] = []
        while len(self._buffer) >= 2:
            packet_size = int.from_bytes(self._buffer[:2], "little")
            if packet_size < 4 or packet_size > self.maximum_packet_size:
                raise GaitwayPacketError(
                    f"invalid gaitway packet size {packet_size}"
                )
            if len(self._buffer) < packet_size:
                break
            packets.append(bytes(self._buffer[:packet_size]))
            del self._buffer[:packet_size]
        return packets


@dataclass(frozen=True, slots=True)
class GaitwayForcePlateConfig:
    device_id: str = "gaitway_force_plate"
    clock_domain: str = "gaitway_stream_clock"
    server_host: str = "127.0.0.1"
    server_port: int = GAITWAY_DEFAULT_PORT
    sample_rate_hz: int = 1000
    trigger_mode: int = 0
    sync_out_enabled: bool = False
    queue_capacity: int = 512
    connect_timeout_s: float = 5.0
    socket_timeout_s: float = 0.2
    stop_timeout_s: float = 8.0
    query_settings_on_connect: bool = True

    def __post_init__(self) -> None:
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if not self.server_host.strip():
            raise ValueError("server_host must not be empty")
        if not 1 <= self.server_port <= 65_535:
            raise ValueError("server_port must be in [1, 65535]")
        if self.sample_rate_hz not in _VALID_SAMPLE_RATES:
            raise ValueError(
                "sample_rate_hz must be one of "
                + ", ".join(str(item) for item in sorted(_VALID_SAMPLE_RATES))
            )
        if self.trigger_mode not in {0, 1, 2, 3}:
            raise ValueError("trigger_mode must be in [0, 3]")
        if self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if min(
            self.connect_timeout_s,
            self.socket_timeout_s,
            self.stop_timeout_s,
        ) <= 0:
            raise ValueError("TCP timeouts must be positive")

    @classmethod
    def from_value(
        cls, value: GaitwayForcePlateConfig | Mapping[str, Any] | None
    ) -> GaitwayForcePlateConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        raw = dict(value)
        parameters = raw.pop("parameters", None)
        if isinstance(parameters, Mapping):
            raw.update(parameters)
        if "id" in raw and "device_id" not in raw:
            raw["device_id"] = raw.pop("id")
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: item for key, item in raw.items() if key in allowed})


class GaitwayForcePlateTcpAdapter(QueuedHardwareAdapter):
    """Loss-intolerant Type-I gaitway stream adapter.

    Type-II per-step packets are deliberately disabled in ``startDS``. The
    continuous Type-I packet already contains the combined force-plate signal
    required for synchronized preview and raw recording.
    """

    def __init__(
        self,
        config: GaitwayForcePlateConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._config = GaitwayForcePlateConfig.from_value(config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._socket: socket.socket | None = None
        self._socket_lock = Lock()
        self._framer = GaitwayPacketFramer()
        self._pending_packets: list[bytes] = []
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._stop_ack = Event()
        self._settings_packet_hex: str | None = None
        self._settings_version: int | None = None
        self._sample_index = 0
        self._last_packet_id: int | None = None
        self._packet_gaps = 0
        self._malformed_packets = 0
        self._type_i_packets = 0

    def descriptor(self) -> ModalityDescriptor:
        return ModalityDescriptor(
            device_id=self._config.device_id,
            modality="force_plate",
            display_name="h/p/cosmos gaitway-3D 测力台",
            clock_domain=self._config.clock_domain,
            event_kind="sample_batch",
            channels=FORCE_PLATE_CHANNELS,
            units=FORCE_PLATE_UNITS,
            nominal_rate_hz=float(self._config.sample_rate_hz),
            sample_shape=(len(FORCE_PLATE_CHANNELS),),
            dtype="<f4",
            metadata={
                "transport": "tcp",
                "protocol": "TM-ICD-0004-ARS A5",
                "server_host": self._config.server_host,
                "server_port": self._config.server_port,
                "packet_type": "Type I",
                "type_ii_enabled": False,
                "trigger_mode": self._config.trigger_mode,
                "sync_out_enabled": self._config.sync_out_enabled,
                "settings_version": self._settings_version,
                "settings_packet_hex": self._settings_packet_hex,
                "preview_labels": list(FORCE_PLATE_CHANNELS),
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return {
            **asdict(self._config),
            "settings_version": self._settings_version,
            "settings_packet_hex": self._settings_packet_hex,
            "baseline_reset_issued": False,
            "type_i_packet_mode": 2,
            "type_ii_packet_mode": 0,
        }

    def _connect_hardware(self) -> None:
        sock = socket.create_connection(
            (self._config.server_host, self._config.server_port),
            timeout=self._config.connect_timeout_s,
        )
        sock.settimeout(self._config.socket_timeout_s)
        self._socket = sock
        self._framer.reset()
        self._pending_packets.clear()
        if self._config.query_settings_on_connect:
            self._send_command("getDSsettings")
            self._expect_ack("getDSsettings")
            settings = self._read_packet_until(
                expected_type=PACKET_SETTINGS,
                timeout_s=self._config.connect_timeout_s,
            )
            self._settings_packet_hex = settings.hex()
            self._settings_version = (
                int(unpack_from("<H", settings, 4)[0])
                if len(settings) >= 6
                else None
            )

    def _reset_trial_state(self) -> None:
        self._stop_event.clear()
        self._stop_ack.clear()
        self._sample_index = 0
        self._last_packet_id = None
        self._packet_gaps = 0
        self._malformed_packets = 0
        self._type_i_packets = 0

    def _start_hardware(self) -> None:
        if self._socket is None:
            raise AdapterError("gaitway TCP socket is not connected")
        command = (
            f"startDS {self._config.sample_rate_hz} 0 "
            f"{self._config.trigger_mode} {int(self._config.sync_out_enabled)} 2 0"
        )
        self._send_command(command)
        self._expect_ack(command)
        self._stop_event.clear()
        self._thread = Thread(
            target=self._read_guarded,
            name=f"gaitway-read-{self._config.device_id}",
            daemon=True,
        )
        self._thread.start()

    def _stop_hardware(self) -> None:
        thread = self._thread
        if thread is None:
            return
        if thread.is_alive():
            self._stop_ack.clear()
            self._send_command("stopDS")
            if not self._stop_ack.wait(self._config.stop_timeout_s):
                raise AdapterError(
                    "timed out waiting for gaitway stopDS acknowledgement"
                )
        self._stop_event.set()
        if thread is not current_thread():
            thread.join(timeout=max(1.0, self._config.socket_timeout_s * 4))
            if thread.is_alive():
                raise AdapterError("gaitway TCP reader did not stop")
        self._thread = None

    def _close_hardware(self) -> None:
        self._stop_event.set()
        sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        self._pending_packets.clear()
        self._framer.reset()

    def _read_guarded(self) -> None:
        try:
            while not self._stop_event.is_set():
                packet = self._read_packet()
                if packet is None:
                    continue
                packet_type = int(unpack_from("<H", packet, 2)[0])
                if packet_type == PACKET_TYPE_I:
                    self._accept_type_i(packet, perf_counter_ns())
                elif packet_type == PACKET_ACK:
                    command = self._response_command(packet)
                    if command == "stopDS":
                        self._stop_ack.set()
                elif packet_type == PACKET_NAK:
                    raise GaitwayCommandRejected(
                        f"gaitway rejected command {self._response_command(packet)!r}"
                    )
                elif packet_type == PACKET_SETTINGS:
                    self._settings_packet_hex = packet.hex()
                else:
                    raise GaitwayPacketError(
                        f"unexpected gaitway packet type 0x{packet_type:04X}"
                    )
        except BaseException as exc:
            if not self._stop_event.is_set():
                self._set_fault(exc)
            self._stop_ack.set()

    def _accept_type_i(self, packet: bytes, received_ns: int) -> None:
        packet_size, packet_type, packet_id = _TYPE_I_HEADER.unpack_from(packet)
        if packet_size != len(packet) or packet_type != PACKET_TYPE_I:
            self._malformed_packets += 1
            raise GaitwayPacketError("invalid Type-I packet header")
        payload_size = packet_size - TYPE_I_HEADER_SIZE
        if payload_size <= 0 or payload_size % TYPE_I_SAMPLE_SIZE:
            self._malformed_packets += 1
            raise GaitwayPacketError(
                f"Type-I payload size {payload_size} is not a multiple of "
                f"{TYPE_I_SAMPLE_SIZE}"
            )
        count = payload_size // TYPE_I_SAMPLE_SIZE
        data = np.empty((count, len(FORCE_PLATE_CHANNELS)), dtype=np.float32)
        offset = TYPE_I_HEADER_SIZE
        for index in range(count):
            fz, fy, fx, cop_y, cop_x, tz, speed, elevation, heart, digital = (
                _TYPE_I_SAMPLE.unpack_from(packet, offset)
            )
            data[index] = (
                fx,
                fy,
                fz,
                cop_x,
                cop_y,
                tz,
                speed,
                elevation,
                heart,
                digital,
            )
            offset += TYPE_I_SAMPLE_SIZE

        if self._last_packet_id is not None:
            expected = (self._last_packet_id + 1) & 0xFFFF_FFFF
            if packet_id != expected:
                self._packet_gaps += (
                    (packet_id - expected) & 0xFFFF_FFFF
                ) or 1
        self._last_packet_id = packet_id
        self._type_i_packets += 1

        first_host_ns = max(
            0,
            received_ns
            - round((count - 1) * 1_000_000_000 / self._config.sample_rate_hz),
        )
        event = SampleBatch(
            session_uuid=(
                str(self._trial.session_uuid)
                if self._trial is not None and self._trial.session_uuid is not None
                else None
            ),
            trial_uuid=str(self._trial.trial_uuid) if self._trial is not None else None,
            device_id=self._config.device_id,
            modality="force_plate",
            clock_domain=self._config.clock_domain,
            host_monotonic_ns=first_host_ns,
            host_utc_ns=time_ns(),
            first_sample_index=self._sample_index,
            sample_count=count,
            sequence_number=packet_id,
            device_timestamp=None,
            sample_rate_hz=float(self._config.sample_rate_hz),
            data=np.ascontiguousarray(data),
        )
        self._publish_raw(
            event,
            item_count=count,
            host_monotonic_ns=first_host_ns,
        )
        self._sample_index += count

    def _send_command(self, command: str) -> None:
        sock = self._socket
        if sock is None:
            raise AdapterError("gaitway TCP socket is not connected")
        payload = command.encode("ascii") + b"\r\n"
        with self._socket_lock:
            sock.sendall(payload)

    def _expect_ack(self, command: str) -> None:
        packet = self._read_packet_until(
            expected_type=PACKET_ACK,
            timeout_s=self._config.connect_timeout_s,
        )
        packet_type = int(unpack_from("<H", packet, 2)[0])
        response_command = self._response_command(packet)
        if packet_type == PACKET_NAK:
            raise GaitwayCommandRejected(
                f"gaitway rejected command {response_command!r}"
            )
        if packet_type != PACKET_ACK:
            raise GaitwayPacketError(
                f"expected ACK, received packet type 0x{packet_type:04X}"
            )
        if response_command != command:
            raise GaitwayPacketError(
                f"ACK command mismatch: expected {command!r}, got "
                f"{response_command!r}"
            )

    def _read_packet_until(
        self,
        *,
        expected_type: int | None,
        timeout_s: float,
    ) -> bytes:
        deadline = perf_counter_ns() + round(timeout_s * 1_000_000_000)
        deferred: list[bytes] = []
        try:
            while perf_counter_ns() < deadline:
                packet = self._read_packet()
                if packet is None:
                    continue
                packet_type = int(unpack_from("<H", packet, 2)[0])
                if expected_type is None or packet_type == expected_type:
                    return packet
                if packet_type == PACKET_NAK:
                    raise GaitwayCommandRejected(
                        "gaitway rejected command "
                        f"{self._response_command(packet)!r}"
                    )
                deferred.append(packet)
            raise AdapterError("timed out waiting for gaitway protocol response")
        finally:
            if deferred:
                self._pending_packets[0:0] = deferred

    def _read_packet(self) -> bytes | None:
        if self._pending_packets:
            return self._pending_packets.pop(0)
        sock = self._socket
        if sock is None:
            raise AdapterError("gaitway TCP socket is not connected")
        while True:
            try:
                chunk = sock.recv(65_535)
            except socket.timeout:
                return None
            if not chunk:
                raise ConnectionError("gaitway TCP server closed the connection")
            packets = self._framer.feed(chunk)
            if packets:
                self._pending_packets.extend(packets[1:])
                return packets[0]

    @staticmethod
    def _response_command(packet: bytes) -> str:
        return packet[4:].rstrip(b"\x00").decode("ascii", errors="replace")

    def _dropped_packets(self) -> int:
        return self._packet_gaps

    def _sequence_gaps(self) -> int:
        return self._packet_gaps

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        return {
            "server_host": self._config.server_host,
            "server_port": self._config.server_port,
            "type_i_packets": self._type_i_packets,
            "packet_id_gaps": self._packet_gaps,
            "malformed_packets": self._malformed_packets,
            "framer_buffered_bytes": self._framer.buffered_bytes,
            "settings_version": self._settings_version,
            "type_ii_enabled": False,
        }


__all__ = [
    "FORCE_PLATE_CHANNELS",
    "FORCE_PLATE_UNITS",
    "GaitwayForcePlateConfig",
    "GaitwayForcePlateTcpAdapter",
    "GaitwayPacketError",
    "GaitwayPacketFramer",
]
