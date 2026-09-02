"""TCP client for the h/p/cosmos gaitway-3D streaming server.

Protocol source: TM-ICD-0004-ARS, issue A, revision 5 (2021-03-16).
The gaitway application owns the TCP server and accepts one client on port
49500. Commands are ASCII/CRLF; responses and samples are little-endian
length-prefixed binary packets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import socket
from struct import Struct, unpack_from
from threading import Event, Lock, Thread, current_thread
from time import perf_counter_ns, time_ns
from typing import Any, Mapping

import numpy as np

from exo_collection.adapters.base import AdapterError, ModalityDescriptor
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.domain.events import GaitwayPacketEvent, SampleBatch


_log = logging.getLogger(__name__)

GAITWAY_DEFAULT_PORT = 49_500
PACKET_SETTINGS = 0x0000
PACKET_TYPE_I = 0x0001
PACKET_TYPE_II = 0x0002
PACKET_ACK = 0x0006
PACKET_NAK = 0x0015
TYPE_I_HEADER_SIZE = 16
TYPE_I_SAMPLE_SIZE = 36
TYPE_II_HEADER_SIZE = 32
TYPE_II_SAMPLE_SIZE = 44
_TYPE_I_HEADER = Struct("<HHI8x")
_TYPE_I_SAMPLE = Struct("<8fHH")
# Type-II header: size(U16) type(U16) id(U32) gait_type(U16) contact_side(U16)
# step_count(U32) + 16 padding bytes.  No timestamp field exists in the ICD.
_TYPE_II_HEADER = Struct("<HHIHHI16x")
# Type-II sample: foot_contact(U16) digital_inputs(U16) then
# FzL FyL FxL COPyL COPxL FzR FyR FxR COPyR COPxR (10 little-endian floats).
_TYPE_II_SAMPLE = Struct("<HH10f")
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

# Type II is NOT a second physical plate: it is gaitway's internal left/right
# decomposition of the single instrumented treadmill.  ``grf_source_type`` is
# recorded verbatim in the CSV/meta provenance so OpenSim keeps that fact.
TYPE_II_CHANNELS = (
    "foot_contact",
    "digital_inputs",
    "fx_l",
    "fy_l",
    "fz_l",
    "cop_x_l",
    "cop_y_l",
    "fx_r",
    "fy_r",
    "fz_r",
    "cop_x_r",
    "cop_y_r",
)
TYPE_II_UNITS = (
    "",  # foot_contact is an enum: 0 aerial / 1 single / 2 double
    "bitmask",  # digital_inputs
    "N",
    "N",
    "N",
    "m",
    "m",
    "N",
    "N",
    "N",
    "m",
    "m",
)
GRF_SOURCE_TYPE_DECOMPOSED = "gaitway_single_platform_decomposed_left_right"


def build_start_ds_command(
    *,
    sample_rate_hz: int,
    trigger_mode: int,
    sync_out_enabled: bool,
    type_i_mode: int,
    type_ii_mode: int,
    seconds: int = 0,
) -> str:
    """Build the ``startDS`` command per TM-ICD-0004-ARS A5.

    Argument order is fixed by the ICD::

        startDS <freq> <seconds> <trigger> <syncout> <typeI> <typeII>

    - ``freq``       sample frequency in Hz (must be one of ``_VALID_SAMPLE_RATES``)
    - ``seconds``    acquisition duration; ``0`` means continuous until ``stopDS``
    - ``trigger``    0..3 external trigger mode
    - ``syncout``    0/1 enable the sync output
    - ``typeI``      0=off / 1=header-only / 2=header+samples (total GRF/COP)
    - ``typeII``     0=off / 1=header-only / 2=header+samples (left/right decomposition)
    """

    return (
        f"startDS {int(sample_rate_hz)} {int(seconds)} {int(trigger_mode)} "
        f"{int(bool(sync_out_enabled))} {int(type_i_mode)} {int(type_ii_mode)}"
    )


def parse_type_i_packet(packet: bytes) -> tuple[dict[str, int], np.ndarray]:
    """Parse a raw Type-I packet into ``(header, samples)``.

    ``header`` holds ``packet_size``/``packet_type``/``packet_id``.  ``samples``
    is a ``(count, 10)`` float32 array in canonical channel order
    (``FORCE_PLATE_CHANNELS``).  Raises :class:`GaitwayPacketError` on a
    malformed packet.
    """

    if len(packet) < TYPE_I_HEADER_SIZE:
        raise GaitwayPacketError("Type-I packet shorter than its header")
    packet_size, packet_type, packet_id = _TYPE_I_HEADER.unpack_from(packet)
    if packet_size != len(packet) or packet_type != PACKET_TYPE_I:
        raise GaitwayPacketError("invalid Type-I packet header")
    payload_size = packet_size - TYPE_I_HEADER_SIZE
    if payload_size <= 0 or payload_size % TYPE_I_SAMPLE_SIZE:
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
        data[index] = (fx, fy, fz, cop_x, cop_y, tz, speed, elevation, heart, digital)
        offset += TYPE_I_SAMPLE_SIZE
    return {
        "packet_size": packet_size,
        "packet_type": packet_type,
        "packet_id": packet_id,
    }, data


def parse_type_ii_packet(packet: bytes) -> tuple[dict[str, int], np.ndarray]:
    """Parse a raw Type-II packet into ``(header, samples)``.

    ``header`` adds ``gait_type``/``contact_side``/``step_count`` on top of the
    shared ``packet_size``/``packet_type``/``packet_id``.  ``samples`` is a
    ``(count, 12)`` float32 array in canonical channel order
    (``TYPE_II_CHANNELS``).  Per the ICD, Type-II packets are emitted per step
    and are NOT a uniform sample stream — no sample-rate assumption is made.
    """

    if len(packet) < TYPE_II_HEADER_SIZE:
        raise GaitwayPacketError("Type-II packet shorter than its header")
    (
        packet_size,
        packet_type,
        packet_id,
        gait_type,
        contact_side,
        step_count,
    ) = _TYPE_II_HEADER.unpack_from(packet)
    if packet_size != len(packet) or packet_type != PACKET_TYPE_II:
        raise GaitwayPacketError("invalid Type-II packet header")
    payload_size = packet_size - TYPE_II_HEADER_SIZE
    if payload_size <= 0 or payload_size % TYPE_II_SAMPLE_SIZE:
        raise GaitwayPacketError(
            f"Type-II payload size {payload_size} is not a multiple of "
            f"{TYPE_II_SAMPLE_SIZE}"
        )
    count = payload_size // TYPE_II_SAMPLE_SIZE
    data = np.empty((count, len(TYPE_II_CHANNELS)), dtype=np.float32)
    offset = TYPE_II_HEADER_SIZE
    for index in range(count):
        foot_contact, digital, fz_l, fy_l, fx_l, cop_y_l, cop_x_l, fz_r, fy_r, fx_r, cop_y_r, cop_x_r = (
            _TYPE_II_SAMPLE.unpack_from(packet, offset)
        )
        data[index] = (
            foot_contact,
            digital,
            fx_l,
            fy_l,
            fz_l,
            cop_x_l,
            cop_y_l,
            fx_r,
            fy_r,
            fz_r,
            cop_x_r,
            cop_y_r,
        )
        offset += TYPE_II_SAMPLE_SIZE
    return {
        "packet_size": packet_size,
        "packet_type": packet_type,
        "packet_id": packet_id,
        "gait_type": gait_type,
        "contact_side": contact_side,
        "step_count": step_count,
    }, data


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
    # startDS packet-mode selectors: 0 = off, 1 = header only, 2 = header + samples.
    type_i_mode: int = 2
    type_ii_mode: int = 2
    # Recording-side storage switches (raw binary is always captured for offline
    # re-parse; these gate the additional per-Trial gaitway/ artefacts).
    save_raw_packets: bool = True
    save_parsed_csv: bool = True
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
        if self.type_i_mode not in {0, 1, 2}:
            raise ValueError("type_i_mode must be in [0, 2]")
        if self.type_ii_mode not in {0, 1, 2}:
            raise ValueError("type_ii_mode must be in [0, 2]")
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
    """Loss-intolerant gaitway stream adapter requesting Type I + Type II.

    Type I is a continuous uniform stream of total GRF/COP/Tz/treadmill status
    and is published both as a :class:`SampleBatch` (for synchronized preview
    and HDF5 recording) and as a byte-preserving :class:`GaitwayPacketEvent`.
    Type II is gaitway's per-step left/right decomposition and is published
    only as a :class:`GaitwayPacketEvent`; it is NOT a physical second plate.
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
        self._settings_query_error: str | None = None
        self._sample_index = 0
        self._last_packet_id: int | None = None
        self._packet_gaps = 0
        self._malformed_packets = 0
        self._type_i_packets = 0
        self._type_ii_sample_index = 0
        self._type_ii_last_packet_id: int | None = None
        self._type_ii_packet_gaps = 0
        self._type_ii_packets = 0

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
                "type_i_mode": self._config.type_i_mode,
                "type_ii_enabled": self._config.type_ii_mode > 0,
                "type_ii_mode": self._config.type_ii_mode,
                "grf_source_type": GRF_SOURCE_TYPE_DECOMPOSED,
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
            "settings_query_error": self._settings_query_error,
            "baseline_reset_issued": False,
            "type_i_packet_mode": self._config.type_i_mode,
            "type_ii_packet_mode": self._config.type_ii_mode,
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
            self._query_settings_best_effort()

    def _query_settings_best_effort(self) -> None:
        """Capture the DS settings packet for provenance, without failing connect.

        The settings packet (GRF range, COP threshold, filter cutoff, ...) is
        metadata only.  The acquisition-critical handshake is ``startDS``; if the
        server does not answer ``getDSsettings`` here the collector still connects
        and ``startDS`` raises its own actionable error when the device is
        genuinely unreachable.  This mirrors the field self-check, which skips
        ``getDSsettings`` and streams via ``startDS`` alone.
        """
        try:
            self._send_command("getDSsettings")
            self._expect_ack("getDSsettings")
            settings = self._read_packet_until(
                expected_type=PACKET_SETTINGS,
                timeout_s=self._config.connect_timeout_s,
            )
        except AdapterError as exc:
            self._settings_query_error = str(exc)
            _log.warning(
                "gaitway getDSsettings failed (%s); continuing without settings "
                "metadata — verify the gaitway-3D GUI is in STREAM DATA mode and "
                "no other client holds the single connection",
                exc,
            )
            return
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
        self._type_ii_sample_index = 0
        self._type_ii_last_packet_id = None
        self._type_ii_packet_gaps = 0
        self._type_ii_packets = 0

    def _start_hardware(self) -> None:
        if self._socket is None:
            raise AdapterError("gaitway TCP socket is not connected")
        command = build_start_ds_command(
            sample_rate_hz=self._config.sample_rate_hz,
            trigger_mode=self._config.trigger_mode,
            sync_out_enabled=self._config.sync_out_enabled,
            type_i_mode=self._config.type_i_mode,
            type_ii_mode=self._config.type_ii_mode,
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
                elif packet_type == PACKET_TYPE_II:
                    self._accept_type_ii(packet, perf_counter_ns())
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
        try:
            header, data = parse_type_i_packet(packet)
        except GaitwayPacketError:
            self._malformed_packets += 1
            raise
        packet_id = header["packet_id"]
        count = data.shape[0]

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
        self._publish_raw(
            SampleBatch(
                session_uuid=self._session_uuid(),
                trial_uuid=self._trial_uuid(),
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
            ),
            item_count=count,
            host_monotonic_ns=first_host_ns,
        )
        self._publish_packet_event(
            packet_type=PACKET_TYPE_I,
            packet_id=packet_id,
            sample_index=self._sample_index,
            raw_bytes=packet,
            received_ns=received_ns,
        )
        self._sample_index += count

    def _accept_type_ii(self, packet: bytes, received_ns: int) -> None:
        """Accept one per-step Type-II packet (left/right decomposition)."""
        try:
            header, _data = parse_type_ii_packet(packet)
        except GaitwayPacketError:
            self._malformed_packets += 1
            raise
        packet_id = header["packet_id"]

        if self._type_ii_last_packet_id is not None:
            expected = (self._type_ii_last_packet_id + 1) & 0xFFFF_FFFF
            if packet_id != expected:
                self._type_ii_packet_gaps += (
                    (packet_id - expected) & 0xFFFF_FFFF
                ) or 1
        self._type_ii_last_packet_id = packet_id
        self._type_ii_packets += 1

        self._publish_packet_event(
            packet_type=PACKET_TYPE_II,
            packet_id=packet_id,
            sample_index=self._type_ii_sample_index,
            raw_bytes=packet,
            received_ns=received_ns,
        )
        # Type II samples are per-step; index counts packets (steps), not Hz.
        self._type_ii_sample_index += 1

    def _publish_packet_event(
        self,
        *,
        packet_type: int,
        packet_id: int,
        sample_index: int,
        raw_bytes: bytes,
        received_ns: int,
    ) -> None:
        """Emit a byte-preserving :class:`GaitwayPacketEvent` for recording."""
        self._publish_raw(
            GaitwayPacketEvent(
                session_uuid=self._session_uuid(),
                trial_uuid=self._trial_uuid(),
                device_id=self._config.device_id,
                modality="force_plate",
                clock_domain=self._config.clock_domain,
                host_monotonic_ns=received_ns,
                host_utc_ns=time_ns(),
                packet_type=packet_type,
                packet_id=packet_id,
                sample_index=sample_index,
                raw_bytes=bytes(raw_bytes),
            ),
            item_count=1,
            host_monotonic_ns=received_ns,
        )

    def _session_uuid(self) -> str | None:
        return (
            str(self._trial.session_uuid)
            if self._trial is not None and self._trial.session_uuid is not None
            else None
        )

    def _trial_uuid(self) -> str | None:
        return str(self._trial.trial_uuid) if self._trial is not None else None

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
            expected = (
                f"0x{expected_type:04X}" if expected_type is not None else "any"
            )
            raise AdapterError(
                "timed out waiting for gaitway protocol response "
                f"(expected type {expected} from "
                f"{self._config.server_host}:{self._config.server_port}) — check "
                "the gaitway-3D GUI is in STREAM DATA mode, that no other client "
                "holds the single connection, and that server_host is the gaitway "
                "PC address"
            )
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
            "type_ii_packets": self._type_ii_packets,
            "type_ii_packet_id_gaps": self._type_ii_packet_gaps,
            "malformed_packets": self._malformed_packets,
            "framer_buffered_bytes": self._framer.buffered_bytes,
            "settings_version": self._settings_version,
            "settings_query_error": self._settings_query_error,
            "type_ii_enabled": self._config.type_ii_mode > 0,
        }


__all__ = [
    "FORCE_PLATE_CHANNELS",
    "FORCE_PLATE_UNITS",
    "TYPE_II_CHANNELS",
    "TYPE_II_UNITS",
    "GRF_SOURCE_TYPE_DECOMPOSED",
    "GAITWAY_DEFAULT_PORT",
    "PACKET_TYPE_I",
    "PACKET_TYPE_II",
    "build_start_ds_command",
    "parse_type_i_packet",
    "parse_type_ii_packet",
    "GaitwayForcePlateConfig",
    "GaitwayForcePlateTcpAdapter",
    "GaitwayPacketError",
    "GaitwayPacketFramer",
]
