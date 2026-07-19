"""Raw Ethernet ultrasound adapter via Scapy + Npcap.

The ultrasound transport is *not* a conventional Ethernet protocol.  The
device places its own 1000-byte record directly in the captured frame:
``00``, channel marker ``01``/``02``/``03``/``04``, sample bytes, ``FF``.
Consequently the bytes Scapy labels as destination/source MAC and EtherType
are protocol data and must never be used as network addresses.  We always
decode :func:`bytes` of the complete captured packet and preserve those bytes
verbatim in the raw binary artifact.

The adapter runs a lightweight internal worker thread so that the
Scapy callback thread never blocks on UUID generation, NumPy
conversion, or the adapter's published raw queue.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter_ns, time_ns
from typing import Any, Callable, Mapping, NamedTuple, Protocol

import numpy as np

from exo_collection.adapters.base import (
    AdapterError,
    ModalityDescriptor,
)
from exo_collection.adapters.hardware_base import QueuedHardwareAdapter
from exo_collection.domain.events import FrameBatch


# ── wire protocol ────────────────────────────────────────────────────

CHANNEL_FROM_WIRE_MARKER: dict[int, int] = {
    0x01: 0,
    0x02: 1,
    0x03: 2,
    0x04: 3,
}
WIRE_FRAME_SIZE = 1000
WIRE_PREFIX = 0x00
WIRE_TRAILER = 0xFF

# Raw artifacts intentionally retain the complete captured 1000-byte record.
US_DEPTH = 1000

_log = logging.getLogger(__name__)


def _load_scapy() -> Any:
    """Import Scapy only inside the hardware process that starts capture."""

    try:
        import scapy.all as scapy
    except ImportError as exc:
        raise AdapterError(
            "未安装 Scapy，无法采集原始以太网超声。请安装 hardware 依赖；"
            "Windows 还必须安装 Npcap（WinPcap API compatible mode）。"
        ) from exc
    return scapy


# ── config ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RawEthernetUltrasoundConfig:
    device_id: str = "ultrasound_raw_ethernet"
    clock_domain: str = "ultrasound_raw_ethernet_clock"
    interface_name: str | None = None
    channels: tuple[int, ...] = (1, 2, 3, 4)
    samples_per_channel: int = 1000
    nominal_rate_hz: float = 20.0
    queue_capacity: int = 64
    inbound_queue_capacity: int = 256
    scan_timeout_s: float = 1.5

    def __post_init__(self) -> None:
        channels = tuple(int(value) for value in self.channels)
        object.__setattr__(self, "channels", channels)
        if not self.device_id.strip() or not self.clock_domain.strip():
            raise ValueError("device_id and clock_domain must not be empty")
        if self.samples_per_channel <= 0 or self.nominal_rate_hz <= 0:
            raise ValueError("sample count and nominal rate must be positive")
        if self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if self.inbound_queue_capacity <= 0:
            raise ValueError("inbound_queue_capacity must be positive")
        if self.scan_timeout_s <= 0:
            raise ValueError("scan_timeout_s must be positive")
        if channels != (1, 2, 3, 4):
            raise ValueError("channels must be exactly (1, 2, 3, 4)")
        if self.samples_per_channel != WIRE_FRAME_SIZE:
            raise ValueError(
                f"raw Ethernet wire frame size must be {WIRE_FRAME_SIZE} bytes"
            )


def _coerce_config(
    value: RawEthernetUltrasoundConfig | Mapping[str, Any] | None,
) -> RawEthernetUltrasoundConfig:
    if value is None:
        return RawEthernetUltrasoundConfig()
    if isinstance(value, RawEthernetUltrasoundConfig):
        return value
    raw = dict(value)
    parameters = raw.pop("parameters", None)
    if isinstance(parameters, Mapping):
        raw.update(parameters)
    if "id" in raw and "device_id" not in raw:
        raw["device_id"] = raw.pop("id")
    allowed = RawEthernetUltrasoundConfig.__dataclass_fields__
    return RawEthernetUltrasoundConfig(
        **{key: item for key, item in raw.items() if key in allowed}
    )


# ── pure packet helpers ──────────────────────────────────────────────────────


class NormalizedPacket(NamedTuple):
    """One classified and timestamped raw Ethernet ultrasound packet."""

    host_monotonic_ns: int
    host_utc_ns: int
    channel: int
    payload: bytes
    has_trailer: bool


class RawEthernetBlockFlags(NamedTuple):
    """Decoded semantic fields from one raw-ultrasound block header."""

    channel: int
    has_trailer: bool


def wire_signature_channel(frame_bytes: bytes) -> int | None:
    """Return the protocol channel without consulting any MAC field."""

    if len(frame_bytes) < 2 or frame_bytes[0] != WIRE_PREFIX:
        return None
    return CHANNEL_FROM_WIRE_MARKER.get(frame_bytes[1])


def decode_ultrasound_wire_frame(
    frame_bytes: bytes,
    *,
    expected_frame_size: int = WIRE_FRAME_SIZE,
) -> tuple[int, bytes] | None:
    """Validate one complete captured record and return channel + raw bytes.

    ``None`` means that the packet does not carry this device's two-byte
    signature.  A matching signature with an invalid size or missing ``FF``
    trailer raises :class:`ValueError`, allowing health telemetry to distinguish
    malformed device frames from unrelated traffic.
    """

    raw_frame = bytes(frame_bytes)
    channel = wire_signature_channel(raw_frame)
    if channel is None:
        return None
    if len(raw_frame) != int(expected_frame_size):
        raise ValueError(
            f"ultrasound frame has {len(raw_frame)} bytes; "
            f"expected {expected_frame_size}"
        )
    if raw_frame[-1] != WIRE_TRAILER:
        raise ValueError("ultrasound frame is missing the terminal 0xFF byte")
    return channel, raw_frame


# ── backend protocol (for testability) ───────────────────────────────────────


class RawEthernetBackend(Protocol):
    """Protocol so tests can supply a fake sniffer."""

    def start(self, on_packet: Callable[[bytes], None]) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class ScapyRawEthernetBackend:
    """Thin owner of Scapy's ``AsyncSniffer`` bound to one interface."""

    def __init__(self, config: RawEthernetUltrasoundConfig) -> None:
        self._config = config
        self._sniffer: Any = None
        self._capture_active = False

    def start(self, on_packet: Callable[[bytes], None]) -> None:
        scapy = _load_scapy()
        iface = self._config.interface_name
        if not iface:
            raise AdapterError("未指定网络接口；请在硬件设置中选择网卡。")
        _log.info("starting Scapy sniffer on %s", iface)

        def handle(packet: Any) -> None:
            try:
                if not packet.haslayer(scapy.Ether):
                    return
                # Do not access Ether.dst/src/type or Raw.load here.  Scapy
                # interprets the first 14 protocol bytes as an Ethernet header,
                # even though this device uses them as part of its data record.
                captured = getattr(packet, "original", None)
                frame_bytes = bytes(captured) if captured else bytes(packet)
                on_packet(frame_bytes)
            except BaseException:
                return

        try:
            sniffer = scapy.AsyncSniffer(
                iface=iface,
                prn=handle,
                store=0,
                promisc=True,
            )
            sniffer.start()
            self._sniffer = sniffer
            self._capture_active = True
        except BaseException as exc:
            raise AdapterError(
                f"启动 Scapy 嗅探器失败 ({iface}): {exc}。"
                "请确认已安装 Npcap，并启用 WinPcap API compatible mode。"
            ) from exc

    def stop(self) -> None:
        if self._sniffer is not None and self._capture_active:
            try:
                self._sniffer.stop()
            except BaseException:
                pass
            self._capture_active = False

    def close(self) -> None:
        try:
            self.stop()
        except BaseException:
            pass
        self._sniffer = None


# ── network interface enumeration ────────────────────────────────────────────


def enumerate_network_interfaces() -> list[dict[str, Any]]:
    """Return a list of candidate Ethernet interfaces for the settings UI.

    Each entry is a small dict with keys ``name``, ``description``, ``mac``.
    The ``name`` is the Npcap / OS-level interface name accepted by
    :func:`scapy.sniff` (e.g. ``\\Device\\NPF_{…}`` on Windows).
    Loopback, Wi‑Fi and virtual/tunnel interfaces are excluded.
    """

    _SKIP_KEYWORDS = (
        "loopback", "wan", "bluetooth", "tailscale", "vethernet",
        "teredo", "6to4", "microsoft", "hyper-v", "virtual",
        "wi-fi", "wireless", "wlan",
    )

    results: list[dict[str, Any]] = []
    try:
        scapy = _load_scapy()
        # Force-load the interface registry on Windows so that Npcap adapters
        # are visible.  On other platforms this is a no-op.
        try:
            _ = scapy.conf.ifaces
        except Exception:
            pass

        for name in scapy.get_if_list():
            iface = scapy.conf.ifaces.get(name)
            if iface is None:
                continue
            desc = iface.description or ""
            lower = (name + " " + desc).lower()
            if any(keyword in lower for keyword in _SKIP_KEYWORDS):
                continue
            results.append(
                {"name": name, "description": desc, "mac": iface.mac or ""}
            )
    except BaseException:
        return []

    _log.debug("枚举到 %d 个候选有线网卡: %s", len(results),
               [r["name"] for r in results])
    return results


def scan_ultrasound_interface(
    interface_name: str,
    *,
    timeout_s: float = 1.5,
    sniff: Callable[..., Any] | None = None,
) -> int:
    """Count valid full-frame ultrasound records on one interface.

    ``sniff`` is injectable so unit tests never touch a real network adapter.
    """

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    scapy = None
    if sniff is None:
        scapy = _load_scapy()
        sniff = scapy.sniff
    total = 0
    count = 0
    sample_channels: list[int] = []

    _log.info("开始扫描网卡 %s（超时 %.1fs）…", interface_name, timeout_s)

    def inspect(packet: Any) -> None:
        nonlocal total, count
        try:
            if scapy is not None:
                if not packet.haslayer(scapy.Ether):
                    return
                total += 1
                captured = getattr(packet, "original", None)
                frame_bytes = bytes(captured) if captured else bytes(packet)
            else:
                frame_bytes = bytes(getattr(packet, "frame_bytes", b""))
                total += 1
            decoded = decode_ultrasound_wire_frame(frame_bytes)
            if decoded is not None:
                channel, _raw_frame = decoded
                count += 1
                if len(sample_channels) < 5:
                    sample_channels.append(channel + 1)
        except (TypeError, ValueError):
            return

    try:
        sniff(
            iface=interface_name,
            prn=inspect,
            store=0,
            timeout=float(timeout_s),
            promisc=True,
        )
    except BaseException as exc:
        _log.error("扫描网卡 %s 失败: %s", interface_name, exc)
        raise AdapterError(
            f"扫描网卡 {interface_name} 失败: {exc}。请确认 Npcap 已安装。"
        ) from exc

    if count > 0:
        _log.info(
            "扫描 %s 完成：共 %d 个捕获帧，%d 个有效超声帧，示例通道: %s",
            interface_name,
            total,
            count,
            ", ".join(str(value) for value in sample_channels),
        )
    else:
        _log.info(
            "扫描 %s 完成：共 %d 个捕获帧，0 个有效超声帧",
            interface_name, total,
        )
    return count


# ── adapter ──────────────────────────────────────────────────────────────────


class RawEthernetUltrasoundAdapter(QueuedHardwareAdapter):
    """Capture one-channel records from the ultrasound Ethernet tap."""

    def __init__(
        self,
        config: RawEthernetUltrasoundConfig | Mapping[str, Any] | None = None,
        *,
        backend: RawEthernetBackend | None = None,
    ) -> None:
        self._config = _coerce_config(config)
        super().__init__(queue_capacity=self._config.queue_capacity)
        self._backend = backend or ScapyRawEthernetBackend(self._config)
        self._inbound_queue: Queue[NormalizedPacket] = Queue(
            maxsize=self._config.inbound_queue_capacity
        )
        self._worker_stop = Event()
        self._worker: Thread | None = None
        self._sequence = 0
        self._frame_index = 0
        self._invalid_length_packets = 0
        self._inbound_queue_overflows = 0
        self._channel_packet_counts = [0, 0, 0, 0]

    # ── descriptor ───────────────────────────────────────────────────────

    def descriptor(self) -> ModalityDescriptor:
        cfg = self._config
        return ModalityDescriptor(
            device_id=cfg.device_id,
            modality="ultrasound",
            display_name="Raw Ethernet four-channel A-mode ultrasound (Scapy + Npcap)",
            clock_domain=cfg.clock_domain,
            event_kind="frame_batch",
            channels=tuple(f"ch_{channel}" for channel in cfg.channels),
            units=("a.u.",) * len(cfg.channels),
            nominal_rate_hz=cfg.nominal_rate_hz,
            sample_shape=(int(cfg.samples_per_channel),),
            dtype=np.dtype(np.uint8).str,
            metadata={
                "simulated": False,
                "manufacturer": "Raw Ethernet (Scapy/Npcap)",
                "geometry": "a_line",
                "channels": list(cfg.channels),
                "samples_per_channel": cfg.samples_per_channel,
                "interface_name": cfg.interface_name,
                "protocol": "raw_ethernet_uint8",
                "transport": "raw_ethernet_scapy_npcap",
                "wire_frame_size": WIRE_FRAME_SIZE,
                "wire_prefix": "00",
                "wire_channel_markers": ["01", "02", "03", "04"],
                "wire_header_bytes": 2,
                "wire_trailer": "FF",
                "wire_adc_byte_count": WIRE_FRAME_SIZE - 3,
                "raw_preservation": "complete captured frame",
                "raw_dtype": "uint8",
                "device_timestamp": "none (network packet timestamps not used)",
            },
        )

    def configuration_snapshot(self) -> Mapping[str, Any]:
        return asdict(self._config)

    # ── lifecycle ────────────────────────────────────────────────────────

    def _connect_hardware(self) -> None:
        # The worker thread is pure Python and does not need Scapy.
        # The real backend will fail in _start_hardware if Scapy is
        # unavailable; fake backends used by tests never hit Scapy.
        self._worker_stop.clear()
        self._worker = Thread(
            target=self._run_worker,
            name=f"raw-eth-{self._config.device_id}",
            daemon=True,
        )
        self._worker.start()

    def _reset_trial_state(self) -> None:
        self._sequence = 0
        self._frame_index = 0
        self._invalid_length_packets = 0
        self._inbound_queue_overflows = 0
        self._channel_packet_counts = [0, 0, 0, 0]
        # Drain any stale packets left from a previous run.
        while not self._inbound_queue.empty():
            try:
                self._inbound_queue.get_nowait()
            except Empty:
                break

    def _start_hardware(self) -> None:
        self._backend.start(on_packet=self._on_ethernet_frame)

    def _stop_hardware(self) -> None:
        # Stop the producer first, then let the conversion worker publish every
        # packet already accepted by the callback.  QueuedHardwareAdapter keeps
        # the public state RUNNING until this method returns, so _publish_raw()
        # remains valid during this bounded drain.
        self._backend.stop()
        self._worker_stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
            if worker.is_alive():
                raise AdapterError(
                    "raw Ethernet ultrasound worker did not drain within 2 seconds"
                )

    def _close_hardware(self) -> None:
        self._worker_stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        self._worker = None
        self._backend.close()

    # ── backend callback (runs on Scapy's internal thread) ───────────────

    def _on_ethernet_frame(self, frame_bytes: bytes) -> None:
        """Validate and enqueue one complete captured frame."""

        try:
            decoded = decode_ultrasound_wire_frame(
                frame_bytes,
                expected_frame_size=self._config.samples_per_channel,
            )
        except ValueError:
            self._invalid_length_packets += 1
            return
        if decoded is None:
            return
        channel, raw_frame = decoded

        host_ns = perf_counter_ns()
        host_utc = time_ns()
        pkt = NormalizedPacket(
            host_monotonic_ns=host_ns,
            host_utc_ns=host_utc,
            channel=channel,
            payload=raw_frame,
            has_trailer=True,
        )
        try:
            self._inbound_queue.put_nowait(pkt)
        except Full:
            self._inbound_queue_overflows += 1
            error = AdapterError(
                f"raw Ethernet inbound queue overflow "
                f"(capacity={self._config.inbound_queue_capacity})"
            )
            self._set_fault(error)

    # ── worker thread (converts NormalizedPacket → FrameBatch) ───────────

    def _run_worker(self) -> None:
        while not self._worker_stop.is_set() or not self._inbound_queue.empty():
            try:
                pkt = self._inbound_queue.get(timeout=0.01)
            except Empty:
                continue
            if self.state.value not in ("running",):
                continue
            try:
                depth = self._config.samples_per_channel
                data = np.frombuffer(pkt.payload, dtype=np.uint8).copy()[None, :]
                if int(data.size) != int(depth):
                    raise AdapterError(
                        f"validated raw frame changed size: {data.size} != {depth}"
                    )
                tail_flags = 1 if pkt.has_trailer else 0
                event = FrameBatch(
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
                    modality="ultrasound",
                    clock_domain=self._config.clock_domain,
                    host_monotonic_ns=pkt.host_monotonic_ns,
                    host_utc_ns=pkt.host_utc_ns,
                    first_frame_index=self._frame_index,
                    frame_count=1,
                    sequence_number=self._sequence,
                    device_timestamp=None,
                    frame_rate_hz=self._config.nominal_rate_hz,
                    channel=pkt.channel,
                    tail_flags=tail_flags,
                    data=data,
                )
                self._publish_raw(
                    event,
                    item_count=1,
                    host_monotonic_ns=pkt.host_monotonic_ns,
                )
                self._frame_index += 1
                self._sequence += 1
                self._channel_packet_counts[pkt.channel] += 1
            except BaseException as exc:
                self._set_fault(exc)

    def _compute_actual_rate(self, elapsed: float) -> float:
        """Per-A-line frame rate (not per-channel event rate).

        Each 4-channel A-line arrives as four separate Ethernet frames.
        Dividing by the channel count gives the true ultrasound frame rate.
        """
        base = super()._compute_actual_rate(elapsed)
        channel_count = len(self._config.channels)
        return base / channel_count if channel_count else base

    def _health_metrics(self) -> dict[str, int | float | str | bool | None]:
        return {
            "invalid_length_packets": self._invalid_length_packets,
            "inbound_queue_overflows": self._inbound_queue_overflows,
            "interface_name": self._config.interface_name,
            "channel_1_packets": self._channel_packet_counts[0],
            "channel_2_packets": self._channel_packet_counts[1],
            "channel_3_packets": self._channel_packet_counts[2],
            "channel_4_packets": self._channel_packet_counts[3],
        }


# ── helpers used by test code ────────────────────────────────────────────────


def encode_raw_ethernet_flags(channel: int | None, tail_flags: int) -> int:
    """Encode channel and trailer flag into a 32-bit block-header flags word.

    bits 0-1 = channel (0-3)
    bit  2   = has_trailer (1 if present)
    bits 3-31 = reserved (0)

    When *channel* is ``None`` (Elonxi / simulated adapter events) this
    returns 0 so existing blocks are unchanged.
    """

    if channel is None:
        return 0
    if int(channel) not in range(4):
        raise ValueError("raw Ethernet ultrasound channel must be in [0, 3]")
    if int(tail_flags) not in (0, 1):
        raise ValueError("raw Ethernet ultrasound tail_flags must be 0 or 1")
    result = int(channel) | (int(tail_flags) << 2)
    return int(result)


def decode_raw_ethernet_flags(flags: int) -> RawEthernetBlockFlags:
    """Decode the channel and optional-trailer bits written by the adapter.

    Bits 3-31 are reserved by format version 1.  Rejecting non-zero reserved
    bits prevents a future packet format from being silently misinterpreted
    as the current four-channel protocol during offline playback.
    """

    if isinstance(flags, bool) or not isinstance(flags, (int, np.integer)):
        raise TypeError("raw Ethernet ultrasound flags must be an integer")
    value = int(flags)
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("raw Ethernet ultrasound flags must be uint32")
    if value & ~0b111:
        raise ValueError("raw Ethernet ultrasound flags contain reserved bits")
    return RawEthernetBlockFlags(
        channel=value & 0b11,
        has_trailer=bool(value & 0b100),
    )


__all__ = [
    "CHANNEL_FROM_WIRE_MARKER",
    "US_DEPTH",
    "WIRE_FRAME_SIZE",
    "WIRE_PREFIX",
    "WIRE_TRAILER",
    "NormalizedPacket",
    "RawEthernetBackend",
    "RawEthernetBlockFlags",
    "RawEthernetUltrasoundAdapter",
    "RawEthernetUltrasoundConfig",
    "ScapyRawEthernetBackend",
    "decode_ultrasound_wire_frame",
    "decode_raw_ethernet_flags",
    "encode_raw_ethernet_flags",
    "enumerate_network_interfaces",
    "scan_ultrasound_interface",
    "wire_signature_channel",
    "_load_scapy",
]
