"""Raw Ethernet ultrasound adapter via Scapy + Npcap.

Captures raw Ethernet frames whose destination MAC byte-1 identifies the
ultrasound channel (0x01/0x02/0x03/0x04 → channel 0/1/2/3).  Each frame
carries approximately 1000 uint8 ADC samples; a trailing 0xFF byte is
detected and recorded as a flag so a reader can reconstruct the exact
original payload.

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


# ── channel mapping (from exo_capture.py, validated first system) ────────────

# Destination MAC byte at offset 1 identifies the channel.
# Examples:  01:xx:xx:xx:xx:xx → channel 0, 02:... → channel 1, etc.
CH_FROM_DST_MAC_BYTE1: dict[int, int] = {0x01: 0, 0x02: 1, 0x03: 2, 0x04: 3}

# Number of meaningful ADC samples per channel per frame.  The frame payload
# is either exactly this size or one byte longer when the trailing 0xFF marker
# is present.
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


def mac_to_channel(dst_mac: str | bytes) -> int | None:
    """Extract the channel index (0-3) from a destination MAC address.

    The destination MAC must start with ``00:`` (unicast-to-self) and the
    second byte must be 0x01-0x04.  Returns ``None`` for broadcast,
    multicast, non-zero first byte, or unknown channel markers.
    """

    if isinstance(dst_mac, str):
        parts = dst_mac.split(":")
        if len(parts) != 6:
            return None
        try:
            byte0 = int(parts[0], 16)
            byte1 = int(parts[1], 16)
        except ValueError:
            return None
    else:
        if len(dst_mac) < 2:
            return None
        byte0 = int(dst_mac[0])
        byte1 = int(dst_mac[1])
    if byte0 != 0x00:
        return None
    return CH_FROM_DST_MAC_BYTE1.get(byte1)


def classify_raw_payload(
    raw_bytes: bytes,
    expected_adc_count: int = US_DEPTH,
) -> tuple[bytes, bool]:
    """Split raw payload into ADC bytes + trailer flag using length rules.

    - ``len(raw_bytes) == expected_adc_count``:  all ADC, ``has_trailer=False``
      (even if the final ADC value happens to be 0xFF).
    - ``len(raw_bytes) == expected_adc_count + 1`` and final byte is 0xFF:
      first *expected_adc_count* bytes are ADC, ``has_trailer=True``.
    - Any other length **or** 1001 bytes whose final byte is NOT 0xFF:
      invalid packet → return ``(b"", False)`` so the caller can count the
      error and raise a fault.
    """

    length = len(raw_bytes)
    if length == expected_adc_count:
        return raw_bytes, False
    if length == expected_adc_count + 1 and raw_bytes[-1] == 0xFF:
        return raw_bytes[:-1], True
    # Invalid length: neither exact-nor-trailer.
    return b"", False


# ── backend protocol (for testability) ───────────────────────────────────────


class RawEthernetBackend(Protocol):
    """Protocol so tests can supply a fake sniffer."""

    def start(
        self, on_packet: Callable[[str | bytes, bytes], None]
    ) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class ScapyRawEthernetBackend:
    """Thin owner of Scapy's ``AsyncSniffer`` bound to one interface."""

    def __init__(self, config: RawEthernetUltrasoundConfig) -> None:
        self._config = config
        self._sniffer: Any = None
        self._capture_active = False

    def start(
        self, on_packet: Callable[[str | bytes, bytes], None]
    ) -> None:
        scapy = _load_scapy()
        iface = self._config.interface_name
        if not iface:
            raise AdapterError("未指定网络接口；请在硬件设置中选择网卡。")
        _log.info("starting Scapy sniffer on %s", iface)

        def handle(packet: Any) -> None:
            try:
                if not (
                    packet.haslayer(scapy.Ether)
                    and packet.haslayer(scapy.Raw)
                ):
                    return
                on_packet(
                    packet[scapy.Ether].dst,
                    bytes(packet[scapy.Raw].load),
                )
            except BaseException:
                return

        try:
            sniffer = scapy.AsyncSniffer(
                iface=iface,
                prn=handle,
                store=0,
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
    Loopback and virtual/tunnel interfaces are excluded.
    """

    import platform

    results: list[dict[str, Any]] = []
    try:
        scapy = _load_scapy()
        if platform.system() == "Windows":
            import scapy.arch.windows as _scapy_win

            raw = _scapy_win.get_windows_if_list()
        else:
            raw = []
            for name in scapy.get_if_list():
                iface = scapy.IFACES.get(name)
                if iface is not None:
                    raw.append(
                        {
                            "name": iface.name,
                            "description": iface.description,
                            "mac": iface.mac,
                        }
                    )
    except BaseException:
        return []

    for entry in raw:
        name = str(entry.get("name", ""))
        desc = str(entry.get("description", ""))
        mac = str(entry.get("mac", ""))
        if not name:
            continue
        lower = (name + " " + desc).lower()
        if any(
            keyword in lower
            for keyword in (
                "loopback", "wan", "bluetooth", "tailscale", "vethernet",
                "teredo", "6to4", "microsoft", "hyper-v", "virtual",
                "wi-fi", "wireless", "wlan",
            )
        ):
            continue
        results.append({"name": name, "description": desc, "mac": mac})
    return results


def scan_ultrasound_interface(
    interface_name: str,
    *,
    timeout_s: float = 1.5,
    sniff: Callable[..., Any] | None = None,
) -> int:
    """Count channel-tagged ultrasound packets on one interface.

    ``sniff`` is injectable so unit tests never touch a real network adapter.
    """

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    scapy = None
    if sniff is None:
        scapy = _load_scapy()
        sniff = scapy.sniff
    count = 0

    def inspect(packet: Any) -> None:
        nonlocal count
        try:
            if scapy is not None:
                if not (
                    packet.haslayer(scapy.Ether)
                    and packet.haslayer(scapy.Raw)
                ):
                    return
                dst = packet[scapy.Ether].dst
                payload = bytes(packet[scapy.Raw].load)
            else:
                dst = getattr(packet, "dst", None)
                payload = bytes(getattr(packet, "payload", b""))
            adc_bytes, _has_trailer = classify_raw_payload(payload)
            if (
                dst is not None
                and mac_to_channel(dst) is not None
                and bool(adc_bytes)
            ):
                count += 1
        except BaseException:
            return

    try:
        sniff(
            iface=interface_name,
            prn=inspect,
            store=0,
            timeout=float(timeout_s),
        )
    except BaseException as exc:
        raise AdapterError(
            f"扫描网卡 {interface_name} 失败: {exc}。请确认 Npcap 已安装。"
        ) from exc
    return count


# ── adapter ──────────────────────────────────────────────────────────────────


class RawEthernetUltrasoundAdapter(QueuedHardwareAdapter):
    """Capture one-channel raw ultrasound frames from a raw Ethernet tap."""

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
                "destination_mac_prefixes": ["00:01", "00:02", "00:03", "00:04"],
                "trailer_policy": "optional 0xFF only at payload byte 1001",
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
        self._backend.start(on_packet=self._on_ethernet_payload)

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

    def _on_ethernet_payload(
        self, dst_mac: str | bytes, raw_bytes: bytes
    ) -> None:
        """Validate and enqueue one packet without doing NumPy work."""

        channel = mac_to_channel(dst_mac)
        if channel is None:
            return
        depth = self._config.samples_per_channel
        adc_bytes, has_trailer = classify_raw_payload(
            bytes(raw_bytes), expected_adc_count=depth
        )
        if not adc_bytes:
            self._invalid_length_packets += 1
            self._set_fault(
                AdapterError(
                    f"invalid raw Ethernet payload length {len(raw_bytes)} "
                    f"(expected {depth} ADC bytes, or {depth + 1} ending in 0xFF)"
                )
            )
            return

        host_ns = perf_counter_ns()
        host_utc = time_ns()
        pkt = NormalizedPacket(
            host_monotonic_ns=host_ns,
            host_utc_ns=host_utc,
            channel=channel,
            payload=adc_bytes,
            has_trailer=has_trailer,
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
                pkt = self._inbound_queue.get(timeout=0.1)
            except Empty:
                continue
            if self.state.value not in ("running",):
                continue
            try:
                depth = self._config.samples_per_channel
                data = np.frombuffer(pkt.payload, dtype=np.uint8).copy()[None, :]
                # Ensure data is exactly one frame of <depth> samples.
                actual_size = int(data.size)
                if actual_size != int(depth):
                    self._set_fault(
                        AdapterError(
                            f"worker received frame with {actual_size} samples, "
                            f"expected {depth}"
                        )
                    )
                    continue
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
    "CH_FROM_DST_MAC_BYTE1",
    "US_DEPTH",
    "NormalizedPacket",
    "RawEthernetBackend",
    "RawEthernetBlockFlags",
    "RawEthernetUltrasoundAdapter",
    "RawEthernetUltrasoundConfig",
    "ScapyRawEthernetBackend",
    "classify_raw_payload",
    "decode_raw_ethernet_flags",
    "encode_raw_ethernet_flags",
    "enumerate_network_interfaces",
    "mac_to_channel",
    "scan_ultrasound_interface",
    "_load_scapy",
]
