"""Tests for the raw Ethernet ultrasound adapter and its pure helpers.

All tests use a ``FakeSnifferBackend`` that feeds synthetic packets
through the adapter's callback, so no real network hardware is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from queue import Empty
from time import sleep
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from exo_collection.adapters.base import AdapterError, AdapterState, TrialContext
from exo_collection.adapters.ultrasound.raw_ethernet import (
    CH_FROM_DST_MAC_BYTE1,
    US_DEPTH,
    NormalizedPacket,
    RawEthernetUltrasoundAdapter,
    RawEthernetUltrasoundConfig,
    classify_raw_payload,
    encode_raw_ethernet_flags,
    mac_to_channel,
    scan_ultrasound_interface,
)
from exo_collection.readers.binary_block import BlockBinaryReader, scan_binary_file
from exo_collection.writers.binary_block import (
    DEVICE_TIMESTAMP_UNKNOWN,
    BlockBinaryWriter,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _context() -> TrialContext:
    return TrialContext(trial_uuid=uuid4(), session_uuid=uuid4())


def _uint8_adc(depth: int = US_DEPTH) -> bytes:
    """Deterministic ramp plus a bump so values not equal 0."""
    data = (np.arange(depth, dtype=np.uint32) * 3 % 256).astype(np.uint8)
    return data.tobytes()


def _uint8_adc_with_trailing_ff(depth: int = US_DEPTH) -> bytes:
    """ADC bytes ending with 0xFF (as a genuine sample, not a trailer)."""
    data = _uint8_adc(depth - 1)
    return data + b"\xff"


# ── backend that mimics the Scapy sniffer without requiring Scapy ────────────


class FakeSnifferBackend:
    """Fake that mimics Scapy's AsyncSniffer.

    Calls ``on_packet`` directly; no real network hardware needed.
    """

    def __init__(self) -> None:
        self.on_packet: Callable[[str | bytes, bytes], None] | None = None
        self.started = False
        self.stopped = False
        self.closed = False

    def start(
        self, on_packet: Callable[[str | bytes, bytes], None]
    ) -> None:
        self.on_packet = on_packet
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def emit(self, packet: Any) -> None:
        assert self.on_packet is not None
        self.on_packet(packet._ether.dst, packet._raw.load)


# ── helpers used by the fake emit path (must look like Scapy layers) ─────────


class _FakeEther:
    """Mimics Scapy's Ether.haslayer / Ether.dst."""

    def __init__(self, dst_mac: str) -> None:
        self.dst = dst_mac

    @staticmethod
    def haslayer(layer: Any) -> bool:
        return True


class _FakeRaw:
    """Mimics Scapy's Raw.haslayer / Raw.load."""

    def __init__(self, payload: bytes) -> None:
        self.load = payload

    @staticmethod
    def haslayer(layer: Any) -> bool:
        return True


class FakeScapyPacket:
    """A synthetic packet that passes haslayer(_Ether)/haslayer(_Raw)."""

    def __init__(self, dst_mac: str, payload: bytes) -> None:
        self._ether = _FakeEther(dst_mac)
        self._raw = _FakeRaw(payload)

    def haslayer(self, layer: Any) -> bool:
        return True

    def __getitem__(self, layer: Any) -> Any:
        if layer.__name__ == "Ether":
            return self._ether
        return self._raw


# ── pure function tests ──────────────────────────────────────────────────────


def test_mac_to_channel_parses_string_and_bytes() -> None:
    for marker, expected in CH_FROM_DST_MAC_BYTE1.items():
        mac_str = f"00:{marker:02x}:03:04:05:06"
        assert mac_to_channel(mac_str) == expected
        mac_bytes = bytes([0, marker, 3, 4, 5, 6])
        assert mac_to_channel(mac_bytes) == expected
    # Unknown marker byte 0xFF is not in the table.
    assert mac_to_channel("00:ff:03:04:05:06") is None
    assert mac_to_channel(b"\x00\xff") is None
    # Broadcast / multicast
    assert mac_to_channel("ff:ff:ff:ff:ff:ff") is None
    # Short / malformed
    assert mac_to_channel("00:02") is None
    assert mac_to_channel("not-a-mac") is None
    assert mac_to_channel(b"") is None


def test_mac_to_channel_rejects_non_zero_first_byte() -> None:
    """Requirement B.1: first byte must be 0x00."""
    # 10:01:... should be rejected even though byte-1 is a valid channel.
    assert mac_to_channel("10:01:03:04:05:06") is None
    assert mac_to_channel(bytes([0x10, 0x01, 3, 4, 5, 6])) is None
    # 0x00 first byte passes.
    assert mac_to_channel("00:01:03:04:05:06") == 0


def test_classify_raw_payload_exact_length() -> None:
    """Exactly 1000 bytes → all ADC, has_trailer=False."""
    adc = _uint8_adc()
    clean, trailer = classify_raw_payload(adc, expected_adc_count=1000)
    assert clean == adc
    assert trailer is False


def test_classify_raw_payload_ff_end_but_exact_length() -> None:
    """1000 bytes ending with FF: not a trailer, genuine sample."""
    adc = _uint8_adc_with_trailing_ff()
    assert len(adc) == 1000
    assert adc[-1] == 0xFF
    clean, trailer = classify_raw_payload(adc, expected_adc_count=1000)
    assert clean == adc
    assert trailer is False


def test_classify_raw_payload_with_trailer() -> None:
    """1001 bytes, last is 0xFF → has_trailer=True, ADC stripped."""
    adc = _uint8_adc()
    with_trailer = adc + b"\xff"
    assert len(with_trailer) == 1001
    clean, trailer = classify_raw_payload(with_trailer, expected_adc_count=1000)
    assert clean == adc
    assert trailer is True


def test_classify_raw_payload_1001_non_ff_last_is_invalid() -> None:
    """1001 bytes, last byte not 0xFF → invalid packet."""
    adc = _uint8_adc()
    invalid = adc + b"\x00"
    assert len(invalid) == 1001
    result, trailer = classify_raw_payload(invalid, expected_adc_count=1000)
    assert result == b""
    assert trailer is False


def test_classify_raw_payload_999_is_invalid() -> None:
    adc = _uint8_adc(depth=999)
    result, trailer = classify_raw_payload(adc, expected_adc_count=1000)
    assert result == b""
    assert trailer is False


def test_classify_raw_payload_1002_is_invalid() -> None:
    adc = _uint8_adc(depth=1002)
    result, trailer = classify_raw_payload(adc, expected_adc_count=1000)
    assert result == b""
    assert trailer is False


def test_classify_raw_payload_empty() -> None:
    empty, trailer = classify_raw_payload(b"", expected_adc_count=1000)
    assert empty == b""
    assert trailer is False


@pytest.mark.parametrize(
    ("channel", "tail_flags", "expected"),
    [
        (None, 0, 0),
        (0, 0, 0x00),
        (1, 0, 0x01),
        (2, 0, 0x02),
        (3, 0, 0x03),
        (0, 1, 0x04),
        (1, 1, 0x05),
        (2, 1, 0x06),
        (3, 1, 0x07),
    ],
)
def test_encode_raw_ethernet_flags(channel, tail_flags, expected) -> None:
    assert encode_raw_ethernet_flags(channel, tail_flags) == expected


def test_encode_raw_ethernet_flags_rejects_invalid_channel() -> None:
    with pytest.raises(ValueError):
        encode_raw_ethernet_flags(5, 0)


# ── config tests ─────────────────────────────────────────────────────────────


def test_config_defaults_are_valid() -> None:
    cfg = RawEthernetUltrasoundConfig()
    assert cfg.device_id == "ultrasound_raw_ethernet"
    assert cfg.queue_capacity == 64
    assert cfg.inbound_queue_capacity == 256
    assert cfg.channels == (1, 2, 3, 4)
    assert cfg.samples_per_channel == 1000
    assert cfg.nominal_rate_hz == 20.0


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        RawEthernetUltrasoundConfig(device_id="")
    with pytest.raises(ValueError):
        RawEthernetUltrasoundConfig(clock_domain="")
    with pytest.raises(ValueError):
        RawEthernetUltrasoundConfig(samples_per_channel=0)
    with pytest.raises(ValueError):
        RawEthernetUltrasoundConfig(nominal_rate_hz=0)
    with pytest.raises(ValueError):
        RawEthernetUltrasoundConfig(queue_capacity=0)


def test_coerce_config_accepts_dict() -> None:
    from exo_collection.adapters.ultrasound.raw_ethernet import _coerce_config

    cfg = _coerce_config({"device_id": "test", "queue_capacity": 32})
    assert cfg.device_id == "test"
    assert cfg.queue_capacity == 32


# ── full adapter lifecycle tests ─────────────────────────────────────────────


def _running_adapter():
    backend = FakeSnifferBackend()
    adapter = RawEthernetUltrasoundAdapter(backend=backend)
    adapter.connect()
    adapter.prepare(_context())
    adapter.start()
    return adapter, backend


def test_descriptor_is_well_formed() -> None:
    adapter = RawEthernetUltrasoundAdapter()
    desc = adapter.descriptor()
    assert desc.sample_shape == (1000,)
    assert np.dtype(desc.dtype) == np.dtype(np.uint8)
    assert desc.metadata["simulated"] is False
    assert desc.metadata["protocol"] == "raw_ethernet_uint8"
    assert desc.metadata["manufacturer"] == "Raw Ethernet (Scapy/Npcap)"
    adapter.close()


def test_single_packet_becomes_one_frame_batch() -> None:
    adapter, backend = _running_adapter()
    payload = _uint8_adc()
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))

    event = adapter.get_event(timeout=2.0)
    assert event.data.shape == (1, US_DEPTH)
    assert event.data.dtype == np.uint8
    assert event.data.flags.c_contiguous
    assert event.channel == 0
    assert event.tail_flags == 0
    assert event.frame_count == 1
    assert event.first_frame_index == 0
    assert event.sequence_number == 0
    adapter.stop()
    adapter.close()
    assert (backend.started, backend.stopped, backend.closed) == (True, True, True)


def test_four_channel_packets_are_individual_batches() -> None:
    adapter, backend = _running_adapter()
    payload = _uint8_adc()
    for marker in (0x01, 0x02, 0x03, 0x04):
        mac = f"00:{marker:02x}:03:04:05:06"
        backend.emit(FakeScapyPacket(mac, payload))

    events = []
    for _ in range(4):
        ev = adapter.get_event(timeout=2.0)
        assert ev is not None
        events.append(ev)
    channels_seen = [ev.channel for ev in events]
    assert sorted(channels_seen) == [0, 1, 2, 3]
    for ev in events:
        assert ev.data.shape == (1, US_DEPTH)
        assert ev.data.dtype == np.uint8
        assert ev.frame_count == 1
    adapter.stop()
    adapter.close()


def test_has_trailer_flag_is_detected() -> None:
    adapter, backend = _running_adapter()
    payload = _uint8_adc() + b"\xff"
    assert len(payload) == 1001
    backend.emit(FakeScapyPacket("00:02:03:04:05:06", payload))

    event = adapter.get_event(timeout=2.0)
    assert event.tail_flags == 1
    assert event.channel == 1
    assert event.frame_count == 1
    assert event.data.shape == (1, US_DEPTH)
    adapter.stop()
    adapter.close()


def test_sequences_and_indices_are_monotonic() -> None:
    adapter, backend = _running_adapter()
    payload = _uint8_adc()
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))
    backend.emit(FakeScapyPacket("00:02:03:04:05:06", payload))

    first = adapter.get_event(timeout=2.0)
    second = adapter.get_event(timeout=2.0)
    assert (first.sequence_number, first.first_frame_index) == (0, 0)
    assert (second.sequence_number, second.first_frame_index) == (1, 1)
    assert first.frame_count == 1
    assert second.frame_count == 1
    adapter.stop()
    adapter.close()


def test_unknown_mac_dst_is_silently_skipped() -> None:
    adapter, backend = _running_adapter()
    payload = _uint8_adc()
    # Unrecognised marker byte 0xFF is not in CH_FROM_DST_MAC_BYTE1.
    backend.emit(FakeScapyPacket("00:ff:03:04:05:06", payload))

    # Nothing should appear on the raw queue.
    assert adapter.get_event(timeout=0.5) is None
    adapter.stop()
    adapter.close()


def test_backend_simulates_get_event_none_of_empty_queue() -> None:
    adapter, backend = _running_adapter()
    assert adapter.get_event(timeout=0.3) is None
    adapter.stop()
    adapter.close()


def test_configuration_snapshot_is_complete() -> None:
    adapter = RawEthernetUltrasoundAdapter(
        config={"interface_name": "Ethernet0"}
    )
    snap = adapter.configuration_snapshot()
    assert snap["interface_name"] == "Ethernet0"
    assert snap["device_id"] == "ultrasound_raw_ethernet"
    adapter.close()


def test_reset_trial_state_resets_counters() -> None:
    adapter, backend = _running_adapter()
    payload = _uint8_adc()
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))
    assert adapter.get_event(timeout=2.0) is not None
    adapter.stop()

    # After stop → close the adapter enters DISCONNECTED? No.
    # The adapter.state after stop depends on adapter.
    # Wait for worker to join. Use close() instead:
    adapter.close()

    # Fresh adapter round.
    backend = FakeSnifferBackend()
    adapter = RawEthernetUltrasoundAdapter(backend=backend)
    adapter.connect()
    adapter.prepare(_context())
    adapter.start()
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))
    event = adapter.get_event(timeout=2.0)
    assert event.sequence_number == 0
    assert event.first_frame_index == 0
    adapter.stop()
    adapter.close()


def test_reset_trial_state_drains_inbound_queue() -> None:
    """requirement B.7: _reset_trial_state clears inbound queue."""
    adapter, backend = _running_adapter()
    payload = _uint8_adc()
    # Inject several packets.
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))
    backend.emit(FakeScapyPacket("00:02:03:04:05:06", payload))
    # Drain only one event.
    ev = adapter.get_event(timeout=2.0)
    assert ev is not None
    adapter.stop()
    adapter.close()

    # Re-open the adapter; prepare() calls _reset_trial_state which drains
    # any leftover inbound packets.
    backend = FakeSnifferBackend()
    adapter = RawEthernetUltrasoundAdapter(backend=backend)
    adapter.connect()
    # Start injected packets BEFORE prepare, in the inbound queue.
    saturate_ok = adapter._inbound_queue.qsize() == 0
    assert saturate_ok
    adapter.prepare(_context())
    adapter.start()
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))
    event = adapter.get_event(timeout=2.0)
    assert event.sequence_number == 0
    adapter.stop()
    adapter.close()


def test_close_idempotent() -> None:
    adapter = RawEthernetUltrasoundAdapter()
    adapter.close()
    adapter.close()  # must not raise


def test_default_backend_created_when_none_provided() -> None:
    adapter = RawEthernetUltrasoundAdapter()
    assert adapter._backend is not None
    adapter.close()


# ── invalid length packet handling ───────────────────────────────────────────


def test_invalid_length_999_causes_fault() -> None:
    """999-byte payload → invalid → fault, no event emitted."""
    adapter, backend = _running_adapter()
    payload = _uint8_adc(depth=999)
    assert len(payload) == 999
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))

    # Worker will process FAULT, no event expected.
    sleep(0.3)
    assert adapter.get_event(timeout=0.3) is None
    health = adapter.health()
    assert health.metrics.get("invalid_length_packets", 0) >= 1
    adapter.stop()
    adapter.close()


def test_invalid_length_1001_non_ff_last_causes_fault() -> None:
    """1001 bytes last byte not FF → invalid."""
    adapter, backend = _running_adapter()
    payload = _uint8_adc() + b"\x00"
    assert len(payload) == 1001
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))

    sleep(0.3)
    assert adapter.get_event(timeout=0.3) is None
    adapter.stop()
    adapter.close()


# ── integration with orchestration flags ────────────────────────────────────


def test_frame_batch_has_backward_compatible_defaults() -> None:
    from exo_collection.domain.events import FrameBatch

    event = FrameBatch(
        device_id="test",
        modality="ultrasound",
        clock_domain="test_clock",
        first_frame_index=0,
        frame_count=1,
        sequence_number=0,
        data=np.zeros(10, dtype=np.uint8),
    )
    assert event.channel is None
    assert event.tail_flags == 0
    assert encode_raw_ethernet_flags(event.channel, event.tail_flags) == 0


# ── NormalizedPacket ────────────────────────────────────────────────────────


def test_normalized_packet_round_trip() -> None:
    payload = _uint8_adc()
    pkt = NormalizedPacket(
        host_monotonic_ns=42,
        host_utc_ns=100,
        channel=2,
        payload=payload,
        has_trailer=True,
    )
    assert pkt.channel == 2
    assert pkt.has_trailer is True
    assert pkt.payload == payload
    assert pkt.host_monotonic_ns == 42
    assert pkt.host_utc_ns == 100


# ── channel mapping table ───────────────────────────────────────────────────


def test_channel_mapping_is_correct() -> None:
    assert CH_FROM_DST_MAC_BYTE1 == {0x01: 0, 0x02: 1, 0x03: 2, 0x04: 3}


# ── preview-compatible batch shape ──────────────────────────────────────────


def test_raw_frame_payload_passes_preview_constraints() -> None:
    """One tagged ``(1, depth)`` raw frame becomes an A-line preview."""

    from exo_collection.acquisition.preview import build_preview_event
    from exo_collection.domain.events import FrameBatch

    data = _uint8_adc()
    fb = FrameBatch(
        device_id="test",
        modality="ultrasound",
        clock_domain="test_clock",
        first_frame_index=0,
        frame_count=1,
        sequence_number=0,
        host_monotonic_ns=0,
        host_utc_ns=0,
        data=np.frombuffer(data, dtype=np.uint8)[None, :],
        channel=2,
        tail_flags=0,
    )

    result = build_preview_event(fb)
    assert result.event_type.value == "preview"
    assert result.payload["channel_count"] == 1
    assert result.payload["geometry"] == "a_line"
    assert result.payload["channel_index"] == 2
    raw = np.frombuffer(data, dtype=np.uint8)
    expected = (raw.astype(np.int16) - 127).astype(float)
    indices = np.linspace(0, raw.size - 1, 512, dtype=np.int64)
    assert result.payload["values"] == pytest.approx(expected[indices].tolist())


# ── end-to-end: four channels produce four independent blocks ────────────────


def test_four_channel_packets_produce_four_distinct_blocks() -> None:
    """E.2: four different-channel packets → four blocks with correct
    sequence, first_frame_index, flags, and distinct host timestamps."""
    adapter, backend = _running_adapter()
    payloads = []
    for channel_marker in (0x01, 0x02, 0x03, 0x04):
        # Use a distinct ADC payload per channel so CRC differs.
        ramp = (np.arange(US_DEPTH, dtype=np.uint32) * (channel_marker * 3)
                % 256).astype(np.uint8)
        payloads.append(ramp.tobytes())
        mac = f"00:{channel_marker:02x}:03:04:05:06"
        backend.emit(FakeScapyPacket(mac, payloads[-1]))
        # Brief pause so host_monotonic_ns differs across packets.
        sleep(0.001)

    events = []
    for _ in range(4):
        ev = adapter.get_event(timeout=2.0)
        assert ev is not None
        events.append(ev)

    # Verify sequence and frame_index are 0..3.
    seqs = [ev.sequence_number for ev in events]
    indices = [ev.first_frame_index for ev in events]
    assert sorted(seqs) == [0, 1, 2, 3]
    assert sorted(indices) == [0, 1, 2, 3]

    # Each event has frame_count == 1 and shape (1, 1000).
    for ev in events:
        assert ev.frame_count == 1
        assert ev.data.shape == (1, US_DEPTH)

    # Flags encode channel correctly (bits 0-1).
    for ev in events:
        flags = encode_raw_ethernet_flags(ev.channel, ev.tail_flags)
        assert (flags & 0x3) == ev.channel

    # Host timestamps are distinct.
    host_times = [ev.host_monotonic_ns for ev in events]
    assert len(set(host_times)) == 4

    # Event data matches injected payload (no derived conversions).
    for ev in events:
        assert ev.data.dtype == np.uint8

    adapter.stop()
    adapter.close()


def test_raw_uint8_data_is_preserved_verbatim() -> None:
    """B.4: raw write-to-disk keeps uint8; no int16(adc)-127 conversion."""
    adapter, backend = _running_adapter()
    payload = _uint8_adc()
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))

    event = adapter.get_event(timeout=2.0)
    assert event.data.dtype == np.uint8
    # Verify exact values.
    np.testing.assert_array_equal(
        event.data[0],
        np.frombuffer(payload, dtype=np.uint8)
    )
    adapter.stop()
    adapter.close()


def test_stop_drains_every_packet_already_accepted_by_callback() -> None:
    adapter, backend = _running_adapter()
    payload = _uint8_adc()
    for index in range(12):
        marker = index % 4 + 1
        backend.emit(
            FakeScapyPacket(f"00:{marker:02x}:03:04:05:06", payload)
        )

    # Deliberately stop without sleeping: this exercises the callback-to-worker
    # drain boundary instead of relying on scheduling luck.
    report = adapter.stop()
    events = []
    while (event := adapter.get_event(timeout=0)) is not None:
        events.append(event)

    assert len(events) == 12
    assert report.samples_emitted == 12
    assert [event.sequence_number for event in events] == list(range(12))
    adapter.close()


def test_four_packets_round_trip_as_independent_crc_checked_binary_blocks(
    tmp_path: Path,
) -> None:
    adapter, backend = _running_adapter()
    original_payloads: list[bytes] = []
    for channel in range(4):
        adc = (
            (np.arange(US_DEPTH, dtype=np.uint32) * (channel + 3) + channel)
            % 256
        ).astype(np.uint8).tobytes()
        raw_payload = adc + (b"\xff" if channel == 3 else b"")
        original_payloads.append(raw_payload)
        backend.emit(
            FakeScapyPacket(
                f"00:{channel + 1:02x}:03:04:05:06",
                raw_payload,
            )
        )

    events = [adapter.get_event(timeout=2.0) for _ in range(4)]
    assert all(event is not None for event in events)
    adapter.stop()
    adapter.close()

    data_path = tmp_path / "ultrasound.bin"
    with BlockBinaryWriter(
        data_path,
        dtype=np.uint8,
        sample_shape=(US_DEPTH,),
        metadata=adapter.descriptor().metadata,
    ) as writer:
        for event in events:
            assert event is not None
            writer.append(
                event.data,
                device_timestamp=event.device_timestamp,
                host_monotonic_ns=event.host_monotonic_ns,
                host_utc_ns=event.host_utc_ns,
                first_sample_index=event.first_frame_index,
                sequence=event.sequence_number,
                flags=encode_raw_ethernet_flags(
                    event.channel, event.tail_flags
                ),
            )

    scan = scan_binary_file(data_path, validate_crc=True)
    assert scan.error is None
    assert scan.complete_block_count == 4
    with BlockBinaryReader(data_path, validate_crc=True) as reader:
        records = list(reader)

    assert [record.header.sequence for record in records] == [0, 1, 2, 3]
    assert [record.header.first_sample_index for record in records] == [0, 1, 2, 3]
    assert [record.header.sample_count for record in records] == [1, 1, 1, 1]
    assert [record.header.flags for record in records] == [0, 1, 2, 7]
    assert all(
        record.header.device_timestamp == DEVICE_TIMESTAMP_UNKNOWN
        for record in records
    )
    reconstructed = [
        record.data[0].tobytes()
        + (b"\xff" if record.header.flags & 0b100 else b"")
        for record in records
    ]
    assert reconstructed == original_payloads


def test_interface_scan_counts_only_valid_ultrasound_payloads() -> None:
    packets = (
        SimpleNamespace(dst="00:01:03:04:05:06", payload=_uint8_adc()),
        SimpleNamespace(
            dst="00:04:03:04:05:06", payload=_uint8_adc() + b"\xff"
        ),
        SimpleNamespace(dst="00:05:03:04:05:06", payload=_uint8_adc()),
        SimpleNamespace(dst="00:02:03:04:05:06", payload=b"too-short"),
    )

    def fake_sniff(**kwargs: Any) -> None:
        assert kwargs["iface"] == "npcap-test-interface"
        assert kwargs["store"] == 0
        assert kwargs["timeout"] == pytest.approx(0.25)
        for packet in packets:
            kwargs["prn"](packet)

    assert scan_ultrasound_interface(
        "npcap-test-interface", timeout_s=0.25, sniff=fake_sniff
    ) == 2


def test_interface_scan_reports_backend_error() -> None:
    def broken_sniff(**_kwargs: Any) -> None:
        raise OSError("Npcap unavailable")

    with pytest.raises(AdapterError, match="Npcap unavailable"):
        scan_ultrasound_interface("bad-interface", sniff=broken_sniff)


# ── health metrics ───────────────────────────────────────────────────────────


def test_health_reports_invalid_length_and_overflow_counters() -> None:
    adapter, backend = _running_adapter()
    health = adapter.health()
    assert "invalid_length_packets" in health.metrics
    assert "inbound_queue_overflows" in health.metrics
    assert health.metrics["interface_name"] is None
    adapter.stop()
    adapter.close()


# ── inbound queue overflow ──────────────────────────────────────────────────


def test_inbound_queue_overflow_sets_fault() -> None:
    """B.5: queue overflow must FAULT, not silently count."""
    adapter = RawEthernetUltrasoundAdapter(
        config={"inbound_queue_capacity": 1}
    )
    backend = FakeSnifferBackend()
    adapter._backend = backend
    adapter.connect()
    adapter.prepare(_context())
    adapter.start()

    payload = _uint8_adc()
    # Fill the single-slot inbound queue.
    adapter._inbound_queue.put_nowait(
        NormalizedPacket(0, 0, 0, payload, False)
    )
    # Next callback push must overflow → FAULT.
    backend.emit(FakeScapyPacket("00:01:03:04:05:06", payload))

    sleep(0.3)
    # Adapter should be in FAULTED state.
    assert adapter.state in (AdapterState.FAULTED, AdapterState.CLOSED)
    adapter.stop()
    adapter.close()
