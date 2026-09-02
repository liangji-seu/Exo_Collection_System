"""Tests for partial-packet (fragmented TCP read) framing."""

from __future__ import annotations

from struct import pack

from exo_collection.adapters.force_plate import GaitwayPacketFramer


def _type_i(packet_id: int = 7) -> bytes:
    sample = pack("<8fHH", 100.0, 20.0, 10.0, 0.2, 0.1, 3.0, 1.5, 2.5, 80, 9)
    return pack("<HHI8x", 16 + len(sample), 1, packet_id) + sample


def test_partial_header_only_is_buffered() -> None:
    framer = GaitwayPacketFramer()
    packet = _type_i()
    assert framer.feed(packet[:2]) == []
    assert framer.buffered_bytes == 2


def test_partial_payload_is_buffered_until_complete() -> None:
    framer = GaitwayPacketFramer()
    packet = _type_i()
    assert framer.feed(packet[:40]) == []
    assert framer.buffered_bytes == 40
    assert framer.feed(packet[40:]) == [packet]
    assert framer.buffered_bytes == 0


def test_partial_then_reset_discards_bytes() -> None:
    framer = GaitwayPacketFramer()
    packet = _type_i()
    framer.feed(packet[:12])
    assert framer.buffered_bytes == 12
    framer.reset()
    assert framer.buffered_bytes == 0
    # A fresh packet can be framed cleanly after the reset.
    assert framer.feed(packet) == [packet]
    assert framer.buffered_bytes == 0


def test_partial_across_every_byte_boundary_emits_once() -> None:
    framer = GaitwayPacketFramer()
    packet = _type_i()
    emitted: list[bytes] = []
    # Feed one byte at a time; the framer must emit exactly one packet at the end.
    for offset in range(1, len(packet) + 1):
        emitted.extend(framer.feed(packet[offset - 1 : offset]))
    assert emitted == [packet]
    assert framer.buffered_bytes == 0
