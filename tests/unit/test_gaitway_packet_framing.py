"""Tests for the gaitway TCP byte-stream framer."""

from __future__ import annotations

from struct import pack

import pytest

from exo_collection.adapters.force_plate import GaitwayPacketError, GaitwayPacketFramer


def _ack() -> bytes:
    return pack("<HH", 8, 6) + b"stop"


def test_framer_returns_nothing_for_empty_feed() -> None:
    framer = GaitwayPacketFramer()
    assert framer.feed(b"") == []
    assert framer.buffered_bytes == 0


def test_framer_emits_a_complete_packet_in_one_feed() -> None:
    packet = _ack()
    assert GaitwayPacketFramer().feed(packet) == [packet]


def test_framer_emits_nothing_until_the_size_bytes_arrive() -> None:
    framer = GaitwayPacketFramer()
    assert framer.feed(b"\x08") == []
    assert framer.buffered_bytes == 1
    assert framer.feed(b"\x00\x06\x00stop") == [_ack()]
    assert framer.buffered_bytes == 0


def test_framer_survives_byte_by_byte_stream() -> None:
    framer = GaitwayPacketFramer()
    packet = _ack()
    emitted: list[bytes] = []
    for byte in packet:
        emitted.extend(framer.feed(bytes([byte])))
    assert emitted == [packet]


def test_framer_rejects_invalid_size_below_four() -> None:
    framer = GaitwayPacketFramer()
    with pytest.raises(GaitwayPacketError):
        framer.feed(b"\x02\x00")


def test_framer_rejects_size_above_maximum() -> None:
    framer = GaitwayPacketFramer(maximum_packet_size=64)
    with pytest.raises(GaitwayPacketError):
        framer.feed(b"\xff\xff")


def test_framer_reset_discards_buffered_bytes() -> None:
    framer = GaitwayPacketFramer()
    framer.feed(_ack()[:3])
    assert framer.buffered_bytes == 3
    framer.reset()
    assert framer.buffered_bytes == 0
