"""Tests for sticky (coalesced TCP read) framing."""

from __future__ import annotations

from struct import pack

from exo_collection.adapters.force_plate import GaitwayPacketFramer


def _ack(command: bytes = b"stop") -> bytes:
    return pack("<HH", 4 + len(command), 6) + command


def _type_i(packet_id: int = 7) -> bytes:
    sample = pack("<8fHH", 100.0, 20.0, 10.0, 0.2, 0.1, 3.0, 1.5, 2.5, 80, 9)
    return pack("<HHI8x", 16 + len(sample), 1, packet_id) + sample


def _type_ii(packet_id: int = 5) -> bytes:
    sample = pack("<HH10f", 1, 0, 500.0, 20.0, 10.0, 0.02, 0.01, 300.0, 15.0, 5.0, 0.03, 0.02)
    return pack("<HHIHHI16x", 32 + len(sample), 2, packet_id, 0, 1, 3) + sample


def test_two_packets_in_one_read_are_both_emitted() -> None:
    first = _ack()
    second = _type_i()
    assert GaitwayPacketFramer().feed(first + second) == [first, second]


def test_three_packets_with_trailing_partial_fourth() -> None:
    framer = GaitwayPacketFramer()
    first = _ack()
    second = _type_i()
    third = _type_ii()
    fourth = _type_i(packet_id=8)
    blob = first + second + third + fourth[:10]
    emitted = framer.feed(blob)
    assert emitted == [first, second, third]
    assert framer.buffered_bytes == 10
    # Complete the fourth packet.
    assert framer.feed(fourth[10:]) == [fourth]


def test_mixed_type_i_and_type_ii_in_one_read() -> None:
    type_i = _type_i()
    type_ii = _type_ii()
    assert GaitwayPacketFramer().feed(type_i + type_ii) == [type_i, type_ii]


def test_many_packets_in_one_read_are_all_emitted_in_order() -> None:
    packets = [_ack(b"run"), _type_i(1), _type_ii(2), _type_i(3)]
    blob = b"".join(packets)
    assert GaitwayPacketFramer().feed(blob) == packets
