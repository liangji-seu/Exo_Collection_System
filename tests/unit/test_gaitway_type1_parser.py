"""Tests for the gaitway Type-I (total GRF/COP) packet parser."""

from __future__ import annotations

from struct import pack

import numpy as np
import pytest

from exo_collection.adapters.force_plate.gaitway_tcp import (
    FORCE_PLATE_CHANNELS,
    GaitwayPacketError,
    parse_type_i_packet,
)


def _type_i(packet_id: int = 7, samples: tuple[tuple, ...] = ()) -> bytes:
    payload = b"".join(pack("<8fHH", *sample) for sample in samples)
    return pack("<HHI8x", 16 + len(payload), 1, packet_id) + payload


# wire order: fz, fy, fx, cop_y, cop_x, tz, speed, elevation, heart, digital
_SAMPLE = (100.0, 20.0, 10.0, 0.2, 0.1, 3.0, 1.5, 2.5, 80, 9)


def test_parse_type_i_returns_header_and_reordered_samples() -> None:
    header, data = parse_type_i_packet(_type_i(packet_id=7, samples=(_SAMPLE,)))
    assert header["packet_size"] == 16 + 36
    assert header["packet_type"] == 1
    assert header["packet_id"] == 7
    assert data.shape == (1, 10)
    # canonical order: fx, fy, fz, cop_x, cop_y, tz, speed, elevation, heart, digital
    np.testing.assert_allclose(
        data[0],
        [10.0, 20.0, 100.0, 0.1, 0.2, 3.0, 1.5, 2.5, 80.0, 9.0],
    )


def test_parse_type_i_parses_multiple_samples() -> None:
    second = (110.0, 21.0, 11.0, 0.3, 0.2, 4.0, 1.6, 2.6, 81, 8)
    _header, data = parse_type_i_packet(_type_i(samples=(_SAMPLE, second)))
    assert data.shape == (2, 10)
    np.testing.assert_allclose(data[1], [11.0, 21.0, 110.0, 0.2, 0.3, 4.0, 1.6, 2.6, 81.0, 8.0])


def test_parse_type_i_channel_order_matches_force_plate_channels() -> None:
    _header, data = parse_type_i_packet(_type_i(samples=(_SAMPLE,)))
    assert FORCE_PLATE_CHANNELS == (
        "fx", "fy", "fz", "cop_x", "cop_y", "tz",
        "treadmill_speed", "treadmill_elevation", "heart_rate", "digital_inputs",
    )
    assert data.shape[1] == len(FORCE_PLATE_CHANNELS)


def test_parse_type_i_rejects_short_packet() -> None:
    with pytest.raises(GaitwayPacketError):
        parse_type_i_packet(b"\x01\x00\x01")


def test_parse_type_i_rejects_wrong_packet_type() -> None:
    payload = pack("<8fHH", *_SAMPLE)
    bad = pack("<HHI8x", 16 + len(payload), 2, 7) + payload  # type 2, not type 1
    with pytest.raises(GaitwayPacketError):
        parse_type_i_packet(bad)


def test_parse_type_i_rejects_size_mismatch() -> None:
    packet = _type_i(packet_id=7, samples=(_SAMPLE,))
    # Rewrite the size field to disagree with the actual length.
    mangled = pack("<HHI8x", 16 + 36 + 1, 1, 7) + packet[16:]
    with pytest.raises(GaitwayPacketError):
        parse_type_i_packet(mangled)


def test_parse_type_i_rejects_non_multiple_payload() -> None:
    bad = pack("<HHI8x", 16 + 20, 1, 7) + b"\x00" * 20
    with pytest.raises(GaitwayPacketError):
        parse_type_i_packet(bad)


def test_parse_type_i_rejects_empty_payload() -> None:
    bad = pack("<HHI8x", 16, 1, 7)
    with pytest.raises(GaitwayPacketError):
        parse_type_i_packet(bad)
