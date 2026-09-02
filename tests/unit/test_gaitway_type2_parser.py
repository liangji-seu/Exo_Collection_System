"""Tests for the gaitway Type-II (left/right decomposition) packet parser."""

from __future__ import annotations

from struct import pack

import numpy as np
import pytest

from exo_collection.adapters.force_plate.gaitway_tcp import (
    TYPE_II_CHANNELS,
    GaitwayPacketError,
    parse_type_ii_packet,
)


def _type_ii(
    packet_id: int = 5,
    gait_type: int = 0,
    contact_side: int = 1,
    step_count: int = 3,
    samples: tuple[tuple, ...] = (),
) -> bytes:
    payload = b"".join(pack("<HH10f", *sample) for sample in samples)
    return (
        pack("<HHIHHI16x", 32 + len(payload), 2, packet_id, gait_type, contact_side, step_count)
        + payload
    )


# wire order: foot_contact, digital, fz_l, fy_l, fx_l, cop_y_l, cop_x_l,
#             fz_r, fy_r, fx_r, cop_y_r, cop_x_r
_SAMPLE = (1, 0, 500.0, 20.0, 10.0, 0.02, 0.01, 300.0, 15.0, 5.0, 0.03, 0.02)


def test_parse_type_ii_returns_full_header_and_reordered_samples() -> None:
    header, data = parse_type_ii_packet(
        _type_ii(packet_id=5, gait_type=0, contact_side=1, step_count=3, samples=(_SAMPLE,))
    )
    assert header["packet_size"] == 32 + 44
    assert header["packet_type"] == 2
    assert header["packet_id"] == 5
    assert header["gait_type"] == 0
    assert header["contact_side"] == 1
    assert header["step_count"] == 3
    assert data.shape == (1, 12)
    # canonical: foot_contact, digital, fx_l, fy_l, fz_l, cop_x_l, cop_y_l,
    #            fx_r, fy_r, fz_r, cop_x_r, cop_y_r
    np.testing.assert_allclose(
        data[0],
        [1.0, 0.0, 10.0, 20.0, 500.0, 0.01, 0.02, 5.0, 15.0, 300.0, 0.02, 0.03],
    )


def test_parse_type_ii_channel_order_matches_type_ii_channels() -> None:
    _header, data = parse_type_ii_packet(_type_ii(samples=(_SAMPLE,)))
    assert TYPE_II_CHANNELS == (
        "foot_contact", "digital_inputs",
        "fx_l", "fy_l", "fz_l", "cop_x_l", "cop_y_l",
        "fx_r", "fy_r", "fz_r", "cop_x_r", "cop_y_r",
    )
    assert data.shape[1] == len(TYPE_II_CHANNELS)


def test_parse_type_ii_parses_multiple_samples() -> None:
    second = (2, 0, 600.0, 22.0, 12.0, 0.04, 0.03, 350.0, 16.0, 6.0, 0.05, 0.04)
    _header, data = parse_type_ii_packet(_type_ii(samples=(_SAMPLE, second)))
    assert data.shape == (2, 12)
    np.testing.assert_allclose(
        data[1],
        [2.0, 0.0, 12.0, 22.0, 600.0, 0.03, 0.04, 6.0, 16.0, 350.0, 0.04, 0.05],
    )


def test_parse_type_ii_rejects_short_packet() -> None:
    with pytest.raises(GaitwayPacketError):
        parse_type_ii_packet(b"\x02\x00\x02")


def test_parse_type_ii_rejects_wrong_packet_type() -> None:
    payload = pack("<HH10f", *_SAMPLE)
    bad = pack("<HHIHHI16x", 32 + len(payload), 1, 5, 0, 1, 3) + payload  # type 1
    with pytest.raises(GaitwayPacketError):
        parse_type_ii_packet(bad)


def test_parse_type_ii_rejects_size_mismatch() -> None:
    packet = _type_ii(samples=(_SAMPLE,))
    mangled = pack("<HHIHHI16x", 32 + 44 + 1, 2, 5, 0, 1, 3) + packet[32:]
    with pytest.raises(GaitwayPacketError):
        parse_type_ii_packet(mangled)


def test_parse_type_ii_rejects_non_multiple_payload() -> None:
    bad = pack("<HHIHHI16x", 32 + 20, 2, 5, 0, 1, 3) + b"\x00" * 20
    with pytest.raises(GaitwayPacketError):
        parse_type_ii_packet(bad)


def test_parse_type_ii_rejects_empty_payload() -> None:
    bad = pack("<HHIHHI16x", 32, 2, 5, 0, 1, 3)
    with pytest.raises(GaitwayPacketError):
        parse_type_ii_packet(bad)
