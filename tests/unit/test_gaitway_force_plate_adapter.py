from __future__ import annotations

from struct import pack

import numpy as np

from exo_collection.acquisition.preview import build_preview_event
from exo_collection.adapters.base import AdapterState, TrialContext
from exo_collection.adapters.force_plate import (
    FORCE_PLATE_CHANNELS,
    GaitwayForcePlateTcpAdapter,
    GaitwayPacketFramer,
)


def _type_i_packet(packet_id: int = 7) -> bytes:
    samples = (
        pack("<8fHH", 100.0, 20.0, 10.0, 0.2, 0.1, 3.0, 1.5, 2.5, 80, 9)
        + pack("<8fHH", 110.0, 21.0, 11.0, 0.3, 0.2, 4.0, 1.6, 2.6, 81, 8)
    )
    size = 16 + len(samples)
    return pack("<HHI8x", size, 1, packet_id) + samples


def test_gaitway_packet_framer_handles_split_and_coalesced_tcp_reads() -> None:
    first = pack("<HH", 8, 6) + b"stop"
    second = _type_i_packet()
    framer = GaitwayPacketFramer()

    assert framer.feed(first[:3]) == []
    assert framer.feed(first[3:] + second[:11]) == [first]
    assert framer.feed(second[11:]) == [second]
    assert framer.buffered_bytes == 0


def test_type_i_packet_is_reordered_to_canonical_force_plate_channels() -> None:
    adapter = GaitwayForcePlateTcpAdapter(
        {
            "device_id": "force",
            "sample_rate_hz": 1000,
            "queue_capacity": 8,
        }
    )
    adapter._trial = TrialContext(trial_uuid="00000000-0000-0000-0000-000000000001")
    adapter._state = AdapterState.RUNNING

    adapter._accept_type_i(_type_i_packet(), 2_000_000)
    event = adapter.get_event(timeout=0)

    assert event is not None
    assert event.sample_count == 2
    assert event.sequence_number == 7
    np.testing.assert_allclose(
        event.data[0],
        [10.0, 20.0, 100.0, 0.1, 0.2, 3.0, 1.5, 2.5, 80.0, 9.0],
    )
    preview = build_preview_event(
        event,
        extra_payload={"channel_names": list(FORCE_PLATE_CHANNELS)},
    )
    assert preview.payload["labels"][:6] == [
        "fx",
        "fy",
        "fz",
        "cop_x",
        "cop_y",
        "tz",
    ]
    assert preview.payload["latest"]["digital_inputs"] == 8.0
