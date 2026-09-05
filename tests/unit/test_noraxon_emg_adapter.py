"""Unit tests for the Noraxon EMG adapter (no vendor SDK required).

The Noraxon AcquireCom SDK is imported lazily inside the sampler thread, so
these tests exercise config validation, sensor-ID normalisation, and the
descriptor's muscle-to-channel correspondence without any COM objects.
"""

from __future__ import annotations

import numpy as np
import pytest

from exo_collection.adapters.emg.noraxon import (
    NoraxonEmgAdapter,
    NoraxonEmgChannel,
    NoraxonEmgConfig,
    _normalise_unit_id,
    _parse_offline_serials,
    _read_mr4_ultium_serials,
)


def test_normalise_unit_id_strips_tag_prefixes() -> None:
    assert _normalise_unit_id("line.noraxon_g3_234fc") == "234fc"
    assert _normalise_unit_id("noraxon_g3_234f5") == "234f5"
    assert _normalise_unit_id("234fc") == "234fc"
    assert _normalise_unit_id("") == ""
    assert _normalise_unit_id(None) == ""


def test_read_mr4_ultium_serials_parses_sensor_nodes(tmp_path) -> None:
    xml = """<?xml version="1.1" encoding="utf-8"?>
<Acquire.Devices.Manager xmlns="noraxon.mr3,1">
    <Struct id="devices">
        <Acquire.Devices.Profile id="default">
            <String id="name" value="默认" />
            <Struct id="devices">
                <Acquire.Noraxon.Ultium id="usb_88024074">
                    <String id="serial" value="88024074" />
                    <Struct id="sensors">
                        <Acquire.Noraxon.Ultium.Sensor id="q4102jh1">
                            <String id="id" value="234fd" />
                            <String id="name" value="肌电图 7" />
                        </Acquire.Noraxon.Ultium.Sensor>
                        <Acquire.Noraxon.Ultium.Sensor id="sj1n2jh1">
                            <String id="id" value="234f2" />
                            <String id="name" value="肌电图 1" />
                        </Acquire.Noraxon.Ultium.Sensor>
                    </Struct>
                </Acquire.Noraxon.Ultium>
            </Struct>
        </Acquire.Devices.Profile>
    </Struct>
</Acquire.Devices.Manager>
"""
    path = tmp_path / "device_manager.object"
    path.write_text(xml, encoding="utf-8")
    assert _read_mr4_ultium_serials(str(path)) == [
        ("234fd", "肌电图 7"),
        ("234f2", "肌电图 1"),
    ]


def test_parse_offline_serials_extracts_hex_serials() -> None:
    assert _parse_offline_serials(
        "Could not find the following sensors: 234f2"
    ) == {"234f2"}


def test_parse_offline_serials_handles_multiple_and_whitespace() -> None:
    assert _parse_offline_serials(
        "Could not find the following sensors: 234fc, 234f5"
    ) == {"234fc", "234f5"}


def test_parse_offline_serials_returns_empty_for_other_errors() -> None:
    assert _parse_offline_serials("(-2147467259, '未指定的错误')") == set()
    assert _parse_offline_serials("") == set()


def test_config_rejects_empty_channels() -> None:
    with pytest.raises(ValueError, match="channels must not be empty"):
        NoraxonEmgConfig(channels=())


def test_config_rejects_duplicate_muscle_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        NoraxonEmgConfig(
            channels=(
                NoraxonEmgChannel(name="股直肌", unit_id="234fc"),
                NoraxonEmgChannel(name="股直肌", unit_id="234f5"),
            )
        )


def test_config_rejects_non_positive_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        NoraxonEmgConfig(sample_rate_hz=0.0)


def test_config_coerces_mapping_and_preserves_channel_order() -> None:
    config = NoraxonEmgAdapter(
        {
            "device_id": "emg_noraxon",
            "clock_domain": "emg_noraxon_clock",
            "sample_rate_hz": 4000.0,
            "unit": "µV",
            "channels": [
                {"name": "股直肌", "unit_id": "noraxon_g3_234fc"},
                {"name": "股内侧肌", "unit_id": "noraxon_g3_234f5"},
                {"name": "股外侧肌", "unit_id": ""},
                {"name": "股中肌", "unit_id": ""},
            ],
        }
    )._config
    assert config.sample_rate_hz == 4000.0
    assert config.unit == "µV"
    assert [channel.name for channel in config.channels] == [
        "股直肌",
        "股内侧肌",
        "股外侧肌",
        "股中肌",
    ]
    assert [channel.unit_id for channel in config.channels] == [
        "noraxon_g3_234fc",
        "noraxon_g3_234f5",
        "",
        "",
    ]


def test_descriptor_records_muscle_to_channel_correspondence() -> None:
    adapter = NoraxonEmgAdapter(
        {
            "device_id": "emg_noraxon",
            "clock_domain": "emg_noraxon_clock",
            "sample_rate_hz": 4000.0,
            "unit": "µV",
            "channels": [
                {"name": "股直肌", "unit_id": "noraxon_g3_234fc"},
                {"name": "股内侧肌", "unit_id": "noraxon_g3_234f5"},
                {"name": "股外侧肌", "unit_id": ""},
                {"name": "股中肌", "unit_id": ""},
            ],
        }
    )
    descriptor = adapter.descriptor()

    assert descriptor.modality == "emg"
    assert descriptor.event_kind == "sample_batch"
    assert descriptor.channels == ("股直肌", "股内侧肌", "股外侧肌", "股中肌")
    assert descriptor.units == ("µV", "µV", "µV", "µV")
    assert descriptor.sample_shape == (4,)
    assert descriptor.dtype == np.dtype(np.float32).str
    assert descriptor.nominal_rate_hz == 4000.0

    metadata = descriptor.metadata
    assert metadata["storage_format"] == "block_binary"
    assert metadata["manufacturer"] == "Noraxon"
    assert metadata["channel_names"] == ["股直肌", "股内侧肌", "股外侧肌", "股中肌"]
    assert metadata["unit_ids"] == [
        "noraxon_g3_234fc",
        "noraxon_g3_234f5",
        "",
        "",
    ]
    assert metadata["unit_serials"] == ["234fc", "234f5", "", ""]
    assert metadata["muscle_to_channel"] == {
        "股直肌": 0,
        "股内侧肌": 1,
        "股外侧肌": 2,
        "股中肌": 3,
    }
    # Before connect, no sensor is detected so every configured slot reports
    # ``connected=False`` while still preserving its muscle/unit-ID pairing.
    assert metadata["channel_mapping"] == [
        {"channel": 0, "muscle": "股直肌", "unit_id": "noraxon_g3_234fc", "connected": False},
        {"channel": 1, "muscle": "股内侧肌", "unit_id": "noraxon_g3_234f5", "connected": False},
        {"channel": 2, "muscle": "股外侧肌", "unit_id": "", "connected": False},
        {"channel": 3, "muscle": "股中肌", "unit_id": "", "connected": False},
    ]

    adapter.close()
