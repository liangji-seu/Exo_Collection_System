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
    _serial_from_tags,
    _ultium_serials_from_components,
)


def test_normalise_unit_id_strips_tag_prefixes() -> None:
    assert _normalise_unit_id("line.noraxon_g3_234fc") == "234fc"
    assert _normalise_unit_id("noraxon_g3_234f5") == "234f5"
    assert _normalise_unit_id("234fc") == "234fc"
    assert _normalise_unit_id("") == ""
    assert _normalise_unit_id(None) == ""


def test_serial_from_tags_returns_bare_serial() -> None:
    assert _serial_from_tags(["line.noraxon_g3_abc12", "type.input.analog.emg"]) == "abc12"
    assert _serial_from_tags(["type.input.analog.emg"]) is None


def test_ultium_serials_from_components_extracts_and_dedupes() -> None:
    tags_per_component = [
        ["type.input.analog.emg", "device.noraxon.ultium", "line.noraxon_g3_234fc"],
        ["type.input.analog.emg", "device.noraxon.ultium", "line.noraxon_g3_234f5"],
        ["type.input.analog.emg", "device.noraxon.ultium", "line.noraxon_g3_234fc"],
        ["device.player.player.record", "line.noraxon_g3_99999"],
        ["type.input.analog.emg", "device.noraxon.ultium"],
    ]
    assert _ultium_serials_from_components(tags_per_component) == ["234f5", "234fc"]


def test_ultium_serials_from_components_ignores_non_ultium() -> None:
    # A replay/player channel carries a g3 tag but no ultium device tag, so it
    # must be excluded from the scan results.
    assert (
        _ultium_serials_from_components(
            [
                ["type.input.analog.emg", "device.player.player.record", "line.noraxon_g3_12345"],
            ]
        )
        == []
    )


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
