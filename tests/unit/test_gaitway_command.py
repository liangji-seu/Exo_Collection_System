"""Tests for gaitway-3D ``startDS`` command construction."""

from __future__ import annotations

from exo_collection.adapters.force_plate.gaitway_tcp import build_start_ds_command


def test_start_ds_defaults_to_continuous_total_and_decomposed() -> None:
    assert (
        build_start_ds_command(
            sample_rate_hz=1000,
            trigger_mode=0,
            sync_out_enabled=False,
            type_i_mode=2,
            type_ii_mode=2,
        )
        == "startDS 1000 0 0 0 2 2"
    )


def test_start_ds_seconds_and_trigger_are_positional() -> None:
    assert (
        build_start_ds_command(
            sample_rate_hz=500,
            trigger_mode=1,
            sync_out_enabled=True,
            type_i_mode=2,
            type_ii_mode=2,
            seconds=30,
        )
        == "startDS 500 30 1 1 2 2"
    )


def test_start_ds_syncout_is_coerced_to_0_or_1() -> None:
    assert build_start_ds_command(
        sample_rate_hz=1000,
        trigger_mode=0,
        sync_out_enabled=False,
        type_i_mode=2,
        type_ii_mode=2,
    ).endswith("0 2 2")
    assert build_start_ds_command(
        sample_rate_hz=1000,
        trigger_mode=0,
        sync_out_enabled=True,
        type_i_mode=2,
        type_ii_mode=2,
    ).endswith("1 2 2")


def test_start_ds_supports_off_and_header_only_modes() -> None:
    assert (
        build_start_ds_command(
            sample_rate_hz=200,
            trigger_mode=0,
            sync_out_enabled=False,
            type_i_mode=0,
            type_ii_mode=0,
        )
        == "startDS 200 0 0 0 0 0"
    )
    assert (
        build_start_ds_command(
            sample_rate_hz=200,
            trigger_mode=2,
            sync_out_enabled=False,
            type_i_mode=1,
            type_ii_mode=1,
        )
        == "startDS 200 0 2 0 1 1"
    )


def test_start_ds_uses_int_casts_for_every_field() -> None:
    # Boolean-like and float inputs must not leak into the ASCII command.
    command = build_start_ds_command(
        sample_rate_hz=1000.0,  # type: ignore[arg-type]
        trigger_mode=0,
        sync_out_enabled=False,
        type_i_mode=2,
        type_ii_mode=2,
        seconds=0,
    )
    assert command == "startDS 1000 0 0 0 2 2"
