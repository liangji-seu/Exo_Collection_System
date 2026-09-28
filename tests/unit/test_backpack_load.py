from __future__ import annotations

import hashlib
import json

import pytest

from exo_collection.apps.calculate.backpack_load import resolve_backpack_load


@pytest.mark.parametrize(
    ("code", "mass"),
    [
        ("LWALK_LEVEL_0KG", 0.0),
        ("lwalk_level_2p5kg_exo", 2.5),
        ("LWALK_LEVEL_5KG_NOEXO", 5.0),
        ("LWALK_LEVEL_10KG_EXO", 10.0),
        ("LWALK_RAMP_2P5D_5KG_NOEXO", 5.0),
        ("LWALK_LEVEL_3P75KG", 3.75),
    ],
)
def test_resolves_lwalk_nominal_loads(code, mass):
    result = resolve_backpack_load(code, {})

    assert result["mass_kg"] == mass
    assert result["enabled"] is (mass > 0)
    assert result["source"] == "NOMINAL"
    assert result["condition_code"] == code
    assert result["spec"]["payload"][0]["mass_kg"] == mass
    assert result["spec"]["parent_body"] == "torso"
    assert result["geometry_status"] == (
        "assumed" if mass > 0 else "not_applicable"
    )
    assert len(result["warnings"]) == (1 if mass > 0 else 0)


@pytest.mark.parametrize("code", ["EXO", "NOARG_0", "LIFT_LOAD"])
def test_legacy_conditions_ignore_load_parameter_and_config(tmp_path, code):
    config_dir = tmp_path / ".exo"
    config_dir.mkdir()
    (config_dir / "backpack_config.json").write_text(
        json.dumps({"unknown": "ignored for legacy"}), encoding="utf-8"
    )

    result = resolve_backpack_load(code, {"load_kg": 3}, tmp_path)

    assert result["enabled"] is False
    assert result["mass_kg"] == 0.0
    assert result["source"] == "NONE"
    assert result["config_path"] is None


@pytest.mark.parametrize(
    "code",
    ["LWALK_LEVEL", "LWALK_LEVEL_5", "LWALK_LEVEL_5KG_OTHER", "LWALK__5KG"],
)
def test_rejects_malformed_lwalk_code(code):
    with pytest.raises(ValueError, match="malformed LWALK"):
        resolve_backpack_load(code, {})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1])
def test_rejects_invalid_parameter_load(value):
    with pytest.raises(ValueError, match="finite non-negative"):
        resolve_backpack_load("LWALK_LEVEL_5KG", {"load_kg": value})


def test_rejects_code_parameter_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        resolve_backpack_load("LWALK_LEVEL_5KG", {"load_kg": 2.5})


def test_valid_geometry_override_is_recorded(tmp_path):
    config_dir = tmp_path / ".exo"
    config_dir.mkdir()
    config_path = config_dir / "backpack_config.json"
    raw = json.dumps(
        {
            "com_m": [-0.2, 0.15, 0],
            "dimensions_xyz_m": [0.08, 0.12, 0.25],
            "orientation_xyz_rad": [0, 0.1, 0],
            "geometry_status": "measured",
        }
    ).encode()
    config_path.write_bytes(raw)

    result = resolve_backpack_load(
        "LWALK_LEVEL_5KG_EXO", {"load_kg": 5}, tmp_path
    )

    payload = result["spec"]["payload"][0]
    assert result["mass_kg"] == 5.0
    assert payload["com_m"] == [-0.2, 0.15, 0.0]
    assert payload["dimensions_xyz_m"] == [0.08, 0.12, 0.25]
    assert payload["orientation_xyz_rad"] == [0.0, 0.1, 0.0]
    assert result["geometry_status"] == "measured"
    assert result["warnings"] == []
    assert result["config_path"] == str(config_path)
    assert result["config_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "config",
    [
        {"mass_kg": 99},
        {"com_m": [0, float("nan"), 0]},
        {"dimensions_xyz_m": [0.1, 0, 0.3]},
        {"orientation_xyz_rad": [0, 0]},
    ],
)
def test_rejects_invalid_geometry_override(tmp_path, config):
    config_dir = tmp_path / ".exo"
    config_dir.mkdir()
    (config_dir / "backpack_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        resolve_backpack_load("LWALK_LEVEL_5KG", {}, tmp_path)


def test_unknown_config_key_is_reported_before_load_mismatch(tmp_path):
    config_dir = tmp_path / ".exo"
    config_dir.mkdir()
    (config_dir / "backpack_config.json").write_text(
        '{"mass_kg": 4}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown backpack geometry config keys"):
        resolve_backpack_load("LWALK_LEVEL_5KG", {"load_kg": 3}, tmp_path)
