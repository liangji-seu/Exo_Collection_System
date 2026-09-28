from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from pipeline.opensim_io.backpack_runtime import BACKPACK_MODEL_NAME, prepare_id_model


def _model(path) -> None:
    root = ET.fromstring(
        '<OpenSimDocument Version="40000"><Model name="toy">'
        '<BodySet name="bodyset"><objects><Body name="torso">'
        '<mass>10</mass><mass_center>0 0 0</mass_center><inertia>1 2 3 0 0 0</inertia>'
        '</Body></objects><groups/></BodySet>'
        '<JointSet name="jointset"><objects/><groups/></JointSet>'
        '</Model></OpenSimDocument>'
    )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _resolved(mass=5.0):
    return {
        "enabled": True,
        "mass_kg": mass,
        "source": "session_config",
        "condition_code": "BP05",
        "geometry_status": "resolved",
        "warnings": ["resolver warning"],
        "spec": {
            "parent_body": "torso",
            "mode": "weld",
            "simulation_only": True,
            "empty_bag": {"mass_kg": 0, "com_m": [0, 0, 0]},
            "payload": [
                {
                    "name": "payload",
                    "mass_kg": mass,
                    "com_m": [-0.12, 0.25, 0],
                    "dimensions_xyz_m": [0.075, 0.5, 0.3],
                }
            ],
        },
    }


def test_missing_resolution_keeps_human_model_and_generates_nothing(tmp_path):
    human = tmp_path / "hh19_static_calibrated.osim"
    _model(human)
    before = human.read_bytes()

    actual, receipt = prepare_id_model(human, None)

    assert actual == human
    assert human.read_bytes() == before
    assert not (tmp_path / BACKPACK_MODEL_NAME).exists()
    assert not (tmp_path / BACKPACK_MODEL_NAME).with_suffix(".json").exists()
    assert receipt["warnings"] == []
    assert receipt["generator_receipt"] is None
    assert receipt["human_model_sha256"] == receipt["actual_model_sha256"]
    json.dumps(receipt, allow_nan=False)


def test_disabled_zero_mass_is_noop_and_preserves_resolver_warnings(tmp_path):
    human = tmp_path / "hh19_static_calibrated.osim"
    _model(human)
    resolved = {
        "enabled": False,
        "mass_kg": 0,
        "source": "legacy_default",
        "condition_code": None,
        "geometry_status": "not_needed",
        "warnings": ["existing warning"],
        "spec": None,
    }

    actual, receipt = prepare_id_model(human, resolved)

    assert actual == human
    assert receipt["warnings"] == ["existing warning"]
    assert receipt["source"] == "legacy_default"
    assert not (tmp_path / BACKPACK_MODEL_NAME).exists()


def test_positive_mass_creates_welded_model_and_complete_receipt(tmp_path):
    human = tmp_path / "hh19_static_calibrated.osim"
    _model(human)
    before = human.read_bytes()

    actual, receipt = prepare_id_model(human, _resolved())

    assert actual == tmp_path / BACKPACK_MODEL_NAME
    assert actual.is_file()
    assert human.read_bytes() == before
    backpack = ET.parse(actual).find(".//Body[@name='backpack']")
    assert backpack is not None
    assert float(backpack.findtext("mass")) == pytest.approx(5.0)
    assert ET.parse(actual).find(".//WeldJoint[@name='backpack_weld']") is not None
    assert receipt["source"] == "session_config"
    assert receipt["geometry_status"] == "resolved"
    assert receipt["totals"] == {"resolved_mass_kg": 5.0, "generated_backpack_mass_kg": 5.0}
    assert receipt["generator_receipt"]["mode"] == "weld"
    assert receipt["actual_model_sha256"] != receipt["human_model_sha256"]
    assert actual.with_suffix(".json").is_file()
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r.update(mass_kg=True), "not bool"),
        (lambda r: r.update(mass_kg=float("nan")), "finite non-negative"),
        (lambda r: r.update(enabled=False), "disabled backpack"),
        (lambda r: r.update(mass_kg=4.0), "does not match spec total"),
        (lambda r: r["spec"].update(mode="merge"), "weld mode"),
    ],
)
def test_invalid_or_contradictory_resolution_is_rejected(tmp_path, mutate, match):
    human = tmp_path / "hh19_static_calibrated.osim"
    _model(human)
    resolved = _resolved()
    mutate(resolved)

    with pytest.raises(ValueError, match=match):
        prepare_id_model(human, resolved)

    assert not (tmp_path / BACKPACK_MODEL_NAME).exists()
