from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from exo_collection.protocols import load_default_protocol, load_protocol


def test_default_protocol_is_versioned_and_has_unique_conditions() -> None:
    protocol = load_default_protocol()
    assert protocol.schema_version == "1.0.0"
    assert protocol.protocol_version == "1.1.0"
    assert len(protocol.conditions) == 19
    assert {condition.condition_code for condition in protocol.conditions} == {
        "STAND",
        "STAND_LOAD_2P5",
        "WALK_LEVEL",
        "WALK_LEVEL_LOAD_2P5",
        "SLOW_LEG_RAISE",
        "WALK_STEADY_0P75",
        "WALK_STEADY_1P00",
        "WALK_STEADY_1P25",
        "WALK_STEADY_1P75",
        "WALK_SLOPE_P05",
        "WALK_SLOPE_P10",
        "WALK_SLOPE_P15",
        "WALK_SLOPE_N05",
        "WALK_SLOPE_N10",
        "WALK_SLOPE_N15",
        "START_STOP_LEFT",
        "START_STOP_RIGHT",
        "SQUAT_STANDARD",
        "SPEED_CHANGE_0P6_TO_0P9",
    }

    by_code = {
        condition.condition_code: condition for condition in protocol.conditions
    }
    assert by_code["WALK_STEADY_1P25"].parameters == {
        "category": "steady_level_walking",
        "recommended_trial_count": 5,
        "target_effective_duration_s": 20,
        "speed_mps": 1.25,
        "load_kg": 0,
        "slope_deg": 0,
        "reference": "Aaron metabolic experiment",
    }
    assert by_code["WALK_SLOPE_N15"].parameters["slope_deg"] == -15
    assert by_code["START_STOP_LEFT"].parameters["lead_foot"] == "left"
    assert by_code["START_STOP_RIGHT"].parameters["lead_foot"] == "right"
    assert (
        by_code["SPEED_CHANGE_0P6_TO_0P9"].parameters["status"]
        == "provisional"
    )


def test_protocol_rejects_duplicate_condition_codes(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "protocol_version": "1.0.0",
                "conditions": [
                    {"condition_code": "A", "condition_name": "A", "parameters": {}},
                    {"condition_code": "A", "condition_name": "Again", "parameters": {}},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="unique"):
        load_protocol(path)
