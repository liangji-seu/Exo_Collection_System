from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from exo_collection.protocols import load_default_protocol, load_protocol


def test_default_protocol_is_versioned_and_has_unique_conditions() -> None:
    protocol = load_default_protocol()
    assert protocol.schema_version == "1.0.0"
    assert protocol.protocol_version == "1.2.0"
    assert len(protocol.conditions) == 26
    assert {condition.condition_code for condition in protocol.conditions} == {
        "FREE_TEST",
        "STATIC_CALIB",
        "STAND_30S_NOEXO",
        "STAND_30S_EXO",
        "WALK_1P0_30S_NOEXO",
        "WALK_1P0_30S_EXO",
        "WALK_0P6_NOEXO",
        "WALK_0P6_EXO",
        "WALK_1P0_NOEXO",
        "WALK_1P0_EXO",
        "WALK_1P4_NOEXO",
        "WALK_1P4_EXO",
        "WALK_2P5D_0P6_NOEXO",
        "WALK_2P5D_0P6_EXO",
        "WALK_2P5D_1P0_NOEXO",
        "WALK_2P5D_1P0_EXO",
        "WALK_5D_0P6_NOEXO",
        "WALK_5D_0P6_EXO",
        "WALK_5D_1P0_NOEXO",
        "WALK_5D_1P0_EXO",
        "SQUAT_STD_EXO",
        "SPEED_RAMP_EXO",
        "START_LEFT_EXO",
        "START_RIGHT_EXO",
        "STOP_LEFT_EXO",
        "STOP_RIGHT_EXO",
    }

    by_code = {
        condition.condition_code: condition for condition in protocol.conditions
    }
    assert by_code["WALK_5D_1P0_EXO"].parameters == {
        "category": "steady_slope_walking",
        "recommended_trial_count": 5,
        "target_effective_duration_s": 30,
        "speed_mps": 1.0,
        "slope_deg": 5,
        "exo": True,
    }
    assert by_code["STAND_30S_NOEXO"].parameters["exo"] is False
    assert (
        by_code["SQUAT_STD_EXO"].parameters["description"]
        == "蹲下，保持3s，站起，共5s"
    )
    assert (
        by_code["SPEED_RAMP_EXO"].parameters["description"]
        == "0.6m/s匀速 → 1.0m/s匀速 → 1.4m/s匀速 → 1.0m/s匀速 → 0.6m/s匀速，加速度1m/s²"
    )
    assert by_code["START_LEFT_EXO"].parameters["lead_foot"] == "left"
    assert by_code["START_RIGHT_EXO"].parameters["lead_foot"] == "right"
    assert by_code["STOP_LEFT_EXO"].parameters["lead_foot"] == "left"
    assert by_code["STATIC_CALIB"].parameters["description"] == (
        "穿戴与不穿戴共用同一 Helen-Hayes 静态标定模型"
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
