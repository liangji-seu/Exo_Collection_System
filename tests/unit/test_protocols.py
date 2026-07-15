from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from exo_collection.protocols import load_default_protocol, load_protocol


def test_default_protocol_is_versioned_and_has_unique_conditions() -> None:
    protocol = load_default_protocol()
    assert protocol.schema_version == "1.0.0"
    assert protocol.protocol_version == "1.0.0"
    assert {condition.condition_code for condition in protocol.conditions} == {"STAND", "WALK_LEVEL"}


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
