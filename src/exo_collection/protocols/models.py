"""Strict loader for versioned condition JSON used by Collector."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator


SemVer = Annotated[str, StringConstraints(pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")]


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConditionDefinition(ProtocolModel):
    condition_code: str = Field(min_length=1)
    condition_name: str = Field(min_length=1)
    condition_level: int | str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ProtocolDefinition(ProtocolModel):
    schema_version: SemVer
    protocol_version: SemVer
    conditions: list[ConditionDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_codes(self) -> ProtocolDefinition:
        codes = [condition.condition_code for condition in self.conditions]
        if len(codes) != len(set(codes)):
            raise ValueError("condition_code values must be unique")
        return self


def load_protocol(path: str | Path) -> ProtocolDefinition:
    source = Path(path).expanduser().resolve()
    return ProtocolDefinition.model_validate_json(source.read_text(encoding="utf-8"))


def _default_protocol_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    if configured := os.environ.get("EXO_CONFIG_ROOT"):
        candidates.append(Path(configured) / "protocols/default.json")
    if bundle_root := getattr(sys, "_MEIPASS", None):
        candidates.append(Path(bundle_root) / "config/protocols/default.json")
    candidates.append(Path.cwd() / "config/protocols/default.json")
    # Editable/source checkout fallback: models.py → protocols → exo_collection
    # → src → project root.
    candidates.append(Path(__file__).resolve().parents[3] / "config/protocols/default.json")
    return tuple(candidates)


def load_default_protocol() -> ProtocolDefinition:
    for candidate in _default_protocol_candidates():
        if candidate.is_file():
            return load_protocol(candidate)
    searched = ", ".join(str(path) for path in _default_protocol_candidates())
    raise FileNotFoundError(f"default condition protocol not found; searched: {searched}")
