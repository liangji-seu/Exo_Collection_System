"""Serializable inputs and outputs for one simulated Trial worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


class TrialRunRequest(OrchestrationModel):
    data_root: Path
    duration_s: float = Field(default=3.0, gt=0, le=86_400)
    project_uuid: UUID = Field(default_factory=uuid4)
    subject_uuid: UUID = Field(default_factory=uuid4)
    session_uuid: UUID = Field(default_factory=uuid4)
    trial_uuid: UUID = Field(default_factory=uuid4)
    project_name: str = "Exoskeleton Study"
    principal_investigator: str = "Not specified"
    subject_code: str = "SIM-001"
    subject_group: str | None = "simulated"
    operator: str = "simulator"
    condition_code: str = "WALK_LEVEL"
    condition_name: str = "Level walking"
    condition_level: int | str | None = 2
    condition_parameters: dict[str, JsonValue] = Field(
        default_factory=lambda: {"speed_mps": 0.8, "assist_level": 3, "simulated": True}
    )
    repeat_index: int = Field(default=1, ge=1)
    protocol_version: str = "1.0.0"
    config_version: str = "1.0.0"
    simulation: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator(
        "project_name",
        "principal_investigator",
        "subject_code",
        "operator",
        "condition_code",
        "condition_name",
        "protocol_version",
        "config_version",
    )
    @classmethod
    def non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


class TrialRunResult(OrchestrationModel):
    trial_uuid: UUID
    state: str
    trial_directory: Path
    manifest_path: Path
    duration_s: float = Field(ge=0)
    modality_counts: dict[str, int] = Field(default_factory=dict)
    pulse_event_count: int = Field(default=0, ge=0)
    quality_grade: str

