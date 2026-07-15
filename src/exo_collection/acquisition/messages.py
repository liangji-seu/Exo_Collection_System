"""JSON-safe control-plane messages shared by UI and collector workers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WorkerEventType(StrEnum):
    STATE = "state"
    PREVIEW = "preview"
    HEALTH = "health"
    METRIC = "metric"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: WorkerEventType
    trial_uuid: str | None = None
    modality: str | None = None
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

