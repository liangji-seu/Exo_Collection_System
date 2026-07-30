"""Human prompt-label events captured from the Collector keyboard buttons."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Iterable, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


PROMPT_LABEL_SCHEMA_VERSION = "1.0.0"
PROMPT_LABEL_RELATIVE_PATH = "raw/prompt_labels.jsonl"


class PromptLabelSource(StrEnum):
    SUBJECT = "SUBJECT"
    OPERATOR = "OPERATOR"

    @property
    def display_name(self) -> str:
        return {
            PromptLabelSource.SUBJECT: "受试者标签",
            PromptLabelSource.OPERATOR: "工作人员标签",
        }[self]

    @property
    def key_text(self) -> str:
        return {
            PromptLabelSource.SUBJECT: "<",
            PromptLabelSource.OPERATOR: ">",
        }[self]


class PromptLabelEvent(BaseModel):
    """One immutable, host-clocked human label event."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = PROMPT_LABEL_SCHEMA_VERSION
    trial_uuid: UUID
    sequence: int = Field(ge=0)
    source: PromptLabelSource
    label: str = Field(min_length=1)
    key: str = Field(min_length=1, max_length=1)
    host_monotonic_ns: int = Field(ge=0)
    host_utc_ns: int = Field(ge=0)


def load_prompt_label_events(path: str | Path) -> tuple[PromptLabelEvent, ...]:
    """Read and validate a finalized prompt-label NDJSON artifact."""

    source = Path(path)
    events: list[PromptLabelEvent] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = PromptLabelEvent.model_validate_json(line)
        except Exception as exc:
            raise ValueError(
                f"invalid prompt-label event at line {line_number}: {exc}"
            ) from exc
        events.append(event)
    _validate_event_sequence(events)
    return tuple(events)


def _validate_event_sequence(events: Iterable[PromptLabelEvent]) -> None:
    materialized = tuple(events)
    sequences = [event.sequence for event in materialized]
    if sequences != list(range(len(materialized))):
        raise ValueError("prompt-label sequence must be contiguous from zero")
    timestamps = [event.host_monotonic_ns for event in materialized]
    if timestamps != sorted(timestamps):
        raise ValueError("prompt-label host timestamps must be monotonic")


__all__ = [
    "PROMPT_LABEL_RELATIVE_PATH",
    "PROMPT_LABEL_SCHEMA_VERSION",
    "PromptLabelEvent",
    "PromptLabelSource",
    "load_prompt_label_events",
]
