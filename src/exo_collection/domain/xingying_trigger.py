"""XINGYING remote-trigger events received on port 7061 (host-clocked).

XINGYING broadcasts a ``CaptureStart``/``CaptureStop`` notification back on its
「捕获--触发」port when a recording actually begins or ends.  The Collector
listens for those notifications and records each one against the host monotonic
clock, mirroring the prompt-label pattern, so the resulting ``.cap`` can later be
aligned with the ultrasound clock.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Iterable, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


XINGYING_TRIGGER_SCHEMA_VERSION = "1.0.0"
XINGYING_TRIGGER_RELATIVE_PATH = "raw/xingying_trigger.jsonl"


class XingYingTriggerKind(StrEnum):
    CAPTURE_START = "capture_start"
    CAPTURE_STOP = "capture_stop"


class XingYingTriggerEvent(BaseModel):
    """One immutable, host-clocked XINGYING start/stop notification."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = XINGYING_TRIGGER_SCHEMA_VERSION
    trial_uuid: UUID
    sequence: int = Field(ge=0)
    kind: XingYingTriggerKind
    capture_name: str = Field(min_length=1)
    database_path: str = ""
    notes: str = ""
    description: str = ""
    delay: str = ""
    timecode: str = ""
    packet_id: str = ""
    host_monotonic_ns: int = Field(ge=0)
    host_utc_ns: int = Field(ge=0)


def load_xingying_trigger_events(
    path: str | Path,
) -> tuple[XingYingTriggerEvent, ...]:
    """Read and validate a finalized XINGYING-trigger NDJSON artifact."""

    source = Path(path)
    events: list[XingYingTriggerEvent] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = XingYingTriggerEvent.model_validate_json(line)
        except Exception as exc:
            raise ValueError(
                f"invalid xingying-trigger event at line {line_number}: {exc}"
            ) from exc
        events.append(event)
    _validate_event_sequence(events)
    return tuple(events)


def _validate_event_sequence(events: Iterable[XingYingTriggerEvent]) -> None:
    materialized = tuple(events)
    sequences = [event.sequence for event in materialized]
    if sequences != list(range(len(materialized))):
        raise ValueError("xingying-trigger sequence must be contiguous from zero")
    timestamps = [event.host_monotonic_ns for event in materialized]
    if timestamps != sorted(timestamps):
        raise ValueError("xingying-trigger host timestamps must be monotonic")


__all__ = [
    "XINGYING_TRIGGER_RELATIVE_PATH",
    "XINGYING_TRIGGER_SCHEMA_VERSION",
    "XingYingTriggerEvent",
    "XingYingTriggerKind",
    "load_xingying_trigger_events",
]
