from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from exo_collection.domain.xingying_trigger import (
    XingYingTriggerEvent,
    XingYingTriggerKind,
    load_xingying_trigger_events,
)


def _event(sequence: int, host_monotonic_ns: int) -> XingYingTriggerEvent:
    kind = (
        XingYingTriggerKind.CAPTURE_START
        if sequence % 2 == 0
        else XingYingTriggerKind.CAPTURE_STOP
    )
    return XingYingTriggerEvent(
        trial_uuid=uuid4(),
        sequence=sequence,
        kind=kind,
        capture_name="trial_01",
        database_path="C:/xing/data",
        notes="note",
        description="desc",
        delay="0",
        timecode="00:00:00:00",
        packet_id="0",
        host_monotonic_ns=host_monotonic_ns,
        host_utc_ns=1_800_000_000_000_000_000 + host_monotonic_ns,
    )


def test_trigger_event_rejects_unknown_fields_and_invalid_kind() -> None:
    event = _event(0, 100)
    with pytest.raises(ValidationError):
        XingYingTriggerEvent.model_validate(
            {**event.model_dump(mode="json"), "kind": "bogus", "extra": True}
        )


def test_trigger_event_requires_nonempty_capture_name() -> None:
    event = _event(0, 100)
    with pytest.raises(ValidationError, match="capture_name"):
        XingYingTriggerEvent.model_validate(
            {**event.model_dump(mode="json"), "capture_name": ""}
        )


def test_load_trigger_events_validates_sequence_and_monotonic_time(
    tmp_path: Path,
) -> None:
    trial_uuid = uuid4()
    events = [
        _event(0, 100).model_copy(update={"trial_uuid": trial_uuid}),
        _event(1, 200).model_copy(update={"trial_uuid": trial_uuid}),
    ]
    path = tmp_path / "xingying_trigger.jsonl"
    path.write_text(
        "".join(
            json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )

    loaded = load_xingying_trigger_events(path)
    assert loaded == tuple(events)
    assert [event.kind.value for event in loaded] == [
        "capture_start",
        "capture_stop",
    ]

    path.write_text(
        "\n".join(
            (
                json.dumps(events[0].model_dump(mode="json"), ensure_ascii=False),
                json.dumps(
                    events[1].model_copy(
                        update={"sequence": 2, "host_monotonic_ns": 50}
                    ).model_dump(mode="json"),
                    ensure_ascii=False,
                ),
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sequence"):
        load_xingying_trigger_events(path)


def test_load_trigger_events_rejects_non_monotonic_timestamps(
    tmp_path: Path,
) -> None:
    trial_uuid = uuid4()
    events = [
        _event(0, 200).model_copy(update={"trial_uuid": trial_uuid}),
        _event(1, 100).model_copy(update={"trial_uuid": trial_uuid}),
    ]
    path = tmp_path / "xingying_trigger.jsonl"
    path.write_text(
        "".join(
            json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="monotonic"):
        load_xingying_trigger_events(path)
