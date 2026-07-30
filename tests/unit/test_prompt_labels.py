from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from exo_collection.domain.prompt_labels import (
    PromptLabelEvent,
    PromptLabelSource,
    load_prompt_label_events,
)


def _event(sequence: int, host_monotonic_ns: int) -> PromptLabelEvent:
    source = (
        PromptLabelSource.SUBJECT
        if sequence % 2 == 0
        else PromptLabelSource.OPERATOR
    )
    return PromptLabelEvent(
        trial_uuid=uuid4(),
        sequence=sequence,
        source=source,
        label=source.display_name,
        key=source.key_text,
        host_monotonic_ns=host_monotonic_ns,
        host_utc_ns=1_800_000_000_000_000_000 + host_monotonic_ns,
    )


def test_prompt_label_event_rejects_unknown_fields_and_invalid_key() -> None:
    event = _event(0, 100)
    with pytest.raises(ValidationError):
        PromptLabelEvent.model_validate(
            {**event.model_dump(mode="json"), "key": "<<", "extra": True}
        )


def test_load_prompt_labels_validates_sequence_and_monotonic_time(
    tmp_path: Path,
) -> None:
    trial_uuid = uuid4()
    events = [
        _event(0, 100).model_copy(update={"trial_uuid": trial_uuid}),
        _event(1, 200).model_copy(update={"trial_uuid": trial_uuid}),
    ]
    path = tmp_path / "prompt_labels.jsonl"
    path.write_text(
        "".join(
            json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )

    loaded = load_prompt_label_events(path)
    assert loaded == tuple(events)
    assert [event.label for event in loaded] == ["受试者标签", "工作人员标签"]

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
        load_prompt_label_events(path)
