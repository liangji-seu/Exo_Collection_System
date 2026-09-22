from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from exo_collection.domain.prompt_annotations import (
    PROMPT_ADDED_LABEL_FILENAME,
    PROMPT_ANNOTATION_FILENAME,
    AddedPromptLabelEvent,
    PromptName,
    added_label_path,
    annotation_path,
    load_added_prompt_labels,
    load_prompt_annotations,
    write_added_prompt_labels,
    write_prompt_annotations,
)
from exo_collection.domain.prompt_labels import PromptLabelSource


def test_prompt_name_values_and_display_names() -> None:
    assert PromptName.START.value == "start"
    assert PromptName.END.value == "end"
    assert PromptName.NAN.value == "nan"
    assert PromptName.START.display_name == "开始"
    assert PromptName.END.display_name == "结束"
    assert PromptName.NAN.display_name == "无效"

    assert PromptName.INTENT_FORWARD_START.value == "intent_forward_start"
    assert PromptName.INTENT_FORWARD_END.value == "intent_forward_end"
    assert PromptName.INTENT_BACKWARD_START.value == "intent_backward_start"
    assert PromptName.INTENT_BACKWARD_END.value == "intent_backward_end"
    assert PromptName.INTENT_LATERAL_START.value == "intent_lateral_start"
    assert PromptName.INTENT_LATERAL_END.value == "intent_lateral_end"
    assert PromptName.INTENT_FORWARD_START.display_name == "意图前-start"
    assert PromptName.INTENT_FORWARD_END.display_name == "意图前-end"
    assert PromptName.INTENT_BACKWARD_START.display_name == "意图后-start"
    assert PromptName.INTENT_BACKWARD_END.display_name == "意图后-end"
    assert PromptName.INTENT_LATERAL_START.display_name == "意图外-start"
    assert PromptName.INTENT_LATERAL_END.display_name == "意图外-end"

    assert PromptName.ACTIVATE_START.value == "activate_start"
    assert PromptName.ACTIVATE_END.value == "activate_end"
    assert PromptName.RELEASE_START.value == "release_start"
    assert PromptName.RELEASE_END.value == "release_end"
    assert PromptName.ACTIVATE_START.display_name == "激活start"
    assert PromptName.ACTIVATE_END.display_name == "激活end"
    assert PromptName.RELEASE_START.display_name == "释放start"
    assert PromptName.RELEASE_END.display_name == "释放end"


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    trial_uuid = uuid4()
    write_prompt_annotations(
        tmp_path,
        trial_uuid,
        {0: PromptName.START, 2: PromptName.NAN, 1: PromptName.END},
    )
    path = annotation_path(tmp_path)
    assert path.name == PROMPT_ANNOTATION_FILENAME
    assert load_prompt_annotations(tmp_path) == {
        0: PromptName.START,
        1: PromptName.END,
        2: PromptName.NAN,
    }


def test_load_returns_empty_when_missing_or_corrupt(tmp_path: Path) -> None:
    assert load_prompt_annotations(tmp_path) == {}
    (tmp_path / PROMPT_ANNOTATION_FILENAME).write_text(
        "{ not valid json", encoding="utf-8"
    )
    assert load_prompt_annotations(tmp_path) == {}
    (tmp_path / PROMPT_ANNOTATION_FILENAME).write_text(
        json.dumps({"schema_version": "1.0.0", "annotations": []}),
        encoding="utf-8",
    )
    # 缺 trial_uuid 无法通过校验，视为损坏返回空。
    assert load_prompt_annotations(tmp_path) == {}


def test_empty_mapping_deletes_sidecar(tmp_path: Path) -> None:
    write_prompt_annotations(tmp_path, uuid4(), {0: PromptName.START})
    assert annotation_path(tmp_path).is_file()
    write_prompt_annotations(tmp_path, uuid4(), {})
    assert not annotation_path(tmp_path).exists()


def test_duplicate_sequence_raises(tmp_path: Path) -> None:
    document = {
        "schema_version": "1.0.0",
        "trial_uuid": str(uuid4()),
        "annotations": [
            {"sequence": 1, "name": "start"},
            {"sequence": 1, "name": "end"},
        ],
    }
    (tmp_path / PROMPT_ANNOTATION_FILENAME).write_text(
        json.dumps(document, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="sequence"):
        load_prompt_annotations(tmp_path)


def test_added_labels_round_trip(tmp_path: Path) -> None:
    write_added_prompt_labels(
        tmp_path,
        uuid4(),
        [
            AddedPromptLabelEvent(time_s=3.0, source=PromptLabelSource.OPERATOR),
            AddedPromptLabelEvent(
                time_s=1.0, source=PromptLabelSource.SUBJECT, name=PromptName.START
            ),
        ],
    )
    assert added_label_path(tmp_path).name == PROMPT_ADDED_LABEL_FILENAME
    loaded = load_added_prompt_labels(tmp_path)
    assert [event.time_s for event in loaded] == [1.0, 3.0]  # 写入时按时间排序
    assert loaded[0].source is PromptLabelSource.SUBJECT
    assert loaded[0].name == PromptName.START
    assert loaded[1].source is PromptLabelSource.OPERATOR
    assert loaded[1].name is None


def test_added_labels_empty_when_missing_or_corrupt(tmp_path: Path) -> None:
    assert load_added_prompt_labels(tmp_path) == ()
    (tmp_path / PROMPT_ADDED_LABEL_FILENAME).write_text(
        "{ not valid json", encoding="utf-8"
    )
    assert load_added_prompt_labels(tmp_path) == ()


def test_added_labels_delete_when_empty(tmp_path: Path) -> None:
    write_added_prompt_labels(
        tmp_path,
        uuid4(),
        [AddedPromptLabelEvent(time_s=1.0, source=PromptLabelSource.BUTTON)],
    )
    assert added_label_path(tmp_path).is_file()
    write_added_prompt_labels(tmp_path, uuid4(), [])
    assert not added_label_path(tmp_path).exists()
