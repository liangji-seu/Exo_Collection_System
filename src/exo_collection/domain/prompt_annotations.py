"""按键打标事件的语义命名标注（start / end / nan）。

采集端写入的 ``raw/prompt_labels.jsonl`` 是不可变、SHA-256 锚定的原始 Artifact，
其 ``label`` 字段存的是来源显示名（受试者标签 / 工作人员标签 / 按钮标签），不是语义名。
本模块把「某个按键事件是什么特殊事件」的结论写到 session 目录下的一个**边车文件**
``prompt_annotations.json``（同 ``apps/calculate/manual_review.py`` 的边车模式），
按事件 ``sequence`` 关联，绝不改写原始 ``prompt_labels.jsonl``。

下游模型训练侧可直接 ``load_prompt_annotations(session_dir)`` +
``load_prompt_label_events(path)`` 按 ``sequence`` 关联得到「事件 → 语义名」。
"""

from __future__ import annotations

import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from exo_collection.domain.prompt_labels import PromptLabelSource

PROMPT_ANNOTATION_SCHEMA_VERSION = "1.0.0"
PROMPT_ANNOTATION_FILENAME = "prompt_annotations.json"

# 手动补录的按键打标事件（采集时漏打，事后在时间轴当前位置补上）。
PROMPT_ADDED_LABEL_SCHEMA_VERSION = "1.0.0"
PROMPT_ADDED_LABEL_FILENAME = "prompt_added_labels.json"

_log = logging.getLogger(__name__)


class PromptName(StrEnum):
    """一个按键事件被赋予的语义名称；``nan`` 表示无效（后续训练应忽略）。"""

    START = "start"
    END = "end"
    NAN = "nan"

    @property
    def display_name(self) -> str:
        return {
            PromptName.START: "开始",
            PromptName.END: "结束",
            PromptName.NAN: "无效",
        }[self]


class PromptAnnotationItem(BaseModel):
    """单条命名标注：``sequence`` 即 ``PromptLabelEvent.sequence``。"""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    name: PromptName


class PromptAnnotationFile(BaseModel):
    """``prompt_annotations.json`` 边车的完整文档。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = PROMPT_ANNOTATION_SCHEMA_VERSION
    trial_uuid: UUID
    annotations: list[PromptAnnotationItem] = Field(default_factory=list)


def annotation_path(session_dir: str | Path) -> Path:
    """返回该 session 的命名标注边车路径（``session_dir / prompt_annotations.json``）。"""
    return Path(session_dir) / PROMPT_ANNOTATION_FILENAME


def load_prompt_annotations(session_dir: str | Path) -> dict[int, PromptName]:
    """读取命名标注，返回 ``{sequence: PromptName}``。

    缺文件 / 损坏 / 缺必要字段时返回空 dict（标注是可选的编辑产物，不应拖垮回放）；
    重复 ``sequence`` 视为数据损坏，抛 ``ValueError``。
    """
    path = annotation_path(session_dir)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    try:
        parsed = PromptAnnotationFile.model_validate(document)
    except Exception as exc:
        _log.warning("解析命名标注失败 %s: %s", path, exc)
        return {}
    mapping: dict[int, PromptName] = {}
    for item in parsed.annotations:
        if item.sequence in mapping:
            raise ValueError(f"prompt annotation sequence 重复: {item.sequence}")
        mapping[item.sequence] = item.name
    return mapping


def write_prompt_annotations(
    session_dir: str | Path,
    trial_uuid: str | UUID,
    mapping: dict[int, PromptName],
) -> None:
    """把命名标注写回边车文件；``mapping`` 为空时删除边车（等价于无标注）。"""
    path = annotation_path(session_dir)
    if not mapping:
        try:
            path.unlink()
        except OSError:
            pass
        return
    document = PromptAnnotationFile(
        trial_uuid=UUID(str(trial_uuid)),
        annotations=[
            PromptAnnotationItem(sequence=sequence, name=name)
            for sequence, name in sorted(mapping.items())
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class AddedPromptLabelEvent(BaseModel):
    """一条手动补录的按键打标事件（采集时漏打，事后在时间轴当前位置补上）。

    ``time_s`` 是相对 Trial formal_t0 的时间（与 ``PromptLabelPlaybackEvent.time_s``
    同基准）；``source`` 决定显示名与按键；``name`` 语义名直接内联存储（补录事件
    没有原始 ``sequence``，无需走 ``prompt_annotations.json`` 的 sequence 关联）。
    """

    model_config = ConfigDict(extra="forbid")

    time_s: float
    source: PromptLabelSource
    name: PromptName | None = None


class AddedPromptLabelFile(BaseModel):
    """``prompt_added_labels.json`` 边车的完整文档。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = PROMPT_ADDED_LABEL_SCHEMA_VERSION
    trial_uuid: UUID
    events: list[AddedPromptLabelEvent] = Field(default_factory=list)


def added_label_path(session_dir: str | Path) -> Path:
    """返回该 session 的补录打标边车路径（``session_dir / prompt_added_labels.json``）。"""
    return Path(session_dir) / PROMPT_ADDED_LABEL_FILENAME


def load_added_prompt_labels(
    session_dir: str | Path,
) -> tuple[AddedPromptLabelEvent, ...]:
    """读取补录打标事件；缺文件 / 损坏时返回空元组（补录是可选的编辑产物）。"""
    path = added_label_path(session_dir)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    try:
        parsed = AddedPromptLabelFile.model_validate(document)
    except Exception as exc:
        _log.warning("解析补录打标失败 %s: %s", path, exc)
        return ()
    return tuple(parsed.events)


def write_added_prompt_labels(
    session_dir: str | Path,
    trial_uuid: str | UUID,
    events: list[AddedPromptLabelEvent],
) -> None:
    """把补录打标事件写回边车（按 ``time_s`` 排序）；``events`` 为空时删除边车。"""
    path = added_label_path(session_dir)
    if not events:
        try:
            path.unlink()
        except OSError:
            pass
        return
    document = AddedPromptLabelFile(
        trial_uuid=UUID(str(trial_uuid)),
        events=sorted(events, key=lambda event: event.time_s),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


__all__ = [
    "PROMPT_ADDED_LABEL_FILENAME",
    "PROMPT_ADDED_LABEL_SCHEMA_VERSION",
    "PROMPT_ANNOTATION_FILENAME",
    "PROMPT_ANNOTATION_SCHEMA_VERSION",
    "AddedPromptLabelEvent",
    "AddedPromptLabelFile",
    "PromptAnnotationFile",
    "PromptAnnotationItem",
    "PromptName",
    "added_label_path",
    "annotation_path",
    "load_added_prompt_labels",
    "load_prompt_annotations",
    "write_added_prompt_labels",
    "write_prompt_annotations",
]
