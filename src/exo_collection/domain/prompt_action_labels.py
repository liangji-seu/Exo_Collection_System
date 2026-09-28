"""意图测试工况的「动作段」标签边车（一段动作 = 连续 4 个按键事件）。

意图测试（``INTENT_TEST_*``）里，受试者每做一个动作按 4 次键（激活 start / 激活 end /
释放 start / 释放 end）。``prompt_annotations.json`` 只能表达「一个按键事件是什么语义」，
表达不了「这一整段动作是什么动作」（如 轻微屈髋 / 明显伸髋）。本模块把「第几段动作＝什么」
写到 session 目录下的另一个**边车文件** ``prompt_action_labels.json``，按段序号 ``index``
与段内事件 ``sequences`` 关联，同样绝不改写原始 ``raw/prompt_labels.jsonl``。

动作名是受试者/操作员给出的自然语言短语（不限于固定枚举），因此 ``name`` 用自由字符串；
约定 ``"none"`` 表示该段无效（多按/误按），下游训练应忽略——与 ``PromptName.nan`` 同义。

``schema_version`` 1.1.0 起追加可选的 ``role``（主动/被动）与 ``pose``（站姿/坐姿）。
**意图测试必须写这两个字段**：``INTENT_TEST_*`` 的 condition 名、``parameters`` 和
``raw/prompt_labels.jsonl`` 都不含主动/被动与站坐信息，只有操作员的 session 编号能区分，
不写下来就永久丢失（关节工况不依赖它——condition 名里已带 ``ACTIVE``/``PASSIVE``/``STAND``/``SIT``）。
1.1.0 是纯追加，``role``/``pose`` 缺省为 ``None``，读旧的 1.0.0 文件仍然通过。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

PROMPT_ACTION_LABEL_SCHEMA_VERSION = "1.1.0"
PROMPT_ACTION_LABEL_FILENAME = "prompt_action_labels.json"

#: 意图测试的动作角色 / 姿态。缺省 ``None`` = 未知（旧的 1.0.0 文件、或操作员未记录）。
ActionRole = Literal["active", "passive"]
ActionPose = Literal["stand", "sit"]

#: name 取该值表示「这一段无效」（多按/误按），与 ``PromptName.nan`` 同义。
INVALID_ACTION_NAME = "none"

_log = logging.getLogger(__name__)


class PromptActionLabelItem(BaseModel):
    """一段动作：``index`` 从 1 开始的段序号，``sequences`` 是该段覆盖的按键事件序列号。"""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1)
    name: str = Field(min_length=1)
    sequences: list[int] = Field(default_factory=list)


class PromptActionLabelFile(BaseModel):
    """``prompt_action_labels.json`` 边车的完整文档。

    ``role`` / ``pose`` 描述**整个 session** 的协议属性（不是逐段的）：意图测试里
    「站姿主动 session1」与「站姿被动 session3」的动作名可以完全一样，只有这两个字段能区分。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0", "1.1.0"] = PROMPT_ACTION_LABEL_SCHEMA_VERSION
    trial_uuid: UUID
    role: ActionRole | None = None
    pose: ActionPose | None = None
    actions: list[PromptActionLabelItem] = Field(default_factory=list)


def action_label_path(session_dir: str | Path) -> Path:
    """返回该 session 的动作段标签边车路径（``session_dir / prompt_action_labels.json``）。"""
    return Path(session_dir) / PROMPT_ACTION_LABEL_FILENAME


def load_action_label_document(session_dir: str | Path) -> PromptActionLabelFile | None:
    """读取完整边车文档（含 ``role``/``pose``）；缺文件或损坏时返回 ``None``。"""
    path = action_label_path(session_dir)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return PromptActionLabelFile.model_validate(document)
    except Exception as exc:
        _log.warning("解析动作段标签失败 %s: %s", path, exc)
        return None


def load_prompt_action_labels(session_dir: str | Path) -> list[PromptActionLabelItem]:
    """读取动作段标签，按 ``index`` 升序返回。

    缺文件 / 损坏 / 缺必要字段时返回空 list（标签是可选的编辑产物，不应拖垮回放）；
    ``index`` 或 ``sequence`` 重复视为数据损坏，抛 ``ValueError``。
    """
    parsed = load_action_label_document(session_dir)
    if parsed is None:
        return []
    seen_index: set[int] = set()
    seen_sequence: set[int] = set()
    for item in parsed.actions:
        if item.index in seen_index:
            raise ValueError(f"prompt action label index 重复: {item.index}")
        seen_index.add(item.index)
        for sequence in item.sequences:
            if sequence in seen_sequence:
                raise ValueError(f"prompt action label sequence 重复: {sequence}")
            seen_sequence.add(sequence)
    return sorted(parsed.actions, key=lambda item: item.index)


def write_prompt_action_labels(
    session_dir: str | Path,
    trial_uuid: str | UUID,
    actions: list[PromptActionLabelItem],
    role: ActionRole | None = None,
    pose: ActionPose | None = None,
) -> None:
    """把动作段标签写回边车文件；``actions`` 为空时删除边车（等价于无标签）。

    ``role``/``pose`` 描述整个 session 的协议属性（意图测试必填），见
    :class:`PromptActionLabelFile`。
    """
    path = action_label_path(session_dir)
    if not actions:
        try:
            path.unlink()
        except OSError:
            pass
        return
    document = PromptActionLabelFile(
        trial_uuid=UUID(str(trial_uuid)),
        role=role,
        pose=pose,
        actions=sorted(actions, key=lambda item: item.index),
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
