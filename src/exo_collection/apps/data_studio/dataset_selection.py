"""训练/测试集「接收」标注的持久化。

run_data_studio 负责人工筛选：研究员可以把质量合格的 session 标记为「接收」，
表示它属于训练/测试数据集。标注写成一个 session 目录下的 ``dataset_selection.json``
边车文件，与 ``manual_review.json``（丢弃）并列、互不影响；被 ``run_data_studio``
的「训练测试集视角」读取，只展示已接收的 session。

与丢弃标记的关系（见用户约定）：被丢弃的 session 不可再接收；丢弃会同时清掉接收标记。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DATASET_FILENAME = "dataset_selection.json"
ACCEPTED = "accepted"


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    """一条数据集接收结论。目前只有 ``accepted`` 一种状态。"""

    status: str
    accepted_at_utc: str
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED


def selection_path(session_dir: Path) -> Path:
    return Path(session_dir) / DATASET_FILENAME


def read_dataset_selection(session_dir: Path) -> DatasetSelection | None:
    """读取接收结论；无文件 / 损坏 / 缺 status 时返回 ``None``。"""
    try:
        document = json.loads(selection_path(session_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or not document.get("status"):
        return None
    return DatasetSelection(
        status=str(document["status"]),
        accepted_at_utc=str(document.get("accepted_at_utc") or ""),
        reason=str(document.get("reason") or ""),
    )


def is_accepted(session_dir: Path) -> bool:
    """该 session 是否被标记为「接收」（属于训练/测试集）。"""
    selection = read_dataset_selection(session_dir)
    return selection is not None and selection.accepted


def write_accepted(session_dir: Path, *, reason: str = "") -> DatasetSelection:
    """标记该 session 为「接收」（幂等：重复写入只更新时间戳与原因）。"""
    selection = DatasetSelection(
        status=ACCEPTED,
        accepted_at_utc=datetime.now(timezone.utc).isoformat(),
        reason=reason,
    )
    path = selection_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": selection.status,
                "accepted_at_utc": selection.accepted_at_utc,
                "reason": selection.reason,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return selection


def clear_dataset_selection(session_dir: Path) -> bool:
    """清除接收结论（移出训练/测试集）；返回是否确实删除了文件。"""
    try:
        selection_path(session_dir).unlink()
        return True
    except OSError:
        return False


__all__ = [
    "ACCEPTED",
    "DATASET_FILENAME",
    "DatasetSelection",
    "clear_dataset_selection",
    "is_accepted",
    "read_dataset_selection",
    "selection_path",
    "write_accepted",
]
