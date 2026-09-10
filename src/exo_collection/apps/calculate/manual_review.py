"""人工复核标注（丢弃 / 覆盖）的持久化。

run_calculate 负责人工复核：复核人可以把不可用的 session 标记为「丢弃」。
标注写成一个 session 目录下的 ``manual_review.json`` 边车文件，被
run_data_studio / run_process 读取展示；run_process 重新解算成功后会清掉它
（覆盖复核结论），从而允许「覆盖结果」。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REVIEW_FILENAME = "manual_review.json"
DISCARDED = "discarded"


@dataclass(frozen=True, slots=True)
class ManualReview:
    """一条人工复核结论。目前只有 ``discarded`` 一种状态。"""

    status: str
    reviewed_at_utc: str
    reason: str = ""

    @property
    def discarded(self) -> bool:
        return self.status == DISCARDED


def review_path(session_dir: Path) -> Path:
    return Path(session_dir) / REVIEW_FILENAME


def read_manual_review(session_dir: Path) -> ManualReview | None:
    """读取复核结论；无文件 / 损坏 / 缺 status 时返回 ``None``。"""
    try:
        document = json.loads(review_path(session_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or not document.get("status"):
        return None
    return ManualReview(
        status=str(document["status"]),
        reviewed_at_utc=str(document.get("reviewed_at_utc") or ""),
        reason=str(document.get("reason") or ""),
    )


def is_discarded(session_dir: Path) -> bool:
    """该 session 是否被人工标记为「丢弃」。"""
    review = read_manual_review(session_dir)
    return review is not None and review.discarded


def write_discard(session_dir: Path, *, reason: str = "") -> ManualReview:
    """标记该 session 为「丢弃」（幂等：重复写入只更新时间戳与原因）。"""
    review = ManualReview(
        status=DISCARDED,
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        reason=reason,
    )
    path = review_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": review.status,
                "reviewed_at_utc": review.reviewed_at_utc,
                "reason": review.reason,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return review


def clear_manual_review(session_dir: Path) -> bool:
    """清除复核结论（覆盖 / 撤销丢弃）；返回是否确实删除了文件。"""
    try:
        review_path(session_dir).unlink()
        return True
    except OSError:
        return False


__all__ = [
    "DISCARDED",
    "ManualReview",
    "clear_manual_review",
    "is_discarded",
    "read_manual_review",
    "review_path",
    "write_discard",
]
