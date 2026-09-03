"""Subject-level freeze lock shared by Data Studio and Collector.

Data Studio 以「受试者（subject）」为单位上锁/解锁。采集端在向该受试者目录写盘
之前必须检查：上锁即拒绝写入，防止受试者编号未更新导致数据混在一起。

锁是操作者手动加/解、存在即锁定的持久标记（无心跳/租约），与
``activity.py`` 的 dataset_root 级采集活动锁（进程租约 + 心跳）互补。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

SUBJECT_LOCK_SCHEMA = "exo.subject-lock/v1"
_SUBJECT_LOCKS_RELATIVE_DIR = ".exo/subject-locks"
_SUBJECT_CODE_RE = re.compile(r"\d{3}")


class SubjectLockedError(RuntimeError):
    """采集端试图写入一个已上锁的受试者目录时抛出。"""


@dataclass(frozen=True, slots=True)
class SubjectLock:
    subject_code: str
    locked_at_utc: str
    locked_by: str = "Data Studio"
    reason: str | None = None


def _validate_subject_code(subject_code: str) -> str:
    """Normalize and validate the three-digit subject code (path-safety guard)."""
    code = str(subject_code).strip()
    if _SUBJECT_CODE_RE.fullmatch(code) is None:
        raise ValueError("subject_code must contain exactly three digits")
    return code


def _locks_root(dataset_root: str | Path) -> Path:
    return Path(dataset_root).expanduser().resolve() / _SUBJECT_LOCKS_RELATIVE_DIR


def subject_lock_path(dataset_root: str | Path, subject_code: str) -> Path:
    return _locks_root(dataset_root) / f"{_validate_subject_code(subject_code)}.json"


def lock_subject(
    dataset_root: str | Path,
    subject_code: str,
    *,
    locked_by: str = "Data Studio",
    reason: str | None = None,
) -> SubjectLock:
    """Atomically write a subject lock file and return the recorded lock."""
    code = _validate_subject_code(subject_code)
    lock = SubjectLock(
        subject_code=code,
        locked_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        locked_by=locked_by,
        reason=reason,
    )
    path = subject_lock_path(dataset_root, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SUBJECT_LOCK_SCHEMA,
        "subject_code": lock.subject_code,
        "locked_at_utc": lock.locked_at_utc,
        "locked_by": lock.locked_by,
        "reason": lock.reason,
    }
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return lock


def read_subject_lock(dataset_root: str | Path, subject_code: str) -> SubjectLock | None:
    """Return the recorded lock, or ``None`` when absent or unreadable."""
    path = subject_lock_path(dataset_root, subject_code)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return SubjectLock(
        subject_code=str(data.get("subject_code", _validate_subject_code(subject_code))),
        locked_at_utc=str(data.get("locked_at_utc", "")),
        locked_by=str(data.get("locked_by", "Data Studio")),
        reason=(str(data["reason"]) if data.get("reason") is not None else None),
    )


def is_subject_locked(dataset_root: str | Path, subject_code: str) -> bool:
    return subject_lock_path(dataset_root, subject_code).is_file()


def unlock_subject(dataset_root: str | Path, subject_code: str) -> bool:
    """Remove the lock file; returns ``False`` when there was nothing to unlock."""
    path = subject_lock_path(dataset_root, subject_code)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def list_locked_subjects(dataset_root: str | Path) -> list[str]:
    root = _locks_root(dataset_root)
    if not root.is_dir():
        return []
    codes: list[str] = []
    for path in root.glob("*.json"):
        if _SUBJECT_CODE_RE.fullmatch(path.stem):
            codes.append(path.stem)
    return sorted(codes)


__all__ = [
    "SubjectLock",
    "SubjectLockedError",
    "is_subject_locked",
    "list_locked_subjects",
    "lock_subject",
    "read_subject_lock",
    "subject_lock_path",
    "unlock_subject",
]
