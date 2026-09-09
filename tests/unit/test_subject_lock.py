from __future__ import annotations

import json
from pathlib import Path

import pytest

from exo_collection.storage.subject_lock import (
    DAY_LOCK_SCHEMA,
    SUBJECT_LOCK_SCHEMA,
    DayLock,
    SubjectLockedError,
    day_lock_path,
    is_day_locked,
    is_subject_locked,
    list_locked_days,
    list_locked_subjects,
    lock_day,
    lock_subject,
    next_expected_day,
    read_day_lock,
    read_subject_lock,
    subject_lock_path,
    unlock_day,
    unlock_subject,
)


def test_lock_path_lives_under_exo_subject_locks(tmp_path: Path) -> None:
    assert subject_lock_path(tmp_path, "001") == (
        tmp_path / ".exo" / "subject-locks" / "001.json"
    )


def test_lock_and_unlock_roundtrip(tmp_path: Path) -> None:
    assert not is_subject_locked(tmp_path, "001")
    lock = lock_subject(tmp_path, "001")
    assert lock.subject_code == "001"
    assert is_subject_locked(tmp_path, "001")
    assert subject_lock_path(tmp_path, "001").is_file()
    assert unlock_subject(tmp_path, "001") is True
    assert not is_subject_locked(tmp_path, "001")


def test_lock_file_has_schema_and_subject(tmp_path: Path) -> None:
    lock_subject(tmp_path, "007", reason="冻结")
    data = json.loads(subject_lock_path(tmp_path, "007").read_text(encoding="utf-8"))
    assert data["schema"] == SUBJECT_LOCK_SCHEMA
    assert data["subject_code"] == "007"
    assert data["reason"] == "冻结"


def test_read_subject_lock_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_subject_lock(tmp_path, "042") is None


def test_read_subject_lock_roundtrip(tmp_path: Path) -> None:
    lock_subject(tmp_path, "042", locked_by="ops", reason="已完成")
    lock = read_subject_lock(tmp_path, "042")
    assert lock is not None
    assert lock.subject_code == "042"
    assert lock.locked_by == "ops"
    assert lock.reason == "已完成"
    assert lock.locked_at_utc


def test_list_locked_subjects_sorted(tmp_path: Path) -> None:
    assert list_locked_subjects(tmp_path) == []
    lock_subject(tmp_path, "010")
    lock_subject(tmp_path, "002")
    lock_subject(tmp_path, "005")
    assert list_locked_subjects(tmp_path) == ["002", "005", "010"]


def test_list_locked_subjects_ignores_non_subject_files(tmp_path: Path) -> None:
    lock_subject(tmp_path, "001")
    root = subject_lock_path(tmp_path, "001").parent
    (root / "notes.json").write_text("{}", encoding="utf-8")
    assert list_locked_subjects(tmp_path) == ["001"]


def test_invalid_subject_code_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        lock_subject(tmp_path, "1")
    with pytest.raises(ValueError):
        lock_subject(tmp_path, "abcd")
    with pytest.raises(ValueError):
        lock_subject(tmp_path, "0000")
    with pytest.raises(ValueError):
        is_subject_locked(tmp_path, "..")


def test_unlock_absent_returns_false(tmp_path: Path) -> None:
    assert unlock_subject(tmp_path, "099") is False


def test_locks_are_isolated_per_subject(tmp_path: Path) -> None:
    lock_subject(tmp_path, "001")
    assert is_subject_locked(tmp_path, "001")
    assert not is_subject_locked(tmp_path, "002")


def test_subject_locked_error_is_runtime_error() -> None:
    assert issubclass(SubjectLockedError, RuntimeError)


# ── Per-day locks ──────────────────────────────────────────────────────


def test_day_lock_path_lives_under_exo_subject_locks(tmp_path: Path) -> None:
    assert day_lock_path(tmp_path, "001", 2) == (
        tmp_path / ".exo" / "subject-locks" / "001.d2.json"
    )


def test_day_lock_and_unlock_roundtrip(tmp_path: Path) -> None:
    assert not is_day_locked(tmp_path, "001", 1)
    lock = lock_day(tmp_path, "001", 1)
    assert lock.subject_code == "001"
    assert lock.day == 1
    assert is_day_locked(tmp_path, "001", 1)
    assert day_lock_path(tmp_path, "001", 1).is_file()
    assert unlock_day(tmp_path, "001", 1) is True
    assert not is_day_locked(tmp_path, "001", 1)


def test_day_lock_file_has_schema_subject_and_day(tmp_path: Path) -> None:
    lock_day(tmp_path, "007", 3, reason="已完成")
    data = json.loads(day_lock_path(tmp_path, "007", 3).read_text(encoding="utf-8"))
    assert data["schema"] == DAY_LOCK_SCHEMA
    assert data["subject_code"] == "007"
    assert data["day"] == 3
    assert data["reason"] == "已完成"


def test_read_day_lock_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_day_lock(tmp_path, "042", 1) is None


def test_read_day_lock_roundtrip(tmp_path: Path) -> None:
    lock_day(tmp_path, "042", 2, locked_by="ops", reason="冻结")
    lock = read_day_lock(tmp_path, "042", 2)
    assert lock is not None
    assert lock.subject_code == "042"
    assert lock.day == 2
    assert lock.locked_by == "ops"
    assert lock.reason == "冻结"
    assert lock.locked_at_utc


def test_unlock_absent_day_returns_false(tmp_path: Path) -> None:
    assert unlock_day(tmp_path, "099", 5) is False


def test_list_locked_days_sorted_and_isolated(tmp_path: Path) -> None:
    assert list_locked_days(tmp_path, "001") == []
    lock_day(tmp_path, "001", 3)
    lock_day(tmp_path, "001", 1)
    lock_day(tmp_path, "001", 10)
    lock_day(tmp_path, "002", 2)  # different subject, must not appear
    assert list_locked_days(tmp_path, "001") == [1, 3, 10]
    assert list_locked_days(tmp_path, "002") == [2]


def test_list_locked_subjects_ignores_day_lock_files(tmp_path: Path) -> None:
    lock_subject(tmp_path, "001")
    lock_day(tmp_path, "001", 1)
    lock_day(tmp_path, "002", 2)
    assert list_locked_subjects(tmp_path) == ["001"]


def test_next_expected_day_scan_from_one(tmp_path: Path) -> None:
    assert next_expected_day(tmp_path, "001") == 1
    lock_day(tmp_path, "001", 1)
    assert next_expected_day(tmp_path, "001") == 2
    lock_day(tmp_path, "001", 2)
    assert next_expected_day(tmp_path, "001") == 3


def test_next_expected_day_skips_locked_prefix_only(tmp_path: Path) -> None:
    # Day 2 locked out-of-order: the first *unlocked* day is still 1.
    lock_day(tmp_path, "001", 2)
    assert next_expected_day(tmp_path, "001") == 1


def test_invalid_day_rejected(tmp_path: Path) -> None:
    for bad in (0, -1, 1.5, "1", True):
        with pytest.raises(ValueError):
            lock_day(tmp_path, "001", bad)
        with pytest.raises(ValueError):
            is_day_locked(tmp_path, "001", bad)
    with pytest.raises(ValueError):
        day_lock_path(tmp_path, "001", 0)


def test_invalid_subject_rejected_for_day_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        lock_day(tmp_path, "12", 1)
    with pytest.raises(ValueError):
        is_day_locked(tmp_path, "abcd", 1)
