from __future__ import annotations

from pathlib import Path

from exo_collection.apps.calculate.manual_review import (
    clear_manual_review,
    is_discarded,
    read_manual_review,
    write_discard,
)


def test_write_and_read_discard(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    assert not is_discarded(session_dir)
    assert read_manual_review(session_dir) is None

    write_discard(session_dir, reason="bad sync")
    assert is_discarded(session_dir)
    review = read_manual_review(session_dir)
    assert review is not None
    assert review.discarded
    assert review.reason == "bad sync"
    assert review.reviewed_at_utc


def test_clear_discard(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    write_discard(session_dir)
    assert is_discarded(session_dir)
    assert clear_manual_review(session_dir)
    assert not is_discarded(session_dir)
    assert read_manual_review(session_dir) is None


def test_clear_returns_false_when_no_marker(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    assert not clear_manual_review(session_dir)


def test_read_corrupt_marker_returns_none(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "manual_review.json").write_text("{not json", encoding="utf-8")
    assert read_manual_review(session_dir) is None
    assert not is_discarded(session_dir)
