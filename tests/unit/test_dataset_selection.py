from __future__ import annotations

from pathlib import Path

from exo_collection.apps.data_studio.dataset_selection import (
    clear_dataset_selection,
    is_accepted,
    read_dataset_selection,
    write_accepted,
)


def test_write_and_read_accepted(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    assert not is_accepted(session_dir)
    assert read_dataset_selection(session_dir) is None

    write_accepted(session_dir, reason="good quality")
    assert is_accepted(session_dir)
    selection = read_dataset_selection(session_dir)
    assert selection is not None
    assert selection.accepted
    assert selection.reason == "good quality"
    assert selection.accepted_at_utc


def test_clear_accepted(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    write_accepted(session_dir)
    assert is_accepted(session_dir)
    assert clear_dataset_selection(session_dir)
    assert not is_accepted(session_dir)
    assert read_dataset_selection(session_dir) is None


def test_clear_returns_false_when_no_marker(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    assert not clear_dataset_selection(session_dir)


def test_read_corrupt_marker_returns_none(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "dataset_selection.json").write_text("{not json", encoding="utf-8")
    assert read_dataset_selection(session_dir) is None
    assert not is_accepted(session_dir)
