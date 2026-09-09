from __future__ import annotations

import re

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exo_collection.apps.collector.sync_filename import (
    _COPY_CONFIRM_MS,
    SyncFilenameBar,
)
from exo_collection.apps.collector.window import CollectorWindow
from exo_collection.orchestration.models import TrialRunRequest


def _app() -> QApplication:
    return QApplication.instance() or QApplication(["test-sync-filename"])


def test_bar_starts_empty_and_disabled() -> None:
    app = _app()
    bar = SyncFilenameBar()
    assert bar.filename() is None
    assert bar.line_edit.text() == ""
    assert not bar.copy_stem_button.isEnabled()
    assert not bar.copy_toast.isVisible()
    bar.close()


def test_set_filename_fills_and_enables() -> None:
    app = _app()
    bar = SyncFilenameBar()
    bar.set_filename("001_STAND_r1_a1b2c3d4")
    assert bar.filename() == "001_STAND_r1_a1b2c3d4"
    assert bar.line_edit.text() == "001_STAND_r1_a1b2c3d4"
    assert bar.copy_stem_button.isEnabled()
    bar.close()


def test_set_filename_none_clears_again() -> None:
    app = _app()
    bar = SyncFilenameBar()
    bar.set_filename("001_STAND_r1_a1b2c3d4")
    bar.set_filename(None)
    assert bar.filename() is None
    assert bar.line_edit.text() == ""
    assert not bar.copy_stem_button.isEnabled()
    bar.close()


def test_copy_button_writes_clipboard_and_shows_toast() -> None:
    app = _app()
    bar = SyncFilenameBar()
    bar.set_filename("001_STAND_r1_a1b2c3d4")

    bar.copy_stem_button.click()
    assert app.clipboard().text() == "001_STAND_r1_a1b2c3d4"
    assert bar.copy_toast.isVisible()
    assert bar.copy_toast.text() == "复制成功"
    assert bar._copy_toast_timer.isActive()
    assert bar._copy_toast_timer.interval() == _COPY_CONFIRM_MS
    bar.close()


def test_copy_toast_auto_hides_after_one_second() -> None:
    app = _app()
    bar = SyncFilenameBar()
    bar.set_filename("001_STAND_r1_a1b2c3d4")

    bar.copy_stem_button.click()
    assert bar.copy_toast.isVisible()

    QTest.qWait(_COPY_CONFIRM_MS + 150)
    assert not bar.copy_toast.isVisible()
    bar.close()


def test_build_xingying_capture_name_format() -> None:
    request = TrialRunRequest.model_validate(
        {
            "data_root": "C:/data",
            "subject_code": "001",
            "condition_code": "STAND",
            "repeat_index": 2,
        }
    )
    # The method never touches ``self``, so it can be driven unbound.
    name = CollectorWindow._build_xingying_capture_name(None, request)
    short_uuid = str(request.trial_uuid).replace("-", "")[:8]
    assert re.fullmatch(r"[0-9a-f]{8}", short_uuid)
    assert name == f"001_STAND_r2_{short_uuid}"
