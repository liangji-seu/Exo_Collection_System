from __future__ import annotations

import re

from PySide6.QtWidgets import QApplication

from exo_collection.apps.collector.sync_filename import SyncFilenameBar
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
    assert not bar.copy_txt_button.isEnabled()
    bar.close()


def test_set_filename_fills_and_enables() -> None:
    app = _app()
    bar = SyncFilenameBar()
    bar.set_filename("001_STAND_r1_a1b2c3d4")
    assert bar.filename() == "001_STAND_r1_a1b2c3d4"
    assert bar.line_edit.text() == "001_STAND_r1_a1b2c3d4"
    assert bar.copy_stem_button.isEnabled()
    assert bar.copy_txt_button.isEnabled()
    bar.close()


def test_set_filename_none_clears_again() -> None:
    app = _app()
    bar = SyncFilenameBar()
    bar.set_filename("001_STAND_r1_a1b2c3d4")
    bar.set_filename(None)
    assert bar.filename() is None
    assert bar.line_edit.text() == ""
    assert not bar.copy_stem_button.isEnabled()
    assert not bar.copy_txt_button.isEnabled()
    bar.close()


def test_copy_stem_and_txt_write_clipboard() -> None:
    app = _app()
    bar = SyncFilenameBar()
    bar.set_filename("001_STAND_r1_a1b2c3d4")

    bar.copy_stem_button.click()
    assert app.clipboard().text() == "001_STAND_r1_a1b2c3d4"

    bar.copy_txt_button.click()
    assert app.clipboard().text() == "001_STAND_r1_a1b2c3d4.txt"
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
