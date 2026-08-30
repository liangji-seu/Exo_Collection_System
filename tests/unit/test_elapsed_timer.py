from __future__ import annotations

import time

from PySide6.QtWidgets import QApplication

from exo_collection.apps.collector.elapsed_timer import (
    ElapsedTimerPanel,
    format_elapsed,
)


def test_format_elapsed_zero_pads_and_rolls_over() -> None:
    assert format_elapsed(0) == "00:00:00"
    assert format_elapsed(59) == "00:00:59"
    assert format_elapsed(60) == "00:01:00"
    assert format_elapsed(3661) == "01:01:01"
    assert format_elapsed(25 * 3600 + 30 * 60 + 7) == "25:30:07"


def test_format_elapsed_clamps_negative_to_zero() -> None:
    assert format_elapsed(-3.5) == "00:00:00"


def test_panel_starts_counting_and_reset_rezeroes() -> None:
    app = QApplication.instance() or QApplication(["test-elapsed-timer"])
    panel = ElapsedTimerPanel()

    assert panel._value_label.text() == "00:00:00"
    assert panel._status_label.text() == "自由计时"
    assert panel.elapsed_seconds() >= 0.0

    time.sleep(0.02)
    assert panel.elapsed_seconds() > 0.0

    panel.reset()
    assert panel.elapsed_seconds() < 0.01
    assert panel._status_label.text() == "自由计时"
    panel.close()


def test_start_recording_resets_and_marks_recording() -> None:
    app = QApplication.instance() or QApplication(["test-elapsed-timer-recording"])
    panel = ElapsedTimerPanel()

    time.sleep(0.02)
    panel.start_recording()
    assert panel.elapsed_seconds() < 0.01
    assert panel._status_label.text() == "● 采集中"

    # Leaving the recording accent must not disturb the running clock.
    before = panel.elapsed_seconds()
    panel.set_recording(False)
    assert panel._status_label.text() == "自由计时"
    assert panel.elapsed_seconds() >= before
    panel.close()


def test_panel_rezeros_on_first_show_only() -> None:
    app = QApplication.instance() or QApplication(["test-elapsed-timer-show"])
    panel = ElapsedTimerPanel()

    first_start = panel._started_at
    assert first_start is not None

    panel.show()
    app.processEvents()
    assert panel._ever_shown
    # First show re-zeros the clock: the start timestamp moved forward.
    assert panel._started_at is not None and panel._started_at > first_start

    # Hiding and re-showing must not reset again (that is the reset button's job).
    second_start = panel._started_at
    panel.hide()
    panel.show()
    app.processEvents()
    assert panel._started_at == second_start
    panel.close()
