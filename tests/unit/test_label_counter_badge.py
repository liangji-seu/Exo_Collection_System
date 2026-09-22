from __future__ import annotations

from PySide6.QtWidgets import QApplication

from exo_collection.apps.collector.label_counter_badge import LabelCounterBadge


def test_set_count_updates_display_and_clamps() -> None:
    app = QApplication.instance() or QApplication(["test-label-badge-count"])
    badge = LabelCounterBadge()

    assert badge._count_label.text() == "0"
    badge.set_count(7)
    assert badge._count_label.text() == "7"
    # 负值钳制为 0，避免显示负数。
    badge.set_count(-3)
    assert badge._count_label.text() == "0"
    badge.close()


def test_flash_switches_style_then_reverts_on_timeout() -> None:
    app = QApplication.instance() or QApplication(["test-label-badge-flash"])
    badge = LabelCounterBadge()

    base = badge.styleSheet()
    badge.flash()
    assert badge.styleSheet() != base

    # 触发 flash 计时器超时，应恢复常态样式。
    badge._flash_timer.timeout.emit()
    assert badge.styleSheet() == base
    badge.close()
