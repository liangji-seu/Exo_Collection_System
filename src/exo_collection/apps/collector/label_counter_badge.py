"""Floating translucent label-counter badge for the Collector.

When the operator connects the USB marker button, the main window shows this
small frameless, always-on-top tool window.  It displays how many button labels
have been recorded in the current Trial and flashes on every physical press.

It has no title bar, so it is draggable by its whole surface (see the mouse
event overrides).  The card is translucent and rounded, and the outer widget is
translucent so the rounded corners render cleanly over whatever is behind it.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

# 常态：白色数字 + 灰色小字说明，半透明深色圆角卡片。
_BASE_STYLE = (
    "#label_counter_badge { background: transparent; }\n"
    "QFrame#badge_card { background: rgba(15, 23, 21, 0.85); border-radius: 14px; }\n"
    "QLabel#badge_count { color: #f8fafc; font-size: 42px; font-weight: 700; background: transparent; }\n"
    "QLabel#badge_caption { color: #94a3b8; font-size: 12px; font-weight: 600; background: transparent; }"
)
# 按下特效：数字与说明短暂变亮绿色。
_FLASH_STYLE = (
    "#label_counter_badge { background: transparent; }\n"
    "QFrame#badge_card { background: rgba(15, 23, 21, 0.85); border-radius: 14px; }\n"
    "QLabel#badge_count { color: #4ade80; font-size: 42px; font-weight: 700; background: transparent; }\n"
    "QLabel#badge_caption { color: #a7f3d0; font-size: 12px; font-weight: 600; background: transparent; }"
)
_FLASH_MS = 180


class LabelCounterBadge(QWidget):
    """A draggable, translucent, always-on-top counter for button labels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("label_counter_badge")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._card = QFrame(self)
        self._card.setObjectName("badge_card")
        self._card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # 卡片只负责展示，鼠标事件穿透到 badge，由 badge 统一处理拖拽。
        self._card.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._count_label = QLabel("0", self._card)
        self._count_label.setObjectName("badge_count")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._caption_label = QLabel("按钮标签", self._card)
        self._caption_label.setObjectName("badge_caption")
        self._caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(24, 14, 24, 14)
        card_layout.setSpacing(0)
        card_layout.addWidget(self._count_label)
        card_layout.addWidget(self._caption_label)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._card)

        self._count = 0
        self._drag_offset: QPoint | None = None

        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._apply_style)

        self._apply_style()

    def set_count(self, count: int) -> None:
        """Update the displayed button-label count (clamped to non-negative)."""
        self._count = max(0, int(count))
        self._count_label.setText(str(self._count))

    def flash(self) -> None:
        """Briefly highlight the badge to signal a physical button press."""
        self.setStyleSheet(_FLASH_STYLE)
        self._flash_timer.start(_FLASH_MS)

    def _apply_style(self) -> None:
        self.setStyleSheet(_BASE_STYLE)

    # ── dragging (frameless → drag by surface) ───────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


__all__ = ["LabelCounterBadge"]
