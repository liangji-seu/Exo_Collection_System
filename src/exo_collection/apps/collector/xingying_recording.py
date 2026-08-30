"""XINGYING capture status panel for the Collector preview workspace.

In hardware mode the motion-capture markers and the six-axis force plate are no
longer streamed through the SDK — XINGYING records them natively as a ``.cap``
file.  This panel replaces the two old preview docks (动捕 Marker / 测力台) with a
single status window that only reflects the XINGYING capture lifecycle:

* 未连接 — dim, static lights.
* 已连接 · 等待采集 — dim, static lights.
* ● 正在录制 .cap — a row of LED-style dots chases back and forth (流水灯).

The widget is driven by :meth:`set_connected` / :meth:`set_recording`, which the
Collector window calls from the XINGYING remote capture lifecycle.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

_DOT_COUNT = 7
_DOT_SIZE = 18
_CHASE_INTERVAL_MS = 120

_LIT_COLOR = "#b02a37"
_DIM_COLOR = "#d1d5db"

_STATUS_STYLES = {
    "recording": "QLabel { color: #b02a37; font-size: 18px; font-weight: 700; }",
    "connected": "QLabel { color: #374151; font-size: 18px; font-weight: 600; }",
    "idle": "QLabel { color: #6b7280; font-size: 18px; font-weight: 600; }",
}


def _dot_style(color: str) -> str:
    return (
        f"QLabel {{ background-color: {color}; border-radius: {_DOT_SIZE // 2}px; "
        f"min-width: {_DOT_SIZE}px; max-width: {_DOT_SIZE}px; "
        f"min-height: {_DOT_SIZE}px; max-height: {_DOT_SIZE}px; }}"
    )


class XingYingRecordingPanel(QWidget):
    """A dockable status widget showing whether XINGYING is recording."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recording = False
        self._connected = False
        self._position = 0
        self._direction = 1

        self._status_label = QLabel("未连接", self)
        self._status_label.setObjectName("xingying_status_text")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._dots: list[QLabel] = []
        dot_row = QHBoxLayout()
        dot_row.setSpacing(8)
        dot_row.addStretch(1)
        for index in range(_DOT_COUNT):
            dot = QLabel(self)
            dot.setObjectName(f"xingying_dot_{index + 1}")
            dot.setStyleSheet(_dot_style(_DIM_COLOR))
            dot_row.addWidget(dot)
            self._dots.append(dot)
        dot_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)
        layout.addStretch(1)
        layout.addWidget(self._status_label)
        layout.addLayout(dot_row)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(_CHASE_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

        self._apply_mode()

    # ── public API ───────────────────────────────────────────────────────

    def set_recording(self, active: bool) -> None:
        """Start/stop the running-light animation (XINGYING is capturing)."""
        self._recording = bool(active)
        if active:
            self._position = 0
            self._direction = 1
            self._timer.start()
        else:
            self._timer.stop()
        self._apply_mode()

    def set_connected(self, connected: bool) -> None:
        """Reflect the XINGYING remote capture link state."""
        self._connected = bool(connected)
        self._apply_mode()

    @property
    def recording(self) -> bool:
        return self._recording

    # ── rendering ────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._position += self._direction
        if self._position >= _DOT_COUNT - 1:
            self._direction = -1
        elif self._position <= 0:
            self._direction = 1
        self._render_dots()

    def _apply_mode(self) -> None:
        if self._recording:
            self._status_label.setText("● 正在录制 .cap")
            self._status_label.setStyleSheet(_STATUS_STYLES["recording"])
        elif self._connected:
            self._status_label.setText("已连接 · 等待采集")
            self._status_label.setStyleSheet(_STATUS_STYLES["connected"])
        else:
            self._status_label.setText("未连接")
            self._status_label.setStyleSheet(_STATUS_STYLES["idle"])
        self._render_dots()

    def _render_dots(self) -> None:
        for index, dot in enumerate(self._dots):
            lit = self._recording and index == self._position
            dot.setStyleSheet(_dot_style(_LIT_COLOR if lit else _DIM_COLOR))


__all__ = ["XingYingRecordingPanel"]
