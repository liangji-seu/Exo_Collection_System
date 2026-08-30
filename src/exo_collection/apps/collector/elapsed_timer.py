"""Elapsed-time stopwatch panel for the Collector preview workspace.

The preview workspace hosts per-modality docks.  This module adds one more
dock, the **timer**, which is not bound to a modality but to the trial
lifecycle:

* While idle it is a free-running stopwatch — it starts counting the first time
  it is shown (i.e. added via the "＋ 添加窗口" menu) and keeps counting until
  the user resets it.
* When the Collector worker enters the ``RECORDING`` state, the window calls
  :meth:`ElapsedTimerPanel.start_recording`, which re-zeros the clock so the
  operator can read exactly how long the current trial has been capturing.

Elapsed time is derived from a single monotonic start timestamp, so the display
never accumulates drift from timer jitter.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


def format_elapsed(seconds: float) -> str:
    """Render a duration as zero-padded ``[时]:[分]:[秒]`` (``HH:MM:SS``)."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


_IDLE_VALUE_STYLE = (
    "QLabel#elapsed_value { color: #26332f; font-size: 42px; font-weight: 700; }"
)
_RECORDING_VALUE_STYLE = (
    "QLabel#elapsed_value { color: #b02a37; font-size: 42px; font-weight: 700; }"
)
_STATUS_STYLE = "QLabel#elapsed_status { color: #6b7280; font-weight: 600; }"


class ElapsedTimerPanel(QWidget):
    """A dockable stopwatch showing elapsed time and a manual reset button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._started_at: float | None = None
        self._recording = False
        self._ever_shown = False

        self._value_label = QLabel(format_elapsed(0.0), self)
        self._value_label.setObjectName("elapsed_value")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._status_label = QLabel("自由计时", self)
        self._status_label.setObjectName("elapsed_status")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(_STATUS_STYLE)

        self._reset_button = QPushButton("重置计时", self)
        self._reset_button.setObjectName("elapsed_reset")
        self._reset_button.setToolTip("将计时归零并重新开始")
        self._reset_button.clicked.connect(self.reset)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.addStretch(1)
        layout.addWidget(self._value_label)
        layout.addWidget(self._status_label)
        layout.addStretch(1)
        layout.addWidget(self._reset_button)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(100)
        self._tick_timer.timeout.connect(self._refresh)

        self.reset()

    # ── clock ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Re-zero the stopwatch and (re)start counting from now."""
        self._started_at = time.perf_counter()
        self._recording = False
        self._apply_mode()
        self._tick_timer.start()

    def start_recording(self) -> None:
        """Re-zero the stopwatch and mark it as timing an active trial."""
        self._started_at = time.perf_counter()
        self._recording = True
        self._apply_mode()
        self._tick_timer.start()

    def set_recording(self, active: bool) -> None:
        """Toggle the recording accent without disturbing the running clock."""
        if self._recording == active:
            return
        self._recording = active
        self._apply_mode()

    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.perf_counter() - self._started_at)

    # ── rendering ────────────────────────────────────────────────────────

    def _apply_mode(self) -> None:
        self._value_label.setText(format_elapsed(self.elapsed_seconds()))
        if self._recording:
            self._status_label.setText("● 采集中")
            self._value_label.setStyleSheet(_RECORDING_VALUE_STYLE)
        else:
            self._status_label.setText("自由计时")
            self._value_label.setStyleSheet(_IDLE_VALUE_STYLE)

    def _refresh(self) -> None:
        self._value_label.setText(format_elapsed(self.elapsed_seconds()))

    # ── Qt overrides ─────────────────────────────────────────────────────

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if not self._ever_shown:
            self._ever_shown = True
            # Start counting from the moment the operator adds the window,
            # unless the timer is already mid-trial (then keep the trial clock).
            if not self._recording:
                self.reset()


__all__ = ["ElapsedTimerPanel", "format_elapsed"]
