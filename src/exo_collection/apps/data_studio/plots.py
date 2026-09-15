"""Light-weight, real-time-axis plots for Data Studio's full-screen viewer.

The old sweep plots kept a fixed ring buffer and rewrote columns every cycle,
which produced point-like, disconnected traces (毛刺) whenever samples landed
between columns or two cycles interleaved.  :class:`TimeSeriesPlot` keeps a real
time axis but renders it oscilloscope-style: the x-axis is fixed at
``[0, window]``, the cursor sweeps left-to-right, and the previous cycle's trace
is overwritten in place (never cleared), so the full window always shows the
trailing ``window`` seconds wrapped at the cursor.  Lines are thicker and
antialiased on a light background.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from exo_collection.domain.prompt_labels import PromptLabelSource

from .gait_events import GaitEvent
from .local_tools import PromptLabelPlaybackEvent

_PLOT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)

# 步态事件竖线：左右脚用可辨色相区分，跟触/尖离用线型区分（身份不靠颜色单通道）。
_EVENT_PEN_BY_SIDE = {"right": "#0072B2", "left": "#D55E00"}
_EVENT_SIDE_LABEL = {"right": "右", "left": "左"}
_EVENT_KIND_LABEL = {"heel_strike": "足跟触地", "toe_off": "足尖离地"}


def _event_label(event: GaitEvent) -> str:
    side = _EVENT_SIDE_LABEL.get(event.side, event.side)
    kind = _EVENT_KIND_LABEL.get(event.kind, event.kind)
    return f"{side}脚{kind}"


def update_event_marker_lines(
    plot: "pg.PlotWidget",
    events: tuple[GaitEvent, ...],
    lines: list["pg.InfiniteLine"],
    *,
    current_s: float,
    cycle_start_s: float,
    window_s: float,
) -> None:
    """在当前循环窗内画步态事件竖线；复用已有线，多余隐藏。"""
    visible = tuple(
        event for event in events if current_s - window_s < event.time_s <= current_s
    )
    while len(lines) < len(visible):
        line = pg.InfiniteLine(pos=0.0, angle=90, movable=False)
        line.setZValue(96)
        plot.addItem(line)
        lines.append(line)
    for index, line in enumerate(lines):
        if index >= len(visible):
            line.hide()
            continue
        event = visible[index]
        line.setPen(
            pg.mkPen(
                _EVENT_PEN_BY_SIDE.get(event.side, "#6B7280"),
                width=1.8,
                style=(
                    Qt.PenStyle.DashLine
                    if event.kind == "toe_off"
                    else Qt.PenStyle.SolidLine
                ),
            )
        )
        line.setPos((event.time_s - cycle_start_s) % window_s)
        line.setToolTip(f"{_event_label(event)} · t={event.time_s:.3f} s")
        line.show()


def update_prompt_marker_lines(
    plot: "pg.PlotWidget",
    events: tuple[PromptLabelPlaybackEvent, ...],
    lines: list["pg.InfiniteLine"],
    *,
    current_s: float,
    cycle_start_s: float,
    window_s: float,
) -> None:
    """在当前循环窗内画按键打标竖线；复用已有线，多余隐藏。"""
    visible = tuple(
        event
        for event in events
        if current_s - window_s < event.time_s <= current_s
    )
    while len(lines) < len(visible):
        line = pg.InfiniteLine(pos=0.0, angle=90, movable=False)
        line.setZValue(95)
        plot.addItem(line)
        lines.append(line)
    for index, line in enumerate(lines):
        if index >= len(visible):
            line.hide()
            continue
        event = visible[index]
        line.setPen(
            pg.mkPen(
                "#ff0000",
                width=1.6,
                style=(
                    Qt.PenStyle.DashLine
                    if event.source is PromptLabelSource.SUBJECT
                    else Qt.PenStyle.SolidLine
                ),
            )
        )
        line.setPos((event.time_s - cycle_start_s) % window_s)
        line.setToolTip(
            f"{event.label}（{event.key}） · t={event.time_s:.3f} s"
        )
        line.show()


class TimeSeriesPlot(pg.PlotWidget):
    """One or more channels on a shared real-time axis with a vertical cursor."""

    def __init__(
        self,
        title: str,
        time_s: np.ndarray,
        values: np.ndarray,
        channels: tuple[str, ...],
        window_s: float = 10.0,
        *,
        fixed_yrange: tuple[float, float] | None = None,
        offset_per_channel: float | None = None,
    ) -> None:
        super().__init__()
        self._times = np.asarray(time_s, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        self._channels = tuple(channels)
        self._window_s = float(window_s)
        self._offset = float(offset_per_channel) if offset_per_channel else 0.0
        self._gait_events: tuple[GaitEvent, ...] = ()
        self._gait_lines: list[pg.InfiniteLine] = []
        self._prompt_events: tuple[PromptLabelPlaybackEvent, ...] = ()
        self._prompt_lines: list[pg.InfiniteLine] = []

        self.setTitle(title)
        self.setBackground("#ffffff")
        self.setLabel("bottom", "时间", units="s")
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.getViewBox().setMouseEnabled(x=False, y=False)

        channel_count = self._values.shape[1] if self._values.ndim == 2 else 0
        self._curves: list[pg.PlotDataItem] = []
        if channel_count > 1:
            self.addLegend(offset=(10, 10))
        for index in range(channel_count):
            label = (
                self._channels[index]
                if index < len(self._channels)
                else f"ch_{index + 1}"
            )
            color = _PLOT_COLORS[index % len(_PLOT_COLORS)]
            curve = self.plot(
                [],
                [],
                name=label,
                pen=pg.mkPen(color, width=2.0),
                antialias=True,
            )
            self._curves.append(curve)

        finite = self._values[np.isfinite(self._values)]
        if fixed_yrange is not None:
            base_low, base_high = float(fixed_yrange[0]), float(fixed_yrange[1])
        elif finite.size:
            base_low, base_high = (float(v) for v in np.percentile(finite, (0.5, 99.5)))
            span = max(base_high - base_low, 1e-6)
            base_low -= 0.08 * span
            base_high += 0.08 * span
        else:
            base_low, base_high = -1.0, 1.0
        if self._offset and channel_count:
            # Stacked hospital-style EMG: the top channel is shifted up by
            # (k-1)*offset, so the axis must span the whole stack.
            self.setYRange(
                base_low,
                base_low + (channel_count - 1) * self._offset + (base_high - base_low),
                padding=0.0,
            )
        else:
            self.setYRange(base_low, base_high, padding=0.0)

        self.cursor = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#dc2626", width=3.0),
        )
        self.cursor.setZValue(100)
        self.addItem(self.cursor)
        self.setXRange(0.0, self._window_s, padding=0.0)

    def set_channel_visible(self, index: int, visible: bool) -> None:
        """显示/隐藏第 ``index`` 条通道曲线（及其图例项）。"""
        if 0 <= index < len(self._curves):
            self._curves[index].setVisible(visible)

    def set_channel_pen(self, index: int, pen: object) -> None:
        """覆盖第 ``index`` 条通道的画笔（如把 baseline 画成虚线）。"""
        if 0 <= index < len(self._curves):
            self._curves[index].setPen(pen)

    def set_gait_events(self, events: tuple[GaitEvent, ...]) -> None:
        """设置（或清空）本图上的步态事件竖线；下次 :meth:`set_time` 摆位。"""
        self._gait_events = tuple(events)

    def set_prompt_events(self, events: tuple[PromptLabelPlaybackEvent, ...]) -> None:
        """设置（或清空）本图上的按键打标竖线；下次 :meth:`set_time` 摆位。"""
        self._prompt_events = tuple(events)

    def set_time(self, current_s: float, cycle_start_s: float | None = None) -> None:
        """Advance the sweep cursor.

        With a fixed ``cycle_start_s`` the plot is an oscilloscope-style cyclic
        sweep on a *fixed* ``[0, window]`` x-axis: the cursor marks the current
        sample and sweeps left-to-right before wrapping to the left edge, and
        each cycle overwrites the previous cycle's trace in place rather than
        clearing it.  The full window always shows the trailing ``window``
        seconds, wrapped at the cursor (the current sweep fills ``[0, phase]``
        and the previous cycle's tail fills ``[phase, window]``).  Without a
        ``cycle_start_s`` the plot falls back to a trailing window ending at the
        cursor.
        """
        current = float(current_s)
        if cycle_start_s is not None:
            left = float(cycle_start_s)
            phase = current - left
            self.setXRange(0.0, self._window_s, padding=0.0)
            self.cursor.setPos(phase)
            update_event_marker_lines(
                self,
                self._gait_events,
                self._gait_lines,
                current_s=current,
                cycle_start_s=left,
                window_s=self._window_s,
            )
            update_prompt_marker_lines(
                self,
                self._prompt_events,
                self._prompt_lines,
                current_s=current,
                cycle_start_s=left,
                window_s=self._window_s,
            )
            if not self._times.size:
                for curve in self._curves:
                    curve.setData([], [])
                return
            new_mask = (self._times >= left) & (self._times <= current)
            old_mask = (self._times >= current - self._window_s) & (self._times < left)
            x = np.concatenate(
                (
                    self._times[new_mask] - left,
                    self._times[old_mask] - left + self._window_s,
                )
            )
            for index, curve in enumerate(self._curves):
                if self._values.ndim == 2 and index < self._values.shape[1]:
                    samples = np.concatenate(
                        (
                            self._values[new_mask, index],
                            self._values[old_mask, index],
                        )
                    )
                    if self._offset:
                        samples = samples + index * self._offset
                else:
                    samples = np.empty(0, dtype=np.float64)
                curve.setData(x, samples)
            return

        left = current - self._window_s
        self.setXRange(left, current, padding=0.0)
        self.cursor.setPos(current)
        for line in self._gait_lines:
            line.hide()
        for line in self._prompt_lines:
            line.hide()
        if not self._times.size:
            for curve in self._curves:
                curve.setData([], [])
            return
        mask = (self._times >= left) & (self._times <= current)
        times = self._times[mask]
        for index, curve in enumerate(self._curves):
            if self._values.ndim == 2 and index < self._values.shape[1]:
                samples = self._values[mask, index]
                if self._offset:
                    samples = samples + index * self._offset
            else:
                samples = np.empty(0, dtype=np.float64)
            curve.setData(times, samples)


__all__ = ["TimeSeriesPlot", "update_prompt_marker_lines"]
