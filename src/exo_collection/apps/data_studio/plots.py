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

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal

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

# 按键打标的语义名配色：start 绿 / end 蓝 / nan 灰（虚线），未命名沿用红。
# 意图类工况按「前/后/外」分色，start 实线 / end 虚线。
_PROMPT_NAME_PEN = {
    "start": ("#16a34a", Qt.PenStyle.SolidLine),
    "end": ("#2563eb", Qt.PenStyle.SolidLine),
    "nan": ("#9ca3af", Qt.PenStyle.DashLine),
    "intent_forward_start": ("#d97706", Qt.PenStyle.SolidLine),
    "intent_forward_end": ("#d97706", Qt.PenStyle.DashLine),
    "intent_backward_start": ("#7c3aed", Qt.PenStyle.SolidLine),
    "intent_backward_end": ("#7c3aed", Qt.PenStyle.DashLine),
    "intent_lateral_start": ("#0ea5e9", Qt.PenStyle.SolidLine),
    "intent_lateral_end": ("#0ea5e9", Qt.PenStyle.DashLine),
}


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
    on_clicked: Callable[[PromptLabelPlaybackEvent], None] | None = None,
) -> None:
    """在当前循环窗内画按键打标竖线；复用已有线，多余隐藏。

    ``on_clicked`` 非空时，点中某条打标竖线会回调该事件（用于直接弹出标注）。
    """
    visible = tuple(
        event
        for event in events
        if current_s - window_s < event.time_s <= current_s
    )
    while len(lines) < len(visible):
        line = pg.InfiniteLine(pos=0.0, angle=90, movable=False)
        line.setZValue(95)
        # 打标竖线本身只有 1.6px，很难点中；给 hoverPen 加宽会撑大
        # boundingRect/shape（点击判定区域），而 movable=False 已关闭 hover 事件，
        # 故该笔不会实际渲染，只是扩大可点范围。
        line.setHoverPen(pg.mkPen("#ff0000", width=8))
        line._prompt_event = None
        if on_clicked is not None:
            line.sigClicked.connect(
                lambda _line, _ev, ln=line: (
                    on_clicked(ln._prompt_event)
                    if ln._prompt_event is not None
                    else None
                )
            )
        plot.addItem(line)
        lines.append(line)
    for index, line in enumerate(lines):
        if index >= len(visible):
            line._prompt_event = None
            line.hide()
            continue
        event = visible[index]
        line._prompt_event = event
        if event.name:
            color, style = _PROMPT_NAME_PEN.get(
                event.name, ("#ff0000", Qt.PenStyle.SolidLine)
            )
        else:
            color, style = "#ff0000", (
                Qt.PenStyle.DashLine
                if event.source is PromptLabelSource.SUBJECT
                else Qt.PenStyle.SolidLine
            )
        line.setPen(pg.mkPen(color, width=1.6, style=style))
        line.setPos((event.time_s - cycle_start_s) % window_s)
        suffix = f" · {event.name}" if event.name else ""
        line.setToolTip(
            f"{event.label}（{event.key}）{suffix} · t={event.time_s:.3f} s"
        )
        line.show()


class TimeSeriesPlot(pg.PlotWidget):
    """One or more channels on a shared real-time axis with a vertical cursor."""

    # 点中某条按键打标竖线时发出，携带被点的 PromptLabelPlaybackEvent。
    prompt_clicked = Signal(object)

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

    def _on_prompt_clicked(self, event: PromptLabelPlaybackEvent) -> None:
        self.prompt_clicked.emit(event)

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
                on_clicked=self._on_prompt_clicked,
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
