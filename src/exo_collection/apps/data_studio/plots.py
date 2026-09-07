"""Light-weight, real-time-axis plots for Data Studio's full-screen viewer.

The old sweep plots kept a fixed ring buffer and rewrote columns every cycle,
which produced point-like, disconnected traces (毛刺) whenever samples landed
between columns or two cycles interleaved.  :class:`TimeSeriesPlot` abandons the
ring buffer entirely: it plots real sample time on the x-axis inside a fixed
trailing scroll window, connects only samples that are truly adjacent in time,
and draws thicker antialiased lines on a light background.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

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
            pen=pg.mkPen("#e11d48", width=2.0),
        )
        self.cursor.setZValue(100)
        self.addItem(self.cursor)
        self.setXRange(0.0, self._window_s, padding=0.0)

    def set_time(self, current_s: float) -> None:
        current = float(current_s)
        left = current - self._window_s
        self.setXRange(left, current, padding=0.0)
        self.cursor.setPos(current)
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


__all__ = ["TimeSeriesPlot"]
