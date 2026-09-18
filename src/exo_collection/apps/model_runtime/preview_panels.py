"""Reusable live-preview widgets for the online-testing window.

The collector renders each modality with modality-specific layouts; the online
runtime reuses the *payload* contract produced by ``build_preview_event`` and
renders it with two lightweight, generic panels:

- ``LivePreviewPanel`` — one ``pg.PlotWidget`` per modality, plotting every
  channel in the payload as a ring-buffer trace (ultrasound frames overwrite).
- ``ModelOutputPanel`` — one ``pg.PlotWidget`` plotting predicted and commanded
  bilateral torque against a shared monotonic time base.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QWidget

# Trace palette (teal / orange / blue / purple …) kept consistent with the
# collector theme so the previews read the same across the two applications.
_TRACE_PEN_COLORS = (
    "#0f766e",
    "#d97706",
    "#2563eb",
    "#7c3aed",
    "#b4533c",
    "#059669",
    "#9333ea",
    "#64748b",
    "#db2777",
    "#0891b2",
    "#ca8a04",
    "#4b5563",
)


def _make_plot(title: str) -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setBackground("w")
    plot.showGrid(x=True, y=True, alpha=0.25)
    plot.setMouseEnabled(x=False, y=False)
    plot.getPlotItem().addLegend(offset=(-2, 2))
    plot.setTitle(title)
    plot.setLabel("bottom", "采样点")
    return plot


class LivePreviewPanel(QGroupBox):
    """Render one modality's PREVIEW events as ring-buffer traces."""

    def __init__(
        self,
        modality: str,
        title: str,
        parent: QWidget | None = None,
        *,
        capacity: int = 800,
    ) -> None:
        super().__init__(title, parent)
        self._modality = modality
        self._capacity = max(16, int(capacity))
        self._overwrite = modality == "ultrasound"
        self._plot = _make_plot(title)
        self._plot.setLabel("left", "幅值")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.addWidget(self._plot)
        self._traces: dict[str, dict[str, Any]] = {}

    def feed(self, payload: dict[str, Any]) -> None:
        channels = payload.get("channels")
        if channels:
            labels = payload.get("labels")
            channel_index = payload.get("channel_index")
            if self._overwrite and channel_index is not None:
                labels = [f"ch_{int(channel_index) + 1}"]
            elif not labels or len(labels) != len(channels):
                labels = [f"ch_{i + 1}" for i in range(len(channels))]
            for index, label in enumerate(labels):
                self._append(str(label), channels[index])
        elif payload.get("values") is not None:
            self._append("signal", payload["values"])

    def _append(self, label: str, values: Any) -> None:
        trace = self._traces.get(label)
        if trace is None:
            color = _TRACE_PEN_COLORS[len(self._traces) % len(_TRACE_PEN_COLORS)]
            curve = self._plot.plot(pen=pg.mkPen(color, width=1.4), name=label)
            trace = {"ys": deque(maxlen=self._capacity), "curve": curve}
            self._traces[label] = trace

        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if self._overwrite:
            # Ultrasound: each event is one full frame — replace, don't append.
            trace["curve"].setData(np.arange(arr.size), arr)
            return

        trace["ys"].extend(arr.tolist())
        ys = list(trace["ys"])
        trace["curve"].setData(np.arange(len(ys), dtype=np.float64), ys)

    def clear(self) -> None:
        for trace in self._traces.values():
            trace["ys"].clear()
            trace["curve"].setData([], [])


class ModelOutputPanel(QGroupBox):
    """Plot predicted and commanded bilateral torque on a monotonic time base."""

    def __init__(
        self,
        title: str = "模型输出",
        parent: QWidget | None = None,
        *,
        capacity: int = 1200,
    ) -> None:
        super().__init__(title, parent)
        self._capacity = max(16, int(capacity))
        self._plot = _make_plot(title)
        self._plot.setLabel("left", "扭矩 (Nm)")
        self._plot.setLabel("bottom", "时间 (s)")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 2)
        layout.addWidget(self._plot)
        self._traces: dict[str, dict[str, Any]] = {}
        self._t0_ns: int | None = None

    def _t(self, ns: int) -> float:
        if self._t0_ns is None:
            self._t0_ns = int(ns)
        return (int(ns) - self._t0_ns) / 1e9

    def add_prediction(self, host_monotonic_ns: int, left_nm: float, right_nm: float) -> None:
        t = self._t(host_monotonic_ns)
        self._append("预测右 (Nm)", t, right_nm)
        self._append("预测左 (Nm)", t, left_nm)

    def add_command(self, command: Any) -> None:
        # ``TorqueCommand`` from torque.py (duck-typed so the panel stays
        # independent of the concrete class).
        ns = int(getattr(command, "host_monotonic_ns", 0))
        t = self._t(ns)
        self._append("指令右 (Nm)", t, float(getattr(command, "right_nm", 0.0)))
        self._append("指令左 (Nm)", t, float(getattr(command, "left_nm", 0.0)))

    def _append(self, label: str, x: float, y: float) -> None:
        trace = self._traces.get(label)
        if trace is None:
            color = _TRACE_PEN_COLORS[len(self._traces) % len(_TRACE_PEN_COLORS)]
            curve = self._plot.plot(pen=pg.mkPen(color, width=1.4), name=label)
            trace = {"xs": deque(maxlen=self._capacity), "ys": deque(maxlen=self._capacity), "curve": curve}
            self._traces[label] = trace
        trace["xs"].append(x)
        trace["ys"].append(y)
        trace["curve"].setData(list(trace["xs"]), list(trace["ys"]))

    def clear(self) -> None:
        self._t0_ns = None
        for trace in self._traces.values():
            trace["xs"].clear()
            trace["ys"].clear()
            trace["curve"].setData([], [])


__all__ = ["LivePreviewPanel", "ModelOutputPanel"]
