"""Compact, read-only preview of online gate decisions."""

from __future__ import annotations

from collections import deque
import math
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from exo_collection.apps.collector.signal_previews import HoverDetailsPlotWidget


_STATE_LABELS = {
    "allow": "放行",
    "rejected": "拒绝",
    "warming": "预热",
    "loading": "载入门控模型",
    "delay_warming": "等待延时样本",
    "stale": "过期",
    "fault": "故障",
    "disabled": "未启用",
}
_MODE_LABELS = {"ao": "AO24", "phase_ensemble": "七相位 TCN", "none": "未启用"}


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class UncertaintyPanel(QGroupBox):
    """Show gate score and threshold on a fixed, wrapping ten-second axis."""

    def __init__(self, parent: QWidget | None = None, *, capacity: int = 2048,
                 window_s: float = 10.0) -> None:
        super().__init__("不确定性判定", parent)
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be finite and positive")
        self.setTitle("")
        self.setObjectName("uncertainty_panel")
        self.setStyleSheet("QGroupBox#uncertainty_panel { margin-top: 0; padding-top: 0; border: 0; }")
        self._window_s = float(window_s)
        self._capacity = max(16, int(capacity))
        self._mode = "none"
        self._origin_ns: int | None = None
        self._latest_t = 0.0
        self._points: deque[tuple[float, float, float, bool]] = deque(maxlen=self._capacity)
        self._break_next = False
        self._seen: set[tuple[Any, ...]] = set()
        self._seen_order: deque[tuple[Any, ...]] = deque(maxlen=self._capacity * 2)

        self._status_label = QLabel("门控：未启用")
        self._status_label.setObjectName("uncertainty_status")
        self._status_label.setWordWrap(True)
        self._plot = HoverDetailsPlotWidget()
        self._plot.setObjectName("uncertainty_plot")
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMinimumHeight(150)
        self._plot.set_details_always_visible(True)
        self._plot.setLabel("bottom", "循环时间 (s)", color="#374151")
        self._plot.setLabel("left", "未启用", color="#374151")
        self._plot.setXRange(0, self._window_s, padding=0)
        self._plot.setLimits(xMin=0, xMax=self._window_s,
                             minXRange=self._window_s, maxXRange=self._window_s)
        self._plot.addLegend(offset=None, colCount=2, frame=False, labelTextColor="#374151")
        for axis in ("left", "bottom"):
            self._plot.getPlotItem().getAxis(axis).setPen("#6b7280")
            self._plot.getPlotItem().getAxis(axis).setTextPen("#374151")
        self._score_curve = self._plot.plot(pen=pg.mkPen("#0f766e", width=2), name="分数")
        self._threshold_curve = self._plot.plot(
            pen=pg.mkPen("#d97706", width=1.6, style=Qt.PenStyle.DashLine), name="阈值"
        )
        self._cursor_line = pg.InfiniteLine(pos=0, angle=90, movable=False,
                                            pen=pg.mkPen("#dc3545", width=2))
        self._cursor_line.setZValue(100)
        self._plot.addItem(self._cursor_line, ignoreBounds=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(2)
        layout.addWidget(self._status_label)
        layout.addWidget(self._plot, 1)

    @property
    def plot(self) -> HoverDetailsPlotWidget:
        """Expose the plot for preview rendering and UI inspection."""
        return self._plot

    def set_mode(self, mode: str) -> None:
        mode = str(mode or "none")
        if mode == self._mode:
            return
        self._mode = mode
        self.clear()
        self._render_status(None, "等待数据" if mode != "none" else None)
        self._set_axis_labels()
        if mode == "ao":
            self._plot.setYRange(0, 100, padding=0)
            self._plot.setLimits(yMin=0, yMax=100, minYRange=100, maxYRange=100)
        else:
            self._plot.setLimits(yMin=None, yMax=None, minYRange=None, maxYRange=None)
            self._plot.setYRange(0, 1, padding=0)

    def add_pipeline(self, payload: dict) -> None:
        status = payload.get("gate_status")
        mode = str(_field(status, "mode", self._mode) or self._mode)
        if mode != self._mode:
            self.set_mode(mode)
        time_ns = payload.get("time_ns", payload.get("host_monotonic_ns"))
        if time_ns is None:
            return
        try:
            time_ns = int(time_ns)
        except (TypeError, ValueError, OverflowError):
            return
        if self._origin_ns is None:
            self._origin_ns = time_ns
        t = max(0.0, (time_ns - self._origin_ns) / 1e9)
        self._latest_t = max(self._latest_t, t)
        self._cursor_line.setPos(self._latest_t % self._window_s)

        if status is None or mode == "none":
            self._render_status(None, "未启用" if mode == "none" else "状态未知")
            self._break_next = True
            self._render_plot()
            return

        state = str(_field(status, "state", "unknown") or "unknown")
        score = self._finite_number(_field(status, "score"))
        threshold = self._finite_number(_field(status, "threshold"))
        self._render_status(status, state)

        # Only a fresh, explicit decision is suitable for the plotted trace.
        # Warmup, stale, fault, unknown and missing values remain text-only.
        if state in {"allow", "rejected", "delay_warming"} and score is not None and threshold is not None:
            raw_source_ns = _field(status, "source_grid_ns")
            try:
                source_ns = int(raw_source_ns) if raw_source_ns is not None else time_ns
            except (TypeError, ValueError, OverflowError):
                source_ns = time_ns
            key = (mode, _field(status, "generation"), _field(status, "sequence"), source_ns)
            if key not in self._seen:
                self._remember(key)
                decision_t = (source_ns - self._origin_ns) / 1e9
                last_decision_t = self._points[-1][0] if self._points else -math.inf
                if (max(0.0, self._latest_t - self._window_s) <= decision_t <= self._latest_t + 0.001
                        and decision_t >= last_decision_t):
                    scale = 100.0 if mode == "ao" else 1.0
                    self._points.append((max(0.0, decision_t), score * scale,
                                         threshold * scale, self._break_next))
                    self._break_next = False
                elif decision_t > last_decision_t:
                    self._break_next = True
        else:
            self._break_next = True
        self._render_plot()

    def mark_inactive(self, *, fault: bool = False) -> None:
        """Mark the retained trace as historical after stop or fault."""
        self._status_label.setText(f"门控：{_MODE_LABELS.get(self._mode, self._mode)} · "
                                   f"{'故障停止' if fault else '已停止'} · 分数 — · 阈值 —")

    def clear(self) -> None:
        self._origin_ns = None
        self._latest_t = 0.0
        self._points.clear()
        self._break_next = False
        self._seen.clear()
        self._seen_order.clear()
        self._cursor_line.setPos(0)
        self._score_curve.setData([], [])
        self._threshold_curve.setData([], [])
        self._render_status(None, "等待数据" if self._mode != "none" else None)

    def _render_status(self, status, override_state: str | None = None) -> None:
        mode = str(_field(status, "mode", self._mode) or self._mode)
        if mode == "none":
            self._status_label.setText("门控：未启用")
            return
        state = override_state or str(_field(status, "state", "unknown") or "unknown")
        state_text = _STATE_LABELS.get(state, "状态未知" if state == "unknown" else state)
        score = self._finite_number(_field(status, "score"))
        threshold = self._finite_number(_field(status, "threshold"))
        if mode == "ao":
            score_text = "—" if score is None else f"{score * 100:.2f}%"
            threshold_text = "—" if threshold is None else f"{threshold * 100:.2f}%"
            meaning = "分类器不适用分数（%，未校准）"
        elif mode == "phase_ensemble":
            score_text = "—" if score is None else f"{score:.5g}"
            threshold_text = "—" if threshold is None else f"{threshold:.5g}"
            meaning = "因果中位数相位预测方差（原始单位）"
        else:
            score_text = threshold_text = "—"
            meaning = "分数"
        age = self._finite_number(_field(status, "age_ms"))
        suffix = f" · 年龄 {age:.0f} ms" if age is not None else ""
        self._status_label.setText(
            f"门控：{_MODE_LABELS.get(mode, mode)} · {state_text} · {meaning} "
            f"{score_text} / 阈值 {threshold_text}{suffix}"
        )

    def _set_axis_labels(self) -> None:
        if self._mode == "ao":
            self._plot.setLabel("left", "不适用分数 (%)", color="#374151")
        elif self._mode == "phase_ensemble":
            self._plot.setLabel("left", "相位方差", color="#374151")
        else:
            self._plot.setLabel("left", "未启用", color="#374151")

    def _render_plot(self) -> None:
        floor = self._latest_t - self._window_s
        while self._points and self._points[0][0] < floor:
            self._points.popleft()
        xs: list[float] = []
        scores: list[float] = []
        thresholds: list[float] = []
        previous = None
        for absolute_t, score, threshold, break_before in self._points:
            x = absolute_t % self._window_s
            if previous is not None and (break_before or x < previous):
                xs.append(float("nan"))
                scores.append(float("nan"))
                thresholds.append(float("nan"))
            xs.append(x)
            scores.append(score)
            thresholds.append(threshold)
            previous = x
        self._score_curve.setData(xs, scores, connect="finite")
        self._threshold_curve.setData(xs, thresholds, connect="finite")
        if self._mode == "phase_ensemble" and self._points:
            values = [value for _, score, threshold, _ in self._points
                      for value in (score, threshold) if math.isfinite(value)]
            if values:
                low = min(0.0, min(values))
                high = max(0.0, max(values))
                span = max(high - low, 1e-9)
                padding = span * 0.08
                self._plot.setYRange(low - padding, high + padding, padding=0)

    def _remember(self, key: tuple[Any, ...]) -> None:
        if len(self._seen_order) == self._seen_order.maxlen:
            self._seen.discard(self._seen_order[0])
        self._seen_order.append(key)
        self._seen.add(key)

    @staticmethod
    def _finite_number(value) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if math.isfinite(result) else None


__all__ = ["UncertaintyPanel"]
