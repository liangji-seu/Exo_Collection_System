"""Model-output charts using the Collector plot style and hover behavior.

Sensor previews are rendered by the shared Collector SignalPreviewMixin.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from exo_collection.apps.collector.signal_previews import HoverDetailsPlotWidget

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
    plot = HoverDetailsPlotWidget()
    plot.setBackground("w")
    plot.showGrid(x=True, y=True, alpha=0.25)
    plot.setMouseEnabled(x=False, y=False)
    plot.addLegend(offset=None, colCount=3, frame=False, labelTextColor="#374151")
    plot.setTitle(title)
    plot.setLabel("bottom", "采样点")
    return plot


class ModelOutputPanel(QGroupBox):
    """One circular Nm display with identical scales on both torque axes."""

    def __init__(
        self,
        title: str = "模型输出",
        parent: QWidget | None = None,
        *,
        capacity: int = 4096,
        window_s: float = 10.0,
    ) -> None:
        super().__init__(title, parent)
        # The dock already supplies the panel title. Keep scarce height for
        # readable traces rather than repeating titles above the axes.
        self.setTitle("")
        self.setObjectName("model_output_panel")
        self.setStyleSheet("QGroupBox#model_output_panel { margin-top: 0; padding-top: 0; border: 0; }")
        self._capacity = max(16, int(capacity))
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be finite and positive")
        self._window_s = float(window_s)
        self._plot = _make_plot("")
        self._plot.setMinimumHeight(252)
        self._plot.set_details_always_visible(True)
        self._plot.setLabel("left", "关节力矩 (Nm)", color="#374151")
        self._plot.setLabel("bottom", "循环时间 (s)", color="#374151")
        self._plot.setXRange(0, self._window_s, padding=0)
        self._plot.setLimits(xMin=0, xMax=self._window_s,
                             minXRange=self._window_s, maxXRange=self._window_s)
        item = self._plot.getPlotItem()
        # Use the former title row for a compact legend outside the data area.
        item.layout.removeItem(item.titleLabel)
        item.legend.setParentItem(item)
        item.legend.layout.setContentsMargins(0, 0, 0, 0)
        item.layout.addItem(item.legend, 0, 0, 1, 3)
        item.layout.setRowFixedHeight(0, 28)
        item.layout.setContentsMargins(8, 0, 8, 0)
        item.showAxis("right")
        # All values are Nm. One ViewBox guarantees identical left/right
        # scales and a range covering prediction, assistance and real sends.
        self._assistance_view = item.vb  # compatibility alias
        item.getAxis("right").linkToView(self._assistance_view)
        self._plot.setLabel("right", "辅助力矩 (Nm)", color="#374151")
        for axis in ("left", "right"):
            item.getAxis(axis).enableAutoSIPrefix(False)
            item.getAxis(axis).setWidth(58)
        for axis in ("left", "right", "bottom"):
            item.getAxis(axis).setPen("#6b7280")
            item.getAxis(axis).setTextPen("#374151")
        self._cursor_line = pg.InfiniteLine(pos=0, angle=90, movable=False,
                                            pen=pg.mkPen("#dc3545", width=3))
        self._cursor_line.setZValue(100)
        self._plot.addItem(self._cursor_line, ignoreBounds=True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(2)
        layout.addWidget(self._plot)
        self._traces: dict[str, dict[str, Any]] = {}
        self._t0_ns: int | None = None
        self._latest_t = 0.0
        # Historical aliases now contain the displayed converted Nm values.
        self._normalized_plot = self._plot
        self._normalized_curve = self._normalized_plot.plot(pen=pg.mkPen("#0f766e", width=2), name="模型预测右 (Nm)")
        self._raw_left_curve = self._normalized_plot.plot(pen=pg.mkPen("#059669", width=1.4), name="模型预测左 (Nm)")
        self._raw_left_x = deque(maxlen=self._capacity)
        self._raw_left_y = deque(maxlen=self._capacity)
        self._normalized_x = deque(maxlen=self._capacity)
        self._normalized_y = deque(maxlen=self._capacity)
        self._raw_traces = [
            self._trace(self._normalized_curve, self._normalized_x, self._normalized_y),
            self._trace(self._raw_left_curve, self._raw_left_x, self._raw_left_y),
        ]
        self._values_label = QLabel("模型预测 — Nm ｜ 期望辅助 — Nm ｜ 发送 — Nm")
        self._values_label.setWordWrap(True)
        self._values_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.insertWidget(0, self._values_label)
        self._feedback_label = QLabel("电机反馈：未连接")
        self._feedback_label.setWordWrap(True)
        layout.addWidget(self._feedback_label)
        self._desired_right_nm = None
        self._desired_left_nm = None
        self._estimate_per_kg = None
        self._raw_right = None
        self._raw_left = None
        self._estimate_right_nm = self._estimate_left_nm = None
        self._output_unit = "Nm/kg"
        self._single_output_maps_to = None
        self._motor_distribution = "per_side"
        self._motor_values_diverged = set()
        self._assistance_errors = []
        self._output_status = "未连接电机 · 未下发"
        self._last_sent_command = None

    def set_output_spec(self, output_spec, *, active_sides=("left", "right"),
                        motor_distribution="per_side") -> None:
        # A scalar mapping is also valid for pair-producing generic models;
        # the caller must explicitly declare any known absent side.
        self._single_output_maps_to = (
            "right_torque" if tuple(active_sides) == ("right",) else
            "left_torque" if tuple(active_sides) == ("left",) else None
        )
        self._motor_distribution = str(motor_distribution)
        self._normalized_curve.setVisible(self._single_output_maps_to != "left_torque")
        self._raw_left_curve.setVisible(self._single_output_maps_to != "right_torque")
        legend = self._plot.getPlotItem().legend
        for curve, label, visible in (
            (self._normalized_curve, "模型预测右 (Nm)", self._single_output_maps_to != "left_torque"),
            (self._raw_left_curve, "模型预测左 (Nm)", self._single_output_maps_to != "right_torque"),
        ):
            legend.removeItem(curve)
            if visible:
                legend.addItem(curve, label)
        self.set_output_unit(output_spec.unit)

    def set_output_unit(self, unit: str) -> None:
        # Called at model installation, including same-unit model switches.
        self.clear()
        self._output_unit = str(unit)
        self._plot.setLabel("left", "关节力矩 (Nm)", color="#374151")
        self._render_values()

    def _sync_legend_row(self) -> None:
        item = self._plot.getPlotItem()
        rows = max(1, math.ceil(len(item.legend.items) / 3))
        item.layout.setRowFixedHeight(0, 28 * rows)
        # A second legend row must not consume the readable waveform height.
        self._plot.setMinimumHeight(252 + 28 * (rows - 1))

    def _trace(self, curve, xs=None, ys=None) -> dict[str, Any]:
        return {"curve": curve, "times": deque(maxlen=self._capacity),
                "xs": xs if xs is not None else deque(maxlen=self._capacity),
                "ys": ys if ys is not None else deque(maxlen=self._capacity)}

    @staticmethod
    def _value(value, decimals=3) -> str:
        return "—" if value is None else f"{float(value):+.{decimals}f}"

    def _render_values(self) -> None:
        missing_conversion = self._output_unit == "Nm/kg" and (
            (self._single_output_maps_to != "left_torque" and self._estimate_right_nm is None)
            or (self._single_output_maps_to != "right_torque" and self._estimate_left_nm is None)
        )
        estimate = ("待填写体重 / 不可换算" if missing_conversion else
                    f"{self._side_values(self._estimate_left_nm, self._estimate_right_nm, 3)} Nm")
        desired = (
            "待填写体重 / 不可换算" if missing_conversion
            else f"{self._side_values(self._desired_left_nm, self._desired_right_nm, motor=True)} Nm"
        )
        reason = "; ".join(map(str, self._assistance_errors))
        command_text = ""
        if self._last_sent_command is not None:
            command = self._last_sent_command
            command_text = (
                f"最近实际下发（限幅后） {self._side_values(command['left_nm'], command['right_nm'], motor=True)} Nm"
                f" ｜该帧阈值 ±{command['limit_nm']:g} Nm ｜{'使能' if command['enabled'] else '失能'}"
            )
        self._values_label.setText(
            f"模型预测 {estimate}"
            f" ｜期望辅助（限幅前） {desired}\n输出：{self._output_status}"
            + (f" ｜ {reason}" if reason else "")
            + (f"\n{command_text}" if command_text else "")
        )

    def _side_values(self, left, right, decimals=3, *, motor=False) -> str:
        if motor and self._motor_distribution == "right_to_both_same_sign":
            return f"左 {self._value(left, decimals)} / 右 {self._value(right, decimals)}"
        if self._single_output_maps_to == "right_torque":
            return f"右 {self._value(right, decimals)}"
        if self._single_output_maps_to == "left_torque":
            return f"左 {self._value(left, decimals)}"
        return f"左 {self._value(left, decimals)} / 右 {self._value(right, decimals)}"

    def add_pipeline(self, payload: dict) -> None:
        distribution = payload.get("motor_distribution", self._motor_distribution)
        if distribution != self._motor_distribution:
            self._motor_distribution = str(distribution)
            self.clear()
        unit = payload.get("output_unit", payload.get("unit", self._output_unit))
        if unit != self._output_unit:
            self.set_output_unit(unit)
        ns = int(payload.get("time_ns", payload.get("host_monotonic_ns", 0)))
        t = self._t(ns)
        estimate = payload.get("raw_estimate_right", payload.get("estimate_nm_per_kg"))
        self._raw_right = estimate
        self._raw_left = payload.get("raw_estimate_left")
        self._estimate_per_kg = estimate if self._output_unit == "Nm/kg" else None
        # The adapter already applied body mass once. Never apply mass, gain,
        # filtering or direction again in the display layer.
        self._estimate_right_nm = payload.get("estimate_nm", estimate if unit == "Nm" else None)
        self._estimate_left_nm = payload.get("estimate_left_nm", self._raw_left if unit == "Nm" else None)
        if self._estimate_right_nm is not None and self._single_output_maps_to != "left_torque":
            self._store(self._raw_traces[0], t, float(self._estimate_right_nm))
        if self._estimate_left_nm is not None and self._single_output_maps_to != "right_torque":
            self._store(self._raw_traces[1], t, float(self._estimate_left_nm))
        self._desired_right_nm = payload.get("desired_right_nm")
        self._desired_left_nm = payload.get("desired_left_nm")
        self._append_motor_values("期望辅助", t, self._desired_left_nm, self._desired_right_nm)
        self._assistance_errors = list(payload.get("assistance_errors", []))
        self._render_traces()
        self._render_values()

    def set_output_status(self, status: str) -> None:
        self._output_status = str(status)
        self._render_values()

    def set_feedback(self, feedback, age_s: float | None = None) -> None:
        if feedback is None:
            self._feedback_label.setText("电机反馈：未收到 / 未连接")
            return
        freshness = "过期" if age_s is None or age_s > 0.25 else f"{age_s * 1000:.0f} ms"
        self._feedback_label.setText(
            f"反馈年龄 {freshness} ｜ state={feedback.state} error={feedback.error}\n"
            f"左：位置 {feedback.left_position_rad:+.3f} rad ｜速度 {feedback.left_velocity_rad_s:+.3f} rad/s ｜力矩 {feedback.left_torque_nm:+.3f} Nm\n"
            f"右：位置 {feedback.right_position_rad:+.3f} rad ｜速度 {feedback.right_velocity_rad_s:+.3f} rad/s ｜力矩 {feedback.right_torque_nm:+.3f} Nm"
        )

    def _t(self, ns: int) -> float:
        if self._t0_ns is None:
            self._t0_ns = int(ns)
        t = (int(ns) - self._t0_ns) / 1e9
        self._latest_t = max(self._latest_t, t)
        self._cursor_line.setPos(self._latest_t % self._window_s)
        return t

    def add_prediction(self, host_monotonic_ns: int, left_nm: float | None, right_nm: float | None) -> None:
        # Converted predictions duplicate the pipeline estimate. Keep the
        # public slot for queued signals without adding another Nm curve.
        del host_monotonic_ns, left_nm, right_nm

    def add_command(self, command: Any) -> None:
        # ``TorqueCommand`` from torque.py (duck-typed so the panel stays
        # independent of the concrete class).
        ns = int(getattr(command, "host_monotonic_ns", 0))
        if self._last_sent_command is not None and ns < self._last_sent_command["time_ns"]:
            return
        self._last_sent_command = {
            "time_ns": ns, "left_nm": float(getattr(command, "left_nm", 0.0)),
            "right_nm": float(getattr(command, "right_nm", 0.0)),
            "limit_nm": float(command.torque_limit_nm), "enabled": bool(command.enabled),
        }
        t = self._t(ns)
        self._append_motor_values("指令", t, self._last_sent_command["left_nm"],
                                  self._last_sent_command["right_nm"])
        self._render_traces()
        self._render_values()

    def clear_command(self) -> None:
        """Forget a detached output's last send, keeping model/desired data."""
        self._last_sent_command = None
        self._motor_values_diverged.discard("指令")
        for label in [key for key in self._traces if key.startswith("指令")]:
            trace = self._traces.pop(label)
            self._assistance_view.removeItem(trace["curve"])
            self._plot.getPlotItem().legend.removeItem(trace["curve"])
        self._sync_legend_row()
        self._render_values()

    def _append_motor_values(self, kind, t, left, right) -> None:
        shared = self._motor_distribution == "right_to_both_same_sign"
        if shared and left is not None and right is not None and left != right:
            # Preserve evidence of unequal real sends instead of calling them
            # synchronized. Keep separate traces until the next clear/model.
            self._motor_values_diverged.add(kind)
        if (shared and kind not in self._motor_values_diverged
                and left is not None and right is not None and left == right):
            caption = "期望" if kind == "期望辅助" else "下发"
            stage = "限幅前" if kind == "期望辅助" else "限幅后"
            self._append(f"{kind}右 (Nm)", t, float(right),
                         legend_label=f"{caption}左右同步(同号)·{stage}")
            return
        for side, value in (("右", right), ("左", left)):
            if not shared and ((side == "左" and self._single_output_maps_to == "right_torque")
                    or (side == "右" and self._single_output_maps_to == "left_torque")):
                continue
            if value is not None:
                self._append(f"{kind}{side} (Nm)", t, float(value))

    def _append(self, label: str, x: float, y: float, *, legend_label=None) -> None:
        if legend_label is None:
            if label.startswith("期望辅助"):
                legend_label = label.replace("期望辅助", "期望").replace(" (Nm)", " (Nm，限幅前)")
            elif label.startswith("指令"):
                legend_label = label.replace("指令", "下发").replace(" (Nm)", " (Nm，限幅后)")
            else:
                legend_label = label
        trace = self._traces.get(label)
        if trace is None:
            color = {"期望辅助右 (Nm)": "#d97706", "期望辅助左 (Nm)": "#7c3aed",
                     "指令右 (Nm)": "#2563eb", "指令左 (Nm)": "#64748b"}.get(label, "#64748b")
            style = Qt.PenStyle.DashLine if label.startswith("指令") else Qt.PenStyle.SolidLine
            curve = pg.PlotDataItem(pen=pg.mkPen(color, width=1.6, style=style), name=label)
            self._assistance_view.addItem(curve)
            self._plot.getPlotItem().legend.addItem(curve, legend_label)
            self._sync_legend_row()
            trace = self._trace(curve)
            trace["legend_label"] = legend_label
            self._traces[label] = trace
        elif trace["legend_label"] != legend_label:
            legend = self._plot.getPlotItem().legend
            legend.removeItem(trace["curve"])
            legend.addItem(trace["curve"], legend_label)
            trace["legend_label"] = legend_label
            self._sync_legend_row()
        self._store(trace, x, y)

    def _store(self, trace: dict, t: float, value: float) -> None:
        if t < 0 or t < self._latest_t - self._window_s:
            return
        # Actual command history is monotonic per trace. Ignore replayed
        # frames when UI feedback polling observes the same send twice.
        if trace["times"] and t <= trace["times"][-1]:
            return
        trace["times"].append(t)
        trace["xs"].append(t % self._window_s)
        trace["ys"].append(value)

    def _render_traces(self) -> None:
        for trace in [*self._raw_traces, *self._traces.values()]:
            while trace["times"] and trace["times"][0] <= self._latest_t - self._window_s:
                for key in ("times", "xs", "ys"):
                    trace[key].popleft()
            xs, ys = [], []
            previous = None
            for x, y in zip(trace["xs"], trace["ys"]):
                if previous is not None and x < previous:
                    xs.append(float("nan"))
                    ys.append(float("nan"))
                xs.append(x)
                ys.append(y)
                previous = x
            trace["curve"].setData(xs, ys, connect="finite")

    def clear(self) -> None:
        self._t0_ns = None
        self._latest_t = 0.0
        self._cursor_line.setPos(0)
        for trace in [*self._raw_traces, *self._traces.values()]:
            trace["times"].clear()
            trace["xs"].clear()
            trace["ys"].clear()
            trace["curve"].setData([], [])
        for trace in self._traces.values():
            self._assistance_view.removeItem(trace["curve"])
            self._plot.getPlotItem().legend.removeItem(trace["curve"])
        self._traces.clear()
        self._sync_legend_row()
        self._estimate_per_kg = None
        self._raw_left = self._raw_right = None
        self._estimate_left_nm = self._estimate_right_nm = None
        self._desired_left_nm = self._desired_right_nm = None
        self._assistance_errors = []
        self._last_sent_command = None
        self._motor_values_diverged.clear()
        self._render_values()


__all__ = ["ModelOutputPanel"]
