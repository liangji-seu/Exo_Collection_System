"""Shared signal preview widgets and rendering for Collector and model runtime.

The preview implementation intentionally owns only display state.  Recording,
worker lifecycle, and application specific alert handling remain in the host
window through the small hooks exposed by :class:`SignalPreviewMixin`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


MAX_PREVIEW_POINTS = 4096
SIGNAL_RING_CAPACITY = 1000
ULTRASOUND_PREVIEW_SAMPLES = 1000
_IMU_SENSOR_LABELS = ("imu_left_leg", "imu_right_leg", "imu_pelvis")
_IMU_AXIS_NAMES = ("acc_x", "acc_y", "acc_z")
_IMU_AXIS_COLORS = {"acc_x": "#dc3545", "acc_y": "#0d6efd", "acc_z": "#198754"}
IMU_PREVIEW_LABELS: tuple[str, ...] = tuple(
    f"{sensor}_{axis}"
    for sensor in _IMU_SENSOR_LABELS
    for axis in _IMU_AXIS_NAMES
)
ENCODER_PREVIEW_LABELS = (
    "left_position",
    "left_velocity",
    "left_torque",
    "right_position",
    "right_velocity",
    "right_torque",
)
EMG_PREVIEW_RING_CAPACITY = 16_000
_SIGNAL_COLORS = (
    "#0d6efd", "#dc3545", "#198754", "#d97706",
    "#6f42c1", "#0dcaf0", "#fd7e14", "#20c997",
)
_ENCODER_METRICS = (
    ("position", "位置", "rad", "#d97706"),
    ("velocity", "速度", "rad/s", "#0d6efd"),
    ("torque", "估算扭矩", "N·m", "#198754"),
)
_ENCODER_SHARED_Y_RANGE = (-13.0, 13.0)

SIGNAL_PREVIEW_PANEL_TITLES = {
    "ultrasound": "超声数据",
    "imu": "IMU 数据",
    "encoder": "电机编码器数据",
    "mocap": "动捕 Marker 数据",
    "emg": "EMG 数据",
}
# Short compatibility spelling for hosts that treat this as the panel title map.
SIGNAL_PREVIEW_TITLES = SIGNAL_PREVIEW_PANEL_TITLES

__all__ = [
    "MAX_PREVIEW_POINTS",
    "SIGNAL_RING_CAPACITY",
    "ULTRASOUND_PREVIEW_SAMPLES",
    "IMU_PREVIEW_LABELS",
    "ENCODER_PREVIEW_LABELS",
    "EMG_PREVIEW_RING_CAPACITY",
    "SIGNAL_PREVIEW_PANEL_TITLES",
    "SIGNAL_PREVIEW_TITLES",
    "HoverDetailsPlotWidget",
    "NoWheelViewBox",
    "RingTrace",
    "SignalPreviewMixin",
]


class NoWheelViewBox(pg.ViewBox):
    """Keep preview scales fixed even when an axis receives a wheel event."""

    def wheelEvent(self, event: Any, axis: Any = None) -> None:  # noqa: N802
        event.ignore()


class HoverDetailsPlotWidget(pg.PlotWidget):
    """Plot that reveals its title, axis labels, and legend only on hover."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._hover_details_visible = False
        self._details_always_visible = False
        kwargs.setdefault("viewBox", NoWheelViewBox())
        super().__init__(*args, **kwargs)
        self._set_hover_details_visible(False)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        # Ignore so an enclosing scroll area may scroll; never dispatch the
        # event to pyqtgraph's scene/axis zoom handlers.
        event.ignore()

    def setTitle(self, *args: Any, **kwargs: Any) -> None:  # noqa: N802
        self.getPlotItem().setTitle(*args, **kwargs)
        self._set_hover_details_visible(self._hover_details_visible)

    def setLabel(self, *args: Any, **kwargs: Any) -> None:  # noqa: N802
        self.getPlotItem().setLabel(*args, **kwargs)
        self._set_hover_details_visible(self._hover_details_visible)

    def addLegend(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
        legend = self.getPlotItem().addLegend(*args, **kwargs)
        self._set_hover_details_visible(self._hover_details_visible)
        return legend

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._set_hover_details_visible(True)
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._set_hover_details_visible(False)
        super().leaveEvent(event)

    def _set_hover_details_visible(self, visible: bool) -> None:
        self._hover_details_visible = bool(visible or self._details_always_visible)
        plot_item = self.getPlotItem()
        title_label = plot_item.titleLabel
        title_label.setVisible(
            self._hover_details_visible and bool(title_label.text)
        )
        for axis_name in ("left", "bottom", "right"):
            axis = plot_item.getAxis(axis_name)
            axis.setStyle(showValues=self._hover_details_visible)
            axis.label.setVisible(
                self._hover_details_visible and bool(axis.labelText)
            )
        legend = plot_item.legend
        if legend is not None:
            legend.setVisible(self._hover_details_visible)

    def set_details_always_visible(self, visible: bool) -> None:
        self._details_always_visible = bool(visible)
        self._set_hover_details_visible(visible)


class RingTrace:
    """Ring-buffer trace backed by a fixed-size numpy array for pyqtgraph."""

    __slots__ = (
        "_buffer", "_capacity", "_count", "_cursor", "_x",
        "_marker_lines", "_render_stride", "curve", "cursor_line", "plot",
    )

    def __init__(
        self, plot: "pg.PlotWidget", pen: str, label: str,
        *, capacity: int = SIGNAL_RING_CAPACITY, render_stride: int = 1,
    ) -> None:
        if capacity < 2:
            raise ValueError("ring trace capacity must be at least two")
        self._capacity = int(capacity)
        self._render_stride = max(1, int(render_stride))
        self._buffer = np.full(self._capacity, np.nan, dtype=np.float64)
        self._x = np.arange(self._capacity, dtype=np.float64)
        self._cursor = 0
        self._count = 0
        self._marker_lines: dict[int, "pg.InfiniteLine"] = {}
        self.plot = plot
        self.curve = plot.plot(pen=pg.mkPen(pen, width=1.2))
        self.cursor_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#dc3545", width=6))
        plot.addItem(self.cursor_line)
        plot.setTitle(label)
        plot.setBackground("w")
        plot.setXRange(0, self._capacity - 1, padding=0)
        plot.setLimits(xMin=0, xMax=self._capacity - 1, minXRange=self._capacity - 1, maxXRange=self._capacity - 1)
        plot.setMouseEnabled(x=False, y=False)
        plot.setLabel("bottom", "循环帧位置")
        plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve.setData(self._x, self._buffer)

    def append(self, values: np.ndarray | list[float]) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        n = int(arr.size)
        if n == 0:
            return
        self._clear_overwritten_markers(n)
        next_cursor = (self._cursor + n) % self._capacity
        if n >= self._capacity:
            tail = arr[-self._capacity :]
            split = self._capacity - next_cursor
            self._buffer[next_cursor:] = tail[:split]
            self._buffer[:next_cursor] = tail[split:]
        else:
            first_count = min(n, self._capacity - self._cursor)
            self._buffer[self._cursor : self._cursor + first_count] = arr[:first_count]
            overflow = n - first_count
            if overflow:
                self._buffer[:overflow] = arr[first_count:]
        self._cursor = next_cursor
        self._count = min(self._capacity, self._count + n)
        self._render()

    def mark_current(self, label: str) -> int:
        slot = (self._cursor - 1) % self._capacity if self._count else self._cursor
        existing = self._marker_lines.get(slot)
        if existing is not None:
            previous = existing.toolTip().strip()
            existing.setToolTip(
                f"{previous}\n{label}" if previous and label not in previous else label
            )
            return slot
        line = pg.InfiniteLine(
            pos=float(slot), angle=90, movable=False,
            pen=pg.mkPen("#ff0000", width=1.5, style=Qt.PenStyle.DashLine),
        )
        line.setZValue(90)
        line.setToolTip(label)
        self.plot.addItem(line)
        self._marker_lines[slot] = line
        return slot

    def _clear_overwritten_markers(self, count: int) -> None:
        if count >= self._capacity:
            overwritten = tuple(self._marker_lines)
        else:
            overwritten = tuple(
                (self._cursor + offset) % self._capacity for offset in range(count)
            )
        for slot in overwritten:
            line = self._marker_lines.pop(slot, None)
            if line is not None:
                self.plot.removeItem(line)

    def _render(self) -> None:
        if self._render_stride > 1:
            step = self._render_stride
            x = self._x[::step]
            display = self._buffer[::step].copy()
            if self._count == self._capacity:
                display[(self._cursor // step) % display.size] = np.nan
            self.curve.setData(x, display)
        else:
            display = self._buffer.copy()
            if self._count == self._capacity:
                display[self._cursor] = np.nan
            self.curve.setData(self._x, display)
        if self._count:
            self.cursor_line.setPos((self._cursor - 1) % self._capacity)

    def reset(self) -> None:
        self._clear_overwritten_markers(self._capacity)
        self._buffer.fill(np.nan)
        self._cursor = 0
        self._count = 0
        self.curve.setData(self._x, self._buffer)
        self.cursor_line.setPos(0.0)


class SignalPreviewMixin:
    """Shared five-modality preview construction and rendering.

    Hosts should call :meth:`init_signal_previews` before constructing panels.
    ``_on_signal_preview_alert`` is intentionally a no-op hook so applications
    can connect their own alert and timeline sinks.
    """

    def init_signal_previews(self) -> None:
        self._us_plots: list[pg.PlotWidget] = []
        self._us_curves: list[pg.PlotDataItem] = []
        self._us_x = np.arange(ULTRASOUND_PREVIEW_SAMPLES, dtype=np.float64)
        self._ultrasound_format_alerted: set[tuple[int, str]] = set()
        self._imu_traces: dict[str, RingTrace] = {}
        self._enc_traces: dict[str, RingTrace] = {}
        self._emg_traces: dict[str, RingTrace] = {}
        self._emg_grid_layout: QVBoxLayout | None = None
        self._emg_grid_content: QWidget | None = None
        self._mocap_table: QTableWidget | None = None
        self._preview_y_ranges: dict[str, tuple[float, float]] = {}

    def _on_signal_preview_alert(self, message: str) -> None:
        del message

    def build_signal_preview_panels(self) -> dict[str, QWidget]:
        """Build the five sensor panels and return them keyed by modality."""
        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")

        us_grid = QGroupBox("超声 · 4 通道当前单帧")
        us_grid.setObjectName("ultrasound_grid")
        us_grid.setMinimumHeight(120)
        us_layout = QGridLayout(us_grid)
        us_layout.setContentsMargins(0, 0, 0, 0)
        for i in range(4):
            plot = HoverDetailsPlotWidget(title=f"超声通道 {i + 1} · 当前帧")
            plot.setObjectName(f"ultrasound_preview_ch{i}")
            plot.setBackground("w")
            plot.setXRange(0, ULTRASOUND_PREVIEW_SAMPLES - 1, padding=0)
            plot.setLimits(
                xMin=0, xMax=ULTRASOUND_PREVIEW_SAMPLES - 1,
                minXRange=ULTRASOUND_PREVIEW_SAMPLES - 1,
                maxXRange=ULTRASOUND_PREVIEW_SAMPLES - 1,
            )
            plot.setMouseEnabled(x=False, y=False)
            plot.setLabel("bottom", "单帧采样点")
            plot.showGrid(x=True, y=True, alpha=0.2)
            curve = plot.plot(pen=pg.mkPen("#0f766e", width=1.2))
            curve.setData(self._us_x, np.full(ULTRASOUND_PREVIEW_SAMPLES, np.nan, dtype=np.float64))
            self._us_plots.append(plot)
            self._us_curves.append(curve)
            us_layout.addWidget(plot, i // 2, i % 2)

        imu_grid = QGroupBox("IMU · 加速度 3 轴循环帧")
        imu_grid.setObjectName("imu_ring_grid")
        imu_grid.setMinimumHeight(120)
        imu_layout = QHBoxLayout(imu_grid)
        imu_layout.setContentsMargins(0, 0, 0, 0)
        sensor_display = ("左腿", "右腿", "盆骨")
        for sensor_idx, sensor_label in enumerate(_IMU_SENSOR_LABELS):
            plot = HoverDetailsPlotWidget()
            plot.setObjectName(f"imu_ring_{sensor_label}")
            plot.addLegend(offset=(-1, 1))
            for axis_name in _IMU_AXIS_NAMES:
                trace_label = f"{sensor_label}_{axis_name}"
                trace = RingTrace(
                    plot, _IMU_AXIS_COLORS.get(axis_name, "#888888"),
                    f"IMU{sensor_idx + 1} {sensor_display[sensor_idx]} {axis_name}",
                    capacity=250,
                )
                self._imu_traces[trace_label] = trace
            plot.setYRange(-10, 10, padding=0)
            plot.setLimits(yMin=-10, yMax=10, minYRange=20, maxYRange=20)
            plot.setMouseEnabled(x=False, y=False)
            imu_layout.addWidget(plot, 1)
        self._preview_y_ranges["imu"] = (-10.0, 10.0)

        enc_grid = QGroupBox("电机编码器 · 每侧同图显示位置 / 速度 / 估算扭矩")
        enc_grid.setObjectName("encoder_ring_grid")
        enc_grid.setMinimumHeight(120)
        enc_layout = QHBoxLayout(enc_grid)
        enc_layout.setContentsMargins(0, 0, 0, 0)
        for side_key, side_name in (("left", "左侧"), ("right", "右侧")):
            plot = HoverDetailsPlotWidget()
            plot.setObjectName(f"encoder_ring_{side_key}")
            legend = plot.addLegend(offset=(5, 5))
            shared_cursor = None
            for metric, metric_name, unit, color in _ENCODER_METRICS:
                trace = RingTrace(plot, color, f"{side_name}电机", capacity=500)
                if shared_cursor is None:
                    shared_cursor = trace.cursor_line
                else:
                    plot.removeItem(trace.cursor_line)
                    trace.cursor_line = shared_cursor
                legend.addItem(trace.curve, f"{metric_name} ({unit})")
                self._enc_traces[f"{side_key}_{metric}"] = trace
            lower, upper = _ENCODER_SHARED_Y_RANGE
            span = upper - lower
            plot.setTitle(f"{side_name}电机编码器")
            plot.setYRange(lower, upper, padding=0)
            plot.setLimits(yMin=lower, yMax=upper, minYRange=span, maxYRange=span)
            plot.setLabel("left", "数值")
            plot.setMouseEnabled(x=False, y=False)
            enc_layout.addWidget(plot, 1)
        self._preview_y_ranges["encoder"] = _ENCODER_SHARED_Y_RANGE

        mocap_grid = QGroupBox("动捕 Marker · 实时坐标")
        mocap_grid.setObjectName("mocap_marker_grid")
        mocap_grid.setMinimumHeight(120)
        mocap_layout = QVBoxLayout(mocap_grid)
        mocap_layout.setContentsMargins(0, 0, 0, 0)
        mocap_table = QTableWidget(0, 4, mocap_grid)
        mocap_table.setObjectName("mocap_marker_table")
        mocap_table.setHorizontalHeaderLabels(["Marker", "X (mm)", "Y (mm)", "Z (mm)"])
        mocap_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        mocap_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        mocap_table.setAlternatingRowColors(True)
        mocap_table.verticalHeader().setVisible(False)
        mocap_header = mocap_table.horizontalHeader()
        mocap_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 4):
            mocap_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self._mocap_table = mocap_table
        mocap_layout.addWidget(mocap_table)

        emg_grid = QGroupBox("表面肌电 EMG · 每通道一个窗口")
        emg_grid.setMinimumHeight(120)
        self._emg_grid_layout = QVBoxLayout(emg_grid)
        self._emg_grid_layout.setContentsMargins(0, 0, 0, 0)
        placeholder = QLabel("等待 EMG 数据…（通道窗口将随配置自动生成）")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("QLabel { color: #6b7280; padding: 16px; }")
        self._emg_grid_layout.addWidget(placeholder)
        self._emg_grid_content = placeholder

        return {
            "ultrasound": us_grid,
            "imu": imu_grid,
            "encoder": enc_grid,
            "mocap": mocap_grid,
            "emg": emg_grid,
        }

    def feed_signal_preview(self, modality: str, payload: Mapping[str, Any]) -> None:
        """Render one raw preview payload into the shared display state."""
        modality = self._normalize_modality(modality)
        if modality == "ultrasound":
            raw_channels = payload.get("channels")
            if not isinstance(raw_channels, (list, tuple)):
                legacy_values = payload.get("values")
                raw_channels = [legacy_values] if legacy_values is not None else []
            raw_channel_index = payload.get("channel_index")
            channel_index: int | None = None
            if raw_channel_index is not None:
                try:
                    candidate = int(raw_channel_index)
                except (TypeError, ValueError):
                    candidate = -1
                if 0 <= candidate < len(self._us_curves):
                    channel_index = candidate
                    if "ultrasound" not in self._preview_y_ranges:
                        lower, upper = -128.0, 128.0
                        span = upper - lower
                        self._preview_y_ranges["ultrasound"] = (lower, upper)
                        for plot in self._us_plots:
                            plot.setYRange(lower, upper, padding=0.0)
                            plot.setLimits(yMin=lower, yMax=upper, minYRange=span, maxYRange=span)
                            plot.setMouseEnabled(x=False, y=False)
            raw_metrics = payload.get("format_metrics")
            if isinstance(raw_metrics, (list, tuple)):
                for metric_offset, metric in enumerate(raw_metrics[:4]):
                    if not isinstance(metric, Mapping) or not bool(metric.get("all_zero")):
                        continue
                    metric_channel = channel_index if channel_index is not None else metric_offset
                    alert_key = (metric_channel, "ALL_ZERO")
                    if alert_key in self._ultrasound_format_alerted:
                        continue
                    message = f"ultrasound 通道 {metric_channel + 1} 当前帧全零；请检查探头、通道和设备连接。"
                    self._on_signal_preview_alert(message)
                    self._ultrasound_format_alerted.add(alert_key)
            prepared_channels: list[tuple[int, list[float]]] = []
            for i, raw_channel in enumerate(raw_channels):
                target_index = channel_index if channel_index is not None else i
                if target_index >= len(self._us_curves):
                    break
                values = self._numeric_values(raw_channel)
                if values:
                    prepared_channels.append((target_index, values))
            self._lock_preview_y_axis("ultrasound", [values for _, values in prepared_channels], self._us_plots)
            for index, values in prepared_channels:
                self._us_curves[index].setData(self._us_x, self._fixed_ultrasound_frame(values))
            return
        if modality == "imu":
            prepared_series: list[tuple[str, list[float]]] = []
            for label, values in self._preview_series(payload, IMU_PREVIEW_LABELS):
                numeric = self._numeric_values(values)
                if label in self._imu_traces and numeric:
                    prepared_series.append((label, numeric))
            self._lock_preview_y_axis("imu", [values for _, values in prepared_series], [trace.plot for trace in self._imu_traces.values()])
            for label, values in prepared_series:
                self._imu_traces[label].append(values)
            return
        if modality == "encoder":
            prepared_series = []
            for label, values in self._preview_series(payload, ENCODER_PREVIEW_LABELS):
                numeric = self._numeric_values(values)
                if label in self._enc_traces and numeric:
                    prepared_series.append((label, numeric))
            for label, values in prepared_series:
                self._enc_traces[label].append(values)
            return
        if modality == "emg":
            channels = payload.get("channels")
            if not isinstance(channels, (list, tuple)):
                return
            labels = self._emg_preview_labels(payload, len(channels))
            if tuple(labels) != tuple(self._emg_traces.keys()):
                self._build_emg_preview(labels)
            for index, raw_values in enumerate(channels):
                values = self._numeric_values(raw_values)
                if values:
                    self._emg_traces[labels[index]].append(values)
            return
        if modality == "mocap":
            self._update_mocap_table(payload)

    def _update_mocap_table(self, payload: Mapping[str, Any]) -> None:
        table = self._mocap_table
        if table is None:
            return
        raw_names = payload.get("marker_names")
        if not isinstance(raw_names, (list, tuple)):
            raw_names = payload.get("labels")
        names = [str(item) for item in raw_names] if isinstance(raw_names, (list, tuple)) else []
        raw_xyz = payload.get("latest_xyz_mm")
        xyz_rows = raw_xyz if isinstance(raw_xyz, (list, tuple)) else ()
        channels = payload.get("channels")
        marker_count = max(len(names), len(xyz_rows), int(payload.get("marker_count") or 0))
        if marker_count <= 0 and isinstance(channels, (list, tuple)):
            marker_count = len(channels)
        table.setRowCount(marker_count)
        for row in range(marker_count):
            name = names[row] if row < len(names) else f"marker_{row + 1:02d}"
            values: list[float | None] = [None, None, None]
            if row < len(xyz_rows):
                raw_row = xyz_rows[row]
                if isinstance(raw_row, (list, tuple)):
                    for axis in range(min(3, len(raw_row))):
                        try:
                            number = float(raw_row[axis])
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(number):
                            values[axis] = number
            elif isinstance(channels, (list, tuple)) and row < len(channels):
                z_values = self._numeric_values(channels[row])
                if z_values:
                    values[2] = z_values[-1]
            cells = [name, *["—" if value is None else f"{value:.3f}" for value in values]]
            for column, text in enumerate(cells):
                item = table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    table.setItem(row, column, item)
                item.setText(text)

    @staticmethod
    def _emg_preview_labels(payload: Mapping[str, Any], count: int) -> list[str]:
        raw = payload.get("labels")
        labels = [str(item) for item in raw] if isinstance(raw, (list, tuple)) and len(raw) == count else []
        if not (labels and all(labels) and len(set(labels)) == count):
            labels = [f"emg_{index + 1:02d}" for index in range(count)]
        raw_serials = payload.get("unit_serials")
        serials = [str(item).strip() for item in raw_serials] if isinstance(raw_serials, (list, tuple)) and len(raw_serials) == count else []
        return [f"{label} ({serial})" if serial else label for label, serial in zip(labels, serials)] if serials else labels

    def _build_emg_preview(self, labels: Sequence[str]) -> None:
        grid_layout = self._emg_grid_layout
        if grid_layout is None:
            return
        if self._emg_grid_content is not None:
            grid_layout.removeWidget(self._emg_grid_content)
            self._emg_grid_content.deleteLater()
            self._emg_grid_content = None
        self._emg_traces = {}
        container = QWidget()
        layout = QGridLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        for index, label in enumerate(labels):
            text = str(label)
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            name_label = QLabel(text)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("QLabel { font-weight: 700; color: #374151; }")
            cell_layout.addWidget(name_label)
            plot = HoverDetailsPlotWidget()
            plot.setObjectName(f"emg_ring_{index + 1}")
            plot.setMinimumHeight(110)
            legend = plot.addLegend(offset=(5, 5))
            trace = RingTrace(plot, _SIGNAL_COLORS[index % len(_SIGNAL_COLORS)], text, capacity=EMG_PREVIEW_RING_CAPACITY, render_stride=4)
            legend.addItem(trace.curve, text)
            plot.setLabel("left", "幅值")
            cell_layout.addWidget(plot, 1)
            layout.addWidget(cell, index // 4, index % 4)
            self._emg_traces[text] = trace
        grid_layout.addWidget(container)
        self._emg_grid_content = container

    def _lock_preview_y_axis(self, modality: str, series: list[list[float]], plots: list[pg.PlotWidget]) -> None:
        if modality in self._preview_y_ranges or not series:
            return
        values = np.concatenate([np.asarray(channel, dtype=np.float64) for channel in series])
        finite = values[np.isfinite(values)]
        if not finite.size:
            return
        minimum, maximum = float(np.min(finite)), float(np.max(finite))
        if modality == "ultrasound" and minimum >= 0:
            lower, upper = 0.0, max(1.0, maximum * 1.1)
        else:
            extent = max(abs(minimum), abs(maximum), 1e-6) * 1.1
            lower, upper = -extent, extent
        span = upper - lower
        self._preview_y_ranges[modality] = (lower, upper)
        for plot in plots:
            plot.setYRange(lower, upper, padding=0)
            plot.setLimits(yMin=lower, yMax=upper, minYRange=span, maxYRange=span)
            plot.setMouseEnabled(x=False, y=False)

    def _fixed_ultrasound_frame(self, values: list[float]) -> np.ndarray:
        source = np.asarray(values, dtype=np.float64)
        if source.size > ULTRASOUND_PREVIEW_SAMPLES:
            source = source[np.linspace(0, source.size - 1, ULTRASOUND_PREVIEW_SAMPLES, dtype=np.int64)]
        display = np.full(ULTRASOUND_PREVIEW_SAMPLES, np.nan, dtype=np.float64)
        display[: source.size] = source
        return display

    @staticmethod
    def _preview_series(payload: Mapping[str, Any], expected_labels: tuple[str, ...]) -> list[tuple[str, object]]:
        channels = payload.get("channels")
        if isinstance(channels, Mapping):
            return [(label, channels[label]) for label in expected_labels if label in channels]
        if isinstance(channels, (list, tuple)):
            labels = payload.get("labels")
            provided_labels = labels if isinstance(labels, (list, tuple)) else ()
            result: list[tuple[str, object]] = []
            for index, values in enumerate(channels[: len(expected_labels)]):
                candidate = str(provided_labels[index]) if index < len(provided_labels) else expected_labels[index]
                result.append((candidate if candidate in expected_labels else expected_labels[index], values))
            return result
        streams = payload.get("streams")
        if isinstance(streams, (list, tuple)):
            result = []
            for index, stream in enumerate(streams[: len(expected_labels)]):
                if isinstance(stream, Mapping):
                    candidate = str(stream.get("label") or expected_labels[index])
                    result.append((candidate if candidate in expected_labels else expected_labels[index], stream.get("values")))
            return result
        legacy_values = payload.get("values")
        return [(expected_labels[0], legacy_values)] if legacy_values is not None else []

    @staticmethod
    def _numeric_values(value: object) -> list[float]:
        if not isinstance(value, (list, tuple)):
            return []
        converted: list[float] = []
        for item in value:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                converted.append(number)
            if len(converted) >= MAX_PREVIEW_POINTS:
                break
        return converted

    @staticmethod
    def _normalize_modality(value: str) -> str:
        normalized = value.strip().lower()
        if "ultrasound" in normalized:
            return "ultrasound"
        if normalized == "imu" or "imu" in normalized:
            return "imu"
        if "encoder" in normalized:
            return "encoder"
        if "mocap" in normalized or "motion" in normalized or "marker" in normalized:
            return "mocap"
        if "force_plate" in normalized or "forceplate" in normalized:
            return "force_plate"
        if "emg" in normalized or "analog" in normalized:
            return "emg"
        if "sync" in normalized or "pulse" in normalized:
            return "sync_pulse"
        return normalized
