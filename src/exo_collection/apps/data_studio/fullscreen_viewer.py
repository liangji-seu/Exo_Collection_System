"""Full-screen, multi-modality trial viewer for Data Studio.

One dock per modality, arranged with the adaptive ``PreviewWorkspace`` layout,
sharing a single global timeline across every panel:

- 动捕 3D: marker points + optional skeleton lines (no OpenSim dependency);
- 超声: waterfall (reused sweep plot);
- 力矩 CSV: real-time line plot with the shared vertical cursor;
- IMU: 3 devices × 3 measurements on independent real-time axes;
- EMG: one window with stacked channels and a fixed y-range;
- 编码器: migrated encoder curve.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from exo_collection.apps.collector.preview_workspace import PreviewWorkspace
from .local_dialogs import (
    _SweepWaterfallPlot,
    _encoder_side_groups,
    _imu_sensor_rows,
)
from .local_tools import MocapPlayback, TrialPlayback
from .plots import TimeSeriesPlot

_WINDOW_SECONDS = 10.0

# Bone segments are matched by marker-name *suffix* (after the marker-set prefix,
# e.g. ``010_no_exo_dynamic/R.ASIS`` → ``R.ASIS``), case-insensitively.  The
# lower-body CAST-style names cover the exoskeleton experiments actually recorded;
# the full-body Plug-in-Gait names remain as a fallback for datasets that use that
# convention.  Only segments whose two endpoints both exist are drawn; unknown
# marker sets fall back to points plus a ground grid.
_BONE_SEGMENTS = (
    # -- lower body: pelvis triangle, thighs, shanks, feet --
    ("R.ASIS", "L.ASIS"),
    ("R.ASIS", "V.Sacral"),
    ("L.ASIS", "V.Sacral"),
    ("R.ASIS", "R.Thigh"),
    ("R.Thigh", "R.Knee"),
    ("R.Knee", "R.Shank"),
    ("R.Shank", "R.Ankle"),
    ("R.Ankle", "R.Heel"),
    ("R.Ankle", "R.Toe"),
    ("R.Heel", "R.Toe"),
    ("L.ASIS", "L.Thigh"),
    ("L.Thigh", "L.Knee"),
    ("L.Knee", "L.Shank"),
    ("L.Shank", "L.Ankle"),
    ("L.Ankle", "L.Heel"),
    ("L.Ankle", "L.Toe"),
    ("L.Heel", "L.Toe"),
    # -- medial knee/ankle markers (static calibration recordings only) --
    ("R.Knee", "R.Knee.Medial"),
    ("R.Ankle", "R.Ankle.Medial"),
    ("L.Knee", "L.Knee.Medial"),
    ("L.Ankle", "L.Ankle.Medial"),
    # -- full-body Plug-in-Gait (fallback for other datasets) --
    ("C7", "T10"),
    ("C7", "CLAV"),
    ("CLAV", "STRN"),
    ("T10", "STRN"),
    ("STRN", "RBAK"),
    ("CLAV", "RSHO"),
    ("CLAV", "LSHO"),
    ("RSHO", "RUPA"),
    ("RUPA", "RELB"),
    ("RELB", "RFRA"),
    ("RFRA", "RWRA"),
    ("LSHO", "LUPA"),
    ("LUPA", "LELB"),
    ("LELB", "LFRA"),
    ("LFRA", "LWRA"),
    ("RASI", "LASI"),
    ("LASI", "LPSI"),
    ("RASI", "RPSI"),
    ("LPSI", "RPSI"),
    ("RASI", "RTHI"),
    ("RTHI", "RKNE"),
    ("RKNE", "RTIB"),
    ("RTIB", "RANK"),
    ("RANK", "RHEE"),
    ("LASI", "LTHI"),
    ("LTHI", "LKNE"),
    ("LKNE", "LTIB"),
    ("LTIB", "LANK"),
    ("LANK", "LHEE"),
)


class Mocap3DCanvas(pg.PlotWidget):
    """Orthographic 3D marker view: rotate/pan/zoom, points plus skeleton."""

    def __init__(self, mocap: MocapPlayback, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mocap = mocap
        self._azimuth = 55.0
        self._elevation = 20.0
        self._frame = 0
        self._last_pos = None
        self._center: np.ndarray | None = None
        self._fit_radius = 500.0
        self.setBackground("#ffffff")
        self.setTitle("动捕 · marker 3D")
        self.setLabel("bottom", "X", units="mm")
        self.setLabel("left", "Z", units="mm")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setMenuEnabled(False)
        self.setAspectLocked(True)
        self._markers = pg.ScatterPlotItem(
            size=9,
            brush=pg.mkBrush("#E15759"),
            pen=pg.mkPen("#7A1F1F", width=1.0),
        )
        self.addItem(self._markers)
        self._segments: list[pg.PlotDataItem] = []
        self._name_to_index = {
            name.casefold().rsplit("/", 1)[-1]: index
            for index, name in enumerate(mocap.marker_names)
        }
        pen = pg.mkPen("#8C8C8C", width=1.6)
        for left, right in _BONE_SEGMENTS:
            if left.casefold() in self._name_to_index and right.casefold() in self._name_to_index:
                line = pg.PlotDataItem([], [], pen=pen, antialias=True)
                self.addItem(line)
                self._segments.append((left, right, line))
        self._auto_fit()
        self.set_frame(0)

    def _auto_fit(self) -> None:
        pts = np.asarray(self._mocap.positions, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.size == 0:
            self._center = None
            self._fit_radius = 500.0
            self.setXRange(-500.0, 500.0, padding=0.0)
            self.setYRange(-500.0, 500.0, padding=0.0)
            return
        center = np.median(pts, axis=0)
        radius = float(np.percentile(np.linalg.norm(pts - center, axis=1), 95))
        self._center = center
        self._fit_radius = max(radius, 150.0) * 1.2
        self._recenter()

    def _recenter(self) -> None:
        """Center the view on the *projected* body, not the raw coordinates.

        The orthographic projection rotates the marker cloud, so fitting the X/Y
        ranges to raw X/Z coordinates (the old behaviour) left an off-origin body
        — e.g. Nokov data sitting around x≈-4.6 m — entirely outside the view.
        A rotation preserves distances, so a square range of the 3D bounding
        sphere radius around the projected centre contains every marker.
        """
        if self._center is None:
            return
        projected = self._project(self._center[None, :])[0]
        radius = self._fit_radius
        self.setXRange(projected[0] - radius, projected[0] + radius, padding=0.0)
        self.setYRange(projected[1] - radius, projected[1] + radius, padding=0.0)

    def _project(self, points: np.ndarray) -> np.ndarray:
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        # Nokov's raw x-axis is medial-lateral with the subject's RIGHT side at
        # the more negative x (e.g. R.ASIS < L.ASIS). Negate it so the right
        # side maps to +screen_x (the plot's right-hand side); otherwise the
        # view is a left-right mirror and the right foot's stomps appear to come
        # from the left foot.
        x = -points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        screen_x = x * math.cos(az) - y * math.sin(az)
        depth = x * math.sin(az) + y * math.cos(az)
        screen_y = z * math.cos(el) - depth * math.sin(el)
        return np.column_stack((screen_x, screen_y))

    def set_frame(self, index: int) -> None:
        positions = np.asarray(self._mocap.positions, dtype=np.float64)
        if positions.shape[0] == 0:
            return
        self._frame = int(np.clip(index, 0, positions.shape[0] - 1))
        frame = positions[self._frame]
        finite = np.isfinite(frame).all(axis=1)
        projected = self._project(frame[finite])
        self._markers.setData(projected[:, 0], projected[:, 1])
        for left, right, line in self._segments:
            li = self._name_to_index[left.casefold()]
            ri = self._name_to_index[right.casefold()]
            if not finite[li] or not finite[ri]:
                line.setData([], [])
                continue
            pts = self._project(frame[[li, ri]])
            line.setData(pts[:, 0], pts[:, 1])

    def rotate(self, delta_azimuth: float, delta_elevation: float) -> None:
        self._azimuth = (self._azimuth + delta_azimuth) % 360.0
        self._elevation = max(-89.0, min(89.0, self._elevation + delta_elevation))
        self._recenter()
        self.set_frame(self._frame)

    def mousePressEvent(self, event: object) -> None:
        self._last_pos = event.position()  # type: ignore[attr-defined]
        event.accept()  # type: ignore[attr-defined]

    def mouseMoveEvent(self, event: object) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:  # type: ignore[attr-defined]
            pos = event.position()  # type: ignore[attr-defined]
            if self._last_pos is not None:
                delta = pos - self._last_pos  # type: ignore[attr-defined]
                self.rotate(delta.x() * 0.4, -delta.y() * 0.4)
            self._last_pos = pos
        event.accept()  # type: ignore[attr-defined]


class FullscreenViewer(PreviewWorkspace):
    """Dockable full-screen trial viewer sharing one global timeline."""

    def __init__(self, playback: TrialPlayback, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.playback = playback
        self.setObjectName("fullscreen_viewer")
        self.setWindowTitle(
            f"全屏可视化 · {playback.condition_code} · {playback.trial_uuid[:8]}"
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._playing = False
        self._last_tick = perf_counter()
        self._time_min, self._time_max = self._playback_bounds(playback)
        self._current_time = self._time_min
        total_span = max(self._time_max - self._time_min, 1e-6)
        self._window_s = min(_WINDOW_SECONDS, max(1.0, total_span))
        self._panels: list[object] = []

        self._build_timeline()
        self._build_panels()
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance_playback)
        self.set_playback_time(self._time_min)

    # -- timeline ----------------------------------------------------------
    def _build_timeline(self) -> None:
        toolbar = QToolBar("全局时间轴", self)
        toolbar.setObjectName("fullscreen_timeline")
        toolbar.setMovable(False)
        self._play_button = QPushButton("▶ 播放")
        self._play_button.setObjectName("fullscreen_play_pause")
        self._play_button.clicked.connect(self.toggle_playback)
        toolbar.addWidget(self._play_button)
        self._timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self._timeline_slider.setObjectName("fullscreen_timeline_slider")
        self._timeline_slider.setRange(0, 10_000)
        self._timeline_slider.valueChanged.connect(self._slider_changed)
        toolbar.addWidget(self._timeline_slider)
        self._time_label = QLabel()
        self._time_label.setObjectName("fullscreen_time_label")
        self._time_label.setMinimumWidth(220)
        toolbar.addWidget(self._time_label)
        toolbar.addWidget(QLabel("速度："))
        self._speed_combo = QComboBox()
        self._speed_combo.setObjectName("fullscreen_speed")
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            self._speed_combo.addItem(f"{speed:g}×", speed)
        self._speed_combo.setCurrentIndex(2)
        toolbar.addWidget(self._speed_combo)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _build_panels(self) -> None:
        playback = self.playback
        if playback.mocap is not None:
            self.register_panel(
                "mocap", "动捕 3D", Mocap3DCanvas(playback.mocap)
            )
            self._panels.append(self.dock_for("mocap").widget())
        if playback.ultrasound is not None:
            self.register_panel("ultrasound", "超声瀑布图", self._build_ultrasound_panel())
            self._panels.append(self.dock_for("ultrasound").widget())
        if playback.imu_sensors or playback.imu is not None:
            self.register_panel("imu", "IMU", self._build_imu_panel())
            self._panels.append(self.dock_for("imu").widget())
        if playback.emg is not None and playback.emg.time_s.size:
            self.register_panel("emg", "EMG", self._build_emg_panel())
            self._panels.append(self.dock_for("emg").widget())
        if playback.encoder is not None and playback.encoder.time_s.size:
            self.register_panel("encoder", "电机编码器", self._build_encoder_panel())
            self._panels.append(self.dock_for("encoder").widget())
        if playback.moment is not None and playback.moment.time_s.size:
            self.register_panel("moment", "力矩真值", self._build_moment_panel())
            self._panels.append(self.dock_for("moment").widget())
        self.reset_default_layout()

    def _build_ultrasound_panel(self) -> QWidget:
        us = self.playback.ultrasound
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        channel_count = min(4, int(us.waterfall.shape[0]))
        for channel in range(channel_count):
            label = us.channels[channel] if channel < len(us.channels) else f"ch_{channel + 1}"
            plot = _SweepWaterfallPlot(
                f"超声通道 {channel + 1} · {label}",
                us.time_s,
                np.asarray(us.waterfall[channel]).T,
                self._window_s,
                self.playback.prompt_labels,
            )
            self._sweep_plots_append(plot)
            layout.addWidget(plot)
        return holder

    def _build_imu_panel(self) -> QWidget:
        # 3 IMUs side-by-side; within each IMU, acc/gyr/mag stack vertically so
        # the three measurements stay visually separated per sensor.
        holder = QWidget()
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        rows = _imu_sensor_rows(self.playback)
        for sensor, series, kinds in rows:
            if series is None:
                continue
            block = QWidget()
            block_layout = QVBoxLayout(block)
            block_layout.setContentsMargins(0, 0, 0, 0)
            block_layout.setSpacing(2)
            for kind, title in (("acc", "加速度计"), ("gyr", "陀螺仪"), ("mag", "磁力计")):
                indices = kinds.get(kind, ())
                if not indices:
                    continue
                values = np.asarray(series.values)[:, indices]
                labels = tuple(series.channels[i] for i in indices if i < len(series.channels))
                plot = TimeSeriesPlot(
                    f"{sensor} · {title}",
                    series.time_s,
                    values,
                    labels,
                    self._window_s,
                )
                self._panels.append(plot)
                block_layout.addWidget(plot, 1)
            layout.addWidget(block, 1)
        return holder

    def _build_emg_panel(self) -> QWidget:
        emg = self.playback.emg
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        values = np.asarray(emg.values)
        channel_count = values.shape[1]
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = np.percentile(finite, (1.0, 99.0))
        else:
            low, high = -1.0, 1.0
        plot = TimeSeriesPlot(
            "EMG · 4 通道",
            emg.time_s,
            values,
            emg.channels,
            self._window_s,
            fixed_yrange=(float(low), float(high)),
            offset_per_channel=max(float(high - low), 1e-6) * 1.4,
        )
        self._panels.append(plot)
        layout.addWidget(plot)
        return holder

    def _build_encoder_panel(self) -> QWidget:
        encoder = self.playback.encoder
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        for title, indices in _encoder_side_groups(encoder):
            values = np.asarray(encoder.values)[:, indices]
            labels = tuple(encoder.channels[i] for i in indices if i < len(encoder.channels))
            plot = TimeSeriesPlot(title, encoder.time_s, values, labels, self._window_s)
            self._panels.append(plot)
            layout.addWidget(plot)
        return holder

    def _build_moment_panel(self) -> QWidget:
        moment = self.playback.moment
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        plot = TimeSeriesPlot(
            "髋关节力矩真值",
            moment.time_s,
            np.asarray(moment.values),
            moment.channels,
            self._window_s,
        )
        self._panels.append(plot)
        layout.addWidget(plot, 1)

        # 左髋/右髋显示勾选：按通道名后缀 _l/_r 映射到对应曲线，左在前右在后。
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("显示："))
        ordered: list[tuple[str, int]] = []
        for side, suffix in (("左髋", "_l"), ("右髋", "_r")):
            for index, name in enumerate(moment.channels):
                if name.casefold().endswith(suffix):
                    ordered.append((side, index))
                    break
        for label, index in ordered:
            box = QCheckBox(label)
            box.setChecked(True)
            box.toggled.connect(
                lambda checked, i=index: plot.set_channel_visible(i, checked)
            )
            controls.addWidget(box)
        controls.addStretch(1)
        layout.addLayout(controls)
        return holder

    def _sweep_plots_append(self, plot: object) -> None:
        self._panels.append(plot)

    # -- playback ----------------------------------------------------------
    @staticmethod
    def _playback_bounds(playback: TrialPlayback) -> tuple[float, float]:
        arrays: list[np.ndarray] = []
        if playback.ultrasound is not None:
            arrays.append(np.asarray(playback.ultrasound.time_s, dtype=float))
        for series in (playback.imu, playback.encoder, playback.sync, playback.emg, playback.moment):
            if series is not None:
                arrays.append(np.asarray(series.time_s, dtype=float))
        for series in playback.imu_sensors:
            arrays.append(np.asarray(series.time_s, dtype=float))
        if playback.mocap is not None:
            arrays.append(np.asarray(playback.mocap.time_s, dtype=float))
        finite = [array[np.isfinite(array)] for array in arrays if array.size]
        if not finite:
            return 0.0, 1.0
        minimum = min(float(array.min()) for array in finite)
        maximum = max(float(array.max()) for array in finite)
        return minimum, maximum if maximum > minimum else minimum + 1.0

    def set_playback_time(self, value: float) -> None:
        bounded = min(max(float(value), self._time_min), self._time_max)
        self._current_time = bounded
        span = self._time_max - self._time_min
        slider_value = int(round((bounded - self._time_min) / span * 10_000))
        with QSignalBlocker(self._timeline_slider):
            self._timeline_slider.setValue(slider_value)
        cycle_index = int((bounded - self._time_min) // self._window_s)
        cycle_start = self._time_min + cycle_index * self._window_s
        if bounded >= self._time_max and bounded == cycle_start:
            cycle_start = max(self._time_min, cycle_start - self._window_s)
        for panel in self._panels:
            if isinstance(panel, TimeSeriesPlot):
                panel.set_time(bounded, cycle_start)
            elif isinstance(panel, _SweepWaterfallPlot):
                panel.update_time(bounded, cycle_start)
            elif isinstance(panel, Mocap3DCanvas):
                mocap = self.playback.mocap
                if mocap is not None and mocap.time_s.size:
                    index = int(np.searchsorted(mocap.time_s, bounded, side="right") - 1)
                    panel.set_frame(max(0, index))
        self._time_label.setText(f"t={bounded:.3f} s / {self._time_max:.3f} s")

    def _slider_changed(self, slider_value: int) -> None:
        fraction = float(slider_value) / 10_000.0
        self.set_playback_time(
            self._time_min + fraction * (self._time_max - self._time_min)
        )
        self._last_tick = perf_counter()

    def toggle_playback(self) -> None:
        self._playing = not self._playing
        if self._playing:
            if self._current_time >= self._time_max:
                self.set_playback_time(self._time_min)
            self._last_tick = perf_counter()
            self._timer.start()
            self._play_button.setText("⏸ 暂停")
        else:
            self._timer.stop()
            self._play_button.setText("▶ 播放")

    def _advance_playback(self) -> None:
        now = perf_counter()
        elapsed = max(0.0, now - self._last_tick)
        self._last_tick = now
        target = self._current_time + elapsed * float(self._speed_combo.currentData() or 1.0)
        if target >= self._time_max:
            self.set_playback_time(self._time_max)
            self._playing = False
            self._timer.stop()
            self._play_button.setText("▶ 播放")
            return
        self.set_playback_time(target)


__all__ = ["FullscreenViewer", "Mocap3DCanvas"]
