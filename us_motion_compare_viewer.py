"""超声原始 ADC 区域移动对比工具。

This module is deliberately separate from ``us_motion_viewer``.  It owns only
the read-only comparison UI and calls the small API in
``us_motion_compare_core`` from worker threads.  The workers are cancellable
at the application boundary and every signal carries a generation number so a
late result cannot be shown for a newer session or channel.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg


APP_TITLE = "超声原始区域移动对比工具"
METHOD_LABELS = {
    "ncc": "传统局部相关",
    "traditional12": "连续控制点拟合",
    "cnn2": "两帧匹配 + CNN",
    "cnn9": "九帧匹配 + CNN",
}
METHOD_ORDER = tuple(METHOD_LABELS)
METHOD_COLORS = (QColor("#2563EB"), QColor("#F08A24"))
DEPTH = np.arange(20, 977, dtype=float)
RAW_DEPTH = np.arange(997, dtype=float)
RAW_POINT_COUNT = 997


def configure_font(app: QApplication) -> None:
    """Use a Chinese-capable Windows font, including offscreen smoke runs."""

    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 9))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _arr(value: Any, dtype=float) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _session_channels(session: Any) -> tuple[int, ...]:
    channels = _field(session, "channels", ())
    return tuple(int(x) for x in channels)


def _session_times(session: Any) -> np.ndarray:
    values = _arr(_field(session, "times", []), float).reshape(-1)
    return values if values.size else np.arange(1, dtype=float)


def _get_raw(session: Any, channel: int) -> np.ndarray | None:
    raw = _field(session, "raw_frames", None)
    if raw is None:
        raw = _field(session, "raw", None)
    if raw is None:
        return None
    raw = _arr(raw, float)
    if raw.ndim == 2:
        return raw
    if raw.ndim == 3:
        channels = _session_channels(session)
        try:
            return raw[:, channels.index(channel), :]
        except (ValueError, IndexError):
            return raw[:, 0, :]
    return None


def _depth_from(obj: Any) -> np.ndarray:
    value = _field(obj, "depth", None)
    if value is None:
        return DEPTH.copy()
    result = _arr(value, float).reshape(-1)
    return result if result.size == 957 else DEPTH.copy()


def _normalise_range(item: Any) -> tuple[float, float] | None:
    """Read a region from common backend shapes without assuming a schema."""

    if isinstance(item, dict):
        for a, b in (("depth_start", "depth_stop"), ("start", "end"),
                     ("start_index", "stop_index"), ("left", "right"),
                     ("x0", "x1")):
            if a in item and b in item:
                try:
                    x0, x1 = float(item[a]), float(item[b])
                    if b in ("stop_index", "depth_stop"):
                        x1 -= 1e-6
                    return min(x0, x1), max(x0, x1)
                except (TypeError, ValueError):
                    pass
    if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
        try:
            return min(float(item[0]), float(item[1])), max(float(item[0]), float(item[1]))
        except (TypeError, ValueError):
            return None
    return None


def _regions_for_frame(entry: Any, key: str) -> list[Any]:
    values = _field(entry, key, [])
    if values is None:
        return []
    if isinstance(values, dict):
        return list(values.values())
    return list(values)


def _region_direction_label(region: Any) -> str:
    """Return the stable display label, preferring a tracker confirmation.

    ``confirmed_label`` is the causal tracker's decision and must win over a
    frame-local ``label`` or ``candidate_label``.  Empty confirmations are
    treated as absent so older result payloads remain displayable.
    """

    confirmed = _field(region, "confirmed_label", None)
    if confirmed is not None and str(confirmed).strip():
        return str(confirmed).strip().lower()
    candidate = _field(region, "candidate_label", None)
    if candidate is not None and str(candidate).strip():
        return str(candidate).strip().lower()
    return str(_field(region, "label", "")).strip().lower()


def _direction_style(label: str) -> tuple[str, str] | None:
    """Map shallow/left to blue and deep/right to orange."""

    value = str(label).strip().lower()
    if any(token in value for token in ("left", "向左", "向浅", "左", "shallow", "negative", "-")):
        return "#2F80ED", "←"
    if any(token in value for token in ("right", "向右", "向深", "右", "deep", "positive", "+")):
        return "#F2994A", "→"
    return None


class _WorkerSignals(QObject):
    progress = Signal(int, int, str)
    loaded = Signal(int, object)
    analyzed = Signal(int, object)
    failed = Signal(str, int, str)


class _LoadWorker(QRunnable):
    def __init__(self, core: Any, path: Path, generation: int, cancel: threading.Event):
        super().__init__()
        self.core, self.path, self.generation, self.cancel = core, path, generation, cancel
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            loader = getattr(self.core, "load_recording", None)
            if loader is None:
                raise AttributeError("us_motion_compare_core 缺少 load_recording")
            session = loader(
                self.path,
                progress=lambda p, m: self.signals.progress.emit(self.generation, int(p), str(m)),
                cancel=self.cancel.is_set,
            )
            if not self.cancel.is_set():
                self.signals.loaded.emit(self.generation, session)
        except Exception as exc:
            if not self.cancel.is_set():
                self.signals.failed.emit("加载", self.generation, str(exc))


class _AnalyzeWorker(QRunnable):
    def __init__(self, engine: Any, session: Any, channel: int, start: float, end: float,
                 generation: int, cancel: threading.Event):
        super().__init__()
        self.engine, self.session, self.channel = engine, session, int(channel)
        self.start, self.end, self.generation, self.cancel = start, end, generation, cancel
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            result = self.engine.analyze(
                self.session, self.channel, self.start, self.end,
                progress=lambda p, m: self.signals.progress.emit(self.generation, int(p), str(m)),
                cancel=self.cancel.is_set,
            )
            if not self.cancel.is_set():
                self.signals.analyzed.emit(self.generation, result)
        except Exception as exc:
            if not self.cancel.is_set():
                self.signals.failed.emit("计算", self.generation, str(exc))


class ComparisonPlot(QWidget):
    """One method's direction and percentage spacing plots."""

    def __init__(self, title: str, color: QColor, parent: QWidget | None = None):
        super().__init__(parent)
        self.color = color
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600; color: #27313B;")
        self.direction_plot = pg.PlotWidget()
        self.spacing_plot = pg.PlotWidget()
        self.direction_plot.setLabel("left", "方向 (ADC点)")
        self.direction_plot.setLabel("bottom", "ADC坐标")
        self.spacing_plot.setLabel("left", "间距变化 (%)")
        self.spacing_plot.setLabel("bottom", "深度")
        for plot, lo, hi in ((self.direction_plot, -24, 24), (self.spacing_plot, -40, 40)):
            plot.setBackground("#FFFFFF")
            plot.showGrid(x=True, y=True, alpha=0.16)
            plot.setXRange(0, 996, padding=0)
            plot.getViewBox().setLimits(xMin=0, xMax=996)
            plot.setYRange(lo, hi, padding=0)
            plot.setMouseEnabled(x=True, y=True)
            plot.setMinimumHeight(105)
        self.direction_curve = self.direction_plot.plot(pen=pg.mkPen(color, width=2))
        self.spacing_curve = self.spacing_plot.plot(pen=pg.mkPen(color, width=2))
        self.direction_pass_curve = self.direction_plot.plot(pen=pg.mkPen(color, width=2.4))
        self.spacing_pass_curve = self.spacing_plot.plot(pen=pg.mkPen(color, width=2.4))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title_label)
        layout.addWidget(self.direction_plot)
        layout.addWidget(self.spacing_plot)
        self._region_items: list[Any] = []

    def clear_regions(self) -> None:
        for item in self._region_items:
            try:
                self.direction_plot.removeItem(item)
                self.spacing_plot.removeItem(item)
            except Exception:
                pass
        self._region_items.clear()

    def render(self, method: dict[str, Any], index: int, depth: np.ndarray) -> None:
        direction_all = _arr(_field(method, "direction", np.empty((0, 957))), float)
        spacing_all = _field(method, "spacing", None)
        if spacing_all is None:
            spacing_all = _field(method, "spacing_fraction", np.full_like(direction_all, np.nan))
        spacing_all = _arr(spacing_all, float)
        direction = direction_all[index] if direction_all.ndim == 2 and index < len(direction_all) else np.full(957, np.nan)
        spacing = spacing_all[index] if spacing_all.ndim == 2 and index < len(spacing_all) else np.full(957, np.nan)
        screen_all = _field(method, "screen", _field(method, "screen_pass", None))
        if screen_all is None:
            screen = np.zeros(direction.shape, dtype=bool)
        else:
            screen_arr = np.asarray(screen_all, dtype=bool)
            screen = screen_arr[index] if screen_arr.ndim == 2 and index < len(screen_arr) else np.zeros(direction.shape, dtype=bool)
        self.direction_curve.setPen(pg.mkPen("#9AA0A6", width=1.2))
        self.spacing_curve.setPen(pg.mkPen("#9AA0A6", width=1.2))
        self.direction_curve.setData(depth, direction)
        self.spacing_curve.setData(depth, spacing * 100.0)
        self.direction_pass_curve.setData(depth, np.where(screen, direction, np.nan))
        self.spacing_pass_curve.setData(depth, np.where(screen, spacing * 100.0, np.nan))
        self.clear_regions()
        regions_all = _field(method, "regions", []) or []
        entry = regions_all[index] if index < len(regions_all) else {}
        candidates = _regions_for_frame(entry, "candidates")
        accepted = _regions_for_frame(entry, "regions")
        # Candidate boundaries provide context; accepted boundaries are colored by depth.
        for candidate in candidates:
            rng = _normalise_range(candidate)
            if rng is None:
                continue
            for x in rng:
                for plot in (self.direction_plot, self.spacing_plot):
                    line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen("#9AA0A6", style=Qt.PenStyle.DashLine, width=1))
                    plot.addItem(line)
                    self._region_items.append(line)
        for region in accepted:
            rng = _normalise_range(region)
            if rng is None:
                continue
            style = _direction_style(_region_direction_label(region))
            if style is None:
                continue
            color, marker = style
            fill = QColor(color); fill.setAlpha(38)
            for plot in (self.direction_plot, self.spacing_plot):
                band = pg.LinearRegionItem(values=rng, orientation="vertical",
                                            brush=pg.mkBrush(fill),
                                            pen=pg.mkPen(color, width=1))
                band.setMovable(False)
                band.setZValue(-10)
                plot.addItem(band)
                self._region_items.append(band)
            text = pg.TextItem(marker, color=color, anchor=(.5, 1.0))
            text.setPos(float(np.mean(rng)), 20.0)
            self.direction_plot.addItem(text)
            self._region_items.append(text)

    def reset_view(self) -> None:
        self.direction_plot.setXRange(0, 996, padding=0)
        self.direction_plot.setYRange(-24, 24, padding=0)
        self.spacing_plot.setXRange(0, 996, padding=0)
        self.spacing_plot.setYRange(-40, 40, padding=0)


class ComparisonViewerWindow(QMainWindow):
    def __init__(self, core_module: Any | None = None, initial_path: str | Path | None = None,
                 initial_channel: int | None = None, initial_start: float | None = None,
                 initial_end: float | None = None, auto_analyze: bool = False):
        super().__init__()
        if core_module is None:
            try:
                from Exo_Collection_System import us_motion_compare_core as core_module  # type: ignore[no-redef]
            except ImportError:
                import us_motion_compare_core as core_module  # type: ignore[no-redef]
        self.core = core_module
        self.session: Any | None = None
        self.analysis: dict[str, Any] | None = None
        self.current_result: dict[str, Any] | None = None
        self.current_index = 0
        self._generation = 0
        self._cancel = threading.Event()
        self._closing = False
        self._syncing_x = False
        self._playing = False
        self._speed = 1.0
        self._play_accum = 0.0
        self._raw_y_range: tuple[float, float] | None = None
        self._auto_analyze = bool(auto_analyze)
        self._initial_channel = initial_channel
        self._initial_start, self._initial_end = initial_start, initial_end
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.setWindowTitle(APP_TITLE)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        if available is None:
            self.resize(1280, 720)
        else:
            width = min(1420, max(1100, int(available.width() * 0.92)))
            height = min(1000, max(680, int(available.height() * 0.88)))
            self.resize(min(width, available.width() - 24), min(height, available.height() - 48))
        self._build_ui()
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(50)
        self.play_timer.timeout.connect(self._play_tick)
        if initial_path:
            QTimer.singleShot(0, lambda: self.load_path(initial_path))

    def _build_ui(self) -> None:
        root = QWidget(self)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 10)
        outer.setSpacing(7)
        title_row = QHBoxLayout()
        title = QLabel(APP_TITLE)
        title.setStyleSheet("font-size: 21px; font-weight: 650; color: #20242A;")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(QLabel("20 Hz · 两侧方法共享原始 ADC 坐标"))
        outer.addLayout(title_row)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("session / ultrasound.bin"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择 session 文件夹或 ultrasound.bin")
        self.folder_button = QPushButton("选文件夹…")
        self.browse_button = QPushButton("选 ultrasound.bin…")
        self.load_button = QPushButton("加载")
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.folder_button)
        path_row.addWidget(self.browse_button)
        path_row.addWidget(self.load_button)
        outer.addLayout(path_row)

        controls = QHBoxLayout()
        form = QFormLayout()
        self.channel_combo = QComboBox()
        self.channel_combo.setObjectName("channelCombo")
        self.start_spin = QDoubleSpinBox(); self.start_spin.setDecimals(2); self.start_spin.setSingleStep(.05); self.start_spin.setRange(0, 0)
        self.end_spin = QDoubleSpinBox(); self.end_spin.setDecimals(2); self.end_spin.setSingleStep(.05); self.end_spin.setRange(0, 0)
        self.full_button = QPushButton("整段")
        form.addRow("通道", self.channel_combo)
        form.addRow("开始 (秒)", self.start_spin)
        form.addRow("结束 (秒)", self.end_spin)
        controls.addLayout(form)
        controls.addWidget(self.full_button)
        self.compute_button = QPushButton("计算")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        controls.addWidget(self.compute_button)
        controls.addWidget(self.cancel_button)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setMinimumWidth(180)
        controls.addWidget(self.progress)
        self.status_label = QLabel("请选择 session；开头不足九帧时显示为未知")
        self.status_label.setStyleSheet("color: #69717C;")
        controls.addWidget(self.status_label, 1)
        outer.addLayout(controls)

        playback = QHBoxLayout()
        playback.setContentsMargins(0, 0, 0, 0)
        self.play_button = QPushButton("播放")
        self.play_button.setObjectName("playButton")
        self.play_button.setStyleSheet("QPushButton#playButton { background: #2563EB; color: white; font-weight: 600; padding: 4px 14px; }")
        self.prev_button = QPushButton("上一帧")
        self.next_button = QPushButton("下一帧")
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setRange(0, 0)
        self.time_label = QLabel("时间 -- / --")
        self.speed_combo = QComboBox()
        for speed in (.25, .5, 1.0): self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)
        self.reset_button = QPushButton("恢复视图")
        playback.addWidget(self.play_button); playback.addWidget(self.prev_button); playback.addWidget(self.next_button)
        playback.addWidget(self.slider, 1); playback.addWidget(self.time_label); playback.addWidget(QLabel("速度")); playback.addWidget(self.speed_combo); playback.addWidget(self.reset_button)
        outer.addLayout(playback)

        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setBackground("#FFFFFF")
        self.raw_plot.showGrid(x=True, y=True, alpha=.16)
        self.raw_plot.setLabel("left", "原始 ADC")
        self.raw_plot.setLabel("bottom", "ADC坐标")
        self.raw_plot.setXRange(0, 996, padding=0)
        self.raw_plot.getViewBox().setLimits(xMin=0, xMax=996)
        self.raw_plot.setMinimumHeight(125)
        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen("#20242A", width=1.5))
        outer.addWidget(self.raw_plot)

        compare_row = QHBoxLayout()
        self.method_a_combo = QComboBox(); self.method_b_combo = QComboBox()
        for combo in (self.method_a_combo, self.method_b_combo):
            for key in METHOD_ORDER:
                combo.addItem(METHOD_LABELS[key], key)
        self.method_a_combo.setCurrentIndex(0); self.method_b_combo.setCurrentIndex(3)
        group_a = QGroupBox(); la = QVBoxLayout(group_a)
        ca = QHBoxLayout(); ca.addWidget(QLabel("A 方法")); ca.addWidget(self.method_a_combo); ca.addStretch(1); la.addLayout(ca)
        self.plot_a = ComparisonPlot("A", METHOD_COLORS[0]); la.addWidget(self.plot_a)
        group_b = QGroupBox(); lb = QVBoxLayout(group_b)
        cb = QHBoxLayout(); cb.addWidget(QLabel("B 方法")); cb.addWidget(self.method_b_combo); cb.addStretch(1); lb.addLayout(cb)
        self.plot_b = ComparisonPlot("B", METHOD_COLORS[1]); lb.addWidget(self.plot_b)
        compare_row.addWidget(group_a, 1); compare_row.addWidget(group_b, 1)
        outer.addLayout(compare_row, 1)
        legend = QLabel(
            "图例：<span style='color:#7A8088'>灰线 = 候选全场</span>　"
            "<span style='color:#2563EB'>彩线 = screen 筛选通过</span>　"
            "<span style='color:#2F80ED'>蓝色色带← = 连续区域向浅/left</span>　"
            "<span style='color:#F2994A'>橙色色带→ = 连续区域向深/right</span>"
        )
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setStyleSheet("color: #69717C; font-size: 11px;")
        outer.addWidget(legend)
        # Keep every ViewBox on exactly the same numeric ADC range.  Native
        # pyqtgraph linked views can align by pixel width, which is wrong when
        # panels have different margins, so use an explicit guarded sync.
        self._x_views = [self.raw_plot.getViewBox(), self.plot_a.direction_plot.getViewBox(),
                         self.plot_a.spacing_plot.getViewBox(), self.plot_b.direction_plot.getViewBox(),
                         self.plot_b.spacing_plot.getViewBox()]
        for view in self._x_views:
            view.sigXRangeChanged.connect(self._sync_x_ranges)

        note = QLabel("先计算后按20Hz回放；候选趋势不等于真实肌肉形变；蓝色=向浅/left，橙色=向深/right；九帧区域改向3帧确认。可缩放，超出视窗不显示。")
        note.setStyleSheet("color: #69717C; font-size: 11px;")
        outer.addWidget(note)
        self.setCentralWidget(root)

        self.folder_button.clicked.connect(self.choose_folder)
        self.browse_button.clicked.connect(self.choose_path)
        self.load_button.clicked.connect(lambda: self.load_path(self.path_edit.text().strip()))
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        self.full_button.clicked.connect(self._set_full_range)
        self.compute_button.clicked.connect(self.start_analysis)
        self.cancel_button.clicked.connect(self.cancel_work)
        self.method_a_combo.currentIndexChanged.connect(self.render_current)
        self.method_b_combo.currentIndexChanged.connect(self.render_current)
        self.slider.valueChanged.connect(self._seek)
        self.play_button.clicked.connect(self.toggle_play)
        self.prev_button.clicked.connect(lambda: self._step(-1))
        self.next_button.clicked.connect(lambda: self._step(1))
        self.speed_combo.currentIndexChanged.connect(lambda: setattr(self, "_speed", float(self.speed_combo.currentData())))
        self.reset_button.clicked.connect(self.reset_views)
        for spin in (self.start_spin, self.end_spin):
            spin.valueChanged.connect(self._range_changed)
        self._clear_plots()

    def _sync_x_ranges(self, source_view: Any, _range: Any) -> None:
        if self._syncing_x:
            return
        self._syncing_x = True
        try:
            x0, x1 = source_view.viewRange()[0]
            for view in self._x_views:
                if view is not source_view:
                    view.setXRange(float(x0), float(x1), padding=0)
        finally:
            self._syncing_x = False

    def choose_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 ultrasound.bin", "", "超声数据 (ultrasound.bin);;所有文件 (*)")
        if path:
            self.path_edit.setText(path)

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择 session 文件夹")
        if folder:
            self.path_edit.setText(folder)

    def _invalidate(self, message: str = "请选择 session 并计算", *, clear_session: bool = False) -> None:
        self._generation += 1
        self._cancel.set()
        self._cancel = threading.Event()
        self.analysis = None
        self.current_result = None
        if clear_session:
            self.session = None
            self.channel_combo.blockSignals(True)
            self.channel_combo.clear()
            self.channel_combo.blockSignals(False)
        self.current_index = 0
        self.slider.setRange(0, 0)
        self._playing = False; self.play_timer.stop(); self.play_button.setText("播放")
        self.cancel_button.setEnabled(False)
        self._clear_plots(); self.status_label.setText(message)

    def load_path(self, path: str | Path) -> None:
        if not str(path).strip():
            self.status_label.setText("请先选择 session 文件夹或 ultrasound.bin")
            return
        self._invalidate("正在加载…", clear_session=True)
        generation = self._generation
        self.path_edit.setText(str(path))
        self.compute_button.setEnabled(False); self.cancel_button.setEnabled(True); self.progress.setValue(0)
        worker = _LoadWorker(self.core, Path(path), generation, self._cancel)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.loaded.connect(self._on_loaded)
        worker.signals.failed.connect(self._on_failed)
        self.pool.start(worker)

    def _on_progress(self, generation: int, percent: int, message: str) -> None:
        if generation != self._generation: return
        self.progress.setValue(max(0, min(100, int(percent)))); self.status_label.setText(message)

    def _on_loaded(self, generation: int, session: Any) -> None:
        if generation != self._generation or self._closing: return
        self.session = session
        channels = _session_channels(session)
        self.channel_combo.blockSignals(True); self.channel_combo.clear()
        for channel in channels: self.channel_combo.addItem(f"通道 {channel}", channel)
        if channels:
            desired = self._initial_channel if self._initial_channel in channels else channels[0]
            self.channel_combo.setCurrentIndex(channels.index(desired))
        self.channel_combo.blockSignals(False)
        times = _session_times(session); end = float(times[-1])
        self.start_spin.blockSignals(True); self.end_spin.blockSignals(True)
        self.start_spin.setRange(0, end); self.end_spin.setRange(0, end)
        start = max(0.0, min(end, self._initial_start if self._initial_start is not None else 0.0))
        stop_default = min(30.0, end)
        stop = max(start, min(end, self._initial_end if self._initial_end is not None else stop_default))
        self.start_spin.setValue(start); self.end_spin.setValue(stop)
        self.start_spin.blockSignals(False); self.end_spin.blockSignals(False)
        self.compute_button.setEnabled(bool(channels)); self.cancel_button.setEnabled(False); self.progress.setValue(100)
        self.status_label.setText(f"已加载 {len(times)} 帧，通道 {', '.join(map(str, channels))}；请点击计算")
        if self._auto_analyze: QTimer.singleShot(0, self.start_analysis)

    def _selected_channel(self) -> int | None:
        value = self.channel_combo.currentData()
        return int(value) if value is not None else None

    def selected_channel(self) -> int | None:
        return self._selected_channel()

    def _channel_changed(self) -> None:
        if self.session is not None:
            self._invalidate("通道已改变，请重新计算（旧结果已清除）")
            self.compute_button.setEnabled(True)

    def _range_changed(self) -> None:
        if self.end_spin.value() < self.start_spin.value():
            if self.sender() is self.start_spin: self.end_spin.setValue(self.start_spin.value())
            else: self.start_spin.setValue(self.end_spin.value())
        if self.session is not None and (self.analysis is not None or self.cancel_button.isEnabled()):
            self._invalidate("时间区间已改变，请重新计算（旧结果已清除）")
            self.compute_button.setEnabled(True)

    def _set_full_range(self) -> None:
        if self.session is None: return
        end = float(_session_times(self.session)[-1]); self.start_spin.setValue(0.0); self.end_spin.setValue(end)

    def start_analysis(self) -> None:
        if self.session is None or self._selected_channel() is None:
            self.status_label.setText("请先加载 session")
            return
        self._invalidate("正在计算…")
        generation = self._generation
        self.compute_button.setEnabled(False); self.cancel_button.setEnabled(True); self.progress.setValue(0)
        engine_cls = getattr(self.core, "ComparisonEngine", None)
        if engine_cls is None: raise RuntimeError("us_motion_compare_core 缺少 ComparisonEngine")
        engine = engine_cls()
        worker = _AnalyzeWorker(engine, self.session, self._selected_channel(), self.start_spin.value(), self.end_spin.value(), generation, self._cancel)
        worker.signals.progress.connect(self._on_progress); worker.signals.analyzed.connect(self._on_analyzed); worker.signals.failed.connect(self._on_failed)
        self.pool.start(worker)

    def cancel_work(self) -> None:
        # Invalidate queued signals before exposing the compute button again.
        # The old event stays set until the next run creates a fresh one.
        self._generation += 1
        self._cancel.set(); self.cancel_button.setEnabled(False)
        self.compute_button.setEnabled(self.session is not None)
        self.status_label.setText("已请求取消；后台线程会在安全点退出")

    def _on_analyzed(self, generation: int, result: dict[str, Any]) -> None:
        if generation != self._generation or self._closing: return
        self.analysis = result
        self.current_result = result
        times = _arr(result.get("times", []), float).reshape(-1)
        self.slider.setRange(0, max(0, len(times) - 1)); self.slider.setValue(0)
        self.raw_plot.setXRange(0, 996, padding=0)
        raw_values = _arr(result.get("raw", []), float)
        finite = raw_values[np.isfinite(raw_values)] if raw_values.size else np.empty(0)
        if finite.size:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            pad = max(1.0, (hi - lo) * .03)
            self._raw_y_range = (lo - pad, hi + pad)
            self.raw_plot.setYRange(*self._raw_y_range, padding=0)
        self.compute_button.setEnabled(True); self.cancel_button.setEnabled(False); self.progress.setValue(100)
        self.status_label.setText(f"结果：通道 {result.get('channel', self._selected_channel())} · {len(times)} 帧 · 两侧方法已配对 50 ms")
        self.render_current()

    def _on_failed(self, stage: str, generation: int, message: str) -> None:
        if generation != self._generation or self._closing: return
        self.compute_button.setEnabled(self.session is not None); self.cancel_button.setEnabled(False)
        self.status_label.setText(f"{stage}失败：{message}")

    def _clear_plots(self) -> None:
        self.raw_curve.clear(); self.plot_a.direction_curve.clear(); self.plot_a.spacing_curve.clear(); self.plot_a.direction_pass_curve.clear(); self.plot_a.spacing_pass_curve.clear(); self.plot_a.clear_regions(); self.plot_b.direction_curve.clear(); self.plot_b.spacing_curve.clear(); self.plot_b.direction_pass_curve.clear(); self.plot_b.spacing_pass_curve.clear(); self.plot_b.clear_regions()
        self.time_label.setText("时间 -- / --")

    def _method(self, combo: QComboBox) -> dict[str, Any] | None:
        if self.analysis is None: return None
        methods = self.analysis.get("methods", {})
        return methods.get(combo.currentData())

    def render_current(self) -> None:
        if self.analysis is None: return
        times = _arr(self.analysis.get("times", []), float).reshape(-1)
        if not times.size: return
        self.current_index = max(0, min(self.current_index, len(times) - 1))
        raw = _arr(self.analysis.get("raw", []), float)
        if raw.ndim == 2 and self.current_index < len(raw):
            self.raw_curve.setData(RAW_DEPTH[:raw.shape[1]], raw[self.current_index])
        depth = _depth_from(self.analysis)
        a, b = self._method(self.method_a_combo), self._method(self.method_b_combo)
        if a is not None: self.plot_a.title_label.setText(f"A · {METHOD_LABELS.get(self.method_a_combo.currentData(), self.method_a_combo.currentText())}"); self.plot_a.render(a, self.current_index, depth)
        if b is not None: self.plot_b.title_label.setText(f"B · {METHOD_LABELS.get(self.method_b_combo.currentData(), self.method_b_combo.currentText())}"); self.plot_b.render(b, self.current_index, depth)
        self.time_label.setText(f"时间 {times[self.current_index]:.2f} / {times[-1]:.2f} 秒")

    def _seek(self, value: int) -> None:
        if self.analysis is None: return
        self._playing = False; self.play_timer.stop(); self.play_button.setText("播放")
        self.current_index = int(value); self.render_current()

    def show_frame(self, index: int, pause: bool = True) -> None:
        if self.analysis is None:
            return
        if pause:
            self._playing = False; self.play_timer.stop(); self.play_button.setText("播放")
        self.slider.setValue(max(0, min(self.slider.maximum(), int(index))))

    def _step(self, amount: int) -> None:
        if self.analysis is None: return
        self.slider.setValue(max(0, min(self.slider.maximum(), self.current_index + int(amount))))

    def toggle_play(self) -> None:
        if self.analysis is None or self.slider.maximum() <= 0: return
        if not self._playing and self.current_index >= self.slider.maximum():
            self.current_index = 0
            self.slider.setValue(0)
        self._playing = not self._playing
        if self._playing:
            self._play_accum = 0.0
            self.play_timer.start(); self.play_button.setText("暂停")
        else: self.play_timer.stop(); self.play_button.setText("播放")

    def _play_tick(self) -> None:
        if not self._playing or self.analysis is None: return
        times = _arr(self.analysis.get("times", []), float)
        self._play_accum += float(self._speed)
        step = int(self._play_accum)
        self._play_accum -= step
        if step <= 0:
            return
        next_index = self.current_index + step
        if next_index >= len(times):
            self._playing = False; self.play_timer.stop(); self.play_button.setText("播放"); next_index = len(times) - 1
        self.slider.blockSignals(True); self.slider.setValue(next_index); self.slider.blockSignals(False); self.current_index = next_index; self.render_current()

    def reset_views(self) -> None:
        self.raw_plot.setXRange(0, 996, padding=0)
        if self._raw_y_range is not None:
            self.raw_plot.setYRange(*self._raw_y_range, padding=0)
        self.plot_a.reset_view(); self.plot_b.reset_view()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._closing = True; self._cancel.set(); self.play_timer.stop(); event.accept()


def create_smoke_core() -> Any:
    """Small deterministic backend used only for offline GUI interaction tests."""

    class SmokeEngine:
        def analyze(self, session, channel, start, end, progress=None, cancel=None):
            times_all = session.times; lo = int(np.searchsorted(times_all, start, side="left")); hi = int(np.searchsorted(times_all, end, side="right")); hi = max(lo + 1, hi)
            times = times_all[lo:hi]; raw = session.raw_frames[lo:hi, session.channels.index(channel)]
            t = np.arange(len(times), dtype=float)[:, None]; x = np.arange(957, dtype=float)[None, :]
            methods = {}
            for mi, name in enumerate(METHOD_ORDER):
                direction = 4 * np.sin(t / 5 + x / 70 + mi * .2).astype(np.float32)
                spacing = .15 * np.cos(t / 7 + x / 80 + mi * .3).astype(np.float32)
                screen = np.ones_like(direction, dtype=bool); screen[:min(8, len(times))] = False
                entries = []
                for i in range(len(times)):
                    entries.append({"candidates": [{"start": 100, "end": 350}, {"start": 600, "end": 820}], "regions": ([{"start": 130, "end": 310, "label": "left"}] if i >= 8 else [])})
                methods[name] = {"direction": direction, "spacing": spacing, "screen": screen, "regions": entries}
                if progress: progress(30 + (METHOD_ORDER.index(name) + 1) * 16, f"计算 {METHOD_LABELS[name]}")
            return {"times": times, "raw": raw, "depth": DEPTH.copy(), "methods": methods, "session_path": str(session.path), "channel": channel, "start": float(times[0]), "end": float(times[-1]), "metadata": {"pairing": "50ms"}}

    class SmokeCore:
        ComparisonEngine = SmokeEngine
        def load_recording(self, path, progress=None, cancel=None):
            if progress: progress(35, "读取 smoke session")
            times = np.arange(0, 6.0, .05); channels = (1, 2, 3); raw = np.empty((len(times), len(channels), 997), dtype=np.float32)
            x = np.arange(997, dtype=float)
            for ci in range(len(channels)): raw[:, ci] = np.clip(120 + 70 * np.sin(x[None, :] / 30 + times[:, None] * (ci + 1)), 0, 255)
            if progress: progress(100, "加载完成")
            return SimpleNamespace(path=Path(path), times=times, raw_frames=raw, channels=channels, depth=DEPTH.copy(), metadata={})
    return SmokeCore()


def launch_viewer(path: str | Path | None = None, channel: int | None = None, start: float | None = None,
                  end: float | None = None, analyze: bool = False) -> int:
    app = QApplication.instance() or QApplication([]); configure_font(app)
    window = ComparisonViewerWindow(initial_path=path, initial_channel=channel, initial_start=start, initial_end=end, auto_analyze=analyze)
    window.show(); return app.exec()


def run_smoke_test(screenshot: str | Path | None = None) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([]); configure_font(app)
    window = ComparisonViewerWindow(create_smoke_core(), initial_path="smoke", initial_channel=2, initial_start=0, initial_end=2.0, auto_analyze=True)
    window.show(); deadline = time.perf_counter() + 4
    while time.perf_counter() < deadline and window.analysis is None:
        app.processEvents(); time.sleep(.01)
    if window.analysis is None: return 2
    window.slider.setValue(min(10, window.slider.maximum())); window.method_a_combo.setCurrentIndex(1); window.method_b_combo.setCurrentIndex(2); app.processEvents()
    if screenshot:
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True); window.grab().save(str(screenshot))
    window.close(); app.processEvents(); return 0


# Short aliases keep the standalone tool pleasant to import in small scripts.
CompareViewerWindow = ComparisonViewerWindow
launch_compare_viewer = launch_viewer

