"""Read-only player for motion estimated from a recorded ultrasound session.

The UI deliberately owns no acquisition or model code.  All file parsing and
motion estimation is delegated to :mod:`us_motion_core` through its small,
fixed API.  Loading and frame estimation run outside the GUI thread.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QRunnable, QSignalBlocker, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg


APP_TITLE = "超声区域移动播放器"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "Exo_Collection_Data"
LABEL_COLORS = {
    "left": QColor("#2F80ED"),
    "right": QColor("#F2994A"),
    "near_zero": QColor("#252525"),
    "unknown": QColor("#A8ADB4"),
}
LABEL_ARROWS = {"left": "←", "right": "→", "near_zero": "≈", "unknown": "?"}
RAW_POINT_COUNT = 997


def configure_font(app: QApplication) -> None:
    """Choose a Chinese-capable font explicitly, including offscreen runs."""

    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 9))


class _WorkerSignals(QObject):
    progress = Signal(int, int, str)
    loaded = Signal(int, object)
    estimated = Signal(int, int, int, int, object)
    failed = Signal(str, int, str)


class _LoadWorker(QRunnable):
    def __init__(self, core: Any, path: Path, generation: int, cancel: threading.Event) -> None:
        super().__init__()
        self.core = core
        self.path = path
        self.generation = generation
        self.cancel = cancel
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            session = self.core.load_session(
                self.path,
                progress=lambda percent, message: self.signals.progress.emit(
                    self.generation, int(percent), str(message)
                ),
                cancel=self.cancel.is_set,
            )
            if not self.cancel.is_set():
                self.signals.loaded.emit(self.generation, session)
        except Exception as exc:  # GUI boundary: the user gets a concise message.
            if not self.cancel.is_set():
                self.signals.failed.emit("load", self.generation, str(exc))


class _EstimateWorker(QRunnable):
    def __init__(
        self,
        core: Any,
        session: Any,
        generation: int,
        serial: int,
        index: int,
        channel: int,
        lag_seconds: float,
    ) -> None:
        super().__init__()
        self.core = core
        self.session = session
        self.generation = generation
        self.serial = serial
        self.index = index
        self.channel = channel
        self.lag_seconds = lag_seconds
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            result = self.core.estimate_frame(
                self.session,
                self.index,
                self.channel,
                lag_seconds=self.lag_seconds,
            )
            self.signals.estimated.emit(
                self.generation, self.serial, self.index, self.channel, result
            )
        except Exception as exc:
            self.signals.failed.emit("estimate", self.generation, str(exc))


class MotionBand(QWidget):
    """Compact depth-aligned 49-point direction regions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(54)
        self.setMinimumWidth(320)
        self._blocks: list[dict[str, Any]] = []
        self._depth = np.arange(RAW_POINT_COUNT, dtype=float)
        self._view_range = (float(self._depth[0]), float(self._depth[-1]))
        self._content_pixels = (0.0, float(self.width()))
        self.setToolTip("每个色块约 49 个深度点；灰色问号表示证据不足，不能解释为静止")

    def set_blocks(self, blocks: list[dict[str, Any]], depth: np.ndarray) -> None:
        self._blocks = list(blocks or [])
        values = np.asarray(depth, dtype=float)
        if values.ndim == 1 and values.size:
            self._depth = values
        self.update()

    def set_axis_mapping(
        self, left_pixel: float, right_pixel: float, x_min: float, x_max: float
    ) -> None:
        self._content_pixels = (float(left_pixel), float(right_pixel))
        self._view_range = (float(x_min), float(x_max))
        self.update()

    def pixel_for_depth(self, depth: float) -> float:
        left, right = self._content_pixels
        x_min, x_max = self._view_range
        if x_max == x_min:
            return left
        return left + (float(depth) - x_min) * (right - left) / (x_max - x_min)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, right = self._content_pixels
        left = max(1.0, min(float(self.width() - 1), left))
        right = max(left + 1.0, min(float(self.width() - 1), right))
        rect = self.rect().adjusted(int(left), 5, int(right - self.width()), -5)
        painter.fillRect(rect, QColor("#F1F3F5"))
        if not self._blocks:
            painter.setPen(QColor("#6C737F"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "尚无可显示的区域结果")
            return
        for block in self._blocks:
            start = max(0, int(block.get("start_index", 0)))
            stop = min(self._depth.size, int(block.get("stop_index", start)))
            if stop <= start:
                continue
            label = str(block.get("label", "unknown"))
            spacing = float(np.median(np.diff(self._depth))) if self._depth.size > 1 else 1.0
            x0 = self.pixel_for_depth(float(self._depth[start]) - spacing / 2.0)
            x1 = self.pixel_for_depth(float(self._depth[stop - 1]) + spacing / 2.0)
            x0 = max(float(rect.left()), min(float(rect.right()), x0))
            x1 = max(float(rect.left()), min(float(rect.right()), x1))
            color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
            painter.fillRect(int(x0), rect.top(), max(1, int(x1 - x0)), rect.height(), color)
            painter.setPen(QPen(QColor(255, 255, 255, 90), 1))
            painter.drawLine(int(x1), rect.top(), int(x1), rect.bottom())
            painter.setPen(QColor("white"))
            painter.drawText(
                int(x0), rect.top(), max(1, int(x1 - x0)), rect.height(),
                Qt.AlignmentFlag.AlignCenter, LABEL_ARROWS.get(label, "?"),
            )
        painter.setPen(QColor("#7B828C"))
        painter.drawRect(rect)


class MotionViewerWindow(QMainWindow):
    """Interactive player backed by ``us_motion_core``."""

    def __init__(
        self,
        core_module: Any | None = None,
        initial_path: str | Path | None = None,
        initial_channel: int | None = None,
        initial_time: float | None = None,
        autoplay: bool = False,
    ) -> None:
        super().__init__()
        if core_module is None:
            import us_motion_core as core_module  # type: ignore[no-redef]

        self.core = core_module
        self.session: Any | None = None
        self.current_result: dict[str, Any] | None = None
        self.current_index = 0
        self._generation = 0
        self._serial = 0
        self._active_serial: int | None = None
        self._pending_index: int | None = None
        self._load_cancel = threading.Event()
        self._loaded_path: Path | None = None
        self._closing = False
        self._initial_channel = initial_channel
        self._initial_time = initial_time
        self._autoplay = bool(autoplay)
        self._playing = False
        self._speed = 1.0
        self._play_wall_start = 0.0
        self._play_session_start = 0.0

        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.setWindowTitle(APP_TITLE)
        self.resize(1220, 790)
        self._build_ui()

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(20)
        self.play_timer.timeout.connect(self._play_tick)

        if initial_path is not None:
            QTimer.singleShot(0, lambda: self.load_path(initial_path))

    def _build_ui(self) -> None:
        root = QWidget(self)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.setSpacing(10)

        title_row = QHBoxLayout()
        title = QLabel("超声区域移动播放器")
        title.setStyleSheet("font-size: 22px; font-weight: 650; color: #20242A;")
        self.readonly_note = QLabel("历史 session · 按真实时间播放 · 只读分析，不连接采集设备")
        self.readonly_note.setStyleSheet("color: #69717C;")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.readonly_note)
        outer.addLayout(title_row)

        source = QFrame()
        source.setFrameShape(QFrame.Shape.StyledPanel)
        source_row = QHBoxLayout(source)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择 session 文件夹或 ultrasound.bin")
        folder_button = QPushButton("选择文件夹")
        file_button = QPushButton("选择 ultrasound.bin")
        self.reload_button = QPushButton("重新加载")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setVisible(False)
        folder_button.clicked.connect(self._choose_folder)
        file_button.clicked.connect(self._choose_file)
        self.reload_button.clicked.connect(lambda: self.load_path(self.path_edit.text()))
        self.cancel_button.clicked.connect(self.cancel_loading)
        source_row.addWidget(QLabel("数据源"))
        source_row.addWidget(self.path_edit, 1)
        source_row.addWidget(folder_button)
        source_row.addWidget(file_button)
        source_row.addWidget(self.reload_button)
        source_row.addWidget(self.cancel_button)
        outer.addWidget(source)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        outer.addWidget(self.progress)

        controls = QHBoxLayout()
        self.prev_button = QPushButton("上一帧")
        self.play_button = QPushButton("播放")
        self.next_button = QPushButton("下一帧")
        self.channel_combo = QComboBox()
        self.lag_combo = QComboBox()
        self.speed_combo = QComboBox()
        self.screenshot_button = QPushButton("导出截图")
        self.csv_button = QPushButton("导出 CSV")
        for lag in (0.05, 0.10, 0.20, 0.50):
            self.lag_combo.addItem(f"{lag:.2f} 秒", lag)
        self.lag_combo.setCurrentIndex(2)
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)
        self.prev_button.clicked.connect(lambda: self.show_frame(self.current_index - 1, pause=True))
        self.next_button.clicked.connect(lambda: self.show_frame(self.current_index + 1, pause=True))
        self.play_button.clicked.connect(self.toggle_playing)
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        self.lag_combo.currentIndexChanged.connect(self._lag_changed)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        self.screenshot_button.clicked.connect(self._export_screenshot_clicked)
        self.csv_button.clicked.connect(self._export_csv_clicked)
        controls.addWidget(self.prev_button)
        controls.addWidget(self.play_button)
        controls.addWidget(self.next_button)
        controls.addSpacing(12)
        controls.addWidget(QLabel("通道"))
        controls.addWidget(self.channel_combo)
        controls.addWidget(QLabel("历史间隔"))
        controls.addWidget(self.lag_combo)
        controls.addWidget(QLabel("倍速"))
        controls.addWidget(self.speed_combo)
        controls.addStretch(1)
        controls.addWidget(self.screenshot_button)
        controls.addWidget(self.csv_button)
        outer.addLayout(controls)

        legend = QHBoxLayout()
        legend.addWidget(QLabel("区域："))
        for label, text in (("left", "蓝 ← 浅层"), ("right", "橙 → 深层"), ("near_zero", "黑 ≈ 接近零"), ("unknown", "灰 ? 未知")):
            item = QLabel(text)
            item.setStyleSheet(
                f"background:{LABEL_COLORS[label].name()}; color:white; padding:4px 9px; border-radius:4px;"
            )
            legend.addWidget(item)
        legend.addSpacing(18)
        legend.addWidget(QLabel("曲线：黑色实线 原始ADC当前帧"))
        legend.addStretch(1)
        outer.addLayout(legend)

        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "深度采样点")
        self.plot.setLabel("left", "原始 ADC 值")
        self.plot.showGrid(x=True, y=True, alpha=0.13)
        self.plot.setMouseEnabled(x=True, y=True)
        self.current_curve = self.plot.plot([], [], pen=pg.mkPen("#17191C", width=2.2))
        self.plot.setLimits(xMin=0.0, xMax=float(RAW_POINT_COUNT - 1), yMin=0.0, yMax=255.0)
        self.plot.setXRange(0.0, float(RAW_POINT_COUNT - 1), padding=0.0)
        self.plot.setYRange(0.0, 255.0, padding=0.0)
        self.plot.enableAutoRange(x=False, y=False)
        self.plot.setTitle("请先加载历史 session")
        self.plot.installEventFilter(self)
        self.plot.getViewBox().sigXRangeChanged.connect(lambda *_: self._sync_band_axis())
        self.plot.getViewBox().sigResized.connect(lambda *_: self._sync_band_axis())
        outer.addWidget(self.plot, 1)

        self.band = MotionBand()
        outer.addWidget(self.band)

        timeline = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(lambda: self.set_playing(False))
        self.slider.valueChanged.connect(self._slider_changed)
        self.frame_label = QLabel("帧 —/—")
        timeline.addWidget(self.frame_label)
        timeline.addWidget(self.slider, 1)
        outer.addLayout(timeline)

        self.metrics = QLabel("当前时间 —　历史时间 —　计算 —　未知比例 —")
        self.metrics.setStyleSheet("font-size: 13px; color: #343A40;")
        self.status = QLabel("请选择数据源。开头没有足够历史时会显示为“未知”，不会标成静止。")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #666E78;")
        outer.addWidget(self.metrics)
        outer.addWidget(self.status)
        self.setCentralWidget(root)
        self._set_session_controls(False)
        QTimer.singleShot(0, self._sync_band_axis)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self.plot and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._sync_band_axis)
        return super().eventFilter(watched, event)

    def _sync_band_axis(self) -> None:
        if not hasattr(self, "plot") or not hasattr(self, "band"):
            return
        view_box = self.plot.getViewBox()
        scene_rect = view_box.sceneBoundingRect()
        left = self.plot.mapFromScene(QPointF(scene_rect.left(), scene_rect.center().y())).x()
        right = self.plot.mapFromScene(QPointF(scene_rect.right(), scene_rect.center().y())).x()
        x_min, x_max = view_box.viewRange()[0]
        self.band.set_axis_mapping(left, right, x_min, x_max)

    def _set_session_controls(self, enabled: bool) -> None:
        for widget in (
            self.prev_button, self.play_button, self.next_button, self.channel_combo,
            self.lag_combo, self.speed_combo, self.slider, self.screenshot_button,
            self.csv_button,
        ):
            widget.setEnabled(enabled)

    def _choose_folder(self) -> None:
        start = self.path_edit.text().strip() or str(DEFAULT_DATA_ROOT)
        selected = QFileDialog.getExistingDirectory(self, "选择包含 ultrasound.bin 的 session", start)
        if selected:
            self.load_path(selected)

    def _choose_file(self) -> None:
        start = self.path_edit.text().strip() or str(DEFAULT_DATA_ROOT)
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择 ultrasound.bin", start, "超声数据 (ultrasound.bin);;二进制文件 (*.bin)"
        )
        if selected:
            self.load_path(selected)

    def load_path(self, path: str | Path) -> None:
        text = str(path).strip()
        if not text:
            self._show_error("没有数据源", "请先选择 session 文件夹或 ultrasound.bin。")
            return
        self.set_playing(False)
        self._load_cancel.set()
        self._load_cancel = threading.Event()
        self._generation += 1
        generation = self._generation
        self._active_serial = None
        self._pending_index = None
        self.path_edit.setText(text)
        self.progress.setValue(0)
        self.progress.setFormat("准备加载… %p%")
        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.reload_button.setEnabled(False)
        self._set_session_controls(False)
        self._clear_result("正在加载新数据…")
        self.status.setText("正在后台读取数据，可随时取消。")
        worker = _LoadWorker(self.core, Path(text), generation, self._load_cancel)
        worker.signals.progress.connect(self._load_progress)
        worker.signals.loaded.connect(self._load_finished)
        worker.signals.failed.connect(self._worker_failed)
        self.pool.start(worker)

    def cancel_loading(self) -> None:
        if not self.progress.isVisible():
            return
        self._load_cancel.set()
        self._generation += 1
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self.reload_button.setEnabled(True)
        self._set_session_controls(self.session is not None)
        if self._loaded_path is not None:
            self.path_edit.setText(str(self._loaded_path))
        self.status.setText("已取消加载；现有 session（如有）保持不变。")
        if self.session is not None:
            self._request_estimate(self.current_index)

    def _load_progress(self, generation: int, percent: int, message: str) -> None:
        if generation != self._generation or self._closing:
            return
        self.progress.setValue(max(0, min(100, percent)))
        self.progress.setFormat(f"{message}　%p%")

    def _validate_session(self, session: Any) -> None:
        times = np.asarray(session.times)
        profiles = np.asarray(session.profiles)
        channels = tuple(int(v) for v in session.channels)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("session 没有可播放的时间帧")
        if profiles.ndim != 3 or profiles.shape[0] != times.size:
            raise ValueError("profiles 形状应为 [时间, 通道, 深度]")
        if profiles.shape[1] != len(channels):
            raise ValueError("profiles 的通道数与 channels 不一致")
        if np.asarray(session.depth).size != profiles.shape[2]:
            raise ValueError("depth 长度与 profiles 深度维不一致")

    def _load_finished(self, generation: int, session: Any) -> None:
        if generation != self._generation or self._closing:
            return
        try:
            self._validate_session(session)
        except Exception as exc:
            self._worker_failed("load", generation, str(exc))
            return
        self.session = session
        self._loaded_path = Path(session.path)
        self.path_edit.setText(str(self._loaded_path))
        self.current_result = None
        self.current_index = 0
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self.reload_button.setEnabled(True)
        with QSignalBlocker(self.channel_combo):
            self.channel_combo.clear()
            for channel in session.channels:
                self.channel_combo.addItem(f"通道 {int(channel)}", int(channel))
            preferred = self._initial_channel
            selected = 0
            if preferred is not None:
                found = self.channel_combo.findData(int(preferred))
                if found >= 0:
                    selected = found
                else:
                    self.status.setText(f"请求的通道 {preferred} 不存在，已显示通道 {session.channels[0]}。")
            self.channel_combo.setCurrentIndex(selected)
        times = np.asarray(session.times, dtype=float)
        self.slider.setRange(0, times.size - 1)
        self._set_session_controls(True)
        target = 0
        if self._initial_time is not None:
            target = int(np.argmin(np.abs(times - float(self._initial_time))))
        self._initial_channel = None
        self._initial_time = None
        self.status.setText(
            f"已加载 {times.size} 帧、{len(session.channels)} 个通道。数据仅在内存中只读分析。"
        )
        self.show_frame(target)
        if self._autoplay:
            self._autoplay = False
            QTimer.singleShot(0, lambda: self.set_playing(True))

    def _worker_failed(self, kind: str, generation: int, message: str) -> None:
        if generation != self._generation or self._closing:
            return
        if kind == "estimate":
            self._active_serial = None
            self.set_playing(False)
            self.status.setText(f"当前帧计算失败：{message}。可更换通道或时间后重试。")
            return
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)
        self.reload_button.setEnabled(True)
        self._set_session_controls(self.session is not None)
        if self._loaded_path is not None:
            self.path_edit.setText(str(self._loaded_path))
        if self.session is not None:
            self._request_estimate(self.current_index)
        self._show_error(
            "加载失败",
            f"{message}\n\n请确认选择的是 session 文件夹或其中的 ultrasound.bin，且索引和元数据完整。",
        )

    def _show_error(self, title: str, message: str) -> None:
        self.status.setText(message.replace("\n", " "))
        if self.isVisible() and os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen":
            QMessageBox.warning(self, title, message)

    def selected_channel(self) -> int | None:
        value = self.channel_combo.currentData()
        return None if value is None else int(value)

    def show_frame(self, index: int, pause: bool = False) -> None:
        if self.session is None:
            return
        if pause:
            self.set_playing(False)
            self._invalidate_estimates()
            self._clear_result("正在计算目标帧…")
        count = int(np.asarray(self.session.times).size)
        bounded = max(0, min(int(index), count - 1))
        self.current_index = bounded
        with QSignalBlocker(self.slider):
            self.slider.setValue(bounded)
        self.frame_label.setText(f"帧 {bounded + 1}/{count}")
        self._request_estimate(bounded)

    def _request_estimate(self, index: int) -> None:
        if self.session is None:
            return
        if self._active_serial is not None:
            self._pending_index = int(index)
            return
        channel = self.selected_channel()
        if channel is None:
            return
        self._serial += 1
        serial = self._serial
        self._active_serial = serial
        self._pending_index = None
        worker = _EstimateWorker(
            self.core,
            self.session,
            self._generation,
            serial,
            int(index),
            channel,
            float(self.lag_combo.currentData()),
        )
        worker.signals.estimated.connect(self._estimate_finished)
        worker.signals.failed.connect(self._worker_failed)
        self.pool.start(worker)

    def _estimate_finished(
        self, generation: int, serial: int, index: int, channel: int, result: object
    ) -> None:
        if generation != self._generation or self._closing:
            return
        if serial != self._active_serial:
            return
        self._active_serial = None
        if channel != self.selected_channel():
            self._pending_index = None
            self._request_estimate(self.current_index)
            return
        if index != self.current_index and not self._playing:
            self._pending_index = None
            self._request_estimate(self.current_index)
            return
        result_dict = dict(result)  # detach from worker-owned mapping
        self.current_result = result_dict
        self._render_result(result_dict)
        pending = self._pending_index
        self._pending_index = None
        if pending is not None and pending != self.current_index:
            pending = self.current_index
        if pending is not None:
            self._request_estimate(pending)

    def _render_result(self, result: dict[str, Any]) -> None:
        depth = np.asarray(result["depth"], dtype=float)
        raw_depth_value = result.get("raw_depth")
        raw_current_value = result.get("raw_current")
        raw_ok = raw_depth_value is not None and raw_current_value is not None
        if raw_ok:
            raw_depth = np.asarray(raw_depth_value, dtype=float)
            raw_current = np.asarray(raw_current_value, dtype=float)
            raw_ok = (
                raw_depth.ndim == 1
                and raw_current.ndim == 1
                and raw_depth.size == RAW_POINT_COUNT
                and raw_current.size == RAW_POINT_COUNT
                and np.all(np.isfinite(raw_depth))
                and np.all(np.isfinite(raw_current))
            )
        if raw_ok:
            self.current_curve.setData(raw_depth, raw_current)
            # Region blocks are indexed in the 957-point analysis depth array.
            # Keep that physical depth coordinate while the raw curve uses 0..996.
            self.band.set_blocks(list(result.get("blocks") or []), depth)
        else:
            # Never substitute the processed 957-point estimate for a missing raw frame.
            self.current_curve.setData([], [])
            self.band.set_blocks(list(result.get("blocks") or []), depth)
        self._sync_band_axis()
        current_time = float(result.get("current_time", 0.0))
        past_time = result.get("past_time")
        past_text = "无（历史不足）" if past_time is None else f"{float(past_time):.3f} s"
        region = self._region_labels(result, depth.size)
        unknown_ratio = float(np.mean(region == "unknown")) if depth.size else 1.0
        compute_ms = float(result.get("compute_ms", 0.0))
        self.metrics.setText(
            f"当前时间 {current_time:.3f} s　历史时间 {past_text}　"
            f"计算 {compute_ms:.1f} ms　分析区未知比例 {unknown_ratio:.1%}"
        )
        valid_pair = bool(result.get("valid_pair", False))
        reason = str(result.get("reason", ""))
        if not raw_ok:
            self.status.setText("原始 ADC 当前帧不可用；未使用预处理曲线代替。")
        elif valid_pair:
            self.status.setText(reason or "当前帧与历史帧均有效。箭头表示相对深度方向。")
        else:
            self.status.setText((reason or "历史帧或数据质量不足。") + " 灰色未知不表示静止。")
        raw_title = "原始ADC当前帧" if raw_ok else "原始ADC不可用"
        self.plot.setTitle(f"通道 {self.selected_channel()} · {raw_title} · t={current_time:.3f} s")

    def _slider_changed(self, value: int) -> None:
        if self.session is not None:
            self.show_frame(int(value), pause=True)

    def _channel_changed(self, _index: int) -> None:
        if self.session is None:
            return
        self.set_playing(False)
        self._invalidate_estimates()
        self._clear_result("正在计算新通道…")
        self.show_frame(self.current_index)

    def _lag_changed(self, _index: int) -> None:
        if self.session is None:
            return
        self.set_playing(False)
        self._invalidate_estimates()
        self._clear_result("正在按新的历史间隔计算…")
        self.show_frame(self.current_index)

    def _speed_changed(self, _index: int) -> None:
        was_playing = self._playing
        if was_playing and self.session is not None:
            current_time = float(np.asarray(self.session.times)[self.current_index])
            self._play_session_start = current_time
            self._play_wall_start = time.perf_counter()
        self._speed = float(self.speed_combo.currentData() or 1.0)

    def _invalidate_estimates(self) -> None:
        self._generation += 1
        self._active_serial = None
        self._pending_index = None

    def toggle_playing(self) -> None:
        if self._playing:
            self.set_playing(False)
            self._invalidate_estimates()
            self._request_estimate(self.current_index)
        else:
            self.set_playing(True)

    def set_playing(self, playing: bool) -> None:
        playing = bool(playing and self.session is not None)
        if playing == self._playing:
            return
        self._playing = playing
        if playing:
            times = np.asarray(self.session.times, dtype=float)
            if self.current_index >= times.size - 1:
                self.current_index = 0
            self._play_session_start = float(times[self.current_index])
            self._play_wall_start = time.perf_counter()
            self._speed = float(self.speed_combo.currentData() or 1.0)
            self.play_timer.start()
            self.play_button.setText("暂停")
        else:
            self.play_timer.stop()
            self.play_button.setText("播放")
            self._pending_index = None

    def _play_tick(self) -> None:
        if not self._playing or self.session is None:
            return
        times = np.asarray(self.session.times, dtype=float)
        target_time = self._play_session_start + (
            time.perf_counter() - self._play_wall_start
        ) * self._speed
        if target_time >= float(times[-1]):
            self.set_playing(False)
            self._invalidate_estimates()
            self._clear_result("正在计算末帧…")
            self.show_frame(times.size - 1)
            return
        index = int(np.searchsorted(times, target_time, side="right") - 1)
        index = max(0, min(index, times.size - 1))
        if index != self.current_index:
            self.show_frame(index)

    def export_screenshot(self, path: str | Path | None = None) -> Path | None:
        if path is None:
            selected, _ = QFileDialog.getSaveFileName(
                self, "导出当前截图", "us_motion_frame.png", "PNG 图片 (*.png)"
            )
            if not selected:
                return None
            path = selected
        target = Path(path)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        if not self.grab().save(str(target), "PNG"):
            raise OSError(f"无法写入截图：{target}")
        self.status.setText(f"截图已导出：{target}")
        return target

    def _export_screenshot_clicked(self) -> None:
        try:
            self.export_screenshot()
        except Exception as exc:
            self._show_error("截图导出失败", f"{exc}\n请换一个可写入的位置后重试。")

    def export_csv(self, path: str | Path | None = None) -> Path | None:
        if self.current_result is None:
            self._show_error("无法导出", "当前还没有完成计算的帧。")
            return None
        if path is None:
            selected, _ = QFileDialog.getSaveFileName(
                self, "导出当前帧 CSV", "us_motion_frame.csv", "CSV 表格 (*.csv)"
            )
            if not selected:
                return None
            path = selected
        target = Path(path)
        if target.suffix.lower() != ".csv":
            target = target.with_suffix(".csv")
        result = self.current_result
        depth = np.asarray(result["depth"])
        current = np.asarray(result["current"])
        past = result.get("past")
        past_values = np.full(depth.shape, np.nan) if past is None else np.asarray(past)
        direction = np.asarray(result.get("direction", np.full(depth.shape, np.nan)))
        screen_pass = np.asarray(result.get("screen_pass", np.zeros(depth.shape, dtype=bool)))
        support = np.asarray(result.get("support", np.zeros(depth.shape)))
        region = self._region_labels(result, depth.size)
        session_path = str(self._loaded_path or getattr(self.session, "path", ""))
        channel = self.selected_channel()
        current_time = result.get("current_time")
        past_time = result.get("past_time")
        lag_seconds = float(self.lag_combo.currentData())
        with target.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "session_path", "channel", "current_time", "past_time", "lag_seconds",
                "depth", "current", "past", "direction", "screen_pass", "support", "region",
            ])
            for values in zip(depth, current, past_values, direction, screen_pass.astype(int), support, region):
                writer.writerow([session_path, channel, current_time, past_time, lag_seconds, *values])
        self.status.setText(f"CSV 已导出：{target}")
        return target

    def _export_csv_clicked(self) -> None:
        try:
            self.export_csv()
        except Exception as exc:
            self._show_error("CSV 导出失败", f"{exc}\n请换一个可写入的位置后重试。")

    @staticmethod
    def _region_labels(result: dict[str, Any], point_count: int) -> np.ndarray:
        region = np.full(int(point_count), "unknown", dtype=object)
        for block in result.get("blocks") or []:
            start = max(0, int(block.get("start_index", 0)))
            stop = min(point_count, int(block.get("stop_index", start)))
            if stop > start:
                region[start:stop] = str(block.get("label", "unknown"))
        return region

    def _clear_result(self, message: str) -> None:
        self.current_result = None
        self.current_curve.setData([], [])
        self.band.set_blocks([], np.arange(RAW_POINT_COUNT, dtype=float))
        self.metrics.setText("当前时间 —　历史时间 —　计算 —　未知比例 —")
        self.plot.setTitle(message)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Space:
            self.toggle_playing()
        elif event.key() == Qt.Key.Key_Left:
            self.show_frame(self.current_index - 1, pause=True)
        elif event.key() == Qt.Key.Key_Right:
            self.show_frame(self.current_index + 1, pause=True)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._closing = True
        self.set_playing(False)
        self._load_cancel.set()
        self._generation += 1
        self.pool.clear()
        self.pool.waitForDone(2000)
        event.accept()


def create_smoke_core() -> Any:
    """Small deterministic core used only by ``--smoke-test`` and UI tests."""

    class SmokeCore:
        def load_session(self, path, progress=None, cancel=None):
            del path
            for percent, message in ((10, "读取元数据"), (55, "读取波形"), (100, "完成")):
                if cancel and cancel():
                    raise RuntimeError("已取消")
                if progress:
                    progress(percent, message)
            times = np.arange(0.0, 9.0, 0.05)
            depth = np.arange(957, dtype=float) + 20.0
            profiles = np.empty((times.size, 4, depth.size), dtype=np.float32)
            raw_depth = np.arange(RAW_POINT_COUNT, dtype=float)
            raw_frames = np.empty((times.size, 4, RAW_POINT_COUNT), dtype=np.uint8)
            for i, current_time in enumerate(times):
                for channel in range(4):
                    center = 24.0 + 3.5 * np.sin(current_time * 1.1 + channel)
                    profiles[i, channel] = 0.15 * np.sin(depth * 0.7) + np.exp(-((depth - center) / 3.2) ** 2)
                    # Deliberately independent from the 957-point processed profile.
                    raw_frames[i, channel] = (
                        np.arange(RAW_POINT_COUNT, dtype=np.int32) * 3
                        + i * 7 + channel * 29
                    ) % 256
            return SimpleNamespace(
                path=Path("smoke"), times=times, profiles=profiles,
                valid=np.ones((times.size, 4), dtype=bool), depth=depth,
                raw_frames=raw_frames, raw_depth=raw_depth,
                channels=(1, 2, 3, 4), metadata={"smoke": True},
            )

        def estimate_frame(self, session, index, channel, lag_seconds=0.20):
            started = time.perf_counter()
            channel_index = session.channels.index(channel)
            lag_frames = max(1, int(round(lag_seconds / 0.05)))
            past_index = index - lag_frames
            valid_pair = past_index >= 0
            current = session.profiles[index, channel_index]
            past = session.profiles[past_index, channel_index] if valid_pair else None
            direction = np.full(957, np.nan if not valid_pair else 0.0)
            blocks = []
            labels = ("left", "right", "near_zero", "unknown")
            for number, start in enumerate(range(0, 957, 49)):
                stop = min(957, start + 49)
                label = labels[(number + index // 8) % len(labels)] if valid_pair else "unknown"
                blocks.append({
                    "start_index": start, "stop_index": stop,
                    "depth_start": float(session.depth[start]),
                    "depth_stop": float(session.depth[stop - 1]), "label": label,
                    "median_direction": 0.0,
                })
            return {
                "current": current, "past": past, "depth": session.depth,
                "raw_current": session.raw_frames[index, channel_index],
                "raw_depth": session.raw_depth,
                "current_time": float(session.times[index]),
                "past_time": float(session.times[past_index]) if valid_pair else None,
                "valid_pair": valid_pair,
                "reason": "历史不足，方向未知" if not valid_pair else "演示数据：区域方向已计算",
                "direction": direction,
                "screen_pass": np.full(957, valid_pair, dtype=bool),
                "support": np.ones(957), "blocks": blocks,
                "compute_ms": (time.perf_counter() - started) * 1000.0,
            }

    return SmokeCore()


def run_smoke_test(
    screenshot: str | Path | None = None,
    session_path: str | Path | None = None,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    configure_font(app)
    core = create_smoke_core() if session_path is None else None
    source = "smoke" if session_path is None else session_path
    window = MotionViewerWindow(core, initial_path=source, initial_channel=2, initial_time=2.0)
    window.show()
    target = Path(screenshot) if screenshot is not None else Path(__file__).with_name("us_motion_viewer_preview.png")
    state = {"stage": 0, "generation": -1}

    def advance() -> None:
        if window.current_result is None or window._active_serial is not None:
            return
        if state["stage"] == 0:
            window.toggle_playing()
            state["stage"] = 1
            return
        if state["stage"] == 1:
            window.set_playing(False)
            window.slider.setValue(min(112, window.slider.maximum()))
            state["stage"] = 2
            return
        if state["stage"] == 2:
            if window.channel_combo.count() > 1:
                window.channel_combo.setCurrentIndex(1)
            state["stage"] = 3
            return
        if state["stage"] == 3:
            state["generation"] = window._generation + 1
            window.load_path(source)
            state["stage"] = 4
            return
        if state["stage"] == 4 and window._generation >= state["generation"]:
            times = np.asarray(window.session.times, dtype=float)
            target_index = int(np.argmin(np.abs(times - 5.6)))
            if window.channel_combo.count() > 1:
                window.channel_combo.setCurrentIndex(1)
            window.show_frame(target_index, pause=True)
            state["stage"] = 5
            return
        if state["stage"] == 5:
            window.export_screenshot(target)
            timer.stop()
            window.close()
            app.quit()

    timer = QTimer(window)
    timer.setInterval(90)
    timer.timeout.connect(advance)
    timer.start()
    QTimer.singleShot(10000, lambda: (timer.stop(), window.close(), app.quit()))
    return int(app.exec())


def launch_viewer(
    path: str | Path | None = None,
    channel: int | None = None,
    start_time: float | None = None,
    autoplay: bool = False,
    *,
    play: bool | None = None,
) -> int:
    if play is not None:
        autoplay = bool(play)
    app = QApplication.instance() or QApplication(sys.argv)
    configure_font(app)
    window = MotionViewerWindow(
        initial_path=path,
        initial_channel=channel,
        initial_time=start_time,
        autoplay=autoplay,
    )
    window.show()
    return int(app.exec())


__all__ = [
    "MotionBand", "MotionViewerWindow", "configure_font", "create_smoke_core",
    "launch_viewer", "run_smoke_test",
]
