"""Ultrasound waterfall region annotator.

This is deliberately a small, read-only viewer around ``us_motion_core``.
Annotations are stored in a separate JSON document and can be rasterized to
an int8 per-frame mask without changing the source binary.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder="row-major")

try:
    from .us_motion_core import load_session, LoadCancelled
    from .us_region_annotation_core import (
        LABELS, add_cut, add_segment, load_document, move_cut, new_document,
        rasterize, region_polygons, remove_cut, save_document, set_label,
        validate_document,
    )
except ImportError:  # running the file directly from Exo_Collection_System
    from us_motion_core import load_session, LoadCancelled
    from us_region_annotation_core import (
        LABELS, add_cut, add_segment, load_document, move_cut, new_document,
        rasterize, region_polygons, remove_cut, save_document, set_label,
        validate_document,
    )


COLORS = {-1: "#94a3b8", 0: "#64748b", 1: "#ef4444", 2: "#22c55e", 3: "#f59e0b"}


def configure_font(app: QtWidgets.QApplication) -> None:
    """Install a Chinese-capable font for Windows and offscreen screenshots."""
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if font_path.is_file():
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QtGui.QFont(families[0], 9))


def _source_for(path: str | Path) -> dict:
    binary = Path(path)
    if binary.is_dir():
        binary = binary / "ultrasound.bin"
    st = binary.stat()
    return {"binary_path": str(binary.resolve()), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def default_annotation_path(source: dict, root: Path | None = None, channel: int | None = None) -> Path:
    key = hashlib.sha1(str(source["binary_path"]).encode("utf-8")).hexdigest()[:16]
    root = root or Path(__file__).resolve().parent / "us_region_annotations"
    suffix = f"_ch{channel}" if channel is not None else ""
    return root / f"{key}{suffix}.json"


class LoadWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.stop_requested = False

    def cancel(self):
        self.stop_requested = True

    @QtCore.Slot()
    def run(self):
        try:
            session = load_session(self.path, self.progress.emit, lambda: self.stop_requested)
        except LoadCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(session)


class PlotWidget(pg.PlotWidget):
    clicked = QtCore.Signal(object)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(event)
        super().mousePressEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    """Main annotator window, also usable from QTest without a real session."""

    def __init__(self, session_path: str | None = None, channel: int | None = None,
                 screenshot: str | None = None):
        super().__init__()
        self.setWindowTitle("超声区域人工标注 · 增强显示")
        self.resize(1366, 768)
        self.setFont(QtGui.QFont("Microsoft YaHei", 9))
        self.session = None
        self.source = None
        self.document = None
        self.annotation_path = None
        self.current_segment = None
        self.current_channel = channel
        self.current_frame = 0
        self._dirty = False
        self._undo = []
        self._loading_path = None
        self._loading_thread = None
        self._loading_worker = None
        self._draw_points = []
        self._line_mode = False
        self.selected_block = None
        self._screenshot_path = screenshot
        self._level_cache = {}
        self._last_levels = None
        self._last_lut = None
        self._build_ui()
        if session_path:
            self.path_edit.setText(session_path)
            QtCore.QTimer.singleShot(0, self.load_path)

    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central); outer.setContentsMargins(8, 6, 8, 6)
        top = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit(); self.path_edit.setPlaceholderText("选择 session 文件夹或 ultrasound.bin")
        self.browse_btn = QtWidgets.QPushButton("选择 session")
        self.channel_combo = QtWidgets.QComboBox(); self.channel_combo.setObjectName("channelCombo")
        self.load_btn = QtWidgets.QPushButton("加载")
        self.display_combo = QtWidgets.QComboBox(); self.display_combo.addItems(["原始 ADC 灰度", "包络增强"]); self.display_combo.setCurrentIndex(1)
        self.save_btn = QtWidgets.QPushButton("保存 JSON"); self.save_as_btn = QtWidgets.QPushButton("另存 JSON"); self.open_btn = QtWidgets.QPushButton("打开 JSON")
        self.export_btn = QtWidgets.QPushButton("导出 NPZ")
        for w in (self.path_edit, self.browse_btn, self.channel_combo, self.load_btn, self.display_combo,
                  self.save_btn, self.save_as_btn, self.open_btn, self.export_btn): top.addWidget(w)
        outer.addLayout(top)
        display_row = QtWidgets.QHBoxLayout(); display_row.addWidget(QtWidgets.QLabel("对比度"))
        self.contrast_spin = QtWidgets.QDoubleSpinBox(); self.contrast_spin.setRange(.5, 3.0); self.contrast_spin.setSingleStep(.1); self.contrast_spin.setValue(1.0); display_row.addWidget(self.contrast_spin)
        display_row.addWidget(QtWidgets.QLabel("亮度")); self.brightness_spin = QtWidgets.QDoubleSpinBox(); self.brightness_spin.setRange(.5, 2.0); self.brightness_spin.setSingleStep(.1); self.brightness_spin.setValue(1.15); display_row.addWidget(self.brightness_spin)
        display_row.addWidget(QtWidgets.QLabel("配色")); self.palette_combo = QtWidgets.QComboBox(); self.palette_combo.addItems(["灰度", "蓝黄伪彩 (viridis)"]); self.palette_combo.setCurrentIndex(1); display_row.addWidget(self.palette_combo)
        self.reset_display_btn = QtWidgets.QPushButton("恢复推荐"); self.mask_checkbox = QtWidgets.QCheckBox("显示标注遮罩"); self.mask_checkbox.setChecked(True); display_row.addWidget(self.reset_display_btn); display_row.addWidget(self.mask_checkbox); display_row.addStretch(1); outer.addLayout(display_row)
        help_label = QtWidgets.QLabel("操作：上方 overview 拖选时间段并点击新增；选段后下方放大。画线模式依次点击左端深度、右端深度，点击小块后用标签下拉框赋值。标签仅表示肉眼定性，不预设物理意义。")
        help_label.setStyleSheet("color:#475569")
        outer.addWidget(help_label)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical); outer.addWidget(splitter, 1)
        overview_box = QtWidgets.QWidget(); ovl = QtWidgets.QVBoxLayout(overview_box); ovl.setContentsMargins(0, 0, 0, 0)
        self.overview = pg.PlotWidget(background="#111827", title="全 session overview（时间 / 深度）")
        self.overview.setLabel("bottom", "时间", units="s"); self.overview.setLabel("left", "深度", units="ADC")
        self.overview.invertY(True); self.overview.showGrid(x=True, y=True, alpha=.2)
        ovl.addWidget(self.overview, 1)
        ovtools = QtWidgets.QHBoxLayout(); self.add_segment_btn = QtWidgets.QPushButton("新增时间段")
        self.delete_segment_btn = QtWidgets.QPushButton("删除时间段"); self.segment_list = QtWidgets.QListWidget()
        self.segment_list.setMaximumHeight(70)
        for w in (self.add_segment_btn, self.delete_segment_btn): ovtools.addWidget(w)
        ovtools.addWidget(self.segment_list, 1); ovl.addLayout(ovtools)
        splitter.addWidget(overview_box)
        detail_box = QtWidgets.QWidget(); dl = QtWidgets.QVBoxLayout(detail_box); dl.setContentsMargins(0, 0, 0, 0)
        self.detail = PlotWidget(background="#111827"); self.detail.setLabel("bottom", "时间", units="s"); self.detail.setLabel("left", "深度", units="ADC")
        self.detail.invertY(True); self.detail.showGrid(x=True, y=True, alpha=.2); dl.addWidget(self.detail, 1)
        row = QtWidgets.QHBoxLayout(); self.line_mode_btn = QtWidgets.QPushButton("画线模式：关"); self.line_mode_btn.setCheckable(True); self.undo_btn = QtWidgets.QPushButton("撤销")
        self.prev_btn = QtWidgets.QPushButton("上一帧"); self.play_btn = QtWidgets.QPushButton("播放"); self.next_btn = QtWidgets.QPushButton("下一帧")
        self.speed_combo = QtWidgets.QComboBox(); self.speed_combo.addItems(["0.5×", "1×", "2×", "4×"]); self.speed_combo.setCurrentIndex(1)
        self.frame_label = QtWidgets.QLabel("帧 0 / 0"); self.block_label = QtWidgets.QLabel("当前块：无")
        row.addWidget(self.line_mode_btn); row.addWidget(self.undo_btn); row.addWidget(self.prev_btn); row.addWidget(self.play_btn); row.addWidget(self.next_btn); row.addWidget(QtWidgets.QLabel("速度")); row.addWidget(self.speed_combo); row.addWidget(self.frame_label); row.addWidget(self.block_label, 1)
        dl.addLayout(row)
        edit = QtWidgets.QHBoxLayout(); self.cut_table = QtWidgets.QTableWidget(0, 3); self.cut_table.setHorizontalHeaderLabels(["线", "左端深度", "右端深度"]); self.cut_table.setMaximumHeight(92); self.cut_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.left_spin = QtWidgets.QDoubleSpinBox(); self.right_spin = QtWidgets.QDoubleSpinBox();
        for s in (self.left_spin, self.right_spin): s.setRange(0, 996); s.setDecimals(1)
        self.update_cut_btn = QtWidgets.QPushButton("更新选中线"); self.remove_cut_btn = QtWidgets.QPushButton("删除选中线")
        self.label_combo = QtWidgets.QComboBox(); self.label_combo.addItem("未标注", -1)
        for k in (0, 1, 2, 3): self.label_combo.addItem(LABELS[k], k)
        self.apply_label_btn = QtWidgets.QPushButton("赋予当前块标签")
        for w in (self.cut_table, QtWidgets.QLabel("左"), self.left_spin, QtWidgets.QLabel("右"), self.right_spin, self.update_cut_btn, self.remove_cut_btn, QtWidgets.QLabel("标签"), self.label_combo, self.apply_label_btn): edit.addWidget(w)
        dl.addLayout(edit); splitter.addWidget(detail_box); splitter.setSizes([330, 390])
        self.status = self.statusBar(); self.progress = QtWidgets.QProgressBar(); self.progress.setMaximumWidth(190); self.status.addPermanentWidget(self.progress)
        self.region = None; self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._advance_frame); self.timer.setInterval(50)
        self._overview_marks = []
        self.browse_btn.clicked.connect(self.choose_session); self.load_btn.clicked.connect(self.load_path); self.channel_combo.currentIndexChanged.connect(self.channel_changed)
        self.display_combo.currentIndexChanged.connect(self._display_changed); self.contrast_spin.valueChanged.connect(self._display_changed); self.brightness_spin.valueChanged.connect(self._display_changed); self.palette_combo.currentIndexChanged.connect(self._display_changed); self.mask_checkbox.toggled.connect(self._display_changed); self.reset_display_btn.clicked.connect(self.reset_display); self.add_segment_btn.clicked.connect(self.add_segment_from_region); self.delete_segment_btn.clicked.connect(self.delete_segment)
        self.segment_list.currentRowChanged.connect(self.segment_changed); self.line_mode_btn.toggled.connect(self.set_line_mode); self.detail.clicked.connect(self.detail_clicked)
        self.cut_table.itemSelectionChanged.connect(self.cut_selected); self.update_cut_btn.clicked.connect(self.update_cut); self.remove_cut_btn.clicked.connect(self.remove_selected_cut); self.apply_label_btn.clicked.connect(self.apply_label)
        self.prev_btn.clicked.connect(lambda: self.set_frame(self.current_frame - 1)); self.next_btn.clicked.connect(lambda: self.set_frame(self.current_frame + 1)); self.play_btn.clicked.connect(self.toggle_play); self.speed_combo.currentIndexChanged.connect(self.speed_changed)
        self.save_btn.clicked.connect(self.save_json); self.save_as_btn.clicked.connect(self.save_json_as); self.open_btn.clicked.connect(self.open_json); self.export_btn.clicked.connect(self.export_npz); self.undo_btn.clicked.connect(self.undo)

    def choose_session(self):
        p = QtWidgets.QFileDialog.getOpenFileName(self, "选择 ultrasound.bin", "", "ultrasound.bin (ultrasound.bin);;所有文件 (*)")[0]
        if p: self.path_edit.setText(p); self.load_path()

    def load_path(self):
        path = self.path_edit.text().strip()
        if not path: return
        if self._loading_thread: return
        if self._dirty and QtWidgets.QMessageBox.question(self, "未保存", "当前标注尚未保存，仍要切换 session 吗？") != QtWidgets.QMessageBox.StandardButton.Yes: return
        self.load_btn.setEnabled(False); self.browse_btn.setEnabled(False); self.path_edit.setEnabled(False); self.progress.setValue(0); self.status.showMessage("正在加载……")
        self._loading_path = path; self._loading_thread = QtCore.QThread(self); self._loading_worker = LoadWorker(path); self._loading_worker.moveToThread(self._loading_thread)
        self._loading_thread.started.connect(self._loading_worker.run); self._loading_worker.progress.connect(self._on_load_progress)
        self._loading_worker.finished.connect(self._loaded); self._loading_worker.failed.connect(self._load_failed); self._loading_worker.cancelled.connect(self._load_cancelled)
        self._loading_worker.finished.connect(self._finish_thread); self._loading_worker.failed.connect(self._finish_thread); self._loading_worker.cancelled.connect(self._finish_thread); self._loading_thread.start()

    @QtCore.Slot()
    def _finish_thread(self, *_):
        if self._loading_thread:
            thread, worker = self._loading_thread, self._loading_worker
            if worker: worker.deleteLater()
            thread.quit(); thread.wait(); thread.deleteLater(); self._loading_thread = None
        self._loading_worker = None; self._loading_path = None; self.load_btn.setEnabled(True); self.browse_btn.setEnabled(True); self.path_edit.setEnabled(True)

    @QtCore.Slot(int, str)
    def _on_load_progress(self, percent, message):
        self.progress.setValue(int(percent)); self.status.showMessage(str(message))

    @QtCore.Slot(str)
    def _load_failed(self, msg): self.status.showMessage("加载失败"); QtWidgets.QMessageBox.critical(self, "加载失败", msg)

    @QtCore.Slot()
    def _load_cancelled(self): self.status.showMessage("已取消加载")

    @QtCore.Slot(object)
    def _loaded(self, session):
        self.session = session; self.source = _source_for(self._loading_path or self.path_edit.text()); self.document = new_document(self.source, (float(session.times[0]), float(session.times[-1])), (0, 996), [str(c) for c in session.channels]); self.annotation_path = None; self.current_segment = None; self.selected_block = None; self._draw_points.clear(); self._undo.clear(); self._level_cache.clear(); self._dirty = False; self.timer.stop(); self.play_btn.setText("播放")
        self.channel_combo.blockSignals(True); self.channel_combo.clear(); self.channel_combo.addItems([f"通道 {c}" for c in session.channels]); idx = session.channels.index(self.current_channel) if self.current_channel in session.channels else 0; self.channel_combo.setCurrentIndex(idx); self.channel_combo.blockSignals(False); self.current_channel = session.channels[idx]
        self._setup_session_plots(); self._refresh_segments(); self.status.showMessage(f"加载完成：{len(session.times)} 帧，{len(session.channels)} 个物理通道")

    def _setup_session_plots(self):
        self.overview.clear(); self.detail.clear(); self.region = pg.LinearRegionItem([float(self.session.times[0]), float(self.session.times[min(len(self.session.times)-1, max(1, len(self.session.times)//5))])], bounds=(float(self.session.times[0]), float(self.session.times[-1])), brush=pg.mkBrush(80, 160, 240, 50)); self.overview.addItem(self.region); self.refresh_images(); self.set_frame(0)

    def channel_changed(self, idx):
        if self.session and idx >= 0:
            self.current_channel = self.session.channels[idx]; self.current_segment = None; self.selected_block = None; self._draw_points.clear(); self.refresh_images(); self._refresh_segments()

    def _matrix(self, start=0, stop=None):
        stop = stop if stop is not None else len(self.session.times)
        c = self.session.channels.index(self.current_channel); arr = self.session.raw_frames[start:stop, c] if self.display_combo.currentIndex() == 0 else self.session.profiles[start:stop, c]
        return np.asarray(arr, dtype=np.float32)

    def _display_key(self):
        source = self.source or {}
        return (source.get("binary_path"), source.get("size_bytes"), source.get("mtime_ns"), int(self.current_channel), int(self.display_combo.currentIndex()))

    def _display_levels(self):
        """Return one valid-pixel p2/p99.5 scale for the current session view."""
        key = self._display_key()
        if key in self._level_cache: return self._level_cache[key]
        c = self.session.channels.index(self.current_channel)
        arr = np.asarray(self.session.raw_frames[:, c] if self.display_combo.currentIndex() == 0 else self.session.profiles[:, c], dtype=np.float32)
        valid = np.asarray(self.session.valid[:, c], dtype=bool)
        values = arr[valid] if valid.any() else arr
        values = values[np.isfinite(values)]
        if values.size == 0: levels = (0.0, 1.0)
        else:
            levels = (float(np.percentile(values, 2)), float(np.percentile(values, 99.5)))
            if not levels[1] > levels[0]: levels = (levels[0], levels[0] + 1.0)
        self._level_cache[key] = levels
        return levels

    def _display_lut(self):
        x = np.linspace(0, 1, 256, dtype=np.float32) ** (1.0 / float(self.brightness_spin.value()))
        if self.palette_combo.currentIndex() == 1:
            base = pg.colormap.get("viridis").getLookupTable(nPts=256, alpha=True).astype(np.float32)
        else:
            g = np.rint(x * 255).astype(np.uint8); return np.column_stack([g, g, g, np.full(256, 255, np.uint8)])
        return np.asarray(base[np.clip(np.rint(x * 255).astype(int), 0, 255)], dtype=np.uint8)

    def reset_display(self):
        for widget, value in ((self.display_combo, 1), (self.contrast_spin, 1.0), (self.brightness_spin, 1.15), (self.palette_combo, 1), (self.mask_checkbox, True)):
            widget.blockSignals(True); (widget.setCurrentIndex(value) if hasattr(widget, "setCurrentIndex") else widget.setValue(value) if hasattr(widget, "setValue") else widget.setChecked(value)); widget.blockSignals(False)
        self._display_changed()

    def _display_changed(self, *_):
        self.refresh_images(preserve_view=True)

    def _show_image(self, plot, arr, times):
        if arr.size == 0: return
        d0, d1 = (0, 996) if arr.shape[1] == 997 else (20, 976)
        dt = float(np.median(np.diff(times))) if len(times) > 1 else .05
        base_lo, base_hi = self._display_levels(); contrast = float(self.contrast_spin.value()); center = (base_lo + base_hi) / 2.0; half = (base_hi - base_lo) / max(contrast, .01) / 2.0; levels = (center - half, center + half); lut = self._display_lut(); display = np.where(np.isfinite(arr), arr, levels[0])
        self._last_levels, self._last_lut = levels, lut
        plot.clear(); item = pg.ImageItem(display.T, axisOrder="row-major"); item.setRect(QtCore.QRectF(float(times[0]) - dt / 2, d0 - .5, len(times) * dt, arr.shape[1])); item.setLevels(levels); item.setLookupTable(lut); plot.addItem(item); plot.invertY(True); plot.setXRange(float(times[0]), float(times[-1]), padding=0); plot.setYRange(d0, d1, padding=0)

    def refresh_images(self, preserve_view=False):
        if not self.session: return
        overview_range = self.overview.viewRange() if preserve_view else None
        detail_range = self.detail.viewRange() if preserve_view else None
        self._show_image(self.overview, self._matrix(), self.session.times)
        if self.region: self.overview.addItem(self.region)
        self._draw_overview_marks()
        self._draw_detail()
        if preserve_view:
            if overview_range: self.overview.setRange(xRange=overview_range[0], yRange=overview_range[1], padding=0)
            if detail_range and self.current_segment: self.detail.setRange(xRange=detail_range[0], yRange=detail_range[1], padding=0)
        # _show_image clears plot items, including the frame cursor. Re-add it
        # after restoring ranges so display-only changes cannot lose playback state.
        if self.session:
            self.set_frame(self.current_frame)

    def _draw_overview_marks(self):
        for item in self._overview_marks:
            try: self.overview.removeItem(item)
            except Exception: pass
        self._overview_marks = []
        if not self.document: return
        for number, seg in enumerate(self.document.get("channels", {}).get(str(self.current_channel), []), 1):
            for edge in (seg["start"], seg["end"]):
                line = pg.InfiniteLine(pos=float(edge), angle=90, movable=False, pen=pg.mkPen("#fbbf24", width=1, style=QtCore.Qt.PenStyle.DashLine)); self.overview.addItem(line); self._overview_marks.append(line)
            text = pg.TextItem(f"段{number}", color="#fde68a", anchor=(.5, 0)); text.setPos((float(seg["start"])+float(seg["end"])) / 2, 30); self.overview.addItem(text); self._overview_marks.append(text)

    def _draw_detail(self):
        if not self.session or not self.current_segment:
            self.detail.clear(); self.cut_table.setRowCount(0); return
        s, e = self.current_segment["start"], self.current_segment["end"]; i0 = max(0, int(np.searchsorted(self.session.times, s, side="left"))); i1 = min(len(self.session.times), int(np.searchsorted(self.session.times, e, side="right")))
        self.detail.clear(); self._show_image(self.detail, self._matrix(i0, i1), self.session.times[i0:i1]); self._draw_overlays(); self._update_segment_row()
        self.detail_frame_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#38bdf8", width=1)); self.detail.addItem(self.detail_frame_line); self.detail_frame_line.setValue(float(self.session.times[self.current_frame]))

    def _draw_overlays(self):
        if not self.current_segment: return
        s, e = self.current_segment["start"], self.current_segment["end"]
        for i, cut in enumerate(self.current_segment.get("cuts", [])):
            pen = pg.mkPen("#f8fafc", width=2); self.detail.plot([s, e], cut, pen=pen, name=f"cut{i}")
        try: polys = region_polygons(self.current_segment, (0, 996))
        except Exception: polys = []
        for i, poly in enumerate(polys):
            if len(poly) < 3: continue
            label = self.current_segment.get("labels", [])[i] if i < len(self.current_segment.get("labels", [])) else -1
            if label == -1:
                continue
            if not self.mask_checkbox.isChecked():
                center = np.mean(poly, axis=0); t = pg.TextItem(LABELS.get(label, ""), color="#f8fafc", anchor=(.5, .5)); t.setPos(float(center[0]), float(center[1])); self.detail.addItem(t)
                continue
            # A closed PlotDataItem gives a stable low-opacity block overlay
            # while retaining the exact sloped boundary coordinates.
            closed = np.vstack([poly, poly[:1]])
            q = pg.PlotDataItem(closed[:, 0], closed[:, 1], pen=pg.mkPen(COLORS.get(label, "#94a3b8"), width=1), brush=pg.mkBrush(QtGui.QColor(COLORS.get(label, "#94a3b8"))), fillLevel=float(np.max(closed[:, 1])))
            q.setOpacity(.22); self.detail.addItem(q)
            if label != -1:
                center = np.mean(poly, axis=0); t = pg.TextItem(LABELS.get(label, ""), color="#f8fafc", anchor=(.5, .5)); t.setPos(float(center[0]), float(center[1])); self.detail.addItem(t)
        self._refresh_cut_table()

    def _update_segment_row(self):
        if not self.current_segment: return
        for row in range(self.segment_list.count()):
            item = self.document["channels"][str(self.current_channel)][row]
            if item.get("id") == self.current_segment.get("id"):
                self.segment_list.item(row).setText(f"{item['start']:.3f}–{item['end']:.3f} s | {len(item.get('cuts', []))} 条线")
                break

    def _refresh_cut_table(self):
        cuts = self.current_segment.get("cuts", []) if self.current_segment else []; self.cut_table.setRowCount(len(cuts))
        for i, (a, b) in enumerate(cuts):
            for j, v in enumerate((i + 1, a, b)):
                item = QtWidgets.QTableWidgetItem(str(v)); item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable); self.cut_table.setItem(i, j, item)

    def _refresh_segments(self):
        self.segment_list.clear()
        if not self.document: return
        for s in self.document.get("channels", {}).get(str(self.current_channel), []): self.segment_list.addItem(f"{s['start']:.3f}–{s['end']:.3f} s | {len(s.get('cuts', []))} 条线")
        if self.segment_list.count(): self.segment_list.setCurrentRow(0)

    def add_segment_from_region(self):
        if not self.document or not self.region: return
        a, b = map(float, self.region.getRegion()); a, b = min(a, b), max(a, b)
        self._push_undo()
        try: seg = add_segment(self.document, str(self.current_channel), a, b)
        except Exception as exc: QtWidgets.QMessageBox.warning(self, "不能新增时间段", str(exc)); return
        self._dirty = True; self.selected_block = None; self._draw_points.clear(); self._refresh_segments(); self._draw_overview_marks()
        for row, item in enumerate(self.document["channels"][str(self.current_channel)]):
            if item["id"] == seg["id"]: self.segment_list.setCurrentRow(row); break
        self.status.showMessage("已新增时间段；可进入画线模式")

    def delete_segment(self):
        if not self.current_segment or not self.document: return
        self._push_undo(); arr = self.document["channels"][str(self.current_channel)]; arr[:] = [s for s in arr if s is not self.current_segment and s.get("id") != self.current_segment.get("id")]; self.current_segment = None; self.selected_block = None; self._draw_points.clear(); self._dirty = True; self._refresh_segments(); self._draw_overview_marks(); self._draw_detail()

    def segment_changed(self, row):
        arr = self.document.get("channels", {}).get(str(self.current_channel), []) if self.document else []; self.current_segment = arr[row] if 0 <= row < len(arr) else None; self.selected_block = None; self._draw_points.clear(); self._draw_detail()

    def set_line_mode(self, checked):
        self._line_mode = checked; self._draw_points.clear(); self.line_mode_btn.setText("结束画线" if checked else "开始画线"); self.status.showMessage("请依次点击左端深度、右端深度" if checked else "画线模式已关闭")

    def detail_clicked(self, event):
        if not self.current_segment or not self.session: return
        # QMouseEvent.position() is in the PlotWidget viewport; convert it
        # through the widget so window offsets never shift the depth value.
        scene_pos = self.detail.mapToScene(event.position().toPoint()) if hasattr(event, "position") else event.scenePosition()
        pos = self.detail.plotItem.vb.mapSceneToView(scene_pos); x, y = float(pos.x()), float(pos.y()); s, e = self.current_segment["start"], self.current_segment["end"]
        if self._line_mode:
            self._draw_points.append(float(np.clip(y, 0, 996)))
            if len(self._draw_points) == 2:
                try: self._push_undo(); add_cut(self.current_segment, self._draw_points[0], self._draw_points[1], (0, 996)); self._dirty = True
                except Exception as exc: QtWidgets.QMessageBox.warning(self, "不能新增分割线", str(exc))
                self._draw_points.clear(); QtCore.QTimer.singleShot(120, self._draw_detail)
            else: self.status.showMessage("已记录左端深度，请点击右端深度")
        else:
            idx = self._block_at(x, y)
            if idx is not None:
                self.selected_block = idx
                self.block_label.setText(f"当前块：第 {idx+1} 块")
                self.status.showMessage(f"已选中第 {idx+1} 个小块，可赋标签")

    def _block_at(self, x, y):
        if not self.current_segment or not (self.current_segment["start"] <= x <= self.current_segment["end"]) or not (0 <= y <= 996): return None
        cuts = self.current_segment.get("cuts", [])
        vals = [0] + [c[0] + (c[1]-c[0]) * (x-self.current_segment["start"]) / max(self.current_segment["end"]-self.current_segment["start"], 1e-9) for c in cuts] + [996]
        for i in range(len(vals)-1):
            low, high = min(vals[i], vals[i+1]), max(vals[i], vals[i+1])
            if low <= y < high or (i == len(vals) - 2 and y == high): return i
        return None

    def cut_selected(self):
        row = self.cut_table.currentRow(); cuts = self.current_segment.get("cuts", []) if self.current_segment else []
        if 0 <= row < len(cuts): self.left_spin.setValue(cuts[row][0]); self.right_spin.setValue(cuts[row][1])

    def update_cut(self):
        row = self.cut_table.currentRow()
        if self.current_segment and row >= 0:
            try: self._push_undo(); move_cut(self.current_segment, row, self.left_spin.value(), self.right_spin.value(), (0, 996)); self._dirty = True; self._draw_detail()
            except Exception as exc: QtWidgets.QMessageBox.warning(self, "不能更新分割线", str(exc))

    def remove_selected_cut(self):
        row = self.cut_table.currentRow()
        if self.current_segment and row >= 0: self._push_undo(); remove_cut(self.current_segment, row); self._dirty = True; self._draw_detail()

    def apply_label(self):
        if not self.current_segment: return
        idx = self.selected_block if self.selected_block is not None else 0
        if self.selected_block is None:
            self.status.showMessage("请先点击下方小块选择标注区域")
            return
        if self.selected_block < len(self.current_segment.get("labels", [])):
            self._push_undo(); set_label(self.current_segment, self.selected_block, int(self.label_combo.currentData())); self._dirty = True; self._draw_detail()

    def set_frame(self, index):
        if not self.session: return
        self.current_frame = max(0, min(int(index), len(self.session.times)-1)); self.frame_label.setText(f"帧 {self.current_frame} / {len(self.session.times)-1}  ({self.session.times[self.current_frame]:.3f}s)")
        if hasattr(self, "frame_line") and self.frame_line in self.overview.plotItem.items: self.frame_line.setValue(float(self.session.times[self.current_frame]))
        else: self.frame_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#38bdf8", width=2)); self.overview.addItem(self.frame_line); self.frame_line.setValue(float(self.session.times[self.current_frame]))
        if hasattr(self, "detail_frame_line") and self.detail_frame_line in self.detail.plotItem.items: self.detail_frame_line.setValue(float(self.session.times[self.current_frame]))

    def toggle_play(self):
        if self.timer.isActive(): self.timer.stop(); self.play_btn.setText("播放")
        else: self.timer.start(); self.play_btn.setText("暂停")

    def speed_changed(self, index):
        self.timer.setInterval(max(10, int(50 / (0.5 * (2 ** int(index))))) if index >= 0 else 50)

    def _advance_frame(self):
        if not self.session: return
        if self.current_frame >= len(self.session.times)-1: self.timer.stop(); self.play_btn.setText("播放")
        else: self.set_frame(self.current_frame + 1)

    def save_json(self):
        if not self.document: return
        path = self.annotation_path or default_annotation_path(self.source, channel=self.current_channel)
        if self.annotation_path is None and path.exists() and QtWidgets.QMessageBox.question(self, "标注已存在", f"默认文件已存在：{path.name}\n确认覆盖吗？", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No, QtWidgets.QMessageBox.StandardButton.No) != QtWidgets.QMessageBox.StandardButton.Yes:
            self.status.showMessage("已取消保存，原有 JSON 未覆盖"); return
        path.parent.mkdir(parents=True, exist_ok=True)
        try: save_document(self.document, path); self.annotation_path = path; self._dirty = False; self.status.showMessage(f"已保存：{path}")
        except Exception as exc: QtWidgets.QMessageBox.critical(self, "保存失败", str(exc))

    def save_json_as(self):
        if not self.document: return
        p = QtWidgets.QFileDialog.getSaveFileName(self, "另存标注 JSON", str(default_annotation_path(self.source, channel=self.current_channel)), "JSON (*.json)")[0]
        if p:
            self.annotation_path = Path(p)
            self.save_json()

    def _push_undo(self):
        if self.document is not None:
            self._undo.append(copy.deepcopy(self.document))
            if len(self._undo) > 30:
                self._undo.pop(0)

    def undo(self):
        if not self._undo:
            self.status.showMessage("没有可撤销的操作")
            return
        self.document = self._undo.pop(); self._dirty = True; self.current_segment = None; self._refresh_segments()
        if self.segment_list.count(): self.segment_list.setCurrentRow(0)
        self._draw_detail(); self.status.showMessage("已撤销")

    def open_json(self):
        if not self.session or not self.source:
            QtWidgets.QMessageBox.information(self, "尚未加载 session", "请先加载 ultrasound.bin，再打开标注 JSON")
            return
        if self._dirty and QtWidgets.QMessageBox.question(self, "未保存", "当前标注尚未保存，仍要打开另一个 JSON 吗？") != QtWidgets.QMessageBox.StandardButton.Yes: return
        p = QtWidgets.QFileDialog.getOpenFileName(self, "打开标注 JSON", str(default_annotation_path(self.source).parent if self.source else Path.cwd()), "JSON (*.json)")[0]
        if not p: return
        try: doc = load_document(p, expected_source=self.source)
        except Exception as exc: QtWidgets.QMessageBox.critical(self, "打开失败", str(exc)); return
        tb, db = doc.get("time_bounds", []), doc.get("depth_bounds", [])
        if str(self.current_channel) not in doc.get("channels", {}) or len(tb) != 2 or len(db) != 2 or not np.allclose(tb, [self.session.times[0], self.session.times[-1]]) or not np.allclose(db, [0, 996]):
            QtWidgets.QMessageBox.critical(self, "打开失败", "JSON 的通道、时间范围或深度范围与当前 session 不一致"); return
        self.document = doc; self.annotation_path = Path(p); self.current_segment = None; self.selected_block = None; self._draw_points.clear(); self._undo.clear(); self._dirty = False; self._refresh_segments(); self._draw_overview_marks(); self._draw_detail(); self.status.showMessage("标注已打开")

    def export_npz(self):
        if not self.document or not self.session: return
        p = QtWidgets.QFileDialog.getSaveFileName(self, "导出逐帧 mask", str(Path(self.annotation_path or default_annotation_path(self.source, channel=self.current_channel)).with_suffix(".npz")), "NPZ (*.npz)")[0]
        if not p: return
        try:
            mask = rasterize(self.document, str(self.current_channel), self.session.times, np.arange(997), self.session.valid[:, self.session.channels.index(self.current_channel)])
            np.savez_compressed(p, mask=mask.astype(np.int8), times=self.session.times, depth=np.arange(997), valid=self.session.valid[:, self.session.channels.index(self.current_channel)], channel=int(self.current_channel), labels=json.dumps(LABELS, ensure_ascii=False), label_semantics=self.document.get("label_semantics", "visual_qualitative_unspecified"), source=json.dumps(self.source, ensure_ascii=False))
            self.status.showMessage(f"已导出：{p}")
        except Exception as exc: QtWidgets.QMessageBox.critical(self, "导出失败", str(exc))

    def _make_synthetic_session(self):
        class S: pass
        s = S(); s.times = np.linspace(0, 10, 201); s.channels = (3,); s.raw_frames = np.random.default_rng(2).normal(128, 28, (201, 1, 997)).astype(np.float32); s.profiles = np.abs(s.raw_frames[:, :, 20:977]-128); s.valid = np.ones((201, 1), bool); s.depth = np.arange(20, 977); s.raw_depth = np.arange(997); s.metadata = {}; self.session = s; self.path_edit.setText(str(Path(tempfile.gettempdir()) / "synthetic_ultrasound.bin")); self.source = {"binary_path": str(Path(self.path_edit.text()).resolve()), "size_bytes": 1, "mtime_ns": 1}; self.document = new_document(self.source, (0., 10.), (0, 996), ["3"]); self.current_channel = 3; self.channel_combo.clear(); self.channel_combo.addItem("通道 3"); self._setup_session_plots(); self.region.setRegion((2., 5.)); self.add_segment_from_region(); self.set_line_mode(True); self._draw_points = [250., 420.]; add_cut(self.current_segment, 250., 420., (0, 996)); set_label(self.current_segment, 0, 1); self._draw_detail(); self._dirty = True

    def run_smoke(self):
        smoke_dir = Path(tempfile.mkdtemp(prefix=".us_region_smoke_", dir=Path(__file__).resolve().parent))
        try:
            self._make_synthetic_session(); self.annotation_path = smoke_dir / "synthetic.json"; self.save_json()
            load_document(self.annotation_path, expected_source=self.source)
            mask = rasterize(self.document, "3", self.session.times, np.arange(997), self.session.valid[:, 0])
            np.savez_compressed(smoke_dir / "synthetic.npz", mask=mask, times=self.session.times, depth=np.arange(997))
            self._screenshot_path and self.grab().save(self._screenshot_path)
        finally:
            shutil.rmtree(smoke_dir, ignore_errors=True)

    def closeEvent(self, event):
        if self._loading_worker:
            self._loading_worker.cancel()
            if self._loading_thread:
                self._loading_thread.quit(); self._loading_thread.wait(2000)
                if self._loading_thread.isRunning():
                    self._loading_thread.finished.connect(self.close); event.ignore(); return
        if self._dirty and QtWidgets.QMessageBox.question(self, "未保存", "当前标注尚未保存，仍要退出吗？") != QtWidgets.QMessageBox.StandardButton.Yes: event.ignore(); return
        self.timer.stop(); event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="超声区域人工标注工具")
    parser.add_argument("--session"); parser.add_argument("--channel", type=int); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--screenshot")
    args = parser.parse_args(argv); app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv); configure_font(app); pg.setConfigOptions(imageAxisOrder="row-major")
    win = MainWindow(args.session, args.channel, args.screenshot); win.show()
    if args.smoke_test:
        QtCore.QTimer.singleShot(250, win.run_smoke)
        QtCore.QTimer.singleShot(900, app.quit)
    return app.exec()


if __name__ == "__main__": main()
