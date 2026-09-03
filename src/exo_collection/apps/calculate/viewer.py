"""Exo Calculate 的 3D + 关节力矩联合回放页（完全离线）。

数据来源是 ``process_session.py``（opensim 环境）导出的 ``viewer/*.npy`` +
``viewer_meta.json``，本模块**不 import opensim、不访问网络**：

- 3D：自绘 QPainter 正交投影（无 OpenGL/PyOpenGL、无 CDN），叠加 OpenSim 骨架、
  19 个模型 marker、15 个实验 marker、左右 COP / GRF 箭头，支持旋转/平移/缩放/预设视角。
- 曲线：pyqtgraph 2D（髋关节力矩 × 左右 + 右腿 IMU 三轴加速度），与 3D 共用同一个时间游标。

设计约束（见 prompt5.md §8/§13）：
- 19 个模型点与 15 个实测点必须同屏、有独立图例与开关；内侧 4 点只作模型预测点；
- 缺失/遮挡的实验 marker 明确隐藏，不得连到错误位置；
- 播放不一次性创建数万 Qt 图元——每帧只重画几十个图元。
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# 颜色（色盲友好分类色，来自 Tableau 10）
# ---------------------------------------------------------------------------
COLOR_MODEL_MARKER = QColor("#4E79A7")     # 蓝 —— 模型 19 点
COLOR_EXP_MARKER = QColor("#E15759")       # 红 —— 实验 15 点
COLOR_RIGHT = QColor("#F28E2B")            # 橙 —— 右脚 COP/GRF
COLOR_LEFT = QColor("#59A14F")             # 绿 —— 左脚 COP/GRF
COLOR_SKELETON = QColor("#8C8C8C")         # 灰 —— 骨架
COLOR_ERROR_LINE = QColor("#B0B0B0")       # 浅灰 —— 误差连线
COLOR_AXIS_X = QColor("#D62728")
COLOR_AXIS_Y = QColor("#2CA02C")
COLOR_AXIS_Z = QColor("#1F77B4")
COLOR_GROUND = QColor("#3A3A3A")
COLOR_BG = QColor("#1E1E1E")

_MOMENT_ROW_LABELS = ["髋关节力矩"]


@dataclass
class ViewerData:
    """viewer 目录的一次加载结果（全部为纯 NumPy 内存视图）。"""

    time_s: np.ndarray
    model_markers: np.ndarray          # (n, 19, 3) mm
    experimental_markers: np.ndarray   # (n, 15, 3) mm
    body_origins: np.ndarray           # (n, N_body, 3) mm
    cop: np.ndarray                    # (n, 2, 3) mm
    grf: np.ndarray                    # (n, 2, 3) N
    moments: np.ndarray                # (n, 6) Nm
    model_marker_names: list[str]
    experimental_marker_names: list[str]
    medial_marker_names: set[str]
    body_names: list[str]
    skeleton_segments: list[tuple[str, str]]
    moment_names: list[str]
    frame_rate_hz: float
    mass_kg: float

    @property
    def n_frames(self) -> int:
        return int(self.time_s.shape[0])


def load_viewer_data(viewer_dir: Path) -> ViewerData:
    """从 ``viewer_dir`` 读取导出数据；文件缺失抛 ``FileNotFoundError``。"""
    viewer_dir = Path(viewer_dir)
    meta_path = viewer_dir / "viewer_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"缺少 viewer_meta.json：{meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    skeleton = [tuple(seg) for seg in meta.get("skeleton_segments", [])]
    body_names = list(meta.get("body_names", []))

    def _load(name: str) -> np.ndarray:
        path = viewer_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 viewer 数据：{path}")
        return np.load(path)

    return ViewerData(
        time_s=_load("time_s"),
        model_markers=_load("model_markers"),
        experimental_markers=_load("experimental_markers"),
        body_origins=_load("body_origins"),
        cop=_load("cop"),
        grf=_load("grf"),
        moments=_load("moments"),
        model_marker_names=list(meta.get("model_marker_names", [])),
        experimental_marker_names=list(meta.get("experimental_marker_names", [])),
        medial_marker_names=set(meta.get("medial_marker_names", [])),
        body_names=body_names,
        skeleton_segments=skeleton,
        moment_names=list(meta.get("moment_names", [])),
        frame_rate_hz=float(meta.get("frame_rate_hz", 100.0)),
        mass_kg=float(meta.get("mass_kg", 75.0)),
    )


# ---------------------------------------------------------------------------
# 3D 场景画布（QPainter 正交投影）
# ---------------------------------------------------------------------------
class Scene3DCanvas(QWidget):
    """自绘 3D 画布：旋转/平移/缩放、骨架、双色 marker、COP/GRF 箭头、地面与轴。"""

    marker_selected = Signal(str, str)   # (marker name, 详情文本)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(560, 420)

        self._data: ViewerData | None = None
        self._frame = 0

        self._azimuth = 55.0
        self._elevation = 20.0
        self._target = np.zeros(3)
        self._zoom_mm = 1200.0

        self._show_model_markers = True
        self._show_exp_markers = True
        self._show_skeleton = True
        self._show_grf = True
        self._show_error_lines = False
        self._force_scale = 0.6          # N -> mm（箭头长度）

        self._dragging = False
        self._panning = False
        self._last_pos = QPoint()
        self._hover_name: str | None = None

    # -- 数据 -----------------------------------------------------------
    def set_data(self, data: ViewerData | None) -> None:
        self._data = data
        self._frame = 0
        if data is not None:
            self._auto_fit()
        self.update()

    def _auto_fit(self) -> None:
        if self._data is None:
            return
        pts = np.concatenate(
            [self._data.model_markers.reshape(-1, 3),
             self._data.experimental_markers.reshape(-1, 3)], axis=0
        )
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.size == 0:
            return
        self._target = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - self._target, axis=1)
        self._zoom_mm = float(max(np.percentile(dist, 95) * 1.4, 300.0))

    # -- 视角 -----------------------------------------------------------
    def set_view(self, name: str) -> None:
        if name == "前":
            self._azimuth, self._elevation = 90.0, 0.0
        elif name == "侧":
            self._azimuth, self._elevation = 0.0, 0.0
        elif name == "后":
            self._azimuth, self._elevation = -90.0, 0.0
        else:  # 复位
            self._azimuth, self._elevation = 55.0, 20.0
            self._auto_fit()
        self.update()

    # -- 图层开关 -------------------------------------------------------
    def set_layer(self, key: str, on: bool) -> None:
        setattr(self, key, on)
        self.update()

    def set_force_scale(self, scale: float) -> None:
        self._force_scale = scale
        self.update()

    def set_frame(self, idx: int) -> None:
        if self._data is None:
            return
        idx = int(np.clip(idx, 0, self._data.n_frames - 1))
        if idx != self._frame:
            self._frame = idx
            self.update()

    # -- 投影 -----------------------------------------------------------
    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        direction = np.array([
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        ])
        eye = self._target + direction  # 正交投影下 eye 只用于方向
        forward = -direction
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return eye, right, up, forward

    def _project(self, p: np.ndarray) -> tuple[float, float, float]:
        eye, right, up, forward = self._basis()
        d = p - eye
        cx = float(np.dot(d, right))
        cy = float(np.dot(d, up))
        cz = float(np.dot(d, forward))
        scale = min(self.width(), self.height()) / (2.0 * self._zoom_mm)
        sx = self.width() / 2.0 + cx * scale
        sy = self.height() / 2.0 - cy * scale
        return sx, sy, cz

    def _draw_line3d(self, painter: QPainter, a: np.ndarray, b: np.ndarray,
                     pen: QPen) -> None:
        x0, y0, _ = self._project(a)
        x1, y1, _ = self._project(b)
        painter.setPen(pen)
        painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    # -- 事件 -----------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
        elif event.button() == Qt.MouseButton.RightButton:
            self._panning = True
        self._last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        if self._dragging:
            self._azimuth += dx * 0.35
            self._elevation = float(np.clip(self._elevation + dy * 0.35, -89.0, 89.0))
            self.update()
        elif self._panning:
            _, right, up, _ = self._basis()
            scale = min(self.width(), self.height()) / (2.0 * self._zoom_mm)
            delta = (-right * dx + up * dy) / max(scale, 1e-9)
            self._target = self._target + delta
            self.update()
        else:
            self._update_hover(pos)
        self._last_pos = pos

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._select_marker(event.position().toPoint())
        elif event.button() == Qt.MouseButton.RightButton:
            self._panning = False

    def wheelEvent(self, event) -> None:
        factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        self._zoom_mm = float(np.clip(self._zoom_mm * factor, 50.0, 20000.0))
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover_name = None
        self.update()

    # -- 拾取 -----------------------------------------------------------
    def _visible_markers(self) -> list[tuple[str, np.ndarray, QColor, bool]]:
        """返回 [(name, xyz_mm, color, is_model)]，只含当前图层可见且有限值的点。"""
        if self._data is None:
            return []
        out: list[tuple[str, np.ndarray, QColor, bool]] = []
        if self._show_model_markers:
            for j, name in enumerate(self._data.model_marker_names):
                p = self._data.model_markers[self._frame, j]
                if np.isfinite(p).all():
                    out.append((name, p, COLOR_MODEL_MARKER, True))
        if self._show_exp_markers:
            for j, name in enumerate(self._data.experimental_marker_names):
                p = self._data.experimental_markers[self._frame, j]
                if np.isfinite(p).all():
                    out.append((name, p, COLOR_EXP_MARKER, False))
        return out

    def _hit_test(self, pos: QPoint) -> str | None:
        best_name = None
        best_d = 1e9
        for name, p, _c, _m in self._visible_markers():
            sx, sy, _ = self._project(p)
            d = math.hypot(sx - pos.x(), sy - pos.y())
            if d < 10.0 and d < best_d:
                best_d = d
                best_name = name
        return best_name

    def _update_hover(self, pos: QPoint) -> None:
        name = self._hit_test(pos)
        if name != self._hover_name:
            self._hover_name = name
            self.update()

    def _select_marker(self, pos: QPoint) -> None:
        name = self._hit_test(pos)
        if name is not None and self._data is not None:
            self.marker_selected.emit(name, self._marker_detail(name))

    def _marker_detail(self, name: str) -> str:
        if self._data is None:
            return name
        lines = [f"{name}"]
        if name in self._data.model_marker_names:
            j = self._data.model_marker_names.index(name)
            p = self._data.model_markers[self._frame, j]
            lines.append(f"模型坐标: ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm")
        if name in self._data.experimental_marker_names:
            j = self._data.experimental_marker_names.index(name)
            p = self._data.experimental_markers[self._frame, j]
            lines.append(f"实验坐标: ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm")
        if (name in self._data.model_marker_names
                and name in self._data.experimental_marker_names):
            m = self._data.model_markers[self._frame, self._data.model_marker_names.index(name)]
            e = self._data.experimental_markers[
                self._frame, self._data.experimental_marker_names.index(name)]
            if np.isfinite(m).all() and np.isfinite(e).all():
                lines.append(f"误差: {np.linalg.norm(m - e):.1f} mm")
        if name in self._data.medial_marker_names:
            lines.append("（内侧模型预测点，动态无实测）")
        return "\n".join(lines)

    # -- 绘制 -----------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), COLOR_BG)

        if self._data is None:
            painter.setPen(QColor("#AAAAAA"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "尚未加载回放数据（解算完成后自动载入）")
            painter.end()
            return

        self._draw_ground(painter)
        if self._show_skeleton:
            self._draw_skeleton(painter)
        if self._show_error_lines:
            self._draw_error_lines(painter)
        if self._show_grf:
            self._draw_grf(painter)
        self._draw_markers(painter)
        self._draw_axes(painter)
        self._draw_hover_overlay(painter)
        painter.end()

    def _draw_ground(self, painter: QPainter) -> None:
        half = self._zoom_mm * 1.6
        step = 200.0
        pen = QPen(COLOR_GROUND, 1)
        painter.setPen(pen)
        # 地面网格在 y=0 平面
        k = int(half // step)
        for i in range(-k, k + 1):
            v = i * step
            self._draw_line3d(painter, np.array([v, 0.0, -half]),
                              np.array([v, 0.0, half]), pen)
            self._draw_line3d(painter, np.array([-half, 0.0, v]),
                              np.array([half, 0.0, v]), pen)

    def _draw_skeleton(self, painter: QPainter) -> None:
        if self._data is None:
            return
        lookup = {n: i for i, n in enumerate(self._data.body_names)}
        pen = QPen(COLOR_SKELETON, 2)
        for a, b in self._data.skeleton_segments:
            if a not in lookup or b not in lookup:
                continue
            pa = self._data.body_origins[self._frame, lookup[a]]
            pb = self._data.body_origins[self._frame, lookup[b]]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                self._draw_line3d(painter, pa, pb, pen)

    def _draw_error_lines(self, painter: QPainter) -> None:
        if self._data is None:
            return
        pen = QPen(COLOR_ERROR_LINE, 1, Qt.PenStyle.DashLine)
        model_idx = {n: i for i, n in enumerate(self._data.model_marker_names)}
        for j, name in enumerate(self._data.experimental_marker_names):
            if name not in model_idx:
                continue
            e = self._data.experimental_markers[self._frame, j]
            m = self._data.model_markers[self._frame, model_idx[name]]
            if np.isfinite(e).all() and np.isfinite(m).all():
                self._draw_line3d(painter, e, m, pen)

    def _draw_grf(self, painter: QPainter) -> None:
        if self._data is None:
            return
        for side, color in ((0, COLOR_RIGHT), (1, COLOR_LEFT)):
            cop = self._data.cop[self._frame, side]
            force = self._data.grf[self._frame, side]
            if not np.isfinite(cop).all() or not np.isfinite(force).all():
                continue
            mag = float(np.linalg.norm(force))
            if mag < 1.0:
                continue
            tip = cop + force / mag * (mag * self._force_scale)
            pen = QPen(color, 3)
            self._draw_line3d(painter, cop, tip, pen)
            # COP 点
            sx, sy, _ = self._project(cop)
            painter.setPen(QPen(color, 1))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(sx, sy), 4.0, 4.0)
            # 箭头头部
            self._draw_arrowhead(painter, cop, tip, color)

    def _draw_arrowhead(self, painter: QPainter, a: np.ndarray, b: np.ndarray,
                        color: QColor) -> None:
        _, right, up, _ = self._basis()
        d = b - a
        n = np.linalg.norm(d)
        if n < 1e-9:
            return
        d = d / n
        # 在垂直于 d 的平面取两个正交向量作为箭头两翼
        ref = up if abs(np.dot(d, up)) < 0.9 else right
        w1 = np.cross(d, ref)
        w1 /= np.linalg.norm(w1)
        w2 = np.cross(d, w1)
        length = min(n * 0.25, 120.0)
        width = length * 0.4
        wing_a = b - d * length + w1 * width
        wing_b = b - d * length - w1 * width
        wing_c = b - d * length + w2 * width
        wing_d = b - d * length - w2 * width
        pen = QPen(color, 2)
        self._draw_line3d(painter, b, wing_a, pen)
        self._draw_line3d(painter, b, wing_b, pen)
        self._draw_line3d(painter, b, wing_c, pen)
        self._draw_line3d(painter, b, wing_d, pen)

    def _draw_markers(self, painter: QPainter) -> None:
        radius = 6.0
        # 先画被选中的点（略大），其余普通点
        for name, p, color, _is_model in self._visible_markers():
            sx, sy, _ = self._project(p)
            painter.setPen(QPen(QColor("#000000"), 1))
            painter.setBrush(color)
            r = radius + 1.0 if name == self._hover_name else radius
            painter.drawEllipse(QPointF(sx, sy), r, r)

    def _draw_axes(self, painter: QPainter) -> None:
        length = 250.0
        o = np.array([0.0, 0.0, 0.0])
        self._draw_line3d(painter, o, np.array([length, 0, 0]), QPen(COLOR_AXIS_X, 2))
        self._draw_line3d(painter, o, np.array([0, length, 0]), QPen(COLOR_AXIS_Y, 2))
        self._draw_line3d(painter, o, np.array([0, 0, length]), QPen(COLOR_AXIS_Z, 2))
        # 轴标签
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        for vec, color, label in ((np.array([length, 0, 0]), COLOR_AXIS_X, "X"),
                                  (np.array([0, length, 0]), COLOR_AXIS_Y, "Y"),
                                  (np.array([0, 0, length]), COLOR_AXIS_Z, "Z")):
            sx, sy, _ = self._project(vec)
            painter.setPen(color)
            painter.drawText(QPointF(sx + 3, sy - 3), label)

    def _draw_hover_overlay(self, painter: QPainter) -> None:
        if self._hover_name is None or self._data is None:
            return
        text = self._marker_detail(self._hover_name)
        painter.setPen(QColor("#FFFFFF"))
        painter.setBrush(QColor(0, 0, 0, 180))
        fm = painter.fontMetrics()
        lines = text.split("\n")
        w = max(fm.horizontalAdvance(ln) for ln in lines) + 16
        h = fm.height() * len(lines) + 12
        rect = QRectF(8, 8, w, h)
        painter.drawRoundedRect(rect, 4, 4)
        painter.drawText(QRectF(16, 12, w, h), Qt.AlignmentFlag.AlignLeft, text)


# ---------------------------------------------------------------------------
# 关节力矩曲线
# ---------------------------------------------------------------------------
class MomentCurvesWidget(pg.GraphicsLayoutWidget):
    """髋/膝/踝 × 左右 力矩曲线，垂直游标与 3D 共用同一帧。"""

    cursor_changed = Signal(float)   # 用户在曲线上拖动游标 -> 时间 s

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setBackground(COLOR_BG)
        self._data: ViewerData | None = None
        self._norm = False
        self._plots: list[pg.PlotItem] = []
        self._curves: list[tuple[pg.PlotDataItem, pg.PlotDataItem]] = []
        self._cursors: list[pg.InfiniteLine] = []
        self._frame = 0
        self._suppress = False

        for i, label in enumerate(_MOMENT_ROW_LABELS):
            plot = self.addPlot(row=i, col=0)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("left", "Nm/kg" if self._norm else "Nm")
            plot.setTitle(label, color="#DDDDDD", size="9pt")
            plot.getAxis("bottom").setStyle(showValues=(i == len(_MOMENT_ROW_LABELS) - 1))
            plot.addLegend(offset=(10, 10))
            self._plots.append(plot)
            r = plot.plot(pen=pg.mkPen(COLOR_RIGHT, width=2), name="右髋")
            l = plot.plot(pen=pg.mkPen(COLOR_LEFT, width=2), name="左髋")
            self._curves.append((r, l))
            cursor = pg.InfiniteLine(angle=90, movable=True,
                                     pen=pg.mkPen("#FFFFFF", width=1))
            cursor.sigPositionChanged.connect(self._on_cursor_moved)
            plot.addItem(cursor)
            self._cursors.append(cursor)

    def set_data(self, data: ViewerData | None) -> None:
        self._data = data
        self._frame = 0
        self._refresh_curves()

    def set_norm(self, norm: bool) -> None:
        self._norm = norm
        for plot in self._plots:
            plot.setLabel("left", "Nm/kg" if norm else "Nm")
        self._refresh_curves()

    def _refresh_curves(self) -> None:
        if self._data is None:
            for r, l in self._curves:
                r.setData([], [])
                l.setData([], [])
            return
        t = self._data.time_s
        div = self._data.mass_kg if self._norm else 1.0
        for row in range(len(self._curves)):
            ri = row * 2
            li = row * 2 + 1
            self._curves[row][0].setData(t, self._data.moments[:, ri] / div)
            self._curves[row][1].setData(t, self._data.moments[:, li] / div)
        self._update_cursors()

    def set_frame(self, idx: int) -> None:
        if self._data is None:
            return
        idx = int(np.clip(idx, 0, self._data.n_frames - 1))
        self._frame = idx
        self._update_cursors()

    def _update_cursors(self) -> None:
        if self._data is None:
            return
        t = float(self._data.time_s[self._frame])
        self._suppress = True
        try:
            for cursor in self._cursors:
                cursor.setValue(t)
        finally:
            self._suppress = False

    def _on_cursor_moved(self) -> None:
        if self._suppress or self._data is None:
            return
        t = float(self._cursors[0].value())
        self.cursor_changed.emit(t)


# ---------------------------------------------------------------------------
# IMU 加速度曲线
# ---------------------------------------------------------------------------
_MAX_IMU_POINTS = 6000


def _downsample(x: np.ndarray, y: np.ndarray, *, max_points: int = _MAX_IMU_POINTS):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size <= max_points:
        return x, y
    stride = int(np.ceil(x.size / max_points))
    return x[::stride], y[::stride]


class IMUCurvesWidget(pg.GraphicsLayoutWidget):
    """右腿 IMU 三轴加速度，垂直游标与 3D / 力矩共用同一时间轴（C3D 时间）。"""

    cursor_changed = Signal(float)   # 用户拖动游标 -> 时间 s

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setBackground(COLOR_BG)
        self._time_s: np.ndarray | None = None
        self._accel: np.ndarray | None = None
        self._suppress = False

        self._plot = self.addPlot(row=0, col=0)
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("left", "加速度", units="m/s²")
        self._plot.setLabel("bottom", "时间", units="s")
        self._plot.setTitle("IMU 加速度（右腿）", color="#DDDDDD", size="9pt")
        self._plot.addLegend(offset=(10, 10))

        self._curves: list[pg.PlotDataItem] = []
        for axis, color in enumerate(("#c0603a", "#4a6fa5", "#7a7a4a")):
            curve = self._plot.plot(pen=pg.mkPen(color, width=1), name=f"acc[{axis}]")
            self._curves.append(curve)

        self._cursor = pg.InfiniteLine(angle=90, movable=True,
                                       pen=pg.mkPen("#FFFFFF", width=1))
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)
        self._plot.addItem(self._cursor)

    def set_data(self, time_s: np.ndarray | None, accel: np.ndarray | None) -> None:
        self._time_s = time_s
        self._accel = accel
        self._refresh()

    def _refresh(self) -> None:
        if self._time_s is None or self._accel is None or self._time_s.size == 0:
            for curve in self._curves:
                curve.setData([], [])
            return
        for axis, curve in enumerate(self._curves):
            x, y = _downsample(self._time_s, self._accel[:, axis])
            curve.setData(x, y)

    def set_time(self, t: float) -> None:
        if self._time_s is None:
            return
        self._suppress = True
        try:
            self._cursor.setValue(float(t))
        finally:
            self._suppress = False

    def _on_cursor_moved(self) -> None:
        if self._suppress or self._time_s is None:
            return
        self.cursor_changed.emit(float(self._cursor.value()))


# ---------------------------------------------------------------------------
# 联合回放页
# ---------------------------------------------------------------------------
class ViewerWidget(QWidget):
    """3D + 曲线 + 统一游标 + 播放控制 + 图层开关 + 导出。"""

    frame_changed = Signal(int, float)   # (frame index, time s)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: ViewerData | None = None
        self._frame = 0
        self._suppress = False
        # 回放以墙钟为基准：记录起播时刻与起播帧，保证 1s 数据 = 1s 墙钟、绝不慢放。
        self._play_start_monotonic: float | None = None
        self._play_start_frame = 0

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._advance)

        self._build_ui()
        self._wire_signals()

    # -- UI -----------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.canvas = Scene3DCanvas()
        self.curves = MomentCurvesWidget()
        self.imu_curves = IMUCurvesWidget()

        top = QHBoxLayout()
        top.addWidget(self.canvas, 1)
        top.addWidget(self._build_controls(), 0)
        root.addLayout(top, 3)

        root.addWidget(self.curves, 2)
        root.addWidget(self.imu_curves, 1)

        root.addLayout(self._build_transport(), 0)

    def _build_controls(self) -> QWidget:
        box = QWidget()
        box.setFixedWidth(240)
        layout = QVBoxLayout(box)

        layout.addWidget(self._section_label("图层"))
        self._chk_model = QCheckBox("模型 marker（19）")
        self._chk_exp = QCheckBox("实验 marker（15）")
        self._chk_skeleton = QCheckBox("骨架")
        self._chk_grf = QCheckBox("左右 COP / GRF")
        self._chk_error = QCheckBox("实验→模型误差连线")
        self._chk_norm = QCheckBox("力矩归一化（Nm/kg）")
        for chk, on in (
            (self._chk_model, True), (self._chk_exp, True), (self._chk_skeleton, True),
            (self._chk_grf, True), (self._chk_error, False), (self._chk_norm, False),
        ):
            chk.setChecked(on)
            layout.addWidget(chk)
        self._chk_model.toggled.connect(lambda on: self.canvas.set_layer("_show_model_markers", on))
        self._chk_exp.toggled.connect(lambda on: self.canvas.set_layer("_show_exp_markers", on))
        self._chk_skeleton.toggled.connect(lambda on: self.canvas.set_layer("_show_skeleton", on))
        self._chk_grf.toggled.connect(lambda on: self.canvas.set_layer("_show_grf", on))
        self._chk_error.toggled.connect(lambda on: self.canvas.set_layer("_show_error_lines", on))
        self._chk_norm.toggled.connect(self.curves.set_norm)

        layout.addWidget(self._section_label("GRF 箭头缩放"))
        self._force_slider = QSlider(Qt.Orientation.Horizontal)
        self._force_slider.setRange(1, 20)
        self._force_slider.setValue(6)
        self._force_slider.valueChanged.connect(self._on_force_scale)
        layout.addWidget(self._force_slider)

        layout.addWidget(self._section_label("视角"))
        view_row = QHBoxLayout()
        for name in ("复位", "前", "侧", "后"):
            btn = QPushButton(name)
            btn.clicked.connect(lambda _=False, n=name: self.canvas.set_view(n))
            view_row.addWidget(btn)
        layout.addLayout(view_row)

        layout.addWidget(self._section_label("Marker 信息"))
        self._marker_info = QLabel("点击 3D 图中的 marker 查看详情")
        self._marker_info.setWordWrap(True)
        self._marker_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._marker_info)

        layout.addWidget(self._section_label("导出"))
        export_row = QHBoxLayout()
        png_btn = QPushButton("PNG")
        png_btn.clicked.connect(self._export_png)
        csv_btn = QPushButton("CSV")
        csv_btn.clicked.connect(self._export_csv)
        export_row.addWidget(png_btn)
        export_row.addWidget(csv_btn)
        layout.addLayout(export_row)

        layout.addStretch(1)
        return box

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("section_label")
        return lbl

    def _build_transport(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._btn_prev = QPushButton("|◀")
        self._btn_play = QPushButton("▶")
        self._btn_next = QPushButton("▶|")
        self._btn_prev.clicked.connect(self._step_back)
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_next.clicked.connect(self._step_forward)
        row.addWidget(self._btn_prev)
        row.addWidget(self._btn_play)
        row.addWidget(self._btn_next)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.valueChanged.connect(self._on_slider)
        row.addWidget(self._slider, 1)

        self._time_label = QLabel("00.000 s")
        self._time_label.setMinimumWidth(90)
        row.addWidget(self._time_label)
        return row

    # -- 数据 -----------------------------------------------------------
    def load(self, viewer_dir: Path | None) -> None:
        """加载（或清空）回放数据。``viewer_dir`` 为 None 时显示空态。"""
        if viewer_dir is None:
            self._data = None
            self.canvas.set_data(None)
            self.curves.set_data(None)
            self.imu_curves.set_data(None, None)
            self._set_enabled(False)
            return
        self._data = load_viewer_data(viewer_dir)
        self.canvas.set_data(self._data)
        self.curves.set_data(self._data)
        self.imu_curves.set_data(None, None)
        self._slider.setRange(0, self._data.n_frames - 1)
        self._set_enabled(True)
        self.set_frame(0)

    def _set_enabled(self, on: bool) -> None:
        for w in (self._btn_prev, self._btn_play, self._btn_next, self._slider):
            w.setEnabled(on)

    def set_imu(self, time_s: np.ndarray | None, accel: np.ndarray | None) -> None:
        """加载（或清空）右腿 IMU 加速度曲线（时间轴已映射到 C3D 时间）。"""
        self.imu_curves.set_data(time_s, accel)

    # -- 播放状态 -------------------------------------------------------
    @property
    def has_data(self) -> bool:
        return self._data is not None

    def set_frame(self, idx: int) -> None:
        if self._data is None:
            return
        idx = int(np.clip(idx, 0, self._data.n_frames - 1))
        self._frame = idx
        self._suppress = True
        try:
            self._slider.setValue(idx)
        finally:
            self._suppress = False
        self.canvas.set_frame(idx)
        self.curves.set_frame(idx)
        t = float(self._data.time_s[idx])
        self.imu_curves.set_time(t)
        self._time_label.setText(f"{t:6.3f} s")
        self.frame_changed.emit(idx, t)

    def _advance(self) -> None:
        if self._data is None or self._play_start_monotonic is None:
            return
        # 目标帧由墙钟流逝时间直接算出：即使定时器抖动或单帧渲染慢，也只会丢帧、
        # 绝不慢放——流逝多少秒就前进多少秒的数据。
        elapsed = time.monotonic() - self._play_start_monotonic
        nxt = self._play_start_frame + int(round(elapsed * self._data.frame_rate_hz))
        nxt %= self._data.n_frames
        self.set_frame(nxt)

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._btn_play.setText("▶")
            self._play_start_monotonic = None
        else:
            if self._data is None:
                return
            self._play_start_monotonic = time.monotonic()
            self._play_start_frame = self._frame
            interval = int(1000.0 / self._data.frame_rate_hz)
            self._timer.start(max(interval, 10))
            self._btn_play.setText("⏸")

    def _step_back(self) -> None:
        self.set_frame(self._frame - 1)

    def _step_forward(self) -> None:
        self.set_frame(self._frame + 1)

    def _on_slider(self, value: int) -> None:
        if not self._suppress:
            self.set_frame(value)

    def _on_force_scale(self, value: int) -> None:
        self.canvas.set_force_scale(value / 10.0)

    # -- 信号接线 -------------------------------------------------------
    def _wire_signals(self) -> None:
        self.curves.cursor_changed.connect(self._on_cursor_time)
        self.imu_curves.cursor_changed.connect(self._on_cursor_time)
        self.canvas.marker_selected.connect(self._on_marker_selected)

    def _on_cursor_time(self, t: float) -> None:
        if self._data is None:
            return
        idx = int(np.argmin(np.abs(self._data.time_s - t)))
        self.set_frame(idx)

    def _on_marker_selected(self, name: str, detail: str) -> None:
        self._marker_info.setText(detail)

    # -- 导出 -----------------------------------------------------------
    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出 3D 快照", "viewer_frame.png",
                                              "PNG 图像 (*.png)")
        if not path:
            return
        pixmap = QPixmap(self.canvas.size())
        self.canvas.render(pixmap)
        pixmap.save(path)

    def _export_csv(self) -> None:
        if self._data is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出关节力矩 CSV", "moments.csv",
                                              "CSV 文件 (*.csv)")
        if not path:
            return
        header = ["time_s", *self._data.moment_names]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for i, t in enumerate(self._data.time_s):
                writer.writerow([f"{t:.6f}"] + [f"{v:.6f}" for v in self._data.moments[i]])
