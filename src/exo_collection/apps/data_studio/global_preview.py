"""Data Studio「全局预览」：完整时间轴的真值全貌（髋力矩 + IMU 姿态）。

与全屏回放（滑动 10s 窗口 + 播放游标）不同，这里把 ``ground_truth.csv`` 的整条
时间轴一次性画出来，让用户快速核对髋关节力矩真值波形与 IMU pitch 的对应关系。
纯 NumPy + pyqtgraph，不依赖 OpenSim。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

# 与 run_calculate 回放一致：右=橙、左=绿；pitch 用蓝。
_COLOR_RIGHT = "#F28E2B"
_COLOR_LEFT = "#59A14F"
_COLOR_PITCH = "#1F77B4"
_COLOR_ROLL = "#2CA02C"
_COLOR_YAW = "#D62728"


@dataclass(frozen=True, slots=True)
class TruthPreview:
    """ground_truth.csv 中「全局预览」所需列的提取结果。"""

    time_s: np.ndarray
    moment_channels: tuple[str, ...]   # 存在的髋力矩通道名（hip_flexion_r/l）
    moment_values: np.ndarray          # (n, len(moment_channels))
    imu_channels: tuple[str, ...]      # 存在的 IMU 姿态通道名（imu_roll/pitch/yaw）
    imu_values: np.ndarray             # (n, len(imu_channels))


def read_truth_preview(path: Path) -> TruthPreview | None:
    """从 ``ground_truth.csv`` 提取髋力矩与 IMU 姿态列；文件缺失/损坏返回 ``None``。

    表头约定 ``time_s`` + 12 个 ``imu_*`` 特征 + 6 个关节力矩列。这里按列名精确
    匹配需要的列，缺失某列时该组为空数组，不会让整个预览失败。
    """
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return None
    rows = [line for line in text.splitlines() if line.strip()]
    if len(rows) < 2:
        return None
    try:
        records = list(csv.reader(rows))
    except csv.Error:
        return None
    header = records[0]
    if not header or header[0].strip().casefold() not in {"time_s", "time", "t"}:
        return None
    names = [str(name).strip().casefold() for name in header[1:]]
    try:
        matrix = np.asarray(
            [[float(cell) for cell in row] for row in records[1:]], dtype=np.float64
        )
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return None

    time_s = np.asarray(matrix[:, 0], dtype=np.float64)
    values = matrix[:, 1:]

    def _pick(needle: str) -> int | None:
        for index, name in enumerate(names):
            if name == needle:
                return index
        return None

    moment_channels: list[str] = []
    moment_indices: list[int] = []
    for channel in ("hip_flexion_r", "hip_flexion_l"):
        index = _pick(channel)
        if index is not None:
            moment_channels.append(channel)
            moment_indices.append(index)

    imu_channels: list[str] = []
    imu_indices: list[int] = []
    for channel in ("imu_roll", "imu_pitch", "imu_yaw"):
        index = _pick(channel)
        if index is not None:
            imu_channels.append(channel)
            imu_indices.append(index)

    if not moment_channels and not imu_channels:
        return None

    return TruthPreview(
        time_s=time_s,
        moment_channels=tuple(moment_channels),
        moment_values=(
            np.asarray(values[:, moment_indices], dtype=np.float64)
            if moment_indices
            else np.empty((time_s.size, 0), dtype=np.float64)
        ),
        imu_channels=tuple(imu_channels),
        imu_values=(
            np.asarray(values[:, imu_indices], dtype=np.float64)
            if imu_indices
            else np.empty((time_s.size, 0), dtype=np.float64)
        ),
    )


_MOMENT_LABELS = {"hip_flexion_r": "右髋", "hip_flexion_l": "左髋"}
_MOMENT_COLORS = {"hip_flexion_r": _COLOR_RIGHT, "hip_flexion_l": _COLOR_LEFT}
_IMU_LABELS = {"imu_roll": "roll", "imu_pitch": "pitch", "imu_yaw": "yaw"}
_IMU_COLORS = {"imu_roll": _COLOR_ROLL, "imu_pitch": _COLOR_PITCH, "imu_yaw": _COLOR_YAW}


class GlobalPreviewWindow(QMainWindow):
    """静态全时间轴真值预览：上为髋力矩（左右可勾选），下为 IMU 姿态角。"""

    def __init__(self, preview: TruthPreview, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"全局预览 · {title}")
        self.resize(1200, 720)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self._graphics, 1)

        # ── 上：髋关节力矩真值 ────────────────────────────────────────────
        self._moment_plot = self._graphics.addPlot(row=0, col=0)
        self._moment_plot.setTitle("髋关节力矩真值", color="#000000", size="10pt")
        self._moment_plot.setLabel("left", "力矩", units="N·m")
        self._moment_plot.getAxis("bottom").setStyle(showValues=False)
        self._moment_plot.showGrid(x=True, y=True, alpha=0.25)
        if preview.moment_channels:
            self._moment_plot.addLegend(offset=(10, 10))
        self._moment_curves: dict[str, pg.PlotDataItem] = {}
        for index, channel in enumerate(preview.moment_channels):
            label = _MOMENT_LABELS.get(channel, channel)
            curve = self._moment_plot.plot(
                pen=pg.mkPen(_MOMENT_COLORS.get(channel, "#000000"), width=2),
                name=label,
            )
            curve.setData(preview.time_s, preview.moment_values[:, index])
            self._moment_curves[label] = curve

        # ── 下：IMU 姿态角（x 轴与力矩图联动缩放/平移）────────────────────
        self._imu_plot = self._graphics.addPlot(row=1, col=0)
        self._imu_plot.getViewBox().setXLink(self._moment_plot.getViewBox())
        self._imu_plot.setTitle("IMU 姿态角（右腿）", color="#000000", size="10pt")
        self._imu_plot.setLabel("left", "角度", units="deg")
        self._imu_plot.setLabel("bottom", "时间", units="s")
        self._imu_plot.showGrid(x=True, y=True, alpha=0.25)
        if preview.imu_channels:
            self._imu_plot.addLegend(offset=(10, 10))
        for index, channel in enumerate(preview.imu_channels):
            label = _IMU_LABELS.get(channel, channel)
            curve = self._imu_plot.plot(
                pen=pg.mkPen(_IMU_COLORS.get(channel, "#000000"), width=2),
                name=label,
            )
            curve.setData(preview.time_s, preview.imu_values[:, index])

        # ── 控制行：左右髋显示开关 ────────────────────────────────────────
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("髋力矩显示："))
        for label, curve in self._moment_curves.items():
            box = QCheckBox(label)
            box.setChecked(True)
            box.toggled.connect(lambda checked, c=curve: c.setVisible(checked))
            controls.addWidget(box)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.setCentralWidget(central)


__all__ = ["GlobalPreviewWindow", "TruthPreview", "read_truth_preview"]
