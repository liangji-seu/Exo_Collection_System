"""Data Studio「全局预览」：完整时间轴的真值全貌（髋力矩 + IMU 姿态）。

与全屏回放（滑动 10s 窗口 + 播放游标）不同，这里把 ``ground_truth.csv`` 的整条
时间轴一次性画出来，让用户快速核对髋关节力矩真值波形与 IMU pitch 的对应关系。
纯 NumPy + pyqtgraph，不依赖 OpenSim。
"""

from __future__ import annotations

import csv
import json
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
_COLOR_SPEED = "#9467BD"
_COLOR_SPEED_TARGET = "#7F7F7F"


@dataclass(frozen=True, slots=True)
class TruthPreview:
    """ground_truth.csv 中「全局预览」所需列的提取结果。"""

    time_s: np.ndarray
    moment_channels: tuple[str, ...]   # 存在的髋力矩通道名（hip_flexion_r/l）
    moment_values: np.ndarray          # (n, len(moment_channels))
    imu_channels: tuple[str, ...]      # 存在的 IMU 姿态通道名（imu_roll/pitch/yaw）
    imu_values: np.ndarray             # (n, len(imu_channels))


@dataclass(frozen=True, slots=True)
class SpeedTrace:
    """Gaitway 跑台速度，时间轴已换算到 C3D（``ground_truth.csv`` 的）时间。"""

    time_s_c3d: np.ndarray        # 采样时间（C3D 时间轴）
    speed: np.ndarray             # 实际速度（m/s）
    speed_target: np.ndarray      # 目标速度（m/s）；导出无此列时为空数组


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


def _parse_speed_txt(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """从单个 Gaitway 跑台导出解析 ``(t_gaitway, speed, speed_target)``。

    列名行以 ``Time (s)`` 开头、含 ``Speed (m/s)`` 列的文件即跑台导出；逐行取
    ``Time (s)`` / ``Speed (m/s)`` / ``Speed target (m/s)``（目标列缺失则返回空
    数组），坏行跳过。非跑台文件 / 列缺失 / 样本不足返回 ``None``。
    """
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return None
    try:
        records = list(csv.reader(lines, delimiter="\t"))
    except csv.Error:
        return None
    header_index = next(
        (
            i
            for i, row in enumerate(records)
            if row and row[0].strip().casefold() == "time (s)"
        ),
        None,
    )
    if header_index is None:
        return None
    names = [str(name).strip().casefold() for name in records[header_index]]
    if "time (s)" not in names or "speed (m/s)" not in names:
        return None
    t_idx = names.index("time (s)")
    s_idx = names.index("speed (m/s)")
    st_idx = names.index("speed target (m/s)") if "speed target (m/s)" in names else None

    times: list[float] = []
    speeds: list[float] = []
    targets: list[float] = []
    for row in records[header_index + 1 :]:
        try:
            times.append(float(row[t_idx]))
            speeds.append(float(row[s_idx]))
            if st_idx is not None:
                targets.append(float(row[st_idx]))
        except (ValueError, IndexError):
            continue
    if len(times) < 2:
        return None
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(speeds, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _find_gaitway_offset_s(session_dir: Path) -> float:
    """读同步偏移（秒），用于把 gaitway 时间换算到 C3D 时间。

    优先 ``derived/opensim/*/manifest.json`` → ``sync.gaitway_offset_s``，回退
    ``derived/opensim/sync_calibration.json`` → ``result.gaitway_offset_s``，都无则 0。
    """
    opensim_dir = Path(session_dir) / "derived" / "opensim"
    if opensim_dir.is_dir():
        for run_manifest in sorted(opensim_dir.glob("*/manifest.json")):
            data = _load_json(run_manifest)
            if data is None:
                continue
            sync = data.get("sync") if isinstance(data, dict) else None
            offset = sync.get("gaitway_offset_s") if isinstance(sync, dict) else None
            if isinstance(offset, (int, float)) and np.isfinite(offset):
                return float(offset)
    calib = _load_json(opensim_dir / "sync_calibration.json")
    if calib is not None:
        result = calib.get("result") if isinstance(calib, dict) else None
        offset = result.get("gaitway_offset_s") if isinstance(result, dict) else None
        if isinstance(offset, (int, float)) and np.isfinite(offset):
            return float(offset)
    return 0.0


def read_gaitway_speed(
    session_dir: Path, *, offset_s: float | None = None
) -> SpeedTrace | None:
    """从 session 目录提取跑台速度，时间轴对齐到 C3D（``ground_truth.csv``）时间。

    遍历 ``*.txt`` 找第一个跑台导出，按约定 ``t_c3d = t_gaitway - offset_s`` 换算。
    ``offset_s`` 为 ``None`` 时用 :func:`_find_gaitway_offset_s` 自动求；找不到有效
    速度数据返回 ``None``。
    """
    for txt in sorted(Path(session_dir).glob("*.txt")):
        parsed = _parse_speed_txt(txt)
        if parsed is None:
            continue
        t_gaitway, speed, speed_target = parsed
        shift = (
            _find_gaitway_offset_s(session_dir)
            if offset_s is None
            else float(offset_s)
        )
        return SpeedTrace(
            time_s_c3d=t_gaitway - shift,
            speed=speed,
            speed_target=speed_target,
        )
    return None


_MOMENT_LABELS = {"hip_flexion_r": "右髋", "hip_flexion_l": "左髋"}
_MOMENT_COLORS = {"hip_flexion_r": _COLOR_RIGHT, "hip_flexion_l": _COLOR_LEFT}
_IMU_LABELS = {"imu_roll": "roll", "imu_pitch": "pitch", "imu_yaw": "yaw"}
_IMU_COLORS = {"imu_roll": _COLOR_ROLL, "imu_pitch": _COLOR_PITCH, "imu_yaw": _COLOR_YAW}


class GlobalPreviewWindow(QMainWindow):
    """静态全时间轴真值预览：上为髋力矩（左右可勾选），下为 IMU 姿态角。"""

    def __init__(
        self,
        preview: TruthPreview,
        title: str,
        parent: QWidget | None = None,
        *,
        speed: SpeedTrace | None = None,
    ) -> None:
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

        # ── 中：跑台速度（x 轴与力矩图联动，便于圈选速度区间对应力矩）────
        self._speed_plot = self._graphics.addPlot(row=1, col=0)
        self._speed_plot.getViewBox().setXLink(self._moment_plot.getViewBox())
        self._speed_plot.setTitle("跑台速度", color="#000000", size="10pt")
        self._speed_plot.setLabel("left", "速度", units="m/s")
        self._speed_plot.getAxis("bottom").setStyle(showValues=False)
        self._speed_plot.showGrid(x=True, y=True, alpha=0.25)
        if speed is not None and speed.speed.size:
            self._speed_plot.addLegend(offset=(10, 10))
            self._speed_plot.plot(
                preview.time_s,
                np.interp(
                    preview.time_s,
                    speed.time_s_c3d,
                    speed.speed,
                    left=np.nan,
                    right=np.nan,
                ),
                pen=pg.mkPen(_COLOR_SPEED, width=2),
                name="实际速度",
            )
            if speed.speed_target.size == speed.speed.size:
                self._speed_plot.plot(
                    preview.time_s,
                    np.interp(
                        preview.time_s,
                        speed.time_s_c3d,
                        speed.speed_target,
                        left=np.nan,
                        right=np.nan,
                    ),
                    pen=pg.mkPen(
                        _COLOR_SPEED_TARGET, width=1, style=Qt.PenStyle.DashLine
                    ),
                    name="目标速度",
                )

        # ── 下：IMU 姿态角（x 轴与力矩图联动缩放/平移）────────────────────
        self._imu_plot = self._graphics.addPlot(row=2, col=0)
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


__all__ = [
    "GlobalPreviewWindow",
    "SpeedTrace",
    "TruthPreview",
    "read_gaitway_speed",
    "read_truth_preview",
]
