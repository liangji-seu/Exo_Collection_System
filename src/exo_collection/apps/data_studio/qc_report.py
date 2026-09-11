"""Data Studio「全局预览」QC 报告：OpenSim 解算结果的步态周期级质控。

从最新一次完整 OpenSim 解算目录（``derived/opensim/run_*``）读取 ``result.json`` +
``viewer/*.npy`` + IK/ID ``.mot``，用纯 NumPy 计算 heel strike、归一化步态周期
（0–100% mean±std）、GRF 同步、IK/ID 残差 QC，并渲染成可导出 PNG 的报告窗口。
分析部分不 import opensim / 不依赖 Qt，可离线单测。

数据来源约定（与 :mod:`opensim_overlay` 一致，已从真实数据核实）：
- ``viewer/grf.npy`` 形状 ``(n, 2, 3)``，``cop_order = [right, left]``，分量 ``[x, y, z]``，
  其中 **y 分量是竖直方向**（着地时 ≈ 体重 N），故右脚 Fz = ``grf[:, 0, 1]``。
- ``viewer/moments.npy`` 形状 ``(n, 6)``，列顺序 ``hip_flexion_r/l, knee_angle_r/l,
  ankle_angle_r/l``（力矩，N·m）。
- 关节角度在 IK ``.mot``（``result.json["files"]["ik"]``），单位为度。
- pelvis 残差力矩在 ID ``.mot`` 的 ``pelvis_*_moment`` 列（现算 RMS/p95）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .global_preview import read_gaitway_speed, read_truth_preview
from .opensim_overlay import _parse_mot, find_latest_run_dir

# 与 run_calculate 回放一致：右=橙、左=绿。
_COLOR_RIGHT = "#F28E2B"
_COLOR_LEFT = "#59A14F"
_COLOR_PITCH = "#1F77B4"
_COLOR_ROLL = "#2CA02C"
_COLOR_YAW = "#D62728"
_COLOR_SPEED = "#9467BD"
_COLOR_SPEED_TARGET = "#7F7F7F"
# QC 状态徽标配色（与 data_studio 树列一致）。
_QC_COLORS = {"PASS": "#20a35a", "FAIL": "#b42318", "WARN": "#b26a00"}

# viewer/moments.npy 的 6 列顺序（export_viewer._SAGITTAL_MOMENTS）。
_MOMENT_NAMES = (
    "hip_flexion_r",
    "hip_flexion_l",
    "knee_angle_r",
    "knee_angle_l",
    "ankle_angle_r",
    "ankle_angle_l",
)
# IK .mot 里抽取的矢状面角度（度），顺序与 _MOMENT_NAMES 对齐。
_ANGLE_NAMES = _MOMENT_NAMES
# ID .mot 里 pelvis 残差力矩列（gait2392）。
_PELVIS_MOMENT_NAMES = ("pelvis_tilt_moment", "pelvis_list_moment", "pelvis_rotation_moment")

_GRF_VERTICAL_AXIS = 1  # grf.npy 竖直分量
_RIGHT_FOOT = 0
_LEFT_FOOT = 1

_MOMENT_LABELS = {
    "hip_flexion_r": "右髋",
    "hip_flexion_l": "左髋",
    "knee_angle_r": "右膝",
    "knee_angle_l": "左膝",
    "ankle_angle_r": "右踝",
    "ankle_angle_l": "左踝",
}
_MOMENT_COLORS = {"hip_flexion_r": _COLOR_RIGHT, "hip_flexion_l": _COLOR_LEFT}
_ANGLE_COLORS = {
    "hip_flexion_r": _COLOR_RIGHT,
    "knee_angle_r": _COLOR_PITCH,
    "ankle_angle_r": _COLOR_ROLL,
}


@dataclass(frozen=True, slots=True)
class GaitQCData:
    """一个 session 的 OpenSim 解算 QC 数据（全部落在同一条时间轴上）。"""

    time_s: np.ndarray
    frame_rate_hz: float
    grf_fz_r: np.ndarray
    grf_fz_l: np.ndarray
    moments: np.ndarray                 # (n, 6) N·m
    moment_names: tuple[str, ...] = field(default=_MOMENT_NAMES)
    angles: np.ndarray | None = None    # (n, 6) 度，IK .mot 缺失时为 None
    angle_names: tuple[str, ...] = field(default=_ANGLE_NAMES)
    marker_qc_overall: dict | None = None
    marker_qc_per_marker: dict | None = None
    id_residual_force: dict | None = None       # {rms_N, p95_N}
    id_residual_moment: dict | None = None      # {rms_Nm, p95_Nm}（从 id.mot 现算）
    marker_cutoff_hz: float | None = None
    grf_cutoff_hz: float | None = None
    qc_status: str | None = None
    qc_summary: str | None = None
    imu_time_s: np.ndarray | None = None
    imu_pitch: np.ndarray | None = None
    imu_roll: np.ndarray | None = None
    imu_yaw: np.ndarray | None = None
    run_dir: Path | None = None
    n_frames: int = 0
    speed: np.ndarray | None = None          # 跑台实际速度（m/s，已插值到 time_s）
    speed_target: np.ndarray | None = None   # 跑台目标速度（m/s，已插值到 time_s）


def _read_angles(ik_path: Path, time_s: np.ndarray) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """从 IK ``.mot`` 抽 6 个矢状面角度并插值到 ``time_s`` 网格；失败返回 ``(None, ())``。"""
    try:
        column_names, data = _parse_mot(ik_path)
    except (OSError, ValueError):
        return None, ()
    indices: list[int] = []
    names: list[str] = []
    for name in _ANGLE_NAMES:
        if name in column_names:
            indices.append(column_names.index(name))
            names.append(name)
    if not indices:
        return None, ()
    ik_time = data[:, 0]
    out = np.empty((time_s.size, len(indices)), dtype=np.float64)
    for column, index in enumerate(indices):
        out[:, column] = np.interp(time_s, ik_time, data[:, index])
    return out, tuple(names)


def _read_residual_moment(id_path: Path) -> dict | None:
    """从 ID ``.mot`` 的 pelvis 残差力矩列算 L2 范数的 RMS/p95；失败返回 ``None``。"""
    try:
        column_names, data = _parse_mot(id_path)
    except (OSError, ValueError):
        return None
    indices = [column_names.index(n) for n in _PELVIS_MOMENT_NAMES if n in column_names]
    if not indices:
        return None
    norm = np.linalg.norm(data[:, indices], axis=1)
    norm = norm[np.isfinite(norm)]
    if norm.size == 0:
        return None
    return {
        "rms_Nm": float(np.sqrt(np.mean(norm**2))),
        "p95_Nm": float(np.percentile(norm, 95)),
    }


def load_gait_qc(session_dir: Path) -> GaitQCData | None:
    """从 session 最新完整解算目录读取 QC 数据；缺关键文件/畸形返回 ``None``。"""
    session_dir = Path(session_dir)
    run_dir = find_latest_run_dir(session_dir)
    if run_dir is None:
        return None
    try:
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    viewer_dir = Path(result.get("files", {}).get("viewer_dir") or (run_dir / "viewer"))
    time_path = viewer_dir / "time_s.npy"
    grf_path = viewer_dir / "grf.npy"
    moments_path = viewer_dir / "moments.npy"
    if not (time_path.is_file() and grf_path.is_file() and moments_path.is_file()):
        return None
    try:
        time_s = np.load(time_path).astype(np.float64)
        grf = np.load(grf_path)
        moments = np.load(moments_path)
    except (OSError, ValueError):
        return None
    if grf.ndim != 3 or grf.shape[1] < 2 or moments.ndim != 2 or moments.shape[1] < 6:
        return None
    n = time_s.size
    if n == 0 or moments.shape[0] < n:
        return None

    ik_path = Path(result.get("files", {}).get("ik") or (run_dir / "hh19_static_calibrated_ik.mot"))
    angles, angle_names = _read_angles(ik_path, time_s)

    id_path = Path(result.get("files", {}).get("id") or (run_dir / "hh19_static_calibrated_id.mot"))
    residual_moment = _read_residual_moment(id_path)

    processing = result.get("processing") or {}
    marker_qc = result.get("marker_qc") or {}
    id_qc = result.get("id_qc") or {}
    qc = result.get("qc") or {}

    truth = read_truth_preview(session_dir / "ground_truth.csv")
    imu_time_s = imu_pitch = imu_roll = imu_yaw = None
    if truth is not None and truth.time_s.size >= 2:
        # IMU 采样网格与 viewer 时间轴起点/间隔一致、仅帧数差 1–2，统一插值到
        # viewer 时间轴，保证 IMU 曲线与 GRF/力矩在横轴上严格对齐。
        imu_time_s = time_s
        mono = bool(np.all(np.diff(truth.time_s) > 0))
        for channel in truth.imu_channels:
            value = truth.imu_values[:, truth.imu_channels.index(channel)]
            if mono:
                value = np.interp(time_s, truth.time_s, value)
            if channel == "imu_pitch":
                imu_pitch = value
            elif channel == "imu_roll":
                imu_roll = value
            elif channel == "imu_yaw":
                imu_yaw = value

    # 跑台速度（Gaitway 导出），同样插值到 viewer 时间轴，保证与力矩/GRF 严格对齐。
    speed = speed_target = None
    speed_trace = read_gaitway_speed(session_dir)
    if speed_trace is not None and speed_trace.speed.size >= 2:
        if bool(np.all(np.diff(speed_trace.time_s_c3d) > 0)):
            speed = np.interp(
                time_s, speed_trace.time_s_c3d, speed_trace.speed,
                left=np.nan, right=np.nan,
            )
            if speed_trace.speed_target.size == speed_trace.speed.size:
                speed_target = np.interp(
                    time_s, speed_trace.time_s_c3d, speed_trace.speed_target,
                    left=np.nan, right=np.nan,
                )

    frame_rate = float((result.get("viewer") or {}).get("frame_rate_hz") or 100.0)

    return GaitQCData(
        time_s=time_s,
        frame_rate_hz=frame_rate,
        grf_fz_r=np.asarray(grf[:, _RIGHT_FOOT, _GRF_VERTICAL_AXIS], dtype=np.float64),
        grf_fz_l=np.asarray(grf[:, _LEFT_FOOT, _GRF_VERTICAL_AXIS], dtype=np.float64),
        moments=np.asarray(moments[:n, :6], dtype=np.float64),
        angles=angles,
        angle_names=angle_names,
        marker_qc_overall=marker_qc.get("overall") or None,
        marker_qc_per_marker=marker_qc.get("markers") or None,
        id_residual_force=id_qc.get("residual_force") or None,
        id_residual_moment=residual_moment,
        marker_cutoff_hz=processing.get("marker_cutoff_hz"),
        grf_cutoff_hz=processing.get("grf_cutoff_hz"),
        qc_status=qc.get("status"),
        qc_summary=qc.get("summary"),
        imu_time_s=imu_time_s,
        imu_pitch=imu_pitch,
        imu_roll=imu_roll,
        imu_yaw=imu_yaw,
        run_dir=run_dir,
        n_frames=n,
        speed=speed,
        speed_target=speed_target,
    )


def detect_heel_strikes(
    fz: np.ndarray,
    *,
    threshold: float | None = None,
    min_gap_s: float = 0.5,
    frame_rate: float = 100.0,
) -> np.ndarray:
    """检测 heel strike（初始着地）：Fz 上升沿越过阈值的帧索引。

    注意：步态里的 heel strike 是 **Fz 从 ≈0 起跳** 的初始着地事件，不是 Fz 峰值
    （峰值出现在支撑中期）。返回按时间升序的帧索引数组。
    """
    fz = np.asarray(fz, dtype=np.float64)
    if fz.size == 0:
        return np.empty(0, dtype=np.int64)
    if threshold is None:
        finite = fz[np.isfinite(fz)]
        peak = float(np.max(finite)) if finite.size else 0.0
        threshold = max(0.03 * peak, 20.0)
    above = fz > threshold
    edges = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    min_gap_frames = max(1, int(round(min_gap_s * frame_rate)))
    heel: list[int] = []
    for edge in edges.tolist():
        if not heel or edge - heel[-1] >= min_gap_frames:
            heel.append(edge)
    return np.asarray(heel, dtype=np.int64)


def normalize_gait_cycles(
    time_s: np.ndarray,
    signal: np.ndarray,
    heel_idx: np.ndarray,
    n_points: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把每个步态周期 ``[HS_i, HS_{i+1}]`` 重采样到 0–100%，返回 ``(x, mean, std)``。

    周期不足 1 个（heel strike < 2）时 mean/std 为空数组。
    """
    x = np.linspace(0.0, 100.0, n_points)
    time_s = np.asarray(time_s, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    if heel_idx.size < 2:
        return x, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    cycles: list[np.ndarray] = []
    for start, end in zip(heel_idx[:-1], heel_idx[1:]):
        if end <= start:
            continue
        t_cycle = time_s[start : end + 1]
        s_cycle = signal[start : end + 1]
        span = t_cycle[-1] - t_cycle[0]
        if span <= 0:
            continue
        t_norm = (t_cycle - t_cycle[0]) / span * 100.0
        cycles.append(np.interp(x, t_norm, s_cycle))
    if not cycles:
        return x, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    stacked = np.asarray(cycles, dtype=np.float64)
    return x, np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def representative_cycle(
    time_s: np.ndarray,
    signal: np.ndarray,
    heel_idx: np.ndarray,
    n_points: int = 101,
) -> tuple[np.ndarray, np.ndarray] | None:
    """取时长最接近中位数的一个周期，重采样到 0–100%；周期不足返回 ``None``。"""
    if heel_idx.size < 2:
        return None
    x = np.linspace(0.0, 100.0, n_points)
    durations = np.asarray(time_s, dtype=np.float64)[heel_idx[1:]] - np.asarray(
        time_s, dtype=np.float64
    )[heel_idx[:-1]]
    best = int(np.argmin(np.abs(durations - np.median(durations))))
    start, end = int(heel_idx[best]), int(heel_idx[best + 1])
    t_cycle = np.asarray(time_s, dtype=np.float64)[start : end + 1]
    s_cycle = np.asarray(signal, dtype=np.float64)[start : end + 1]
    span = t_cycle[-1] - t_cycle[0]
    if span <= 0:
        return None
    return x, np.interp(x, (t_cycle - t_cycle[0]) / span * 100.0, s_cycle)


def _fmt(value: float | None, digits: int = 2, unit: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{unit}"


class GaitQCReportWindow(QMainWindow):
    """静态全时间轴 + 归一化步态周期 + IK/ID QC 的报告窗口，支持导出 PNG。"""

    def __init__(self, data: GaitQCData, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data = data
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"全局预览 QC · {title}")
        self.resize(1240, 900)

        self._heel_idx = detect_heel_strikes(data.grf_fz_r, frame_rate=data.frame_rate_hz)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        root.addLayout(self._build_toolbar())

        self._graphics = pg.GraphicsLayoutWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._graphics)
        root.addWidget(scroll, 1)

        self._build_plots()
        root.addWidget(self._build_qc_text())

        # 让滚动区按内容高度撑开（每行约 240px）。
        self._graphics.setMinimumHeight(240 * self._plot_row_count + 40)

        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        title = QLabel(self._header_html())
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setWordWrap(True)
        bar.addWidget(title, 1)
        self._hip_checkboxes: dict[str, QCheckBox] = {}
        for label, channel in (("右髋", "hip_flexion_r"), ("左髋", "hip_flexion_l")):
            box = QCheckBox(label)
            box.setChecked(True)
            self._hip_checkboxes[channel] = box
            bar.addWidget(box)
        export_btn = QPushButton("导出 PNG")
        export_btn.clicked.connect(self._export_png)
        bar.addWidget(export_btn)
        return bar

    def _header_html(self) -> str:
        data = self._data
        status = data.qc_status or "未知"
        color = _QC_COLORS.get(status, "#b26a00")
        n_cycles = max(0, self._heel_idx.size - 1)
        parts = [
            f"QC 状态：<b style=\"color:{color}\">{status}</b>",
            f"滤波 marker {_fmt(data.marker_cutoff_hz, 1)} Hz · GRF {_fmt(data.grf_cutoff_hz, 1)} Hz",
            f"帧 {data.n_frames} @ {data.frame_rate_hz:.0f} Hz · 步态周期 {n_cycles}",
        ]
        if data.run_dir is not None:
            parts.append(f"run：{data.run_dir.name}")
        return "　".join(parts)

    def _build_plots(self) -> None:
        data = self._data
        row = 0

        # S1 GRF 同步图（右 Fz + 右髋力矩，x 联动）。
        fz_plot = self._graphics.addPlot(row=row, col=0)
        row += 1
        fz_plot.setTitle("GRF 同步 · 右 Fz", color="#000000", size="10pt")
        fz_plot.setLabel("left", "Fz", units="N")
        fz_plot.getAxis("bottom").setStyle(showValues=False)
        fz_plot.showGrid(x=True, y=True, alpha=0.25)
        fz_plot.plot(data.time_s, data.grf_fz_r, pen=pg.mkPen(_COLOR_RIGHT, width=2), name="右 Fz")

        moment_plot = self._graphics.addPlot(row=row, col=0)
        row += 1
        moment_plot.getViewBox().setXLink(fz_plot.getViewBox())
        moment_plot.setTitle("右髋力矩", color="#000000", size="10pt")
        moment_plot.setLabel("left", "力矩", units="N·m")
        moment_plot.setLabel("bottom", "时间", units="s")
        moment_plot.showGrid(x=True, y=True, alpha=0.25)
        moment_plot.plot(
            data.time_s, data.moments[:, 0], pen=pg.mkPen(_COLOR_RIGHT, width=2), name="右髋力矩"
        )
        self._mark_heel_strikes(fz_plot)
        self._mark_heel_strikes(moment_plot)

        # S2 跑台速度（实际 vs 目标，x 联动；便于圈选速度区间对应力矩真值）。
        if data.speed is not None:
            speed_plot = self._graphics.addPlot(row=row, col=0)
            row += 1
            speed_plot.getViewBox().setXLink(fz_plot.getViewBox())
            speed_plot.setTitle("跑台速度 · 实际(紫) 目标(灰)", color="#000000", size="10pt")
            speed_plot.setLabel("left", "速度", units="m/s")
            speed_plot.getAxis("bottom").setStyle(showValues=False)
            speed_plot.showGrid(x=True, y=True, alpha=0.25)
            speed_plot.addLegend(offset=(10, 10))
            speed_plot.plot(
                data.time_s, data.speed, pen=pg.mkPen(_COLOR_SPEED, width=2), name="实际速度"
            )
            if data.speed_target is not None:
                speed_plot.plot(
                    data.time_s,
                    data.speed_target,
                    pen=pg.mkPen(_COLOR_SPEED_TARGET, width=1, style=Qt.PenStyle.DashLine),
                    name="目标速度",
                )

        # S3 髋力矩真值（全时间轴，左右可勾选）。
        hip_plot = self._graphics.addPlot(row=row, col=0)
        row += 1
        hip_plot.getViewBox().setXLink(fz_plot.getViewBox())
        hip_plot.setTitle("髋关节力矩真值 · 右髋(橙) 左髋(绿)", color="#000000", size="10pt")
        hip_plot.setLabel("left", "力矩", units="N·m")
        hip_plot.getAxis("bottom").setStyle(showValues=False)
        hip_plot.showGrid(x=True, y=True, alpha=0.25)
        self._hip_curves: dict[str, pg.PlotDataItem] = {}
        for index, channel in enumerate(("hip_flexion_r", "hip_flexion_l")):
            label = _MOMENT_LABELS[channel]
            curve = hip_plot.plot(
                pen=pg.mkPen(_MOMENT_COLORS[channel], width=2), name=label
            )
            curve.setData(data.time_s, data.moments[:, index])
            self._hip_curves[channel] = curve
        self._mark_heel_strikes(hip_plot)
        for channel, box in self._hip_checkboxes.items():
            box.toggled.connect(
                lambda checked, c=self._hip_curves[channel]: c.setVisible(checked)
            )

        # S4 IMU pitch。
        imu_plot = self._graphics.addPlot(row=row, col=0)
        row += 1
        imu_plot.getViewBox().setXLink(fz_plot.getViewBox())
        imu_plot.setTitle("IMU 姿态角（右腿）· pitch(蓝) roll(绿) yaw(红)", color="#000000", size="10pt")
        imu_plot.setLabel("left", "角度", units="deg")
        imu_plot.getAxis("bottom").setStyle(showValues=False)
        imu_plot.showGrid(x=True, y=True, alpha=0.25)
        if data.imu_time_s is not None:
            if data.imu_roll is not None:
                imu_plot.plot(data.imu_time_s, data.imu_roll, pen=pg.mkPen(_COLOR_ROLL, width=1), name="roll")
            if data.imu_pitch is not None:
                imu_plot.plot(data.imu_time_s, data.imu_pitch, pen=pg.mkPen(_COLOR_PITCH, width=2), name="pitch")
            if data.imu_yaw is not None:
                imu_plot.plot(data.imu_time_s, data.imu_yaw, pen=pg.mkPen(_COLOR_YAW, width=1), name="yaw")

        # S5a 归一化力矩 mean±std（hip_r / hip_l）。
        if self._heel_idx.size >= 2:
            mom_plot = self._graphics.addPlot(row=row, col=0)
            row += 1
            mom_plot.setTitle("归一化步态周期 · 力矩 mean±std · 右髋(橙) 左髋(绿)", color="#000000", size="10pt")
            mom_plot.setLabel("left", "力矩", units="N·m")
            mom_plot.setLabel("bottom", "步态周期", units="%")
            mom_plot.showGrid(x=True, y=True, alpha=0.25)
            for index, channel in enumerate(("hip_flexion_r", "hip_flexion_l")):
                x, mean, std = normalize_gait_cycles(data.time_s, data.moments[:, index], self._heel_idx)
                self._add_mean_std(mom_plot, x, mean, std, _MOMENT_COLORS[channel], _MOMENT_LABELS[channel])

        # S5b 归一化角度 mean±std（hip_r / knee_r / ankle_r）。
        if self._heel_idx.size >= 2 and data.angles is not None:
            ang_plot = self._graphics.addPlot(row=row, col=0)
            row += 1
            ang_plot.setTitle("归一化步态周期 · 角度 mean±std · 右髋(橙) 右膝(蓝) 右踝(绿)", color="#000000", size="10pt")
            ang_plot.setLabel("left", "角度", units="deg")
            ang_plot.setLabel("bottom", "步态周期", units="%")
            ang_plot.showGrid(x=True, y=True, alpha=0.25)
            for index, channel in enumerate(("hip_flexion_r", "knee_angle_r", "ankle_angle_r")):
                if channel not in data.angle_names:
                    continue
                col = data.angle_names.index(channel)
                x, mean, std = normalize_gait_cycles(data.time_s, data.angles[:, col], self._heel_idx)
                self._add_mean_std(ang_plot, x, mean, std, _ANGLE_COLORS[channel], _MOMENT_LABELS[channel])

        # S6 左右髋对比（代表性周期，右 vs 左）。
        if self._heel_idx.size >= 2:
            lr_plot = self._graphics.addPlot(row=row, col=0)
            row += 1
            lr_plot.setTitle("左右髋对比 · 右(橙) 左(绿)", color="#000000", size="10pt")
            lr_plot.setLabel("left", "力矩", units="N·m")
            lr_plot.setLabel("bottom", "步态周期", units="%")
            lr_plot.showGrid(x=True, y=True, alpha=0.25)
            for index, channel in enumerate(("hip_flexion_r", "hip_flexion_l")):
                rep = representative_cycle(data.time_s, data.moments[:, index], self._heel_idx)
                if rep is None:
                    continue
                x, values = rep
                lr_plot.plot(
                    x, values, pen=pg.mkPen(_MOMENT_COLORS[channel], width=2), name=_MOMENT_LABELS[channel]
                )

        self._plot_row_count = row

    def _mark_heel_strikes(self, plot: pg.PlotItem) -> None:
        pen = pg.mkPen("#999999", width=1, style=Qt.PenStyle.DashLine)
        for index in self._heel_idx:
            line = pg.InfiniteLine(pos=self._data.time_s[int(index)], angle=90, pen=pen)
            plot.addItem(line, ignoreBounds=True)

    def _add_mean_std(
        self,
        plot: pg.PlotItem,
        x: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        color: str,
        label: str,
    ) -> None:
        if mean.size == 0:
            return
        upper = mean + std
        lower = mean - std
        lower_curve = plot.plot(pen=pg.mkPen(None))
        upper_curve = plot.plot(pen=pg.mkPen(None))
        lower_curve.setData(x, lower)
        upper_curve.setData(x, upper)
        fill_color = QColor(color)
        fill_color.setAlpha(60)
        plot.addItem(pg.FillBetweenItem(lower_curve, upper_curve, brush=pg.mkBrush(fill_color)))
        plot.plot(x, mean, pen=pg.mkPen(color, width=2), name=label)

    def _build_qc_text(self) -> QWidget:
        data = self._data
        label = QLabel(self._qc_html())
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        return label

    def _qc_html(self) -> str:
        data = self._data
        overall = data.marker_qc_overall or {}
        lines = ["<b>IK QC（动态）</b>"]
        lines.append(
            "静态标记误差：未采集（需重新解算才可获得）"
        )
        lines.append(
            "动态标记误差：RMS 均值 "
            f"{_fmt(overall.get('rms_mean_cm'))} cm · RMS p95 {_fmt(overall.get('rms_p95_cm'))} cm · "
            f"Max p95 {_fmt(overall.get('max_marker_p95_cm'))} cm · Max {_fmt(overall.get('max_marker_max_cm'))} cm"
        )
        worst = self._worst_markers()
        if worst:
            lines.append("最差标记（按 max_cm）：")
            lines.extend(f"　{name}：{_fmt(max_cm)} cm" for name, max_cm in worst)

        lines.append("")
        lines.append("<b>ID QC</b>")
        force = data.id_residual_force or {}
        moment = data.id_residual_moment or {}
        lines.append(
            "pelvis 残差力：RMS " + _fmt(force.get("rms_N"), 1) + " N · p95 "
            + _fmt(force.get("p95_N"), 1) + " N"
        )
        lines.append(
            "pelvis 残差力矩：RMS " + _fmt(moment.get("rms_Nm"), 2) + " N·m · p95 "
            + _fmt(moment.get("p95_Nm"), 2) + " N·m"
        )
        return "<br>".join(lines)

    def _worst_markers(self, top: int = 8) -> list[tuple[str, float]]:
        markers = self._data.marker_qc_per_marker or {}
        ranked = sorted(
            ((name, float(stat.get("max_cm", 0.0))) for name, stat in markers.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top]

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出 PNG", "qc_report.png", "PNG 图片 (*.png)")
        if not path:
            return
        # 抓取前先让图形场景完成布局（legend 等 scene item 若未初始化就 grab 会段错误，
        # 且无法被 except 捕获）。直接抓取整个 GraphicsLayoutWidget 位图，widget 已按
        # 内容撑到完整高度，grab() 会截取全部 section。
        QApplication.processEvents()
        self._graphics.grab().save(path, "PNG")
        QMessageBox.information(self, "导出成功", f"已导出：\n{path}")


__all__ = [
    "GaitQCData",
    "GaitQCReportWindow",
    "detect_heel_strikes",
    "load_gait_qc",
    "normalize_gait_cycles",
    "representative_cycle",
]
