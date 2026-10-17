"""解算页：参数、OpenSim 子环境、进度日志、取消与 QC 状态。

主界面进程不 import ``opensim``；OpenSim 子环境只在这里被**选择 / 校验**，实际
解算由 ``workers.OpenSimProcessWorker`` 启动子进程完成。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from exo_collection.apps.calculate.models import ProcessingConfig
from exo_collection.apps.calculate.opensim_env import (
    discover_opensim_python,
    pick_default_opensim_python,
    validate_opensim_python,
)
from exo_collection.configuration import SharedAppSettings

_log = logging.getLogger(__name__)


class ProcessingView(QWidget):
    """参数 + OpenSim 子环境 + 解算进度。"""

    process_requested = Signal(object)   # ProcessingConfig
    cancel_requested = Signal()

    def __init__(self, settings: SharedAppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._running = False
        self._solve_allowed = False
        self._condition_code = ""
        self._analysis_last_time_s: float | None = None

        layout = QVBoxLayout(self)

        # OpenSim 子环境
        env_group = QGroupBox("OpenSim 子环境（Scale/IK/ID）")
        env_form = QFormLayout(env_group)
        self._opensim_path = QLineEdit()
        self._opensim_path.setReadOnly(True)
        self._opensim_path.setPlaceholderText("未选择")
        browse_row = QHBoxLayout()
        self._browse_button = QPushButton("浏览…")
        self._browse_button.clicked.connect(self._browse_opensim)
        self._discover_button = QPushButton("自动发现")
        self._discover_button.clicked.connect(self._auto_discover)
        self._validate_button = QPushButton("校验")
        self._validate_button.clicked.connect(self._validate)
        for button in (self._browse_button, self._discover_button, self._validate_button):
            browse_row.addWidget(button)
        browse_row.addStretch(1)
        env_form.addRow("python.exe：", self._opensim_path)
        env_form.addRow("", _wrap(browse_row))
        self._env_status = QLabel("")
        self._env_status.setWordWrap(True)
        env_form.addRow("", self._env_status)
        layout.addWidget(env_group)

        # 处理参数
        param_group = QGroupBox("解算参数")
        param_form = QFormLayout(param_group)
        self._mass = QDoubleSpinBox()
        self._mass.setRange(20.0, 300.0)
        self._mass.setValue(75.0)
        self._mass.setSuffix(" kg")
        self._mass.setDecimals(1)
        self._height = QDoubleSpinBox()
        self._height.setRange(1.0, 2.5)
        self._height.setValue(1.75)
        self._height.setSuffix(" m")
        self._height.setDecimals(2)
        self._marker_cutoff = QDoubleSpinBox()
        self._marker_cutoff.setRange(1.0, 30.0)
        self._marker_cutoff.setValue(6.0)
        self._marker_cutoff.setSuffix(" Hz")
        self._marker_cutoff.setDecimals(1)
        self._grf_cutoff = QDoubleSpinBox()
        self._grf_cutoff.setRange(1.0, 60.0)
        self._grf_cutoff.setValue(6.0)
        self._grf_cutoff.setSuffix(" Hz")
        self._grf_cutoff.setDecimals(1)
        self._grf_cutoff.setToolTip("默认 6 Hz，与 marker/逆动力学运动学低通匹配；匹配截止频率不保证力矩准确，仍需检查 QC。")
        param_form.addRow("体重：", self._mass)
        param_form.addRow("身高：", self._height)
        param_form.addRow("marker 低通：", self._marker_cutoff)
        param_form.addRow("GRF 低通：", self._grf_cutoff)
        layout.addWidget(param_group)

        # 区间（可选）：稳态分析区间（§3.5）+ 静态稳定窗口（§3.6），默认自动。
        range_group = QGroupBox("分析区间与静态标定窗口")
        range_form = QFormLayout(range_group)
        self._analysis_auto = QCheckBox("自动检测稳态分析区间（推荐）")
        self._analysis_auto.setChecked(True)
        self._analysis_start = QDoubleSpinBox()
        self._analysis_end = QDoubleSpinBox()
        for spin in (self._analysis_start, self._analysis_end):
            spin.setRange(0.0, 600.0)
            spin.setSuffix(" s")
            spin.setDecimals(2)
            spin.setEnabled(False)
        self._analysis_auto.toggled.connect(
            lambda checked: self._set_range_enabled(self._analysis_start, self._analysis_end, not checked)
        )
        range_form.addRow("", self._analysis_auto)
        range_form.addRow("分析区间（起/止）：", _span(self._analysis_start, self._analysis_end))
        self._analysis_hint = QLabel()
        self._analysis_hint.setWordWrap(True)
        range_form.addRow("", self._analysis_hint)

        self._static_auto = QCheckBox("自动选择静态稳定窗口（推荐）")
        self._static_auto.setChecked(True)
        self._static_start = QDoubleSpinBox()
        self._static_end = QDoubleSpinBox()
        for spin in (self._static_start, self._static_end):
            spin.setRange(0.0, 600.0)
            spin.setSuffix(" s")
            spin.setDecimals(2)
            spin.setEnabled(False)
        self._static_auto.toggled.connect(
            lambda checked: self._set_range_enabled(self._static_start, self._static_end, not checked)
        )
        range_form.addRow("", self._static_auto)
        range_form.addRow("静态窗口（起/止）：", _span(self._static_start, self._static_end))
        layout.addWidget(range_group)

        # 运行控制
        run_row = QHBoxLayout()
        self._process_button = QPushButton("开始解算")
        self._process_button.setProperty("buttonRole", "primary")
        self._process_button.clicked.connect(self._request_process)
        self._cancel_button = QPushButton("取消")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self.cancel_requested.emit)
        run_row.addWidget(self._process_button)
        run_row.addWidget(self._cancel_button)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        # 进度日志
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        layout.addWidget(self._log, 1)

        self._load_persisted_env()
        for signal in (self._analysis_auto.toggled, self._analysis_start.valueChanged, self._analysis_end.valueChanged):
            signal.connect(self._refresh_analysis_state)
        self._refresh_analysis_state()
        self._refresh_process_button()

    # ------------------------------------------------------------------
    # OpenSim 子环境
    # ------------------------------------------------------------------
    def _load_persisted_env(self) -> None:
        path = self._settings.opensim_python_executable
        if path is not None and path.is_file():
            self._opensim_path.setText(str(path))
            self._env_status.setText(f"已载入：{path}")
        else:
            self._env_status.setText("未配置 OpenSim 子环境。点击「自动发现」或「浏览」。")

    def _auto_discover(self) -> None:
        self._env_status.setText("正在自动发现 OpenSim 环境…")
        self._env_status.repaint()
        discovered = discover_opensim_python()
        if not discovered:
            self._env_status.setText("未找到可 import opensim 的 Python 环境。")
            return
        chosen = pick_default_opensim_python(discovered)
        self._set_env(chosen.executable, f"发现 {len(discovered)} 个环境，默认：{chosen.executable}")

    def _browse_opensim(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择 OpenSim 环境的 python.exe", "", "python.exe (python.exe)"
        )
        if not filename:
            return
        info = validate_opensim_python(filename)
        if info is None:
            self._env_status.setText("该解释器无法 import opensim，请重新选择。")
            return
        self._set_env(Path(filename), f"校验通过：OpenSim {info.version}")

    def _validate(self) -> None:
        text = self._opensim_path.text().strip()
        if not text:
            self._env_status.setText("请先选择或发现 OpenSim 环境。")
            return
        info = validate_opensim_python(text)
        if info is None:
            self._env_status.setText("校验失败：该解释器无法 import opensim。")
            return
        self._set_env(Path(text), f"校验通过：OpenSim {info.version}")

    def _set_env(self, path: Path, message: str) -> None:
        self._opensim_path.setText(str(path))
        self._env_status.setText(message)
        self._settings.set_opensim_python_executable(path)

    def opensim_python(self) -> Path | None:
        text = self._opensim_path.text().strip()
        return Path(text) if text else None

    # ------------------------------------------------------------------
    # 参数 / 运行
    # ------------------------------------------------------------------
    @staticmethod
    def _set_range_enabled(start: QDoubleSpinBox, end: QDoubleSpinBox, enabled: bool) -> None:
        start.setEnabled(enabled)
        end.setEnabled(enabled)

    def apply_patient_info(self, info: dict) -> None:
        """用测力台头部读到的受试者信息预填体重/身高（用户仍可手动覆盖）。"""
        weight = info.get("weight_kg")
        if isinstance(weight, (int, float)):
            self._mass.setValue(float(weight))
        height = info.get("height_m")
        if isinstance(height, (int, float)):
            self._height.setValue(float(height))

    @staticmethod
    def requires_manual_analysis(condition_code: str) -> bool:
        return "STAND_BWALK_STAND" in condition_code.upper()

    def set_condition(self, condition_code: str) -> None:
        """Apply the current trial's policy without carrying another trial's window."""
        self._condition_code = condition_code
        self._analysis_last_time_s = None
        required = self.requires_manual_analysis(condition_code)
        self._analysis_auto.setEnabled(not required)
        self._analysis_auto.setText("自动稳态截取不适用于站立→倒走→站立" if required else "自动检测稳态分析区间（推荐）")
        self._analysis_auto.setChecked(not required)
        self._analysis_start.setValue(0.0)
        self._analysis_end.setValue(0.0)
        self._refresh_analysis_state()

    def set_analysis_last_time(self, last_time_s: float) -> None:
        self._analysis_last_time_s = float(last_time_s)
        self._refresh_analysis_state()

    @staticmethod
    def analysis_window_error(window, *, required=False, last_time_s=None) -> str | None:
        if window is None:
            return "此瞬态工况必须手动填写分析区间，不能自动截取稳态。" if required else None
        try:
            start, end = window
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            return "分析区间必须包含起始和结束两个秒数。"
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            return "请填写有效分析区间：0 ≤ 起始 < 结束（动态 C3D 时间，秒）。"
        if last_time_s is not None and end > last_time_s + 1e-6:
            return f"分析区间超出当前动态 C3D 末帧 {last_time_s:.2f} s，请缩小结束时间。"
        return None

    def _current_analysis_range(self):
        return None if self._analysis_auto.isChecked() else (self._analysis_start.value(), self._analysis_end.value())

    def _refresh_analysis_state(self) -> None:
        required = self.requires_manual_analysis(self._condition_code)
        if required:
            text = ("此工况包含起停，不能自动截取稳态。请只填写上方「分析区间（起/止）」。"
                    "时间以当前动态 C3D 第0帧为0 s，不是TXT原始时间或同步offset；"
                    "下方「静态窗口」用于另一条静态标定记录，可保持自动。"
                    "请选择身体朝向固定且左右分力有效的片段。手选全段不能使左右分力缺失的静止期成为有效力矩真值。")
        else:
            text = "分析起止使用当前动态 C3D 时间（第0帧为0 s）；静态窗口属于独立的静态标定记录。"
        if self._analysis_last_time_s is not None:
            text += f" 当前动态 C3D 范围：0–{self._analysis_last_time_s:.2f} s。"
        error = self.analysis_window_error(self._current_analysis_range(), required=required,
                                           last_time_s=self._analysis_last_time_s)
        if error:
            text += " " + error
        self._analysis_hint.setText(text)
        self._refresh_process_button()

    def _request_process(self) -> None:
        error = self.analysis_window_error(self._current_analysis_range(),
            required=self.requires_manual_analysis(self._condition_code), last_time_s=self._analysis_last_time_s)
        if error:
            self.append_log("无法开始解算：" + error)
            return
        analysis_range = None
        if not self._analysis_auto.isChecked():
            analysis_range = (
                float(self._analysis_start.value()),
                float(self._analysis_end.value()),
            )
        static_range = None
        if not self._static_auto.isChecked():
            static_range = (
                float(self._static_start.value()),
                float(self._static_end.value()),
            )
        config = ProcessingConfig(
            mass_kg=float(self._mass.value()),
            height_m=float(self._height.value()),
            marker_cutoff_hz=float(self._marker_cutoff.value()),
            grf_cutoff_hz=float(self._grf_cutoff.value()),
            analysis_time_range_s=analysis_range,
            static_time_range_s=static_range,
        )
        self.process_requested.emit(config)

    def append_log(self, line: str) -> None:
        self._log.appendPlainText(line)
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log.setTextCursor(cursor)

    def set_running(self, running: bool) -> None:
        self._running = running
        self._cancel_button.setEnabled(running)
        self._refresh_process_button()

    def set_solve_enabled(self, enabled: bool) -> None:
        """门禁「开始解算」：仅当同步已确认（§3.2）时才允许点按。"""
        self._solve_allowed = enabled
        self._refresh_process_button()

    def _refresh_process_button(self) -> None:
        valid_window = self.analysis_window_error(self._current_analysis_range(),
            required=self.requires_manual_analysis(self._condition_code), last_time_s=self._analysis_last_time_s) is None
        self._process_button.setEnabled(self._solve_allowed and not self._running and valid_window)

    def clear_log(self) -> None:
        self._log.clear()


def _wrap(layout: QHBoxLayout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget


def _span(start: QDoubleSpinBox, end: QDoubleSpinBox) -> QWidget:
    widget = QWidget()
    row = QHBoxLayout(widget)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(start)
    row.addWidget(end)
    row.addStretch(1)
    return widget


__all__ = ["ProcessingView"]
