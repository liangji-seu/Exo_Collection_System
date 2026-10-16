"""Operator controls for the IMU torque pipeline; construction never opens devices."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)
from exo_collection.domain.motor_protocol import TORQUE_MAX_NM
from exo_collection.apps.model_runtime.serial_ports import scan_motor_ports


class MotorPortComboBox(QComboBox):
    scan_requested = Signal()

    def showPopup(self):
        self.scan_requested.emit()
        super().showPopup()

    def wheelEvent(self, event):
        event.ignore()


class KeyboardNumberEdit(QDoubleSpinBox):
    """Validated direct numeric entry without wheel or button/key stepping."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event):
        event.ignore()  # Let the surrounding controls scroll without editing.

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down,
                           Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            event.accept()
            return
        super().keyPressEvent(event)

    def stepBy(self, steps):
        return

    def stepEnabled(self):
        return QAbstractSpinBox.StepEnabledFlag.StepNone


def _number(low: float, high: float, value: float, suffix: str, decimals: int = 2):
    widget = KeyboardNumberEdit()
    widget.setRange(low, high)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setSuffix(suffix)
    widget.setKeyboardTracking(False)
    return widget


class AssistanceControls(QWidget):
    apply_requested = Signal()
    preflight_requested = Signal()
    motor_requested = Signal()
    stop_requested = Signal()
    changed = Signal()
    parameter_committed = Signal(str, float)
    ports_scanned = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.mass = _number(0, 300, 0, " kg")
        self.mass.setSpecialValueText("请填写被试体重")
        self.mass.setToolTip("仅 Nm/kg → Nm 助力换算需要正体重；只看模型估计或 Nm 模型可留空。")
        self.gain = _number(0, 1, 1, "", 3)
        self.cutoff = _number(0, 9, 3, " Hz")
        self.cutoff.setSpecialValueText("关闭低通")
        self.delay = _number(0, 2000, 0, " ms", 0)
        self.limit = _number(0, TORQUE_MAX_NM, 0, " Nm")
        self.limit.setSpecialValueText("0 Nm · 禁止辅助")
        self.limit.setToolTip(f"固件协议范围 0–{TORQUE_MAX_NM:g} Nm；正负方向均按该阈值限幅，0 表示禁止辅助。")
        for label, widget in (
            ("体重", self.mass), ("力矩阈值", self.limit),
            ("助力比例 0–1", self.gain), ("延时", self.delay),
            ("低通截止频率", self.cutoff),
        ):
            form.addRow(label, widget)
        self.cutoff.setToolTip("辅助力矩因果低通滤波器的截止频率（Hz）；0 表示关闭。")
        layout.addLayout(form)
        self.preflight_label = QLabel("连接 IMU → 加载模型和填写参数 → 运行预测")
        self.preflight_label.setWordWrap(True)
        layout.addWidget(self.preflight_label)

        # Kept as a separate group so the main window can place the physical
        # motor connection controls beside the other devices.  The fields keep
        # their historical names for settings/tests, but the operator flow no
        # longer depends on the hidden mode/enable compatibility controls.
        self.motor_group = QGroupBox("输出电机")
        motor_layout = QVBoxLayout(self.motor_group)
        motor_form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Mock · 不连接电机", "mock")
        self.mode.addItem("真实电机 · ExoCode", "serial")
        self.mode.hide()
        port_row = QHBoxLayout()
        self.port = MotorPortComboBox()
        self.port.setEditable(True)
        # Scan-result placeholder text can be much wider than a COM port.
        # Keep that editable field within the existing control-column width.
        self.port.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.port.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.port.setPlaceholderText("选择电机 COM 口")
        self.port.scan_requested.connect(self.refresh_ports)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh_ports)
        port_row.addWidget(self.port, 1)
        port_row.addWidget(refresh)
        motor_form.addRow("电机串口", port_row)
        motor_layout.addLayout(motor_form)
        self.motor_button = QPushButton("连接电机（保持失能）")
        self.motor_button.clicked.connect(self.motor_requested)
        motor_layout.addWidget(self.motor_button)
        self.motor_status = QLabel("电机未连接；当前仅显示模型估计")
        self.motor_status.setWordWrap(True)
        motor_layout.addWidget(self.motor_status)
        layout.addWidget(self.motor_group)
        self.enable = QCheckBox("启动辅助输出")
        self.enable.hide()
        self.command_label = QLabel("发送指令：左 — Nm ｜右 — Nm")
        self.command_label.setWordWrap(True)
        layout.addWidget(self.command_label)
        self.log_label = QLabel("运行日志：尚未开始")
        self.log_label.setWordWrap(True)
        self.log_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.log_label.hide()

        # A wrapped QLabel still reports an unbroken path/error as its minimum
        # width. Let the existing column determine width and wrap text vertically.
        for label in (self.preflight_label, self.motor_status, self.command_label, self.log_label):
            policy = label.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
            label.setSizePolicy(policy)

        self.parameter_fields = {"mass_kg": self.mass, "torque_limit_nm": self.limit,
                                 "gain": self.gain, "delay_ms": self.delay,
                                 "lowpass_hz": self.cutoff}
        self._protected_parameters = set()
        for name, widget in self.parameter_fields.items():
            widget.valueChanged.connect(lambda _value, key=name: self._parameter_changed(key))
            widget.editingFinished.connect(lambda key=name: self._parameter_committed(key))
        self._right_sign = 1
        self._known_ports = set()
        self.last_port_scan = {}
        self.refresh_ports()

    def _parameter_changed(self, name):
        self._protected_parameters.add(name)
        self.changed.emit()

    def _parameter_committed(self, name):
        self._protected_parameters.add(name)
        self.parameter_committed.emit(name, self.parameter_fields[name].value())

    def restore_parameters(self, values):
        for name, value in values.items():
            widget = self.parameter_fields[name]
            widget.blockSignals(True)
            try:
                widget.setValue(value)
                self._protected_parameters.add(name)
            finally:
                widget.blockSignals(False)

    def load_spec(self, spec):
        """Use model defaults only for fields without a saved or edited value."""
        widgets = (self.mass, self.gain, self.cutoff, self.delay, self.limit)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            config = spec.assistance
            defaults = {"mass_kg": config.mass_kg, "gain": config.gain,
                        "lowpass_hz": config.lowpass_hz, "delay_ms": config.delay_s * 1000,
                        "torque_limit_nm": spec.output.torque_limit_nm}
            for name, value in defaults.items():
                if name not in self._protected_parameters:
                    self.parameter_fields[name].setValue(value)
            self._right_sign = config.right_sign
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def refresh_ports(self):
        current = self.port.currentText().strip()
        report = scan_motor_ports()
        report["previous"] = current
        report["previous_removed"] = False
        if report["error"]:
            # An enumeration error is not evidence that a selected/connected
            # device vanished. Preserve the previous list and show the error.
            report.update(selected=current, selection_reason="scan_error")
            message = report["error"]
        else:
            ports = {item["device"] for item in report["ports"]}
            report["previous_removed"] = bool(current in self._known_ports and current not in ports)
            manual = bool(current and current not in self._known_ports and current not in ports)
            if current in ports:
                selected, reason = current, "preserved"
            elif manual:
                selected, reason = current, "manual_unverified"
            elif len(report["candidates"]) == 1:
                selected, reason = report["candidates"][0], "unique_teensy"
            else:
                selected = ""
                reason = ("removed_previous" if current else "ambiguous" if report["candidates"]
                          else "no_candidate" if ports else "empty")
            self.port.blockSignals(True)
            try:
                self.port.clear()
                for item in report["ports"]:
                    self.port.addItem(item["device"])
                    index = self.port.count() - 1
                    kind = "Teensy 电机候选" if item["is_teensy"] else "未确认为电机"
                    self.port.setItemData(index, f"{kind} · {item['description']}\n{item['hwid']}", Qt.ItemDataRole.ToolTipRole)
                self.port.setCurrentIndex(-1)
                self.port.setCurrentText(selected)
            finally:
                self.port.blockSignals(False)
            self._known_ports = ports
            report.update(selected=selected, selection_reason=reason)
            if reason == "manual_unverified":
                message = f"手动端口 {selected} 未在本次枚举中发现；扫描未连接设备。"
            elif selected:
                message = f"已发现 {len(ports)} 个串口；已选择 {selected}，尚未连接。"
                if report["previous_removed"]:
                    message = f"原端口 {current} 已拔出；已选择唯一 Teensy 候选 {selected}，尚未连接。"
            elif current:
                message = f"原端口 {current} 已拔出；请选择电机串口或手动输入。"
            elif len(report["candidates"]) > 1:
                message = "发现多个 Teensy 电机候选；请选择对应串口。"
            elif ports:
                message = "未发现 Teensy 电机候选；可查看端口描述并手动输入。"
            else:
                message = "未发现可用串口；请插入电机控制器后刷新，或手动输入。"
        self.port.setPlaceholderText(message if not report["selected"] else "选择电机 COM 口")
        self.port.lineEdit().setPlaceholderText(message)
        self.port.setToolTip(message + "\n" + "\n".join(
            f"{item['device']}: {item['description']}" for item in report["ports"]))
        if not self.motor_button.text().startswith("断开"):
            self.motor_status.setText("电机未连接；" + message)
        self.last_port_scan = report
        self.ports_scanned.emit(dict(report))

    def assistance_values(self):
        return {
            "mass_kg": self.mass.value(), "gain": self.gain.value(),
            "lowpass_hz": self.cutoff.value(), "delay_s": self.delay.value() / 1000,
            "right_sign": self._right_sign,
        }

    def signature(self):
        return (*self.assistance_values().values(), self.limit.value())
