"""Independent persistent settings dialogs for Collector modalities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from exo_collection.adapters.ultrasound.raw_ethernet import (
    enumerate_network_interfaces,
    scan_ultrasound_interface,
)
from exo_collection.configuration import load_device_profile


def _validated_override(modality: str, override: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one override against the built-in hardware profile."""

    profile = load_device_profile("hardware")
    device = profile.by_modality()[modality]
    base = device.parameters.model_dump(exclude_none=True)
    parameter_type = type(device.parameters)
    parameter_type.model_validate({**base, **dict(override)})
    return dict(override)


class ModalityDeviceSettingsDialog(QDialog):
    """Base contract shared by the four independent settings dialogs."""

    modality: str

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._validated_override: dict[str, Any] | None = None

    @property
    def validated_override(self) -> dict[str, Any]:
        if self._validated_override is None:
            raise RuntimeError("device settings have not been accepted")
        return dict(self._validated_override)

    def _button_box(self) -> QDialogButtonBox:
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        return buttons

    def _finish_accept(self, override: Mapping[str, Any]) -> None:
        try:
            self._validated_override = _validated_override(self.modality, override)
        except Exception as exc:
            QMessageBox.warning(self, "设备设置无效", str(exc))
            return
        super().accept()


class UltrasoundInterfaceScanWorker(QThread):
    """Scan candidate NICs without blocking the Collector GUI thread."""

    result_ready = Signal(str, int)
    scan_failed = Signal(str, str)

    def __init__(
        self,
        interface_names: list[str],
        *,
        timeout_s: float = 1.5,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._interface_names = list(interface_names)
        self._timeout_s = float(timeout_s)

    def run(self) -> None:
        for interface_name in self._interface_names:
            if self.isInterruptionRequested():
                break
            try:
                count = scan_ultrasound_interface(
                    interface_name, timeout_s=self._timeout_s
                )
            except Exception as exc:
                self.scan_failed.emit(interface_name, str(exc))
                continue
            self.result_ready.emit(interface_name, count)


class UltrasoundDeviceSettingsDialog(ModalityDeviceSettingsDialog):
    modality = "ultrasound"

    def __init__(
        self,
        current: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("超声设备设置")
        self.setMinimumWidth(680)
        self._scan_worker: UltrasoundInterfaceScanWorker | None = None

        outer = QVBoxLayout(self)
        intro = QLabel(
            "真实设备：Raw Ethernet / Npcap。请选择与超声采集板直连的有线网卡。"
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)
        form = QFormLayout()

        interface_widget = QWidget(self)
        interface_layout = QHBoxLayout(interface_widget)
        interface_layout.setContentsMargins(0, 0, 0, 0)
        self.interface_combo = QComboBox(interface_widget)
        self.interface_combo.setObjectName("ultrasound_interface")
        interface_layout.addWidget(self.interface_combo, 1)
        self.refresh_button = QPushButton("刷新网卡", interface_widget)
        self.refresh_button.clicked.connect(self._populate_interfaces)
        interface_layout.addWidget(self.refresh_button)
        self.scan_button = QPushButton("扫描超声帧", interface_widget)
        self.scan_button.clicked.connect(self._scan_interfaces)
        interface_layout.addWidget(self.scan_button)
        form.addRow("采集网卡：", interface_widget)

        self.scan_status = QLabel("请选择连接超声设备的有线网卡。")
        self.scan_status.setWordWrap(True)
        form.addRow("扫描状态：", self.scan_status)

        self.nominal_rate_spin = QDoubleSpinBox()
        self.nominal_rate_spin.setObjectName("ultrasound_nominal_rate_hz")
        self.nominal_rate_spin.setRange(0.1, 10_000.0)
        self.nominal_rate_spin.setDecimals(2)
        self.nominal_rate_spin.setSuffix(" Hz")
        self.nominal_rate_spin.setValue(float(current.get("nominal_rate_hz", 20.0)))
        form.addRow("标称帧率：", self.nominal_rate_spin)

        fixed = QLabel("固定格式：4 通道；每个网络包对应一个通道的 1000 个 uint8 采样点。")
        fixed.setWordWrap(True)
        outer.addLayout(form)
        outer.addWidget(fixed)
        outer.addWidget(self._button_box())
        self._populate_interfaces(preferred=str(current.get("interface_name") or ""))

    @Slot()
    def _populate_interfaces(self, preferred: str = "") -> None:
        current = preferred or str(self.interface_combo.currentData() or "")
        self.interface_combo.clear()
        self.interface_combo.addItem("请选择有线网卡", None)
        entries = enumerate_network_interfaces()
        for entry in entries:
            name = str(entry.get("name") or "")
            if not name:
                continue
            description = str(entry.get("description") or name)
            self.interface_combo.addItem(f"{description} [{name}]", name)
        if current:
            index = self.interface_combo.findData(current)
            if index < 0:
                self.interface_combo.addItem(f"已保存的网卡 [{current}]", current)
                index = self.interface_combo.count() - 1
            self.interface_combo.setCurrentIndex(index)
        if not entries:
            self.scan_status.setText(
                "未枚举到可用有线网卡；请检查 Scapy/Npcap 安装。"
            )

    @Slot()
    def _scan_interfaces(self) -> None:
        if self._scan_worker is not None:
            return
        names = [
            str(self.interface_combo.itemData(index) or "")
            for index in range(self.interface_combo.count())
        ]
        names = [name for name in names if name]
        if not names:
            self.scan_status.setText("没有可扫描的有线网卡。")
            return
        self.scan_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.scan_status.setText("正在后台扫描超声目标 MAC 帧…")
        worker = UltrasoundInterfaceScanWorker(names, parent=self)
        worker.result_ready.connect(self._on_scan_result)
        worker.scan_failed.connect(self._on_scan_failed)
        worker.finished.connect(self._on_scan_finished)
        self._scan_worker = worker
        worker.start()

    @Slot(str, int)
    def _on_scan_result(self, interface_name: str, count: int) -> None:
        if count <= 0:
            return
        index = self.interface_combo.findData(interface_name)
        if index >= 0:
            self.interface_combo.setCurrentIndex(index)
        self.scan_status.setText(
            f"已在 {interface_name} 检测到 {count} 个超声通道帧。"
        )

    @Slot(str, str)
    def _on_scan_failed(self, interface_name: str, message: str) -> None:
        self.scan_status.setText(f"扫描 {interface_name} 失败：{message}")

    @Slot()
    def _on_scan_finished(self) -> None:
        worker = self._scan_worker
        self._scan_worker = None
        self.scan_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    def _stop_scan_worker(self) -> bool:
        worker = self._scan_worker
        if worker is None:
            return True
        if worker.isRunning():
            worker.requestInterruption()
            if not worker.wait(2_500):
                self.scan_status.setText("正在停止网卡扫描，请稍后再关闭或保存。")
                return False
        self._scan_worker = None
        self.scan_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        worker.deleteLater()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._stop_scan_worker():
            event.ignore()
            return
        super().closeEvent(event)

    @Slot()
    def reject(self) -> None:
        if self._stop_scan_worker():
            super().reject()

    @Slot()
    def accept(self) -> None:
        if not self._stop_scan_worker():
            return
        interface_name = str(self.interface_combo.currentData() or "").strip()
        self._finish_accept(
            {
                "interface_name": interface_name or None,
                "nominal_rate_hz": self.nominal_rate_spin.value(),
            }
        )


class ImuDeviceSettingsDialog(ModalityDeviceSettingsDialog):
    modality = "imu"

    def __init__(
        self,
        current: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("IMU 设备设置")
        self.setMinimumWidth(560)
        outer = QVBoxLayout(self)
        intro = QLabel("真实设备：Xsens Awinda；固定连接 3 个 MTw 传感器。")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        form = QFormLayout()

        self.channel_spin = QSpinBox()
        self.channel_spin.setObjectName("imu_radio_channel")
        self.channel_spin.setRange(11, 25)
        self.channel_spin.setValue(int(current.get("radio_channel", 25)))
        form.addRow("Awinda 无线信道：", self.channel_spin)

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setObjectName("imu_sample_rate_hz")
        self.rate_spin.setRange(1.0, 2_000.0)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.setValue(float(current.get("sample_rate_hz", 120.0)))
        form.addRow("采样率：", self.rate_spin)

        self.sensor_ids_edit = QLineEdit(
            ", ".join(str(item) for item in current.get("sensor_ids", ()))
        )
        self.sensor_ids_edit.setObjectName("imu_sensor_ids")
        self.sensor_ids_edit.setPlaceholderText(
            "可留空；或按躯干、左腿、右腿顺序填写 3 个 MTw ID"
        )
        form.addRow("3 个 MTw ID：", self.sensor_ids_edit)
        outer.addLayout(form)
        outer.addWidget(self._button_box())

    @Slot()
    def accept(self) -> None:
        sensor_ids = tuple(
            item.strip()
            for item in self.sensor_ids_edit.text().split(",")
            if item.strip()
        )
        self._finish_accept(
            {
                "radio_channel": self.channel_spin.value(),
                "sample_rate_hz": self.rate_spin.value(),
                "sensor_ids": sensor_ids,
            }
        )


def enumerate_serial_ports() -> list[tuple[str, str]]:
    """Return serial port and description pairs without requiring pyserial at import."""

    try:
        import serial.tools.list_ports
    except ImportError:
        return []
    return [
        (str(port.device), str(port.description or port.device))
        for port in serial.tools.list_ports.comports()
    ]


class EncoderDeviceSettingsDialog(ModalityDeviceSettingsDialog):
    modality = "encoder"

    def __init__(
        self,
        current: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("电机编码器设备设置")
        self.setMinimumWidth(560)
        outer = QVBoxLayout(self)
        intro = QLabel("真实设备：Teensy 串口编码器；固定记录左右两侧位置。")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        form = QFormLayout()

        port_widget = QWidget(self)
        port_layout = QHBoxLayout(port_widget)
        port_layout.setContentsMargins(0, 0, 0, 0)
        self.port_combo = QComboBox(port_widget)
        self.port_combo.setObjectName("encoder_serial_port")
        self.port_combo.setEditable(True)
        port_layout.addWidget(self.port_combo, 1)
        self.refresh_button = QPushButton("刷新串口", port_widget)
        self.refresh_button.clicked.connect(self._populate_ports)
        port_layout.addWidget(self.refresh_button)
        form.addRow("Teensy 串口：", port_widget)

        self.baud_spin = QSpinBox()
        self.baud_spin.setObjectName("encoder_baudrate")
        self.baud_spin.setRange(1, 10_000_000)
        self.baud_spin.setValue(int(current.get("baudrate", 1_000_000)))
        form.addRow("波特率：", self.baud_spin)

        self.vid_edit = QLineEdit(f"0x{int(current.get('vid', 0x16C0)):04X}")
        self.vid_edit.setObjectName("encoder_vid")
        form.addRow("USB VID：", self.vid_edit)
        self.pid_edit = QLineEdit(f"0x{int(current.get('pid', 0x0483)):04X}")
        self.pid_edit.setObjectName("encoder_pid")
        form.addRow("USB PID：", self.pid_edit)

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setObjectName("encoder_nominal_rate_hz")
        self.rate_spin.setRange(1.0, 10_000.0)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.setValue(float(current.get("nominal_rate_hz", 200.0)))
        form.addRow("标称采样率：", self.rate_spin)

        outer.addLayout(form)
        outer.addWidget(self._button_box())
        self._populate_ports(preferred=str(current.get("port") or ""))

    @Slot()
    def _populate_ports(self, preferred: str = "") -> None:
        current = preferred or self._selected_port()
        self.port_combo.clear()
        self.port_combo.addItem("自动发现（按 VID/PID）", None)
        ports = enumerate_serial_ports()
        for port, description in ports:
            self.port_combo.addItem(f"{port} — {description}", port)
        if current:
            index = self.port_combo.findData(current)
            if index < 0:
                self.port_combo.addItem(current, current)
                index = self.port_combo.count() - 1
            self.port_combo.setCurrentIndex(index)
        else:
            self.port_combo.setCurrentIndex(0)

    def _selected_port(self) -> str:
        data = self.port_combo.currentData()
        if data:
            return str(data).strip()
        text = self.port_combo.currentText().strip()
        if self.port_combo.currentIndex() == 0 and text == "自动发现（按 VID/PID）":
            return ""
        return text

    @Slot()
    def accept(self) -> None:
        try:
            vid = int(self.vid_edit.text().strip(), 0)
            pid = int(self.pid_edit.text().strip(), 0)
        except ValueError as exc:
            QMessageBox.warning(self, "设备设置无效", f"VID/PID 格式无效：{exc}")
            return
        self._finish_accept(
            {
                "port": self._selected_port() or None,
                "baudrate": self.baud_spin.value(),
                "vid": vid,
                "pid": pid,
                "nominal_rate_hz": self.rate_spin.value(),
            }
        )


class SyncPulseDeviceSettingsDialog(ModalityDeviceSettingsDialog):
    modality = "sync_pulse"

    def __init__(
        self,
        current: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("同步脉冲设置")
        self.setMinimumWidth(560)
        outer = QVBoxLayout(self)
        warning = QLabel(
            "当前同步脉冲仍为模拟台架信号。真实测力台/动捕模拟输入将在硬件协议确认后接入。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            "QLabel { color:#664d03; background:#fff3cd; padding:8px; "
            "border:1px solid #ffecb5; border-radius:4px; }"
        )
        outer.addWidget(warning)
        form = QFormLayout()

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setObjectName("sync_sample_rate_hz")
        self.rate_spin.setRange(1.0, 100_000.0)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.setValue(float(current.get("sample_rate_hz", 1_000.0)))
        form.addRow("采样率：", self.rate_spin)

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setObjectName("sync_pulse_interval_s")
        self.interval_spin.setRange(0.001, 3_600.0)
        self.interval_spin.setDecimals(4)
        self.interval_spin.setSuffix(" s")
        self.interval_spin.setValue(float(current.get("pulse_interval_s", 1.0)))
        form.addRow("脉冲间隔：", self.interval_spin)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setObjectName("sync_pulse_width_s")
        self.width_spin.setRange(0.0001, 3_600.0)
        self.width_spin.setDecimals(4)
        self.width_spin.setSuffix(" s")
        self.width_spin.setValue(float(current.get("pulse_width_s", 0.02)))
        form.addRow("脉冲宽度：", self.width_spin)

        self.first_spin = QDoubleSpinBox()
        self.first_spin.setObjectName("sync_first_pulse_s")
        self.first_spin.setRange(0.0, 3_600.0)
        self.first_spin.setDecimals(4)
        self.first_spin.setSuffix(" s")
        self.first_spin.setValue(float(current.get("first_pulse_s", 0.25)))
        form.addRow("首次脉冲延迟：", self.first_spin)

        outer.addLayout(form)
        outer.addWidget(self._button_box())

    @Slot()
    def accept(self) -> None:
        self._finish_accept(
            {
                "sample_rate_hz": self.rate_spin.value(),
                "pulse_interval_s": self.interval_spin.value(),
                "pulse_width_s": self.width_spin.value(),
                "first_pulse_s": self.first_spin.value(),
            }
        )


DEVICE_SETTINGS_DIALOGS: dict[str, type[ModalityDeviceSettingsDialog]] = {
    "ultrasound": UltrasoundDeviceSettingsDialog,
    "imu": ImuDeviceSettingsDialog,
    "encoder": EncoderDeviceSettingsDialog,
    "sync_pulse": SyncPulseDeviceSettingsDialog,
}


__all__ = [
    "DEVICE_SETTINGS_DIALOGS",
    "EncoderDeviceSettingsDialog",
    "ImuDeviceSettingsDialog",
    "ModalityDeviceSettingsDialog",
    "SyncPulseDeviceSettingsDialog",
    "UltrasoundDeviceSettingsDialog",
    "UltrasoundInterfaceScanWorker",
    "enumerate_serial_ports",
]
