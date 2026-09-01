"""Independent persistent settings dialogs for Collector modalities."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from exo_collection.adapters.emg.noraxon import _normalise_unit_id, scan_ultium_units
from exo_collection.adapters.ultrasound.raw_ethernet import (
    enumerate_network_interfaces,
    scan_ultrasound_interface,
)
from exo_collection.configuration import load_device_profile

LOG = logging.getLogger("exo_collection.collector.ui")


def _validated_override(modality: str, override: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one override against the built-in hardware profile."""

    profile = load_device_profile("hardware")
    device = profile.by_modality()[modality]
    base = device.parameters.model_dump(exclude_none=True)
    parameter_type = type(device.parameters)
    parameter_type.model_validate({**base, **dict(override)})
    return dict(override)


class ModalityDeviceSettingsDialog(QDialog):
    """Base contract shared by the independent modality settings dialogs."""

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
            LOG.debug("扫描超声接口: %s", interface_name)
            try:
                count = scan_ultrasound_interface(
                    interface_name, timeout_s=self._timeout_s
                )
            except Exception as exc:
                LOG.error("扫描 %s 失败: %s", interface_name, exc)
                self.scan_failed.emit(interface_name, str(exc))
                continue
            LOG.info("扫描 %s 完成: %d 帧", interface_name, count)
            self.result_ready.emit(interface_name, count)


class NoraxonUnitScanWorker(QThread):
    """Enumerate online Noraxon Ultium sensors without blocking the GUI."""

    result_ready = Signal(list)
    scan_failed = Signal(str)

    def run(self) -> None:
        LOG.debug("检测 Noraxon Ultium 传感器…")
        try:
            serials = scan_ultium_units()
        except Exception as exc:
            LOG.error("Noraxon 传感器检测失败: %s", exc)
            self.scan_failed.emit(str(exc))
            return
        LOG.info("Noraxon 传感器检测完成: %s", serials)
        self.result_ready.emit(serials)


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
        self._scan_results: dict[str, int] = {}

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
        self._scan_results.clear()
        self.scan_status.setText("正在后台扫描超声协议帧…")
        worker = UltrasoundInterfaceScanWorker(names, parent=self)
        worker.result_ready.connect(self._on_scan_result)
        worker.scan_failed.connect(self._on_scan_failed)
        worker.finished.connect(self._on_scan_finished)
        self._scan_worker = worker
        worker.start()

    @Slot(str, int)
    def _on_scan_result(self, interface_name: str, count: int) -> None:
        self._scan_results[interface_name] = count
        if count <= 0:
            return
        index = self.interface_combo.findData(interface_name)
        if index >= 0:
            self.interface_combo.setCurrentIndex(index)
        self.scan_status.setText(
            f"已在 {interface_name} 检测到 {count} 个超声通道帧。"
        )
        LOG.info("超声扫描结果: %s → %d 帧（已自动选中）", interface_name, count)

    @Slot(str, str)
    def _on_scan_failed(self, interface_name: str, message: str) -> None:
        self.scan_status.setText(f"扫描 {interface_name} 失败：{message}")
        LOG.error("超声扫描失败: %s → %s", interface_name, message)

    @Slot()
    def _on_scan_finished(self) -> None:
        worker = self._scan_worker
        self._scan_worker = None
        self.scan_button.setEnabled(True)
        self.refresh_button.setEnabled(True)
        if worker is not None:
            worker.deleteLater()
        LOG.debug("超声扫描流程结束")

        # 自动选出检测到帧数最多的网口
        best = max(self._scan_results, key=self._scan_results.get, default=None)
        best_count = self._scan_results.get(best, 0) if best else 0
        if best is not None and best_count > 0:
            index = self.interface_combo.findData(best)
            if index >= 0:
                self.interface_combo.setCurrentIndex(index)
            self.scan_status.setText(
                f"扫描完成：已自动选中 {best}（{best_count} 帧）。"
            )
            LOG.info("超声扫描自动选中: %s（%d 帧）", best, best_count)
        else:
            self.scan_status.setText(
                "扫描完成：未检测到超声帧，请确认超声设备已上电并连接。"
            )
            LOG.warning("超声扫描：所有网口均未检测到超声帧，结果: %s",
                        self._scan_results)

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
        intro = QLabel(
            "真实设备：Xsens Awinda。填写哪个 IMU 槽位的 ID 就启用哪个；"
            "留空的槽位不会参与连接、对齐或丢包统计。"
        )
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

        ids_layout = QHBoxLayout()
        ids_layout.setSpacing(8)
        current_ids = tuple(str(item).strip() for item in current.get("sensor_ids", ()))
        sensor_slots = (*current_ids[:3], *("" for _ in range(max(0, 3 - len(current_ids)))))
        self.id_1_edit = QLineEdit(sensor_slots[0])
        self.id_1_edit.setObjectName("imu_sensor_id_1")
        self.id_1_edit.setPlaceholderText("左腿(IMU1) ID")
        ids_layout.addWidget(QLabel("左腿(IMU1)："))
        ids_layout.addWidget(self.id_1_edit)
        self.id_2_edit = QLineEdit(sensor_slots[1])
        self.id_2_edit.setObjectName("imu_sensor_id_2")
        self.id_2_edit.setPlaceholderText("右腿(IMU2) ID")
        ids_layout.addWidget(QLabel("右腿(IMU2)："))
        ids_layout.addWidget(self.id_2_edit)
        self.id_3_edit = QLineEdit(sensor_slots[2])
        self.id_3_edit.setObjectName("imu_sensor_id_3")
        self.id_3_edit.setPlaceholderText("盆骨(IMU3) ID")
        ids_layout.addWidget(QLabel("盆骨(IMU3)："))
        ids_layout.addWidget(self.id_3_edit)
        # Compatibility aliases for code that used the first positional UI.
        self.id_left_edit = self.id_1_edit
        self.id_mid_edit = self.id_2_edit
        self.id_right_edit = self.id_3_edit
        form.addRow("MTw 传感器 ID：", ids_layout)
        outer.addLayout(form)
        outer.addWidget(self._button_box())

    @Slot()
    def accept(self) -> None:
        sensor_slots = tuple(
            edit.text().strip()
            for edit in (self.id_1_edit, self.id_2_edit, self.id_3_edit)
        )
        sensor_ids = sensor_slots if any(sensor_slots) else ()
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
    ports: list[tuple[str, str]] = []
    for port in serial.tools.list_ports.comports():
        description = str(port.description or port.device)
        hwid = str(getattr(port, "hwid", "") or "")
        normalized = f"{description} {hwid}".upper()
        if (
            "BTHENUM" in normalized
            or "BLUETOOTH" in normalized
            or "蓝牙" in normalized
        ):
            continue
        ports.append((str(port.device), description))
    return ports


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
        intro = QLabel(
            "真实设备：Teensy 串口状态流；记录左右电机的位置、速度和 "
            "Iq 估算扭矩。留空串口时按 VID/PID 自动发现，并排除蓝牙虚拟串口。"
        )
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
            "当前同步脉冲仍为模拟台架信号，尚未接入真实测力台的同步触发输出。"
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


class MocapDeviceSettingsDialog(ModalityDeviceSettingsDialog):
    modality = "mocap"

    def __init__(
        self,
        current: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("动捕设备设置")
        self.setMinimumWidth(560)
        outer = QVBoxLayout(self)
        intro = QLabel(
            "动捕 Marker 与测力台数据由 XINGYING 原生录制为 .cap。"
            "采集脚本不再从 SDK 读取原始数据，而是在 Trial 开始/结束时"
            "通过「远程控制」端口触发 XINGYING 录制。"
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)
        form = QFormLayout()
        self.server_edit = QLineEdit(str(current.get("server_ip", "10.1.1.198")))
        self.server_edit.setObjectName("mocap_server_ip")
        form.addRow("Seeker 服务器 IP：", self.server_edit)
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(1.0, 1000.0)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.setValue(float(current.get("nominal_rate_hz", 100.0)))
        form.addRow("后备帧率：", self.rate_spin)
        self.marker_count_spin = QSpinBox()
        self.marker_count_spin.setRange(0, 1000)
        self.marker_count_spin.setSpecialValueText("自动读取")
        self.marker_count_spin.setValue(int(current.get("marker_count_fallback", 0)))
        form.addRow("后备 Marker 数量：", self.marker_count_spin)
        self.remote_control_ip_edit = QLineEdit(
            str(current.get("remote_control_ip", "127.0.0.1"))
        )
        self.remote_control_ip_edit.setObjectName("mocap_remote_control_ip")
        form.addRow("远程控制 IP：", self.remote_control_ip_edit)
        self.remote_control_port_spin = QSpinBox()
        self.remote_control_port_spin.setRange(1, 65535)
        self.remote_control_port_spin.setValue(int(current.get("remote_control_port", 7060)))
        form.addRow("远程控制端口：", self.remote_control_port_spin)
        self.remote_trigger_port_spin = QSpinBox()
        self.remote_trigger_port_spin.setRange(1, 65535)
        self.remote_trigger_port_spin.setValue(
            int(current.get("remote_trigger_port", 7061))
        )
        self.remote_trigger_port_spin.setObjectName("mocap_remote_trigger_port")
        form.addRow("远程触发端口：", self.remote_trigger_port_spin)
        self.database_path_edit = QLineEdit(
            str(current.get(
                "database_path",
                "C:/Users/Admin/Desktop/SEU_liangji/software/Exo_Collection_Calibration_XINGYING",
            ))
        )
        self.database_path_edit.setObjectName("mocap_database_path")
        form.addRow("工程目录（DatabasePath）：", self.database_path_edit)
        outer.addLayout(form)
        outer.addWidget(self._button_box())

    @Slot()
    def accept(self) -> None:
        self._finish_accept(
            {
                "server_ip": self.server_edit.text().strip(),
                "nominal_rate_hz": self.rate_spin.value(),
                "marker_count_fallback": self.marker_count_spin.value(),
                "remote_control_ip": self.remote_control_ip_edit.text().strip(),
                "remote_control_port": self.remote_control_port_spin.value(),
                "remote_trigger_port": self.remote_trigger_port_spin.value(),
                "database_path": self.database_path_edit.text().strip(),
            }
        )


class EmgDeviceSettingsDialog(ModalityDeviceSettingsDialog):
    modality = "emg"

    _DEFAULT_CHANNELS: tuple[tuple[str, str], ...] = (
        ("股直肌", "noraxon_g3_234fc"),
        ("股内侧肌", "noraxon_g3_234f5"),
        ("股外侧肌", ""),
        ("股中肌", ""),
    )

    def __init__(
        self,
        current: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("EMG 设备设置（Noraxon）")
        self.setMinimumWidth(640)
        self._scan_worker: NoraxonUnitScanWorker | None = None
        self._detected_serials: list[str] = []
        outer = QVBoxLayout(self)
        intro = QLabel(
            "Noraxon Ultium/G3 表面肌电。每块肌肉对应一个传感器 unit ID"
            "（可填完整标签 noraxon_g3_<序列号> 或纯序列号）；"
            "未填写 unit ID 的通道会以 NaN 记录，并在连接时触发「缺失传感器」告警。"
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        scan_row = QHBoxLayout()
        scan_row.setSpacing(8)
        self.scan_button = QPushButton("检测传感器")
        self.scan_button.setObjectName("emg_scan_units")
        self.scan_button.clicked.connect(self._start_unit_scan)
        scan_row.addWidget(self.scan_button)
        self.scan_status = QLabel("点击「检测传感器」扫描接收盒上在线的 unit。")
        self.scan_status.setWordWrap(True)
        scan_row.addWidget(self.scan_status, 1)
        outer.addLayout(scan_row)

        form = QFormLayout()

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setObjectName("emg_sample_rate_hz")
        self.rate_spin.setRange(1.0, 100_000.0)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.setValue(float(current.get("sample_rate_hz", 4000.0)))
        form.addRow("EMG 采样率：", self.rate_spin)

        self.unit_edit = QLineEdit(str(current.get("unit", "µV")))
        self.unit_edit.setObjectName("emg_unit")
        form.addRow("单位：", self.unit_edit)

        channels = self._resolve_channels(current)
        self._channel_name_edits: list[QLineEdit] = []
        self._channel_unit_combos: list[QComboBox] = []
        self._channel_rows: list[QWidget] = []
        channel_box = QWidget()
        channel_box_layout = QVBoxLayout(channel_box)
        channel_box_layout.setContentsMargins(0, 0, 0, 0)
        channel_box_layout.setSpacing(6)
        self._channel_list_layout = QVBoxLayout()
        self._channel_list_layout.setContentsMargins(0, 0, 0, 0)
        self._channel_list_layout.setSpacing(6)
        channel_box_layout.addLayout(self._channel_list_layout)
        for name, unit_id in channels:
            self._add_channel_row(name, unit_id)
        add_button = QPushButton("＋ 添加通道")
        add_button.setObjectName("emg_add_channel")
        add_button.setToolTip("新增一个自定义肌肉通道")
        add_button.clicked.connect(lambda: self._add_channel_row("", ""))
        channel_box_layout.addWidget(add_button, 0, Qt.AlignmentFlag.AlignLeft)
        form.addRow("肌肉通道：", channel_box)
        outer.addLayout(form)
        outer.addWidget(self._button_box())

    @staticmethod
    def _resolve_channels(current: Mapping[str, Any]) -> list[tuple[str, str]]:
        raw = current.get("channels", ())
        resolved: list[tuple[str, str]] = []
        if raw:
            for channel in raw:
                name = (
                    channel.get("name", "")
                    if isinstance(channel, Mapping)
                    else getattr(channel, "name", "")
                )
                unit_id = (
                    channel.get("unit_id", "")
                    if isinstance(channel, Mapping)
                    else getattr(channel, "unit_id", "")
                )
                resolved.append((str(name), str(unit_id)))
        if not resolved:
            resolved = list(EmgDeviceSettingsDialog._DEFAULT_CHANNELS)
        while len(resolved) < 4:
            resolved.append(("", ""))
        return resolved

    def _add_channel_row(self, name: str = "", unit_id: str = "") -> None:
        """Append one editable muscle-channel row to the dialog."""
        index = len(self._channel_name_edits) + 1
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        name_edit = QLineEdit(name)
        name_edit.setObjectName(f"emg_channel_name_{index}")
        name_edit.setPlaceholderText(f"肌肉 {index} 名称")
        row_layout.addWidget(QLabel(f"通道 {index}："))
        row_layout.addWidget(name_edit, 1)
        unit_combo = QComboBox()
        unit_combo.setObjectName(f"emg_channel_unit_id_{index}")
        unit_combo.setPlaceholderText("unit ID（未分配，记录 NaN）")
        if self._detected_serials:
            # 已扫描过传感器：直接填入检测结果，并按归一化序列号选中当前 unit；
            # 未分配 unit 的新行保持「未分配」占位（currentIndex = -1）。
            unit_combo.addItems(self._detected_serials)
            if unit_id:
                unit_combo.setCurrentIndex(
                    unit_combo.findText(_normalise_unit_id(unit_id))
                )
            else:
                unit_combo.setCurrentIndex(-1)
        elif unit_id:
            unit_combo.addItem(unit_id)
            unit_combo.setCurrentIndex(0)
        row_layout.addWidget(unit_combo, 2)
        remove_button = QPushButton("−")
        remove_button.setObjectName(f"emg_channel_remove_{index}")
        remove_button.setFixedWidth(28)
        remove_button.setToolTip("移除该通道")
        remove_button.clicked.connect(
            lambda _=False, w=row: self._remove_channel_row(w)
        )
        row_layout.addWidget(remove_button)
        self._channel_list_layout.addWidget(row)
        self._channel_name_edits.append(name_edit)
        self._channel_unit_combos.append(unit_combo)
        self._channel_rows.append(row)

    def _remove_channel_row(self, row: QWidget) -> None:
        """Remove a muscle-channel row (name edit + unit combo) from the dialog."""
        try:
            index = self._channel_rows.index(row)
        except ValueError:
            return
        self._channel_list_layout.removeWidget(row)
        row.deleteLater()
        self._channel_rows.pop(index)
        self._channel_name_edits.pop(index)
        self._channel_unit_combos.pop(index)

    def _start_unit_scan(self) -> None:
        if self._scan_worker is not None:
            return
        self._scan_worker = NoraxonUnitScanWorker(self)
        worker = self._scan_worker
        worker.result_ready.connect(self._on_scan_result)
        worker.scan_failed.connect(self._on_scan_failed)
        worker.finished.connect(self._on_scan_finished)
        self.scan_button.setEnabled(False)
        self.scan_status.setText("正在检测接收盒上在线的 Noraxon 传感器…")
        LOG.info("开始检测 Noraxon Ultium 传感器")
        worker.start()

    def _on_scan_result(self, serials: list[str]) -> None:
        self._detected_serials = list(serials)
        for combo in self._channel_unit_combos:
            current = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            for serial in serials:
                combo.addItem(serial)
            # 只读下拉框：把当前通道的 unit ID 归一化后匹配到裸序列号并选中；
            # 匹配不到（如旧配置串号有误）则清空，让用户从检测结果里重新选择。
            combo.setCurrentIndex(combo.findText(_normalise_unit_id(current)))
            combo.blockSignals(False)
        if serials:
            self.scan_status.setText(
                f"检测到 {len(serials)} 个在线传感器：{', '.join(serials)}"
            )
        else:
            self.scan_status.setText("未检测到在线传感器，请确认接收器已连接并上电。")

    def _on_scan_failed(self, message: str) -> None:
        self.scan_status.setText(f"检测失败：{message}")
        LOG.error("Noraxon 传感器检测失败: %s", message)

    def _on_scan_finished(self) -> None:
        self._scan_worker = None
        self.scan_button.setEnabled(True)

    def _stop_scan_worker(self) -> bool:
        worker = self._scan_worker
        if worker is None:
            return True
        if worker.isRunning():
            worker.requestInterruption()
            if not worker.wait(2_500):
                self.scan_status.setText("正在停止传感器检测，请稍后再关闭或保存。")
                return False
        self._scan_worker = None
        self.scan_button.setEnabled(True)
        worker.deleteLater()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._stop_scan_worker():
            event.ignore()
            return
        super().closeEvent(event)

    @Slot()
    def reject(self) -> None:
        if not self._stop_scan_worker():
            return
        super().reject()

    @Slot()
    def accept(self) -> None:
        if not self._stop_scan_worker():
            return
        channels: list[dict[str, str]] = []
        for name_edit, unit_combo in zip(
            self._channel_name_edits, self._channel_unit_combos
        ):
            name = name_edit.text().strip()
            if not name:
                continue
            channels.append({"name": name, "unit_id": unit_combo.currentText().strip()})
        self._finish_accept(
            {
                "sample_rate_hz": self.rate_spin.value(),
                "unit": self.unit_edit.text().strip(),
                "channels": channels,
            }
        )


class ForcePlateDeviceSettingsDialog(ModalityDeviceSettingsDialog):
    modality = "force_plate"

    def __init__(
        self,
        current: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("测力台设备设置")
        self.setMinimumWidth(600)
        outer = QVBoxLayout(self)
        intro = QLabel(
            "测力台六维力已由动捕供应商集成进 XING/Nokov 服务器的 Analog Channel 广播。"
            "本软件直接监听同一 Seeker 广播，通道数必须与服务器当前输出严格一致。"
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)
        form = QFormLayout()

        self.server_edit = QLineEdit(str(current.get("server_ip", "10.1.1.198")))
        self.server_edit.setObjectName("force_plate_server_ip")
        form.addRow("Seeker 服务器 IP：", self.server_edit)

        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(1.0, 100_000.0)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setSuffix(" Hz")
        self.rate_spin.setValue(float(current.get("sample_rate_hz", 100.0)))
        form.addRow("采样率：", self.rate_spin)

        self.channel_count_spin = QSpinBox()
        self.channel_count_spin.setRange(1, 80)
        self.channel_count_spin.setValue(int(current.get("channel_count", 6)))
        form.addRow("通道数：", self.channel_count_spin)

        self.channel_names_edit = QLineEdit(
            ", ".join(str(item) for item in current.get("channel_names", ()))
        )
        self.channel_names_edit.setPlaceholderText(
            "留空自动命名；或输入：fx, fy, fz, mx, my, mz"
        )
        form.addRow("通道名称（逗号分隔）：", self.channel_names_edit)

        self.units_edit = QLineEdit(
            ", ".join(str(item) for item in current.get("units", ()))
        )
        self.units_edit.setPlaceholderText(
            "留空自动；或输入：N, N, N, N*m, N*m, N*m"
        )
        form.addRow("单位（逗号分隔）：", self.units_edit)

        outer.addLayout(form)
        outer.addWidget(self._button_box())

    @Slot()
    def accept(self) -> None:
        names = tuple(
            item.strip()
            for item in self.channel_names_edit.text().replace("，", ",").split(",")
            if item.strip()
        )
        units = tuple(
            item.strip()
            for item in self.units_edit.text().replace("，", ",").split(",")
            if item.strip()
        )
        self._finish_accept(
            {
                "server_ip": self.server_edit.text().strip(),
                "sample_rate_hz": self.rate_spin.value(),
                "channel_count": self.channel_count_spin.value(),
                "channel_names": names,
                "units": units,
            }
        )


class MocapForcePlateDeviceSettingsDialog(ModalityDeviceSettingsDialog):
    """合并「动捕 Marker + 六维力测力台」的设置对话框。

    两者由动捕供应商集成进同一 XING/Nokov Seeker 广播，共享同一服务器 IP，
    因此设备连接表折叠为一项；本对话框一次同时保存两个模态的覆盖值。
    """

    # 基类占位；本对话框通过 validated_override 返回 {modality: override} 映射。
    modality = "mocap"

    def __init__(
        self,
        current_mocap: Mapping[str, Any],
        current_force_plate: Mapping[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("动捕与测力台设备设置")
        self.setMinimumWidth(640)
        outer = QVBoxLayout(self)
        intro = QLabel(
            "动捕 Marker 与测力台共享同一 Seeker 服务器。采集脚本通过 SDK 流式接收"
            "动捕 marker（带主机时间戳，与超声等模态同步写盘），同时经「远程控制」"
            "端口在 Trial 开始/结束时触发 XINGYING 录制 .cap（含 marker 与测力台）。"
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        form = QFormLayout()
        # 共享服务器 IP：同时写入 mocap 与 force_plate 两个覆盖值。
        self.server_edit = QLineEdit(
            str(current_mocap.get("server_ip", "10.1.1.198"))
        )
        self.server_edit.setObjectName("xingying_server_ip")
        form.addRow("Seeker 服务器 IP：", self.server_edit)
        outer.addLayout(form)

        # ── 动捕 Marker ──
        mocap_box = QGroupBox("动捕 Marker")
        mocap_form = QFormLayout(mocap_box)
        self.mocap_rate_spin = QDoubleSpinBox()
        self.mocap_rate_spin.setRange(1.0, 1000.0)
        self.mocap_rate_spin.setDecimals(2)
        self.mocap_rate_spin.setSuffix(" Hz")
        self.mocap_rate_spin.setValue(
            float(current_mocap.get("nominal_rate_hz", 100.0))
        )
        mocap_form.addRow("后备帧率：", self.mocap_rate_spin)
        self.marker_count_spin = QSpinBox()
        self.marker_count_spin.setRange(0, 1000)
        self.marker_count_spin.setSpecialValueText("自动读取")
        self.marker_count_spin.setValue(
            int(current_mocap.get("marker_count_fallback", 0))
        )
        mocap_form.addRow("后备 Marker 数量：", self.marker_count_spin)
        self.remote_control_ip_edit = QLineEdit(
            str(current_mocap.get("remote_control_ip", "127.0.0.1"))
        )
        self.remote_control_ip_edit.setObjectName("mocap_remote_control_ip")
        mocap_form.addRow("远程控制 IP：", self.remote_control_ip_edit)
        self.remote_control_port_spin = QSpinBox()
        self.remote_control_port_spin.setRange(1, 65535)
        self.remote_control_port_spin.setValue(
            int(current_mocap.get("remote_control_port", 7060))
        )
        mocap_form.addRow("远程控制端口：", self.remote_control_port_spin)
        self.remote_trigger_port_spin = QSpinBox()
        self.remote_trigger_port_spin.setRange(1, 65535)
        self.remote_trigger_port_spin.setValue(
            int(current_mocap.get("remote_trigger_port", 7061))
        )
        self.remote_trigger_port_spin.setObjectName("mocap_remote_trigger_port")
        mocap_form.addRow("远程触发端口：", self.remote_trigger_port_spin)
        self.database_path_edit = QLineEdit(
            str(current_mocap.get(
                "database_path",
                "C:/Users/Admin/Desktop/SEU_liangji/software/Exo_Collection_Calibration_XINGYING",
            ))
        )
        self.database_path_edit.setObjectName("mocap_database_path")
        mocap_form.addRow("工程目录（DatabasePath）：", self.database_path_edit)
        outer.addWidget(mocap_box)

        # ── 六维力测力台 ──
        force_box = QGroupBox("六维力测力台")
        force_form = QFormLayout(force_box)
        self.force_rate_spin = QDoubleSpinBox()
        self.force_rate_spin.setRange(1.0, 100_000.0)
        self.force_rate_spin.setDecimals(2)
        self.force_rate_spin.setSuffix(" Hz")
        self.force_rate_spin.setValue(
            float(current_force_plate.get("sample_rate_hz", 100.0))
        )
        force_form.addRow("采样率：", self.force_rate_spin)
        self.channel_count_spin = QSpinBox()
        self.channel_count_spin.setRange(1, 80)
        self.channel_count_spin.setValue(
            int(current_force_plate.get("channel_count", 6))
        )
        force_form.addRow("通道数：", self.channel_count_spin)
        self.channel_names_edit = QLineEdit(
            ", ".join(str(item) for item in current_force_plate.get("channel_names", ()))
        )
        self.channel_names_edit.setPlaceholderText(
            "留空自动命名；或输入：fx, fy, fz, mx, my, mz"
        )
        force_form.addRow("通道名称（逗号分隔）：", self.channel_names_edit)
        self.units_edit = QLineEdit(
            ", ".join(str(item) for item in current_force_plate.get("units", ()))
        )
        self.units_edit.setPlaceholderText(
            "留空自动；或输入：N, N, N, N*m, N*m, N*m"
        )
        force_form.addRow("单位（逗号分隔）：", self.units_edit)
        outer.addWidget(force_box)

        outer.addWidget(self._button_box())

    @property
    def validated_override(self) -> dict[str, dict[str, Any]]:
        if self._validated_override is None:
            raise RuntimeError("device settings have not been accepted")
        return dict(self._validated_override)

    @Slot()
    def accept(self) -> None:
        names = tuple(
            item.strip()
            for item in self.channel_names_edit.text().replace("，", ",").split(",")
            if item.strip()
        )
        units = tuple(
            item.strip()
            for item in self.units_edit.text().replace("，", ",").split(",")
            if item.strip()
        )
        server_ip = self.server_edit.text().strip()
        mocap_override = {
            "server_ip": server_ip,
            "nominal_rate_hz": self.mocap_rate_spin.value(),
            "marker_count_fallback": self.marker_count_spin.value(),
            "remote_control_ip": self.remote_control_ip_edit.text().strip(),
            "remote_control_port": self.remote_control_port_spin.value(),
            "remote_trigger_port": self.remote_trigger_port_spin.value(),
            "database_path": self.database_path_edit.text().strip(),
        }
        force_override = {
            "server_ip": server_ip,
            "sample_rate_hz": self.force_rate_spin.value(),
            "channel_count": self.channel_count_spin.value(),
            "channel_names": names,
            "units": units,
        }
        try:
            self._validated_override = {
                "mocap": _validated_override("mocap", mocap_override),
                "force_plate": _validated_override("force_plate", force_override),
            }
        except Exception as exc:
            QMessageBox.warning(self, "设备设置无效", str(exc))
            return
        super().accept()


DEVICE_SETTINGS_DIALOGS: dict[str, type[ModalityDeviceSettingsDialog]] = {
    "ultrasound": UltrasoundDeviceSettingsDialog,
    "imu": ImuDeviceSettingsDialog,
    "encoder": EncoderDeviceSettingsDialog,
    "mocap": MocapDeviceSettingsDialog,
    "emg": EmgDeviceSettingsDialog,
    "force_plate": ForcePlateDeviceSettingsDialog,
    "sync_pulse": SyncPulseDeviceSettingsDialog,
}


__all__ = [
    "DEVICE_SETTINGS_DIALOGS",
    "EncoderDeviceSettingsDialog",
    "ImuDeviceSettingsDialog",
    "MocapDeviceSettingsDialog",
    "EmgDeviceSettingsDialog",
    "ForcePlateDeviceSettingsDialog",
    "ModalityDeviceSettingsDialog",
    "MocapForcePlateDeviceSettingsDialog",
    "SyncPulseDeviceSettingsDialog",
    "UltrasoundDeviceSettingsDialog",
    "UltrasoundInterfaceScanWorker",
    "enumerate_serial_ports",
]
