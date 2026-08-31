"""Responsive PySide6 shell for the Collector worker process.

Per-modality preview connect/disconnect with independent subprocess workers.
Trial lifecycle: continuous preview workers forward raw events to the
CollectorWorker through bounded IPC queues (StreamProxyAdapter).
Preview workers are never stopped or reconnected during a Trial.
"""

from __future__ import annotations

import inspect
import logging
import math
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QEvent,
    QLocale,
    QObject,
    QRegularExpression,
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QDoubleValidator,
    QIntValidator,
    QKeyEvent,
    QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.acquisition.recording_stream import RecordingStreamEndpoint
from exo_collection.acquisition.workers import CollectorWorker
from exo_collection.apps.collector.button_marker import (
    ButtonMarkerListener,
    START_STOP_VK,
)
from exo_collection.apps.collector.device_preview import (
    AdapterFactory,
    ModalityPreviewHandle,
    ModalityPreviewProcessHandle,
    ProfileModalityAdapterFactory,
)
from exo_collection.apps.collector.device_settings import (
    DEVICE_SETTINGS_DIALOGS,
    MocapForcePlateDeviceSettingsDialog,
)
from exo_collection.apps.collector.xingying_remote import (
    XingYingRemoteCapture,
    XingYingRemoteTrigger,
)
from exo_collection.apps.collector.preflight import (
    CollectorPreflightReport,
    CollectorPreflightWorker,
    run_simulated_preflight,
)
from exo_collection.apps.collector.elapsed_timer import ElapsedTimerPanel
from exo_collection.apps.collector.preview_workspace import PreviewWorkspace
from exo_collection.apps.collector.xingying_recording import XingYingRecordingPanel
from exo_collection.apps.collector.theme import COLLECTOR_STYLESHEET
from exo_collection.configuration import (
    SharedAppSettings,
    build_adapters,
    load_device_profile,
)
from exo_collection.adapters.ultrasound.raw_ethernet import (
    enumerate_network_interfaces,
    scan_ultrasound_interface,
)
from exo_collection.orchestration.models import (
    MeasuredConditionMetadata,
    TrialExperimentMetadata,
    TrialRunRequest,
)
from exo_collection.domain.project_codes import (
    COLLECTOR_PROJECTS,
    SUPPORTED_PROJECT_CODES,
    project_accepts_condition_level,
)
from exo_collection.domain.prompt_labels import PromptLabelEvent, PromptLabelSource
from exo_collection.domain.xingying_trigger import XingYingTriggerKind
from exo_collection.protocols import load_default_protocol
from exo_collection.quality import load_storage_policy

LOG = logging.getLogger("exo_collection.collector.ui")

MODALITIES = (
    "ultrasound",
    "imu",
    "encoder",
    "mocap",
    "emg",
    "force_plate",
)
# 动捕 Marker 与测力台绑定为「XINGYING 远程触发」：连接其一即连接另一，
# 且不再从 SDK 读取原始数据，而是在 Trial 开始/结束时触发 .cap 录制。
XINGYING_LINKED_MODALITIES = ("mocap", "force_plate")


# 设备连接表把「动捕 Marker + 六维力测力台」折叠成一行（共享同一 Seeker 服务器）。
XINGYING_GROUP_KEY = "mocap_force_plate"
XINGYING_GROUP_DISPLAY_NAME = "动捕 Marker + 六维力测力台"
# 按钮标签行不是数据模态，仅作为「连接 = 启用按钮记录」的 UI 开关。
BUTTON_ROW_KEY = "button_label"
BUTTON_ROW_DISPLAY_NAME = "按钮标签（,）"
# 设备连接表的显示行：绑定组作为一项合并展示，其余模态各占一行。
CONNECTION_ROWS: tuple[tuple[str, ...], ...] = (
    ("ultrasound",),
    ("imu",),
    ("encoder",),
    XINGYING_LINKED_MODALITIES,
    ("emg",),
)
CONNECTION_ROW_DISPLAY_NAMES: dict[tuple[str, ...], str] = {
    XINGYING_LINKED_MODALITIES: XINGYING_GROUP_DISPLAY_NAME,
}
# 模态 → 设备连接表行键：合并组共用同一状态点与连接/断开按钮。
_MODALITY_ROW_KEY = {
    "mocap": XINGYING_GROUP_KEY,
    "force_plate": XINGYING_GROUP_KEY,
}
PROMPT_HEALTH_ROWS = ("subject_prompt", "operator_prompt", "button_prompt")
HEALTH_ROWS = MODALITIES + PROMPT_HEALTH_ROWS
MODALITY_DISPLAY_NAMES = {
    "ultrasound": "超声",
    "imu": "IMU",
    "encoder": "电机编码器",
    "mocap": "动捕 Marker",
    "force_plate": "六维力测力台",
    "emg": "表面肌电 EMG",
    "subject_prompt": "受试者标签（<）",
    "operator_prompt": "工作人员标签（>）",
    "button_prompt": "按钮标签（,）",
}
CRITICAL_MODALITIES = frozenset(
    {"ultrasound", "imu", "encoder", "mocap", "emg"}
)
HEALTH_COLUMN_MODALITY = 0
HEALTH_COLUMN_SAMPLE_COUNT = 1
HEALTH_COLUMN_NOMINAL_RATE = 2
HEALTH_COLUMN_RATE = 3
HEALTH_COLUMN_DROPPED = 4
HEALTH_COLUMN_SYNC = 5


def _nominal_rate_from_device(device: Any) -> float | None:
    """Extract the nominal/set rate from a device profile's parameters.

    Parameter models name this field inconsistently across modalities
    (``nominal_rate_hz`` vs ``sample_rate_hz`` vs ``frame_rate_hz``), so we
    inspect all three names and return the first non-None value.
    """

    parameters = getattr(device, "parameters", None)
    if parameters is None:
        return None
    dumped = parameters.model_dump(exclude_none=True)
    for key in ("nominal_rate_hz", "sample_rate_hz", "frame_rate_hz"):
        value = dumped.get(key)
        if value is not None:
            return float(value)
    return None


MAX_PREVIEW_POINTS = 4096
MAX_TIMELINE_EVENTS = 300
SIGNAL_RING_CAPACITY = 1000
ULTRASOUND_PREVIEW_SAMPLES = 1000
_IMU_SENSOR_LABELS = ("imu_left_leg", "imu_right_leg", "imu_pelvis")
_IMU_AXIS_NAMES = ("acc_x", "acc_y", "acc_z")
_IMU_AXIS_COLORS = {"acc_x": "#dc3545", "acc_y": "#0d6efd", "acc_z": "#198754"}
IMU_PREVIEW_LABELS: tuple[str, ...] = tuple(
    f"{sensor}_{axis}"
    for sensor in _IMU_SENSOR_LABELS
    for axis in _IMU_AXIS_NAMES
)
ENCODER_PREVIEW_LABELS = (
    "left_position",
    "left_velocity",
    "left_torque",
    "right_position",
    "right_velocity",
    "right_torque",
)
# Noraxon EMG runs at 4000 Hz; 4 s of signal = 16 000 samples per ring window.
EMG_PREVIEW_RING_CAPACITY = 16_000
_SIGNAL_COLORS = (
    "#0d6efd", "#dc3545", "#198754", "#d97706",
    "#6f42c1", "#0dcaf0", "#fd7e14", "#20c997",
)
_ENCODER_METRICS = (
    ("position", "位置", "rad", "#d97706"),
    ("velocity", "速度", "rad/s", "#0d6efd"),
    ("torque", "估算扭矩", "N·m", "#198754"),
)
# Five-times magnification relative to the original +/-65 rad preview range.
_ENCODER_SHARED_Y_RANGE = (-13.0, 13.0)
DEFAULT_OPERATOR = "not_recorded"
DEFAULT_CONTROLLED_STOP_TIMEOUT_S = 30.0

PROJECTS: tuple[dict[str, str], ...] = tuple(
    dict(project) for project in COLLECTOR_PROJECTS
)

_PROTOCOL = load_default_protocol()
CONDITIONS: tuple[dict[str, Any], ...] = tuple(
    condition.model_dump(mode="json") for condition in _PROTOCOL.conditions
)


class WorkerHandle(Protocol):
    @property
    def is_alive(self) -> bool: ...

    @property
    def exitcode(self) -> int | None: ...

    def start(self) -> None: ...

    def request_stop(self) -> None: ...

    def record_prompt_label(
        self,
        source: PromptLabelSource | str,
        *,
        host_monotonic_ns: int,
        host_utc_ns: int,
    ) -> PromptLabelEvent: ...

    def record_xingying_trigger(
        self,
        kind: XingYingTriggerKind | str,
        *,
        capture_name: str,
        database_path: str,
        notes: str,
        description: str,
        delay: str,
        timecode: str,
        packet_id: str,
        host_monotonic_ns: int,
        host_utc_ns: int,
    ) -> Any: ...

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]: ...

    def join(self, timeout: float | None = None) -> int | None: ...

    def terminate_for_recovery(self, timeout: float = 5.0) -> int | None: ...

    def close(self) -> None: ...


WorkerFactory = Callable[
    [TrialRunRequest, Mapping[str, RecordingStreamEndpoint]], WorkerHandle
]


class PreflightWorkerHandle(Protocol):
    @property
    def is_alive(self) -> bool: ...

    @property
    def exitcode(self) -> int | None: ...

    def start(self) -> None: ...

    def poll_result(self) -> tuple[str, object] | None: ...

    def join(self, timeout: float | None = None) -> int | None: ...

    def terminate(self, timeout: float = 1.0) -> int | None: ...

    def close(self) -> None: ...


PreflightWorkerFactory = Callable[..., PreflightWorkerHandle]


def simulated_preflight_worker_factory(
    data_root: Path,
    device_profile_key: str = "simulated",
    device_overrides: dict[str, dict[str, Any]] | None = None,
) -> PreflightWorkerHandle:
    storage_policy = load_storage_policy()
    return CollectorPreflightWorker(
        data_root,
        device_profile_key=device_profile_key,
        device_overrides=device_overrides,
        minimum_free_space_gib=storage_policy.minimum_free_space_gib,
    )


def simulated_profile_preflight(
    data_root: Path,
) -> CollectorPreflightReport:
    storage_policy = load_storage_policy()
    return run_simulated_preflight(
        data_root,
        minimum_free_space_gib=storage_policy.minimum_free_space_gib,
    )


# ── Hardware Device Settings Dialog ────────────────────────────────────────


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


class HardwareDeviceSettingsDialog(QDialog):
    """Legacy combined settings dialog retained for compatibility tests."""

    def __init__(
        self,
        overrides: Mapping[str, Mapping[str, Any]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("真实设备设置")
        self.setMinimumWidth(680)
        self._validated_overrides: dict[str, dict[str, Any]] | None = None
        self._ultrasound_scan_worker: UltrasoundInterfaceScanWorker | None = None
        self._ultrasound_scan_results: dict[str, int] = {}
        current = {name: dict(values) for name, values in overrides.items()}
        self._initial_overrides = current
        ultrasound = current.get("ultrasound", {})
        imu = current.get("imu", {})
        encoder = current.get("encoder", {})

        outer = QVBoxLayout(self)
        form = QFormLayout()

        interface_widget = QWidget(self)
        interface_layout = QHBoxLayout(interface_widget)
        interface_layout.setContentsMargins(0, 0, 0, 0)
        self.ultrasound_interface_combo = QComboBox(interface_widget)
        self.ultrasound_interface_combo.setObjectName("hardware_ultrasound_interface")
        interface_layout.addWidget(self.ultrasound_interface_combo, 1)
        self.ultrasound_refresh_button = QPushButton("刷新网卡", interface_widget)
        self.ultrasound_refresh_button.clicked.connect(self._populate_ultrasound_interfaces)
        interface_layout.addWidget(self.ultrasound_refresh_button)
        self.ultrasound_scan_button = QPushButton("扫描超声帧", interface_widget)
        self.ultrasound_scan_button.clicked.connect(self._scan_ultrasound_interfaces)
        interface_layout.addWidget(self.ultrasound_scan_button)
        form.addRow("超声采集网卡：", interface_widget)
        self.ultrasound_scan_status = QLabel("请选择连接超声设备的有线网卡。")
        self.ultrasound_scan_status.setWordWrap(True)
        form.addRow("超声扫描状态：", self.ultrasound_scan_status)
        self._populate_ultrasound_interfaces(
            preferred=str(ultrasound.get("interface_name") or "")
        )

        self.awinda_channel_edit = QLineEdit(
            str(imu.get("radio_channel", 25))
        )
        self.awinda_channel_edit.setValidator(QIntValidator(11, 25, self))
        form.addRow("Awinda 无线信道：", self.awinda_channel_edit)
        self.awinda_rate_edit = QLineEdit(str(imu.get("sample_rate_hz", 120.0)))
        self.awinda_rate_edit.setValidator(QDoubleValidator(1.0, 2000.0, 3, self))
        form.addRow("Awinda 采样率 (Hz)：", self.awinda_rate_edit)
        imu_ids_layout = QHBoxLayout()
        imu_ids_layout.setSpacing(8)
        current_ids = tuple(
            str(item).strip() for item in imu.get("sensor_ids", ())
        )
        sensor_slots = (*current_ids[:3], *("" for _ in range(max(0, 3 - len(current_ids)))))
        self.awinda_id_left = QLineEdit(sensor_slots[0])
        self.awinda_id_left.setPlaceholderText("左腿(IMU1) ID")
        imu_ids_layout.addWidget(QLabel("左腿(IMU1)："))
        imu_ids_layout.addWidget(self.awinda_id_left)
        self.awinda_id_mid = QLineEdit(sensor_slots[1])
        self.awinda_id_mid.setPlaceholderText("右腿(IMU2) ID")
        imu_ids_layout.addWidget(QLabel("右腿(IMU2)："))
        imu_ids_layout.addWidget(self.awinda_id_mid)
        self.awinda_id_right = QLineEdit(sensor_slots[2])
        self.awinda_id_right.setPlaceholderText("盆骨(IMU3) ID")
        imu_ids_layout.addWidget(QLabel("盆骨(IMU3)："))
        imu_ids_layout.addWidget(self.awinda_id_right)
        form.addRow("MTw 传感器 ID：", imu_ids_layout)

        self.encoder_port_edit = QLineEdit(str(encoder.get("port") or ""))
        self.encoder_port_edit.setPlaceholderText("留空时按 VID/PID 自动发现")
        form.addRow("Teensy 串口：", self.encoder_port_edit)
        self.encoder_baud_edit = QLineEdit(str(encoder.get("baudrate", 1_000_000)))
        self.encoder_baud_edit.setValidator(QIntValidator(1, 10_000_000, self))
        form.addRow("Teensy 波特率：", self.encoder_baud_edit)
        self.encoder_vid_edit = QLineEdit(
            f"0x{int(encoder.get('vid', 0x16C0)):04X}"
        )
        form.addRow("Teensy VID：", self.encoder_vid_edit)
        self.encoder_pid_edit = QLineEdit(
            f"0x{int(encoder.get('pid', 0x0483)):04X}"
        )
        form.addRow("Teensy PID：", self.encoder_pid_edit)

        fixed = QLabel(
            "固定配置：超声 4 通道×1000点；IMU 3 台；编码器左右 2 侧。"
            "密码或凭据不会写入这里。"
        )
        fixed.setWordWrap(True)
        outer.addLayout(form)
        outer.addWidget(fixed)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    @property
    def validated_overrides(self) -> dict[str, dict[str, Any]]:
        if self._validated_overrides is None:
            raise RuntimeError("hardware settings have not been accepted")
        return self._validated_overrides

    @Slot()
    def _populate_ultrasound_interfaces(self, preferred: str = "") -> None:
        current = preferred or str(
            self.ultrasound_interface_combo.currentData() or ""
        )
        self.ultrasound_interface_combo.clear()
        self.ultrasound_interface_combo.addItem("请选择有线网卡", None)
        entries = enumerate_network_interfaces()
        for entry in entries:
            name = str(entry.get("name") or "")
            if not name:
                continue
            description = str(entry.get("description") or name)
            self.ultrasound_interface_combo.addItem(
                f"{description} [{name}]", name
            )
        if current:
            index = self.ultrasound_interface_combo.findData(current)
            if index < 0:
                self.ultrasound_interface_combo.addItem(
                    f"已保存的网卡 [{current}]", current
                )
                index = self.ultrasound_interface_combo.count() - 1
            self.ultrasound_interface_combo.setCurrentIndex(index)
        if not entries:
            self.ultrasound_scan_status.setText(
                "未枚举到可用有线网卡；请检查 Scapy/Npcap 安装。"
            )

    @Slot()
    def _scan_ultrasound_interfaces(self) -> None:
        if self._ultrasound_scan_worker is not None:
            return
        names = [
            str(self.ultrasound_interface_combo.itemData(index) or "")
            for index in range(self.ultrasound_interface_combo.count())
        ]
        names = [name for name in names if name]
        if not names:
            self.ultrasound_scan_status.setText("没有可扫描的有线网卡。")
            return
        self.ultrasound_scan_button.setEnabled(False)
        self.ultrasound_refresh_button.setEnabled(False)
        self._ultrasound_scan_results.clear()
        self.ultrasound_scan_status.setText("正在后台扫描超声协议帧…")
        worker = UltrasoundInterfaceScanWorker(names, parent=self)
        worker.result_ready.connect(self._on_ultrasound_scan_result)
        worker.scan_failed.connect(self._on_ultrasound_scan_failed)
        worker.finished.connect(self._on_ultrasound_scan_finished)
        self._ultrasound_scan_worker = worker
        worker.start()

    @Slot(str, int)
    def _on_ultrasound_scan_result(self, interface_name: str, count: int) -> None:
        self._ultrasound_scan_results[interface_name] = count
        if count <= 0:
            return
        index = self.ultrasound_interface_combo.findData(interface_name)
        if index >= 0:
            self.ultrasound_interface_combo.setCurrentIndex(index)
        self.ultrasound_scan_status.setText(
            f"已在 {interface_name} 检测到 {count} 个超声通道帧。"
        )
        LOG.info("超声扫描结果: %s → %d 帧（已自动选中）", interface_name, count)

    @Slot(str, str)
    def _on_ultrasound_scan_failed(self, interface_name: str, message: str) -> None:
        self.ultrasound_scan_status.setText(
            f"扫描 {interface_name} 失败：{message}"
        )
        LOG.error("超声扫描失败: %s → %s", interface_name, message)

    @Slot()
    def _on_ultrasound_scan_finished(self) -> None:
        worker = self._ultrasound_scan_worker
        self._ultrasound_scan_worker = None
        self.ultrasound_scan_button.setEnabled(True)
        self.ultrasound_refresh_button.setEnabled(True)
        if worker is not None:
            worker.deleteLater()
        LOG.debug("超声扫描流程结束")

        # 自动选出检测到帧数最多的网口
        best = max(self._ultrasound_scan_results, key=self._ultrasound_scan_results.get, default=None)
        best_count = self._ultrasound_scan_results.get(best, 0) if best else 0
        if best is not None and best_count > 0:
            index = self.ultrasound_interface_combo.findData(best)
            if index >= 0:
                self.ultrasound_interface_combo.setCurrentIndex(index)
            self.ultrasound_scan_status.setText(
                f"扫描完成：已自动选中 {best}（{best_count} 帧）。"
            )
            LOG.info("超声扫描自动选中: %s（%d 帧）", best, best_count)
        else:
            self.ultrasound_scan_status.setText(
                "扫描完成：未检测到超声帧，请确认超声设备已上电并连接。"
            )
            LOG.warning("超声扫描：所有网口均未检测到超声帧，结果: %s",
                        self._ultrasound_scan_results)

    def _stop_ultrasound_scan_worker(self) -> bool:
        worker = self._ultrasound_scan_worker
        if worker is None:
            return True
        if worker.isRunning():
            worker.requestInterruption()
            if not worker.wait(2_500):
                self.ultrasound_scan_status.setText(
                    "正在停止网卡扫描，请稍后再关闭或保存。"
                )
                return False
        self._ultrasound_scan_worker = None
        self.ultrasound_scan_button.setEnabled(True)
        self.ultrasound_refresh_button.setEnabled(True)
        worker.deleteLater()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._stop_ultrasound_scan_worker():
            event.ignore()
            return
        super().closeEvent(event)

    @Slot()
    def reject(self) -> None:
        if self._stop_ultrasound_scan_worker():
            super().reject()

    @Slot()
    def accept(self) -> None:
        try:
            sensor_slots = tuple(
                edit.text().strip()
                for edit in (self.awinda_id_left, self.awinda_id_mid, self.awinda_id_right)
            )
            sensor_ids = sensor_slots if any(sensor_slots) else ()
            encoder_port = self.encoder_port_edit.text().strip()
            interface_name = str(
                self.ultrasound_interface_combo.currentData() or ""
            ).strip()
            overrides: dict[str, dict[str, Any]] = {
                key: dict(value)
                for key, value in self._initial_overrides.items()
                if key not in {"ultrasound", "imu", "encoder"}
            }
            overrides.update({
                "ultrasound": {
                    "interface_name": interface_name or None,
                },
                "imu": {
                    "radio_channel": int(self.awinda_channel_edit.text()),
                    "sample_rate_hz": float(self.awinda_rate_edit.text()),
                    "sensor_ids": sensor_ids,
                },
                "encoder": {
                    "port": encoder_port or None,
                    "baudrate": int(self.encoder_baud_edit.text()),
                    "vid": int(self.encoder_vid_edit.text().strip(), 0),
                    "pid": int(self.encoder_pid_edit.text().strip(), 0),
                },
            })
            build_adapters(load_device_profile("hardware"), overrides)
        except Exception as exc:
            QMessageBox.warning(self, "真实设备设置无效", str(exc))
            return
        if not self._stop_ultrasound_scan_worker():
            return
        self._validated_overrides = overrides
        super().accept()


# ── Experiment Metadata Dialog ─────────────────────────────────────────────


class ExperimentMetadataDialog(QDialog):
    """Compact editor for optional, structured experimental records."""

    def __init__(
        self,
        metadata: TrialExperimentMetadata,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._validated_metadata: TrialExperimentMetadata | None = None
        self.setWindowTitle("详细信息")
        self.setMinimumWidth(760)
        outer = QVBoxLayout(self)
        grid = QGridLayout()

        subject_box = QGroupBox("受试者（可选）")
        subject_form = QFormLayout(subject_box)
        self.height_edit = self._float_edit("subject_height_cm", 30, 250)
        subject_form.addRow("身高 (cm)：", self.height_edit)
        self.weight_edit = self._float_edit("subject_weight_kg", 1, 500)
        subject_form.addRow("体重 (kg)：", self.weight_edit)
        self.leg_length_edit = self._float_edit("subject_leg_length_cm", 10, 200)
        subject_form.addRow("腿长 (cm)：", self.leg_length_edit)
        self.sex_combo = self._choice_combo(
            "subject_sex",
            (("未填写", None), ("女", "female"), ("男", "male"), ("其他", "other")),
        )
        subject_form.addRow("性别：", self.sex_combo)
        self.age_edit = QLineEdit()
        self.age_edit.setObjectName("subject_age_years")
        self.age_edit.setPlaceholderText("未填写")
        self.age_edit.setValidator(QIntValidator(0, 120, self))
        subject_form.addRow("年龄：", self.age_edit)
        grid.addWidget(subject_box, 0, 0)

        condition_box = QGroupBox("工况实测（可选）")
        condition_form = QFormLayout(condition_box)
        self.speed_edit = self._float_edit("treadmill_speed_mps", 0, 15)
        condition_form.addRow("跑台速度 (m/s)：", self.speed_edit)
        self.assist_edit = self._float_edit("assist_level", 0, 100)
        condition_form.addRow("助力等级：", self.assist_edit)
        self.load_edit = self._float_edit("load_kg", 0, 500)
        condition_form.addRow("负载 (kg)：", self.load_edit)
        self.slope_edit = self._float_edit("slope_deg", -45, 45)
        condition_form.addRow("坡度 (deg)：", self.slope_edit)
        grid.addWidget(condition_box, 0, 1)

        probe_box = QGroupBox("超声探头（可选）")
        probe_grid = QGridLayout(probe_box)
        self.muscle_edit = QLineEdit()
        self.muscle_edit.setObjectName("probe_muscle")
        self.muscle_edit.setPlaceholderText("例如：股外侧肌")
        probe_grid.addWidget(QLabel("肌肉："), 0, 0)
        probe_grid.addWidget(self.muscle_edit, 0, 1)
        self.laterality_combo = self._choice_combo(
            "probe_laterality",
            (("未填写", None), ("左腿", "left"), ("右腿", "right")),
        )
        probe_grid.addWidget(QLabel("侧别："), 0, 2)
        probe_grid.addWidget(self.laterality_combo, 0, 3)
        self.position_combo = self._choice_combo(
            "probe_longitudinal_position",
            (
                ("未填写", None),
                ("近端", "proximal"),
                ("中段", "middle"),
                ("远端", "distal"),
            ),
        )
        probe_grid.addWidget(QLabel("纵向位置："), 0, 4)
        probe_grid.addWidget(self.position_combo, 0, 5)

        self.channel_mapping_edits: list[QLineEdit] = []
        for channel_index in range(4):
            edit = QLineEdit()
            edit.setObjectName(f"probe_channel_{channel_index + 1}")
            edit.setPlaceholderText("未填写")
            self.channel_mapping_edits.append(edit)
            probe_grid.addWidget(QLabel(f"通道 {channel_index + 1}："), 1, channel_index * 2)
            probe_grid.addWidget(edit, 1, channel_index * 2 + 1)

        self.fixation_edit = QLineEdit()
        self.fixation_edit.setObjectName("probe_fixation_method")
        self.fixation_edit.setPlaceholderText("例如：弹力绑带 + 胶带")
        probe_grid.addWidget(QLabel("固定方式："), 2, 0)
        probe_grid.addWidget(self.fixation_edit, 2, 1, 1, 3)
        self.strap_pressure_edit = QLineEdit()
        self.strap_pressure_edit.setObjectName("probe_strap_pressure")
        self.strap_pressure_edit.setPlaceholderText("按实验刻度/描述记录，不假定单位")
        probe_grid.addWidget(QLabel("绑带压力："), 2, 4)
        probe_grid.addWidget(self.strap_pressure_edit, 2, 5)
        self.reapplied_combo = self._choice_combo(
            "probe_reapplied",
            (("未填写", None), ("否", False), ("是", True)),
        )
        probe_grid.addWidget(QLabel("重新贴探头："), 2, 6)
        probe_grid.addWidget(self.reapplied_combo, 2, 7)
        grid.addWidget(probe_box, 1, 0, 1, 2)
        outer.addLayout(grid)

        notes_box = QGroupBox("Trial 备注（可选）")
        notes_layout = QVBoxLayout(notes_box)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setObjectName("trial_notes")
        self.notes_edit.setMaximumHeight(90)
        self.notes_edit.setPlaceholderText("记录动作异常、探头滑移、临时调整等。")
        notes_layout.addWidget(self.notes_edit)
        outer.addWidget(notes_box)

        self.validation_label = QLabel()
        self.validation_label.setObjectName("experiment_metadata_validation")
        self.validation_label.setStyleSheet("color:#842029;")
        self.validation_label.setWordWrap(True)
        outer.addWidget(self.validation_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.set_metadata(metadata)

    def _float_edit(self, object_name: str, bottom: float, top: float) -> QLineEdit:
        edit = QLineEdit()
        edit.setObjectName(object_name)
        edit.setPlaceholderText("未填写")
        validator = QDoubleValidator(bottom, top, 3, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        validator.setLocale(QLocale.c())
        edit.setValidator(validator)
        return edit

    def _choice_combo(
        self, object_name: str, choices: tuple[tuple[str, object], ...]
    ) -> QComboBox:
        combo = QComboBox()
        combo.setObjectName(object_name)
        for label, value in choices:
            combo.addItem(label, value)
        return combo

    @staticmethod
    def _set_optional_number(edit: QLineEdit, value: float | int | None) -> None:
        edit.setText("" if value is None else f"{value:g}")

    @staticmethod
    def _select_data(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(max(0, index))

    def set_metadata(self, metadata: TrialExperimentMetadata) -> None:
        subject = metadata.subject
        self._set_optional_number(self.height_edit, subject.height_cm)
        self._set_optional_number(self.weight_edit, subject.weight_kg)
        self._set_optional_number(self.leg_length_edit, subject.leg_length_cm)
        self._select_data(self.sex_combo, subject.sex)
        self._set_optional_number(self.age_edit, subject.age_years)
        probe = metadata.ultrasound_probe
        self.muscle_edit.setText(probe.muscle or "")
        self._select_data(self.laterality_combo, probe.laterality)
        self._select_data(self.position_combo, probe.longitudinal_position)
        for edit, value in zip(self.channel_mapping_edits, probe.channel_mapping, strict=True):
            edit.setText(value or "")
        self.fixation_edit.setText(probe.fixation_method or "")
        self.strap_pressure_edit.setText(probe.strap_pressure or "")
        self._select_data(self.reapplied_combo, probe.probe_reapplied)
        measured = metadata.measured_condition
        self._set_optional_number(self.speed_edit, measured.treadmill_speed_mps)
        self._set_optional_number(self.assist_edit, measured.assist_level)
        self._set_optional_number(self.load_edit, measured.load_kg)
        self._set_optional_number(self.slope_edit, measured.slope_deg)
        self.notes_edit.setPlainText(metadata.trial_notes or "")

    @staticmethod
    def _optional_float(edit: QLineEdit, label: str) -> float | None:
        raw = edit.text().strip()
        if not raw:
            return None
        if not edit.hasAcceptableInput():
            raise ValueError(f"{label}超出允许范围")
        return float(raw)

    @staticmethod
    def _optional_int(edit: QLineEdit, label: str) -> int | None:
        raw = edit.text().strip()
        if not raw:
            return None
        if not edit.hasAcceptableInput():
            raise ValueError(f"{label}超出允许范围")
        return int(raw)

    def build_metadata(self) -> TrialExperimentMetadata:
        return TrialExperimentMetadata.model_validate(
            {
                "subject": {
                    "height_cm": self._optional_float(self.height_edit, "身高"),
                    "weight_kg": self._optional_float(self.weight_edit, "体重"),
                    "leg_length_cm": self._optional_float(self.leg_length_edit, "腿长"),
                    "sex": self.sex_combo.currentData(),
                    "age_years": self._optional_int(self.age_edit, "年龄"),
                },
                "ultrasound_probe": {
                    "muscle": self.muscle_edit.text(),
                    "laterality": self.laterality_combo.currentData(),
                    "longitudinal_position": self.position_combo.currentData(),
                    "channel_mapping": [edit.text() for edit in self.channel_mapping_edits],
                    "fixation_method": self.fixation_edit.text(),
                    "strap_pressure": self.strap_pressure_edit.text(),
                    "probe_reapplied": self.reapplied_combo.currentData(),
                },
                "measured_condition": {
                    "treadmill_speed_mps": self._optional_float(self.speed_edit, "跑台速度"),
                    "assist_level": self._optional_float(self.assist_edit, "助力等级"),
                    "load_kg": self._optional_float(self.load_edit, "负载"),
                    "slope_deg": self._optional_float(self.slope_edit, "坡度"),
                },
                "trial_notes": self.notes_edit.toPlainText(),
            }
        )

    def metadata(self) -> TrialExperimentMetadata:
        return self._validated_metadata or self.build_metadata()

    @Slot()
    def accept(self) -> None:
        try:
            self._validated_metadata = self.build_metadata()
        except (TypeError, ValueError) as exc:
            self.validation_label.setText(f"无法保存：{exc}")
            return
        self.validation_label.clear()
        super().accept()


# ── Ring Trace (preview display) ───────────────────────────────────────────


class HoverDetailsPlotWidget(pg.PlotWidget):
    """Plot that reveals its title, axis labels, and legend only on hover."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._hover_details_visible = False
        super().__init__(*args, **kwargs)
        self._set_hover_details_visible(False)

    def setTitle(self, *args: Any, **kwargs: Any) -> None:  # noqa: N802
        self.getPlotItem().setTitle(*args, **kwargs)
        self._set_hover_details_visible(self._hover_details_visible)

    def setLabel(self, *args: Any, **kwargs: Any) -> None:  # noqa: N802
        self.getPlotItem().setLabel(*args, **kwargs)
        self._set_hover_details_visible(self._hover_details_visible)

    def addLegend(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
        legend = self.getPlotItem().addLegend(*args, **kwargs)
        self._set_hover_details_visible(self._hover_details_visible)
        return legend

    def enterEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        self._set_hover_details_visible(True)
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        self._set_hover_details_visible(False)
        super().leaveEvent(event)

    def _set_hover_details_visible(self, visible: bool) -> None:
        self._hover_details_visible = bool(visible)
        plot_item = self.getPlotItem()
        title_label = plot_item.titleLabel
        title_label.setVisible(
            self._hover_details_visible and bool(title_label.text)
        )
        for axis_name in ("left", "bottom"):
            axis = plot_item.getAxis(axis_name)
            axis.setStyle(showValues=self._hover_details_visible)
            axis.label.setVisible(
                self._hover_details_visible and bool(axis.labelText)
            )
        legend = plot_item.legend
        if legend is not None:
            legend.setVisible(self._hover_details_visible)


class RingTrace:
    """Ring-buffer trace backed by a fixed-size numpy array for pyqtgraph."""

    __slots__ = (
        "_buffer", "_capacity", "_count", "_cursor", "_x",
        "_marker_lines", "_render_stride", "curve", "cursor_line", "plot",
    )

    def __init__(
        self, plot: "pg.PlotWidget", pen: str, label: str,
        *, capacity: int = SIGNAL_RING_CAPACITY, render_stride: int = 1,
    ) -> None:
        if capacity < 2:
            raise ValueError("ring trace capacity must be at least two")
        self._capacity = int(capacity)
        self._render_stride = max(1, int(render_stride))
        self._buffer = np.full(self._capacity, np.nan, dtype=np.float64)
        self._x = np.arange(self._capacity, dtype=np.float64)
        self._cursor = 0
        self._count = 0
        self._marker_lines: dict[int, "pg.InfiniteLine"] = {}
        self.plot = plot
        self.curve = plot.plot(pen=pg.mkPen(pen, width=1.2))
        self.cursor_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#dc3545", width=2))
        plot.addItem(self.cursor_line)
        plot.setTitle(label)
        plot.setBackground("w")
        plot.setXRange(0, self._capacity - 1, padding=0)
        plot.setLimits(xMin=0, xMax=self._capacity - 1, minXRange=self._capacity - 1, maxXRange=self._capacity - 1)
        plot.setMouseEnabled(x=False, y=False)
        plot.setLabel("bottom", "循环帧位置")
        plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve.setData(self._x, self._buffer)

    def append(self, values: np.ndarray | list[float]) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        n = int(arr.size)
        if n == 0:
            return
        self._clear_overwritten_markers(n)
        next_cursor = (self._cursor + n) % self._capacity
        if n >= self._capacity:
            tail = arr[-self._capacity :]
            split = self._capacity - next_cursor
            self._buffer[next_cursor:] = tail[:split]
            self._buffer[:next_cursor] = tail[split:]
        else:
            first_count = min(n, self._capacity - self._cursor)
            self._buffer[self._cursor : self._cursor + first_count] = arr[:first_count]
            overflow = n - first_count
            if overflow:
                self._buffer[:overflow] = arr[first_count:]
        self._cursor = next_cursor
        self._count = min(self._capacity, self._count + n)
        self._render()

    def mark_current(self, label: str) -> int:
        """Mark the latest displayed sample until its slot is overwritten."""

        slot = (self._cursor - 1) % self._capacity if self._count else self._cursor
        existing = self._marker_lines.get(slot)
        if existing is not None:
            previous = existing.toolTip().strip()
            existing.setToolTip(
                f"{previous}\n{label}" if previous and label not in previous else label
            )
            return slot
        line = pg.InfiniteLine(
            pos=float(slot),
            angle=90,
            movable=False,
            pen=pg.mkPen(
                "#ff0000",
                width=1.5,
                style=Qt.PenStyle.DashLine,
            ),
        )
        line.setZValue(90)
        line.setToolTip(label)
        self.plot.addItem(line)
        self._marker_lines[slot] = line
        return slot

    def _clear_overwritten_markers(self, count: int) -> None:
        if count >= self._capacity:
            overwritten = tuple(self._marker_lines)
        else:
            overwritten = tuple(
                (self._cursor + offset) % self._capacity
                for offset in range(count)
            )
        for slot in overwritten:
            line = self._marker_lines.pop(slot, None)
            if line is not None:
                self.plot.removeItem(line)

    def _render(self) -> None:
        if self._render_stride > 1:
            step = self._render_stride
            x = self._x[::step]
            display = self._buffer[::step].copy()
            if self._count == self._capacity:
                display[(self._cursor // step) % display.size] = np.nan
            self.curve.setData(x, display)
        else:
            display = self._buffer.copy()
            if self._count == self._capacity:
                display[self._cursor] = np.nan
            self.curve.setData(self._x, display)
        if self._count:
            self.cursor_line.setPos((self._cursor - 1) % self._capacity)

    def reset(self) -> None:
        self._clear_overwritten_markers(self._capacity)
        self._buffer.fill(np.nan)
        self._cursor = 0
        self._count = 0
        self.curve.setData(self._x, self._buffer)
        self.cursor_line.setPos(0.0)


class DismissibleToastLabel(QLabel):
    """Opaque notification card dismissed by one click anywhere on it."""

    dismissed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("点击关闭")
        self.setAccessibleName("通知；点击关闭")

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self.dismissed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() in {
            Qt.Key.Key_Escape,
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
            Qt.Key.Key_Space,
        }:
            self.dismissed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


# ── Preview Worker Factory Helpers ─────────────────────────────────────────

# ── CollectorWindow ────────────────────────────────────────────────────────


class CollectorWindow(QMainWindow):
    """Collect one Trial at a time with per-modality preview workers."""

    trial_started = Signal(object)
    trial_finished = Signal(bool)
    # XINGYING 起停通知来自后台监听线程；经此信号 queued 到 GUI 线程再弹 toast。
    xingying_alert_requested = Signal(str)

    def __init__(
        self,
        data_root: str | Path,
        *,
        settings: SharedAppSettings | None = None,
        worker_factory: WorkerFactory = CollectorWorker,
        preflight_worker_factory: PreflightWorkerFactory = simulated_preflight_worker_factory,
        preview_worker_factory: AdapterFactory | None = None,
        button_marker_factory: Callable[..., ButtonMarkerListener] | None = ButtonMarkerListener,
        poll_interval_ms: int = 20,
        controlled_stop_timeout_s: float = DEFAULT_CONTROLLED_STOP_TIMEOUT_S,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be positive")
        if controlled_stop_timeout_s <= 0:
            raise ValueError("controlled_stop_timeout_s must be positive")
        self._settings = settings if settings is not None else SharedAppSettings()
        self._worker_factory = worker_factory
        self._preflight_worker_factory = preflight_worker_factory
        self._button_marker_factory = button_marker_factory
        self._controlled_stop_timeout_s = float(controlled_stop_timeout_s)
        self._worker: WorkerHandle | None = None
        self._active_trial_uuid: str | None = None
        self._active_request: TrialRunRequest | None = None
        self._terminal_event_received = False
        self._dead_poll_count = 0
        self._stop_requested = False
        self._stop_requested_at: float | None = None
        self._forced_stop_alerted = False
        self._close_when_finished = False
        self._close_started_at: float | None = None
        self._configuration_locked = False
        self._preflight_busy = False
        self._preflight_ready = False
        self._preflight_worker: PreflightWorkerHandle | None = None
        self._preflight_result_handled = False
        self._preflight_empty_exit_polls = 0
        self._preflight_root: Path | None = None
        self._worker_state = "IDLE"
        self._trial_succeeded = False
        self._recording_branch_fault: str | None = None

        # Per-modality preview workers
        self._preview_workers: dict[str, ModalityPreviewHandle] = {}
        self._preview_connected_modalities: set[str] = set()
        self._preview_connection_status: dict[str, str] = {
            m: "未连接" for m in MODALITIES
        }
        self._preview_disconnect_deadlines: dict[str, float] = {}
        self._recording_preview_handles: dict[str, ModalityPreviewHandle] = {}
        self._recording_streams_ended = False
        self._injected_preview_factory = preview_worker_factory

        # XINGYING 远程触发（动捕 Marker + 测力台绑定，不经过预览 worker）。
        self._xingying_remote: XingYingRemoteCapture | None = None
        self._xingying_capture_name: str | None = None
        self._xingying_trigger: XingYingRemoteTrigger | None = None

        # 按钮标签（USB HID 键盘，逗号键）：全局钩子监听，连接时启用。
        self._button_marker: ButtonMarkerListener | None = None

        # 开始/停止 USB 按钮（句号键）：全局钩子监听，启动即启用。
        self._start_stop_button: ButtonMarkerListener | None = None

        self._experiment_metadata = TrialExperimentMetadata()
        self._experiment_metadata_by_identity: dict[tuple[str, str], TrialExperimentMetadata] = {}
        self._metadata_identity_key: tuple[str, str] | None = None
        self._metadata_condition_code: str | None = None

        self._session_key: tuple[str, str, str] | None = None
        self._session_uuid = uuid4()

        self._health_rows = {name: index for index, name in enumerate(HEALTH_ROWS)}
        self._prompt_label_counts = {
            PromptLabelSource.SUBJECT: 0,
            PromptLabelSource.OPERATOR: 0,
            PromptLabelSource.BUTTON: 0,
        }
        self._last_health_status: dict[str, str] = {}
        self._us_plots: list["pg.PlotWidget"] = []
        self._us_curves: list["pg.PlotDataItem"] = []
        self._us_x = np.arange(ULTRASOUND_PREVIEW_SAMPLES, dtype=np.float64)
        self._ultrasound_format_alerted: set[tuple[int, str]] = set()
        self._imu_traces: dict[str, RingTrace] = {}
        self._enc_traces: dict[str, RingTrace] = {}
        self._emg_traces: dict[str, RingTrace] = {}
        self._emg_grid_layout: QVBoxLayout | None = None
        self._emg_grid_content: QWidget | None = None
        self._xingying_status_panel: XingYingRecordingPanel | None = None
        self.preview_workspace: PreviewWorkspace | None = None
        self._elapsed_timer: ElapsedTimerPanel | None = None
        self._preview_focus_previous_sizes: list[int] | None = None
        self._preview_y_ranges: dict[str, tuple[float, float]] = {}
        self._timeline_started_at = time.monotonic()
        self._timeline_x: deque[float] = deque(maxlen=MAX_TIMELINE_EVENTS)
        self._timeline_y: deque[float] = deque(maxlen=MAX_TIMELINE_EVENTS)
        self._timeline_text: deque[str] = deque(maxlen=MAX_TIMELINE_EVENTS)

        # Per-modality connect buttons
        self._connect_buttons: dict[str, QPushButton] = {}
        self._disconnect_buttons: dict[str, QPushButton] = {}
        self._configure_buttons: dict[str, QPushButton] = {}
        self._connect_status_labels: dict[str, QLabel] = {}

        self.setWindowTitle("Exo Collector")
        self.setStyleSheet(COLLECTOR_STYLESHEET)
        self.resize(1280, 820)
        self._create_ui(Path(data_root).expanduser().resolve())
        self._populate_nominal_rates()
        self.xingying_alert_requested.connect(self._append_alert)
        self._prompt_event_filter_installed = False
        application = QApplication.instance()
        if application is not None:
            # Capture hardware-button key events before a focused child widget
            # (for example QLineEdit) can consume them.
            application.installEventFilter(self)
            self._prompt_event_filter_installed = True
        self.project_combo.currentIndexChanged.connect(self._handle_project_changed)
        self.subject_code_edit.textChanged.connect(self._activate_selected_metadata_identity)
        self._activate_selected_metadata_identity()
        self.condition_combo.currentIndexChanged.connect(self._handle_metadata_condition_changed)
        self._metadata_condition_code = self._selected_condition_code()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(poll_interval_ms)
        self._poll_timer.timeout.connect(self.poll_worker_events)
        self._preflight_timer = QTimer(self)
        self._preflight_timer.setInterval(max(20, poll_interval_ms))
        self._preflight_timer.timeout.connect(self.poll_preflight_worker)
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(max(20, poll_interval_ms))
        self._preview_timer.timeout.connect(self._poll_preview_workers)
        self._button_poll_timer = QTimer(self)
        self._button_poll_timer.setInterval(50)
        self._button_poll_timer.timeout.connect(self._poll_button_marker)
        self._start_stop_poll_timer = QTimer(self)
        self._start_stop_poll_timer.setInterval(50)
        self._start_stop_poll_timer.timeout.connect(self._poll_start_stop_button)
        self._set_trial_state("IDLE")
        self._update_start_button()
        self._start_start_stop_listener()

        LOG.info(
            "CollectorWindow 已初始化 data_root=%s profile=%s",
            data_root, self._settings.device_profile_key,
        )

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Esc 退出全屏→最大化；F11 切换全屏。"""
        if event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showMaximized()
                self.statusBar().showMessage("已退出全屏（按 F11 重新进入）。", 5000)
                return
        elif event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.showFullScreen()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """Capture prompt HID keys application-wide while a Trial is writing."""

        if (
            event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and self._worker is not None
            and not self._stop_requested
            and self._worker_state in {"WAITING_SYNC", "RECORDING"}
        ):
            text = event.text()
            key = event.key()
            modifiers = event.modifiers()
            LOG.debug(
                "Trial keypress: text=%r key=%d modifiers=%d "
                "native_scan_code=%d native_virtual_key=%d auto_repeat=%s",
                text,
                key,
                int(modifiers.value),
                event.nativeScanCode(),
                event.nativeVirtualKey(),
                event.isAutoRepeat(),
            )
            source: PromptLabelSource | None = None
            if (
                text == "<"
                or key == Qt.Key.Key_Less
                or (
                    key == Qt.Key.Key_Comma
                    and bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                )
            ):
                source = PromptLabelSource.SUBJECT
            elif (
                text == ">"
                or key == Qt.Key.Key_Greater
                or (
                    key == Qt.Key.Key_Period
                    and bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
                )
            ):
                source = PromptLabelSource.OPERATOR
            if source is not None:
                if event.isAutoRepeat():
                    LOG.debug(
                        "Ignoring auto-repeat prompt key: source=%s",
                        source.value,
                    )
                else:
                    LOG.info(
                        "Prompt hardware key captured: source=%s text=%r "
                        "key=%d native_scan_code=%d",
                        source.value,
                        text,
                        key,
                        event.nativeScanCode(),
                    )
                    self._capture_prompt_label(source)
                return True
        return super().eventFilter(watched, event)

    @Slot()
    def _capture_prompt_label(
        self,
        source: PromptLabelSource,
        *,
        host_monotonic_ns: int | None = None,
        host_utc_ns: int | None = None,
    ) -> None:
        """Capture one hardware-button keystroke only while writing a Trial."""

        worker = self._worker
        if (
            worker is None
            or self._stop_requested
            or self._worker_state not in {"WAITING_SYNC", "RECORDING"}
        ):
            self.statusBar().showMessage(
                f"{source.display_name}未记录：当前没有正在写盘的 Trial。",
                2500,
            )
            return
        record_prompt = getattr(worker, "record_prompt_label", None)
        if not callable(record_prompt):
            self._append_alert("当前 Collector Worker 不支持人工标签记录。")
            return
        if host_monotonic_ns is None:
            host_monotonic_ns = time.perf_counter_ns()
        if host_utc_ns is None:
            host_utc_ns = time.time_ns()
        try:
            event = record_prompt(
                source,
                host_monotonic_ns=host_monotonic_ns,
                host_utc_ns=host_utc_ns,
            )
        except Exception as exc:
            self._append_alert(
                f"{source.display_name}记录失败：{type(exc).__name__}: {exc}"
            )
            LOG.exception("Prompt label enqueue failed: source=%s", source.value)
            return
        label = (
            event.label
            if isinstance(event, PromptLabelEvent)
            else source.display_name
        )
        self._mark_prompt_on_previews(label)
        self.statusBar().showMessage(
            f"已捕获{label}（{source.key_text}）",
            2500,
        )
        LOG.info(
            "Prompt label enqueued: source=%s host_monotonic_ns=%d",
            source.value,
            host_monotonic_ns,
        )

    def _mark_prompt_on_previews(self, label: str) -> None:
        marked_plots: set[int] = set()
        for trace in (
            *self._imu_traces.values(),
            *self._enc_traces.values(),
            *self._emg_traces.values(),
        ):
            plot_identity = id(trace.plot)
            if plot_identity in marked_plots:
                continue
            trace.mark_current(label)
            marked_plots.add(plot_identity)

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def worker(self) -> WorkerHandle | None:
        return self._worker

    @property
    def configuration_locked(self) -> bool:
        return self._configuration_locked

    @property
    def preflight_ready(self) -> bool:
        return self._preflight_ready

    @property
    def preflight_in_progress(self) -> bool:
        return self._preflight_worker is not None

    @property
    def device_profile_label(self) -> QLabel:
        """Backward-compatible alias for _device_profile_label."""
        return self._device_profile_label

    @property
    def overall_status(self) -> str:
        return self.state_label.text().removeprefix("总状态：")

    def _create_ui(self, data_root: Path) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        # ── Header ──
        header = QHBoxLayout()
        title = QLabel("Exo Collector · 多模态数据采集")
        title.setObjectName("page_title")
        header.addWidget(title)
        header.addStretch(1)
        self.state_label = QLabel()
        self.state_label.setObjectName("trial_state")
        self.state_label.setMinimumWidth(170)
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.state_label)
        outer.addLayout(header)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setObjectName("collector_body")
        body.setChildrenCollapsible(False)
        self._body_splitter = body

        # The control column is deliberately scrollable.  On a 1080p Windows
        # desktop the taskbar, title bar and per-monitor DPI scaling leave less
        # than 1000 logical pixels of usable height.  Letting this large form
        # participate directly in the main window minimum-size calculation
        # made showMaximized() request an impossible geometry; Qt then crushed
        # rows and buttons together.  Keeping the form at its real minimum
        # height and scrolling only this column prevents both clipping and
        # overlap while the live plots continue to use the full viewport.
        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("controls_scroll")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        controls_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        controls_scroll.setMinimumWidth(610)
        controls_scroll.setMaximumWidth(650)
        self._controls_scroll = controls_scroll

        controls = QWidget()
        controls.setObjectName("controls_content")
        controls.setStyleSheet(
            "QWidget#controls_content QPushButton { "
            "min-height: 22px; max-height: 22px; padding: 2px 7px; }"
            "QWidget#controls_content QPushButton[buttonRole='deviceConfig'] { "
            "min-height: 21px; max-height: 21px; }"
            "QWidget#controls_content QPushButton#connect_all, "
            "QWidget#controls_content QPushButton#start_trial { "
            "min-height: 28px; max-height: 28px; }"
            "QWidget#controls_content QPushButton#edit_experiment_metadata { "
            "min-height: 24px; max-height: 24px; }"
        )
        controls.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        controls_layout.setContentsMargins(6, 0, 6, 4)
        controls_layout.setSpacing(4)

        # ── Trial Settings ──
        metadata_box = QGroupBox("Trial 设置")
        form = QFormLayout(metadata_box)
        form.setContentsMargins(8, 12, 8, 7)
        form.setHorizontalSpacing(7)
        form.setVerticalSpacing(4)
        root_row = QHBoxLayout()
        root_row.setSpacing(5)
        self.data_root_edit = QLineEdit(str(data_root))
        self.data_root_edit.setObjectName("data_root")
        self.data_root_edit.textChanged.connect(self._invalidate_preflight)
        root_row.addWidget(self.data_root_edit, 1)
        self.browse_button = QPushButton("选择…")
        self.browse_button.setFixedHeight(28)
        self.browse_button.clicked.connect(self.choose_data_root)
        root_row.addWidget(self.browse_button)
        form.addRow("数据根目录：", root_row)

        # Row 1: 项目 + 受试者编码
        row1 = QGridLayout()
        row1.setHorizontalSpacing(7)
        row1.setVerticalSpacing(0)
        self.project_combo = QComboBox()
        self.project_combo.setObjectName("project")
        for project in PROJECTS:
            self.project_combo.addItem(
                project["project_name"],
                dict(project),
            )
        self.project_combo.setCurrentIndex(0)
        row1.addWidget(QLabel("项目："), 0, 0)
        row1.addWidget(self.project_combo, 0, 1)

        self.subject_code_edit = QLineEdit("001")
        self.subject_code_edit.setObjectName("subject_code")
        self.subject_code_edit.setMaxLength(3)
        self.subject_code_edit.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"\d{3}"), self)
        )
        self.subject_code_edit.editingFinished.connect(self.normalize_subject_code)
        self.subject_code_edit.textChanged.connect(self._update_start_button)
        row1.addWidget(QLabel("受试者编码："), 0, 2)
        row1.addWidget(self.subject_code_edit, 0, 3)

        self.day_spin = QSpinBox()
        self.day_spin.setObjectName("day")
        self.day_spin.setRange(1, 9999)
        self.day_spin.setValue(1)
        self.day_spin.setMinimumWidth(78)
        self.day_spin.setMaximumWidth(105)
        row1.addWidget(QLabel("第几天："), 0, 4)
        row1.addWidget(self.day_spin, 0, 5)
        row1.setColumnStretch(1, 1)
        row1.setColumnStretch(3, 1)
        row1.setColumnStretch(5, 0)
        form.addRow(row1)

        # Row 2: 工况 + 重复轮次
        row2 = QGridLayout()
        row2.setHorizontalSpacing(7)
        row2.setVerticalSpacing(0)
        self.condition_combo = QComboBox()
        self.condition_combo.setObjectName("condition")
        self.condition_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.condition_combo.setMinimumContentsLength(12)
        self.condition_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self.condition_combo.view().setMinimumWidth(620)
        self._populate_condition_combo(preferred_code="WALK_LEVEL")
        row2.addWidget(QLabel("工况："), 0, 0)
        row2.addWidget(self.condition_combo, 0, 1)

        self.repeat_spin = QSpinBox()
        self.repeat_spin.setObjectName("repeat_index")
        self.repeat_spin.setRange(1, 9999)
        self.repeat_spin.setValue(1)
        self.repeat_spin.setMinimumWidth(78)
        self.repeat_spin.setMaximumWidth(105)
        row2.addWidget(QLabel("重复轮次："), 0, 2)
        row2.addWidget(self.repeat_spin, 0, 3)
        row2.setColumnStretch(1, 1)
        row2.setColumnStretch(3, 0)
        form.addRow(row2)
        for compact_control in (
            self.data_root_edit,
            self.project_combo,
            self.subject_code_edit,
            self.day_spin,
            self.condition_combo,
            self.repeat_spin,
        ):
            compact_control.setFixedHeight(28)
        controls_layout.addWidget(metadata_box)

        experiment_box = QGroupBox("详细信息")
        experiment_layout = QHBoxLayout(experiment_box)
        experiment_layout.setContentsMargins(8, 11, 8, 6)
        experiment_layout.setSpacing(8)
        self.experiment_metadata_button = QPushButton("填写 / 修改…")
        self.experiment_metadata_button.setObjectName("edit_experiment_metadata")
        self.experiment_metadata_button.clicked.connect(self.edit_experiment_metadata)
        self.experiment_metadata_button.setFixedHeight(30)
        experiment_layout.addWidget(self.experiment_metadata_button)
        self.experiment_metadata_summary = QLabel("未填写；不影响采集")
        self.experiment_metadata_summary.setObjectName("experiment_metadata_summary")
        self.experiment_metadata_summary.setWordWrap(True)
        self.experiment_metadata_summary.setMaximumHeight(32)
        experiment_layout.addWidget(self.experiment_metadata_summary, 1)
        controls_layout.addWidget(experiment_box)

        # ── Trial buttons ──
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.connect_all_button = QPushButton("全部连接")
        self.connect_all_button.setObjectName("connect_all")
        self.connect_all_button.clicked.connect(self._toggle_connect_all)
        self.connect_all_button.setMinimumWidth(105)
        self.connect_all_button.setFixedHeight(34)
        buttons.addWidget(self.connect_all_button)
        self.start_button = QPushButton("开始写盘")
        self.start_button.setObjectName("start_trial")
        self.start_button.setStyleSheet(
            "QPushButton { font-weight: 600; padding: 4px 8px; color: #ffffff; background: #0f766e; border: 1px solid #115e59; border-radius: 4px; }"
        )
        self.start_button.clicked.connect(self._toggle_write)
        self.start_button.setMinimumWidth(105)
        self.start_button.setFixedHeight(34)
        buttons.addWidget(self.start_button)
        controls_layout.addLayout(buttons)

        # ── Device Connection Area ──
        connection_box = QGroupBox("设备连接")
        connection_layout = QGridLayout(connection_box)
        connection_layout.setContentsMargins(8, 11, 8, 6)
        connection_layout.setHorizontalSpacing(6)
        connection_layout.setVerticalSpacing(1)
        connection_layout.addWidget(QLabel("模态（点击设置）"), 0, 0)
        status_header = QLabel("状态")
        status_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        connection_layout.addWidget(status_header, 0, 1)
        connection_layout.addWidget(QLabel("操作"), 0, 2)
        connection_layout.setColumnStretch(0, 1)
        connection_layout.setColumnStretch(1, 0)
        connection_layout.setColumnStretch(2, 0)
        connection_layout.setColumnMinimumWidth(1, 52)
        connection_layout.setColumnMinimumWidth(2, 158)

        self._device_profile_label = QLabel()
        self._device_profile_label.setObjectName("device_profile")
        self._device_profile_label.setWordWrap(True)
        self._device_profile_label.setMaximumHeight(38)
        connection_legend = QLabel(
            "<span style='color:#64748B'>● 未连接</span>&nbsp;&nbsp;"
            "<span style='color:#2563EB'>● 连接中</span>&nbsp;&nbsp;"
            "<span style='color:#D97706'>● 等数据</span>&nbsp;&nbsp;"
            "<span style='color:#15803D'>● 正常</span>&nbsp;&nbsp;"
            "<span style='color:#EA580C'>● 异常</span>&nbsp;&nbsp;"
            "<span style='color:#B91C1C'>● 故障</span>"
        )
        connection_legend.setObjectName("connection_status_legend")
        connection_legend.setTextFormat(Qt.TextFormat.RichText)
        connection_legend.setMaximumHeight(19)
        connection_legend.setToolTip(
            "灰：未连接；蓝：正在连接或断开；黄：已连接但尚无数据；"
            "绿：持续收到正常数据；橙：数据中断、丢包、队列压力或健康降级；"
            "红：设备故障或连接失败。"
        )
        connection_layout.addWidget(
            connection_legend,
            len(CONNECTION_ROWS) + 2,
            0,
            1,
            3,
        )
        connection_layout.addWidget(
            self._device_profile_label,
            len(CONNECTION_ROWS) + 3,
            0,
            1,
            3,
        )
        self._connection_status_legend = connection_legend

        # Per-row groups（动捕 Marker 与六维力测力台合并为一项）
        for row_idx, group in enumerate(CONNECTION_ROWS, start=1):
            row_key = group[0] if len(group) == 1 else XINGYING_GROUP_KEY
            display_name = CONNECTION_ROW_DISPLAY_NAMES.get(
                group, MODALITY_DISPLAY_NAMES[group[0]]
            )
            configure_btn = QPushButton(display_name)
            configure_btn.setObjectName(f"configure_{row_key}")
            configure_btn.setProperty("buttonRole", "deviceConfig")
            configure_btn.setFixedHeight(27)
            configure_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            configure_btn.setToolTip(f"设置{display_name}设备参数（自动保存）")
            if len(group) == 1:
                configure_btn.clicked.connect(
                    lambda _checked=False, selected=group[0]: self.edit_modality_device_settings(selected)
                )
            else:
                configure_btn.clicked.connect(
                    lambda _checked=False: self._edit_xingying_group_settings()
                )
            connection_layout.addWidget(configure_btn, row_idx, 0)
            self._configure_buttons[row_key] = configure_btn

            status_label = QLabel("")
            status_label.setObjectName(f"connect_status_{row_key}")
            status_label.setFixedSize(16, 16)
            self._style_connection_indicator(status_label, "未连接")
            status_label.setToolTip("状态：未连接")
            connection_layout.addWidget(
                status_label,
                row_idx,
                1,
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
            self._connect_status_labels[row_key] = status_label

            btn_container = QHBoxLayout()
            btn_container.setContentsMargins(0, 0, 0, 0)
            btn_container.setSpacing(4)
            connect_btn = QPushButton("连接")
            connect_btn.setObjectName(f"connect_{row_key}")
            connect_btn.setProperty("buttonRole", "connect")
            disconnect_btn = QPushButton("断开")
            disconnect_btn.setObjectName(f"disconnect_{row_key}")
            disconnect_btn.setProperty("buttonRole", "disconnect")

            def _make_connect_handler(grp: tuple[str, ...]):
                return lambda: self._connect_group(grp)
            def _make_disconnect_handler(grp: tuple[str, ...]):
                return lambda: self._disconnect_group(grp)

            connect_btn.clicked.connect(_make_connect_handler(group))
            disconnect_btn.clicked.connect(_make_disconnect_handler(group))
            disconnect_btn.setEnabled(False)
            connect_btn.setMinimumWidth(72)
            disconnect_btn.setMinimumWidth(72)
            connect_btn.setFixedHeight(28)
            disconnect_btn.setFixedHeight(28)

            btn_container.addWidget(connect_btn)
            btn_container.addWidget(disconnect_btn)
            connection_layout.addLayout(btn_container, row_idx, 2)
            self._connect_buttons[row_key] = connect_btn
            self._disconnect_buttons[row_key] = disconnect_btn

        # ── 按钮标签行（非数据模态：连接 = 启用全局键盘钩子监听逗号键） ──
        button_row_idx = len(CONNECTION_ROWS) + 1
        button_label_btn = QPushButton(BUTTON_ROW_DISPLAY_NAME)
        button_label_btn.setObjectName(f"configure_{BUTTON_ROW_KEY}")
        button_label_btn.setProperty("buttonRole", "deviceConfig")
        button_label_btn.setFixedHeight(27)
        button_label_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_label_btn.setToolTip("按钮无需设置；按下 USB 按钮（逗号键）即记录一次标记。")
        button_label_btn.clicked.connect(
            lambda _checked=False: self.statusBar().showMessage(
                "按钮标签：按下 USB 按钮（逗号键）即记录一次标记，无需额外设置。", 4000
            )
        )
        connection_layout.addWidget(button_label_btn, button_row_idx, 0)
        self._configure_buttons[BUTTON_ROW_KEY] = button_label_btn

        button_status_label = QLabel("")
        button_status_label.setObjectName(f"connect_status_{BUTTON_ROW_KEY}")
        button_status_label.setFixedSize(16, 16)
        self._style_connection_indicator(button_status_label, "未连接")
        button_status_label.setToolTip("状态：未连接")
        connection_layout.addWidget(
            button_status_label,
            button_row_idx,
            1,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        self._connect_status_labels[BUTTON_ROW_KEY] = button_status_label

        button_btn_container = QHBoxLayout()
        button_btn_container.setContentsMargins(0, 0, 0, 0)
        button_btn_container.setSpacing(4)
        button_connect_btn = QPushButton("连接")
        button_connect_btn.setObjectName(f"connect_{BUTTON_ROW_KEY}")
        button_connect_btn.setProperty("buttonRole", "connect")
        button_disconnect_btn = QPushButton("断开")
        button_disconnect_btn.setObjectName(f"disconnect_{BUTTON_ROW_KEY}")
        button_disconnect_btn.setProperty("buttonRole", "disconnect")
        button_connect_btn.clicked.connect(self._start_button_marker)
        button_disconnect_btn.clicked.connect(self._stop_button_marker)
        button_disconnect_btn.setEnabled(False)
        button_connect_btn.setMinimumWidth(72)
        button_disconnect_btn.setMinimumWidth(72)
        button_connect_btn.setFixedHeight(28)
        button_disconnect_btn.setFixedHeight(28)
        button_btn_container.addWidget(button_connect_btn)
        button_btn_container.addWidget(button_disconnect_btn)
        connection_layout.addLayout(button_btn_container, button_row_idx, 2)
        self._connect_buttons[BUTTON_ROW_KEY] = button_connect_btn
        self._disconnect_buttons[BUTTON_ROW_KEY] = button_disconnect_btn

        controls_layout.addWidget(connection_box)

        # ── Health Table ──
        health_box = QGroupBox("设备健康与样本计数")
        health_layout = QVBoxLayout(health_box)
        health_layout.setContentsMargins(8, 11, 8, 6)
        health_layout.setSpacing(2)
        self.health_table = QTableWidget(len(HEALTH_ROWS), 6)
        self.health_table.setObjectName("health_table")
        self.health_table.setHorizontalHeaderLabels(
            ["模态", "样本/帧", "设置频率", "实际速率", "丢包", "同步"]
        )
        self.health_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.health_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.health_table.setAlternatingRowColors(True)
        self.health_table.verticalHeader().setVisible(False)
        for row, modality in enumerate(HEALTH_ROWS):
            self.health_table.setItem(
                row,
                HEALTH_COLUMN_MODALITY,
                QTableWidgetItem(MODALITY_DISPLAY_NAMES[modality]),
            )
            self.health_table.setItem(row, HEALTH_COLUMN_SAMPLE_COUNT, QTableWidgetItem("0"))
            self.health_table.setItem(row, HEALTH_COLUMN_NOMINAL_RATE, QTableWidgetItem("-"))
            self.health_table.setItem(row, HEALTH_COLUMN_RATE, QTableWidgetItem("-"))
            self.health_table.setItem(row, HEALTH_COLUMN_DROPPED, QTableWidgetItem("-"))
            sync_placeholder = QTableWidgetItem("—")
            sync_placeholder.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.health_table.setItem(row, HEALTH_COLUMN_SYNC, sync_placeholder)

        self.health_table.resizeColumnsToContents()
        for row in range(self.health_table.rowCount()):
            self.health_table.setRowHeight(row, 22)
        health_header = self.health_table.horizontalHeader()
        health_header.setFixedHeight(25)
        health_header.setSectionResizeMode(
            HEALTH_COLUMN_MODALITY,
            QHeaderView.ResizeMode.Stretch,
        )
        for column in (
            HEALTH_COLUMN_SAMPLE_COUNT,
            HEALTH_COLUMN_NOMINAL_RATE,
            HEALTH_COLUMN_RATE,
            HEALTH_COLUMN_DROPPED,
        ):
            health_header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        health_header.setSectionResizeMode(HEALTH_COLUMN_SYNC, QHeaderView.ResizeMode.Fixed)
        self.health_table.setColumnWidth(HEALTH_COLUMN_SYNC, 56)
        self.health_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        compact_height = (
            self.health_table.horizontalHeader().height()
            + sum(self.health_table.rowHeight(row) for row in range(self.health_table.rowCount()))
            + self.health_table.frameWidth() * 2
            + 4
        )
        self.health_table.setFixedHeight(compact_height)
        health_layout.addWidget(self.health_table)
        health_box.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        controls_layout.addWidget(health_box)
        controls_layout.addStretch(1)

        # ── Toast overlay for alerts ──
        self._toast_label = DismissibleToastLabel(self)
        self._toast_label.setObjectName("toast")
        self._toast_label.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        self._toast_label.setWordWrap(True)
        self._toast_label.setMinimumWidth(340)
        self._toast_label.setMaximumWidth(520)
        self._toast_label.setMinimumHeight(48)
        self._toast_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._toast_label.setVisible(False)
        self._toast_label.setContentsMargins(14, 9, 14, 9)
        self._toast_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction
        )
        self._toast_label.dismissed.connect(self._hide_toast)
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._hide_toast)
        controls_scroll.setWidget(controls)
        body.addWidget(controls_scroll)

        # ── Dockable preview workspace ──
        preview_workspace = PreviewWorkspace(self)
        self.preview_workspace = preview_workspace
        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")

        us_grid = QGroupBox("超声 · 4 通道当前单帧")
        us_grid.setObjectName("ultrasound_grid")
        us_grid.setMinimumHeight(120)
        us_grid_layout = QGridLayout(us_grid)
        us_grid_layout.setContentsMargins(0, 0, 0, 0)
        for i in range(4):
            plot = HoverDetailsPlotWidget(
                title=f"超声通道 {i + 1} · 当前帧"
            )
            plot.setObjectName(f"ultrasound_preview_ch{i}")
            plot.setBackground("w")
            plot.setXRange(0, ULTRASOUND_PREVIEW_SAMPLES - 1, padding=0)
            plot.setLimits(
                xMin=0, xMax=ULTRASOUND_PREVIEW_SAMPLES - 1,
                minXRange=ULTRASOUND_PREVIEW_SAMPLES - 1,
                maxXRange=ULTRASOUND_PREVIEW_SAMPLES - 1,
            )
            plot.setMouseEnabled(x=False, y=False)
            plot.setLabel("bottom", "单帧采样点")
            plot.showGrid(x=True, y=True, alpha=0.2)
            curve = plot.plot(pen=pg.mkPen("#0f766e", width=1.2))
            curve.setData(self._us_x, np.full(ULTRASOUND_PREVIEW_SAMPLES, np.nan, dtype=np.float64))
            self._us_plots.append(plot)
            self._us_curves.append(curve)
            us_grid_layout.addWidget(plot, i // 2, i % 2)
        preview_workspace.register_panel(
            "ultrasound",
            "超声数据",
            us_grid,
        )

        imu_grid = QGroupBox("IMU · 加速度 3 轴循环帧")
        imu_grid.setObjectName("imu_ring_grid")
        imu_grid.setMinimumHeight(120)
        imu_layout = QHBoxLayout(imu_grid)
        imu_layout.setContentsMargins(0, 0, 0, 0)
        _sensor_display = ("左腿", "右腿", "盆骨")
        for sensor_idx, sensor_label in enumerate(_IMU_SENSOR_LABELS):
            plot = HoverDetailsPlotWidget()
            plot.setObjectName(f"imu_ring_{sensor_label}")
            plot.addLegend(offset=(-1, 1))
            for axis_name in _IMU_AXIS_NAMES:
                trace_label = f"{sensor_label}_{axis_name}"
                color = _IMU_AXIS_COLORS.get(axis_name, "#888888")
                trace = RingTrace(
                    plot, color,
                    f"IMU{sensor_idx+1} {_sensor_display[sensor_idx]} {axis_name}",
                    capacity=250,
                )
                self._imu_traces[trace_label] = trace
            plot.setYRange(-10, 10, padding=0)
            plot.setLimits(yMin=-10, yMax=10, minYRange=20, maxYRange=20)
            plot.setMouseEnabled(x=False, y=False)
            imu_layout.addWidget(plot, 1)
        # Pre-register IMU Y range so auto-scale doesn't override the fixed range.
        self._preview_y_ranges["imu"] = (-10.0, 10.0)
        preview_workspace.register_panel("imu", "IMU 数据", imu_grid)

        enc_grid = QGroupBox("电机编码器 · 每侧同图显示位置 / 速度 / 估算扭矩")
        enc_grid.setObjectName("encoder_ring_grid")
        enc_grid.setMinimumHeight(120)
        enc_layout = QHBoxLayout(enc_grid)
        enc_layout.setContentsMargins(0, 0, 0, 0)
        for side_key, side_name in (("left", "左侧"), ("right", "右侧")):
            plot = HoverDetailsPlotWidget()
            plot.setObjectName(f"encoder_ring_{side_key}")
            legend = plot.addLegend(offset=(5, 5))
            shared_cursor = None
            for metric, metric_name, unit, color in _ENCODER_METRICS:
                label = f"{side_key}_{metric}"
                trace = RingTrace(
                    plot,
                    color,
                    f"{side_name}电机",
                    capacity=500,
                )
                if shared_cursor is None:
                    shared_cursor = trace.cursor_line
                else:
                    plot.removeItem(trace.cursor_line)
                    trace.cursor_line = shared_cursor
                legend.addItem(trace.curve, f"{metric_name} ({unit})")
                self._enc_traces[label] = trace
            lower, upper = _ENCODER_SHARED_Y_RANGE
            span = upper - lower
            plot.setTitle(f"{side_name}电机编码器")
            plot.setYRange(lower, upper, padding=0)
            plot.setLimits(
                yMin=lower,
                yMax=upper,
                minYRange=span,
                maxYRange=span,
            )
            plot.setLabel("left", "数值")
            plot.setMouseEnabled(x=False, y=False)
            enc_layout.addWidget(plot, 1)
        self._preview_y_ranges["encoder"] = _ENCODER_SHARED_Y_RANGE
        preview_workspace.register_panel(
            "encoder",
            "电机编码器数据",
            enc_grid,
        )

        self._xingying_status_panel = XingYingRecordingPanel()
        preview_workspace.register_panel(
            "xingying",
            "动捕 + 测力台（XINGYING）",
            self._xingying_status_panel,
        )

        emg_grid = QGroupBox("表面肌电 EMG · 每通道一个窗口")
        emg_grid.setMinimumHeight(120)
        self._emg_grid_layout = QVBoxLayout(emg_grid)
        self._emg_grid_layout.setContentsMargins(0, 0, 0, 0)
        placeholder = QLabel("等待 EMG 数据…（通道窗口将随配置自动生成）")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("QLabel { color: #6b7280; padding: 16px; }")
        self._emg_grid_layout.addWidget(placeholder)
        self._emg_grid_content = placeholder
        preview_workspace.register_panel("emg", "EMG 数据", emg_grid)

        self._elapsed_timer = ElapsedTimerPanel()
        preview_workspace.register_panel(
            "timer",
            "计时器",
            self._elapsed_timer,
            visible_by_default=False,
        )

        body.addWidget(preview_workspace)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([630, 1270])
        outer.addWidget(body, 1)

        self.setCentralWidget(central)
        preview_workspace.focus_mode_requested.connect(self._set_preview_focus_mode)
        preview_workspace.restore_layout(self._settings.preview_workspace_state)
        preview_workspace.layout_changed.connect(self._save_preview_workspace_layout)
        self.statusBar().showMessage("就绪；原始写盘仅由 Collector Worker 负责。")

        self._configuration_widgets = (
            self.data_root_edit,
            self.browse_button,
            self.project_combo,
            self.subject_code_edit,
            self.condition_combo,
            self.repeat_spin,
            self.experiment_metadata_button,
            self.connect_all_button,
            *self._configure_buttons.values(),
        )
        self._render_device_profile()

    # ── Profile / Device Metadata ──────────────────────────────────────

    def _selected_device_profile_key(self) -> str:
        return self._settings.device_profile_key

    def _populate_nominal_rates(self) -> None:
        """预填健康表「设置频率」列：直接读设备配置的标称频率。

        该列是静态配置值，未连接设备时也应可见（例如动捕/测力台硬件模式
        100 Hz）。运行时 health 事件仍会以实际设备标称值覆盖它。
        """

        try:
            profile = load_device_profile(self._selected_device_profile_key())
            devices = profile.by_modality()
        except Exception as exc:  # noqa: BLE001 - 预填失败不应阻断 UI 启动
            LOG.warning("预填设置频率失败: %s", exc)
            return
        for modality, row in self._health_rows.items():
            device = devices.get(modality)
            if device is None:
                continue
            nominal = _nominal_rate_from_device(device)
            if nominal is not None:
                self.health_table.item(row, HEALTH_COLUMN_NOMINAL_RATE).setText(
                    f"{nominal:.1f} Hz"
                )

    @staticmethod
    def _condition_tooltip(condition: Mapping[str, Any]) -> str:
        parameters = dict(condition.get("parameters", {}))
        parts = [
            (
                f"建议 Trial：{parameters['recommended_trial_count']}"
                if "recommended_trial_count" in parameters
                else ""
            ),
            (
                f"目标有效时长：{parameters['target_effective_duration_s']} s"
                if "target_effective_duration_s" in parameters
                else ""
            ),
            (
                "有效时长："
                f"{parameters['effective_duration_s_min']}–"
                f"{parameters['effective_duration_s_max']} s"
                if "effective_duration_s_min" in parameters
                and "effective_duration_s_max" in parameters
                else ""
            ),
            (
                f"速度：{parameters['speed_mps']} m/s"
                if "speed_mps" in parameters
                else ""
            ),
            (
                f"坡度：{parameters['slope_deg']}°"
                if "slope_deg" in parameters
                else ""
            ),
            (
                f"负重：{parameters['load_kg']} kg"
                if "load_kg" in parameters
                else ""
            ),
        ]
        return "\n".join(part for part in parts if part)

    def _populate_condition_combo(self, *, preferred_code: str | None = None) -> None:
        project = self.project_combo.currentData()
        project_code = (
            str(project.get("project_code") or "").strip().upper()
            if isinstance(project, dict)
            else ""
        )
        visible = [
            condition
            for condition in CONDITIONS
            if project_accepts_condition_level(
                project_code,
                condition.get("condition_level"),
            )
        ]
        previous_blocked = self.condition_combo.blockSignals(True)
        try:
            self.condition_combo.clear()
            for condition in visible:
                # The stable English code remains in item data and the
                # Manifest, while operators see only the Chinese condition.
                self.condition_combo.addItem(
                    str(condition["condition_name"]),
                    dict(condition),
                )
                self.condition_combo.setItemData(
                    self.condition_combo.count() - 1,
                    self._condition_tooltip(condition),
                    Qt.ItemDataRole.ToolTipRole,
                )
            selected_index = next(
                (
                    index
                    for index, condition in enumerate(visible)
                    if condition["condition_code"] == preferred_code
                ),
                0,
            )
            if visible:
                self.condition_combo.setCurrentIndex(selected_index)
        finally:
            self.condition_combo.blockSignals(previous_blocked)

    @Slot()
    def _handle_project_changed(self, *_args: object) -> None:
        current = self.condition_combo.currentData()
        preferred_code = (
            str(current.get("condition_code") or "")
            if isinstance(current, dict)
            else None
        )
        self._populate_condition_combo(preferred_code=preferred_code)
        self._activate_selected_metadata_identity()
        self._handle_metadata_condition_changed()
        self._update_start_button()

    def _render_device_profile(self) -> None:
        hardware = self._selected_device_profile_key() == "hardware"

        if hardware:
            self._device_profile_label.setText(
                "真实设备模式：超声 / Xsens / Teensy / XING；"
                "同步脉冲为模拟信号。"
            )
            self._device_profile_label.setToolTip(
                "Raw Ethernet 超声 + Xsens MTw IMU + Teensy 编码器 + "
                "XING/Nokov 动捕 Marker、EMG 与六维力测力台；"
                "同步脉冲仍为模拟台架信号。"
            )
            self._device_profile_label.setStyleSheet("color:#842029;font-weight:600;")
        else:
            self._device_profile_label.setText(
                "模拟设备模式；保存任一设备设置后切换为真实设备。"
            )
            self._device_profile_label.setToolTip(
                "当前为自动化测试用模拟设备；正常启动并保存任一设备设置后"
                "切换为真实设备模式。"
            )
            self._device_profile_label.setStyleSheet("")

    @Slot(str)
    def edit_modality_device_settings(self, modality: str) -> None:
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality: {modality!r}")
        if modality in self._preview_workers or (
            self._selected_device_profile_key() != "hardware" and self._preview_workers
        ):
            QMessageBox.information(
                self,
                "请先断开设备",
                "修改该设备设置前，请先断开对应预览连接；从模拟模式切换时需全部断开。",
            )
            return
        if self._configuration_locked or self._preflight_busy:
            QMessageBox.information(self, "当前不可修改", "采集或预检期间不能修改设备设置。")
            return

        current = self._settings.hardware_device_overrides.get(modality, {})
        dialog_type = DEVICE_SETTINGS_DIALOGS[modality]
        dialog = dialog_type(current, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._settings.set_hardware_device_override(modality, dialog.validated_override)
        # Saving any per-device settings is an explicit request to use the
        # laboratory hardware profile. The choice and values are both synced
        # immediately by SharedAppSettings and survive process restarts.
        self._settings.set_device_profile_key("hardware")
        self._invalidate_preflight()
        self._render_device_profile()
        self._populate_nominal_rates()
        display = MODALITY_DISPLAY_NAMES[modality]
        self.statusBar().showMessage(
            f"{display}设备设置已保存；下次启动将自动恢复。", 8000
        )
        LOG.info("%s 设备设置已保存并持久化", modality)

    @Slot()
    def _edit_xingying_group_settings(self) -> None:
        """配置合并后的「动捕 Marker + 六维力测力台」设备参数。"""
        if (
            "mocap" in self._preview_workers
            or "force_plate" in self._preview_workers
            or (
                self._selected_device_profile_key() != "hardware"
                and self._preview_workers
            )
        ):
            QMessageBox.information(
                self,
                "请先断开设备",
                "修改该设备设置前，请先断开对应预览连接；从模拟模式切换时需全部断开。",
            )
            return
        if self._configuration_locked or self._preflight_busy:
            QMessageBox.information(self, "当前不可修改", "采集或预检期间不能修改设备设置。")
            return

        current_mocap = self._settings.hardware_device_overrides.get("mocap", {})
        current_force = self._settings.hardware_device_overrides.get("force_plate", {})
        dialog = MocapForcePlateDeviceSettingsDialog(current_mocap, current_force, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        for modality, override in dialog.validated_override.items():
            self._settings.set_hardware_device_override(modality, override)
        # Saving any per-device settings is an explicit request to use the
        # laboratory hardware profile. The choice and values are both synced
        # immediately by SharedAppSettings and survive process restarts.
        self._settings.set_device_profile_key("hardware")
        self._invalidate_preflight()
        self._render_device_profile()
        self._populate_nominal_rates()
        self.statusBar().showMessage(
            "动捕与测力台设备设置已保存；下次启动将自动恢复。", 8000
        )
        LOG.info("mocap + force_plate 设备设置已保存并持久化")

    @Slot()
    def choose_data_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "选择外骨骼数据根目录",
            self.data_root_edit.text(), QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.set_data_root(selected)

    def set_data_root(self, data_root: str | Path) -> Path:
        normalized = self._settings.set_data_root(data_root)
        self.data_root_edit.setText(str(normalized))
        return normalized

    @Slot()
    def _invalidate_preflight(self) -> None:
        if self._worker is not None or not self._preflight_ready:
            return
        self._preflight_ready = False
        for row in self._health_rows.values():
            self.health_table.item(row, HEALTH_COLUMN_MODALITY).setToolTip("")
        self._last_health_status.clear()
        self._set_trial_state("IDLE")
        self.statusBar().showMessage("配置或存储目标已变化，请重新连接设备。")
        self._update_start_button()

    @property
    def experiment_metadata(self) -> TrialExperimentMetadata:
        return self._experiment_metadata

    def set_experiment_metadata(self, metadata: TrialExperimentMetadata | Mapping[str, Any]) -> None:
        self._experiment_metadata = TrialExperimentMetadata.model_validate(metadata)
        if self._metadata_identity_key is not None:
            self._experiment_metadata_by_identity[self._metadata_identity_key] = self._experiment_metadata
        self._render_experiment_metadata_summary()

    @staticmethod
    def _experiment_metadata_value_count(metadata: TrialExperimentMetadata) -> int:
        payload = metadata.model_dump(mode="python")
        def count_values(value: object) -> int:
            if isinstance(value, dict):
                return sum(count_values(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(count_values(item) for item in value)
            return int(value is not None)
        return count_values(payload)

    def _render_experiment_metadata_summary(self, *, transition: str | None = None) -> None:
        value_count = self._experiment_metadata_value_count(self._experiment_metadata)
        identity = "未识别受试者" if self._metadata_identity_key is None else f"{self._metadata_identity_key[0]}/{self._metadata_identity_key[1]}"
        text = f"{identity} 已填写 {value_count} 项；同一受试者后续 Trial 默认沿用" if value_count else f"{identity} 未填写；不影响采集"
        if transition:
            text = f"{transition}；{text}"
        self.experiment_metadata_summary.setText(text)

    def _selected_metadata_identity(self) -> tuple[str, str] | None:
        project = self.project_combo.currentData()
        subject_code = self.subject_code_edit.text().strip()
        if not isinstance(project, dict):
            return None
        project_code = str(project.get("project_code") or "").strip().upper()
        if project_code not in SUPPORTED_PROJECT_CODES:
            return None
        if not subject_code.isascii() or not subject_code.isdigit() or len(subject_code) != 3:
            return None
        return project_code, subject_code

    @Slot()
    def _activate_selected_metadata_identity(self, *_args: object) -> None:
        selected = self._selected_metadata_identity()
        if selected is None or selected == self._metadata_identity_key:
            return
        previous_key = self._metadata_identity_key
        previous_metadata = self._experiment_metadata
        if previous_key is not None:
            self._experiment_metadata_by_identity[previous_key] = previous_metadata
        restored = self._experiment_metadata_by_identity.get(selected)
        self._metadata_identity_key = selected
        if restored is None:
            self._experiment_metadata = TrialExperimentMetadata()
            transition = (
                "已切换受试者，实验元数据已清空以避免串写"
                if previous_key is not None and self._experiment_metadata_value_count(previous_metadata)
                else None
            )
            if transition:
                self._append_alert(
                    f"{transition}：{previous_key[0]}/{previous_key[1]} → "
                    f"{selected[0]}/{selected[1]}。切回原受试者时会恢复其会话缓存。"
                )
        else:
            self._experiment_metadata = restored
            transition = "已恢复该受试者在本次会话中的实验元数据"
        self._render_experiment_metadata_summary(transition=transition)

    def _selected_condition_code(self) -> str | None:
        condition = self.condition_combo.currentData()
        if not isinstance(condition, dict):
            return None
        value = str(condition.get("condition_code") or "").strip()
        return value or None

    @Slot()
    def _handle_metadata_condition_changed(self, *_args: object) -> None:
        selected = self._selected_condition_code()
        previous = self._metadata_condition_code
        if selected is None or selected == previous:
            return
        self._metadata_condition_code = selected
        had_condition_values = bool(
            self._experiment_metadata_value_count(
                TrialExperimentMetadata(
                    measured_condition=self._experiment_metadata.measured_condition,
                    trial_notes=self._experiment_metadata.trial_notes,
                )
            )
        )
        self._experiment_metadata = self._experiment_metadata.model_copy(
            update={"measured_condition": MeasuredConditionMetadata(), "trial_notes": None}
        )
        for identity, cached in tuple(self._experiment_metadata_by_identity.items()):
            self._experiment_metadata_by_identity[identity] = cached.model_copy(
                update={"measured_condition": MeasuredConditionMetadata(), "trial_notes": None}
            )
        if self._metadata_identity_key is not None:
            self._experiment_metadata_by_identity[self._metadata_identity_key] = self._experiment_metadata
        transition = "工况已切换，实测工况与 Trial 备注已清空"
        self._render_experiment_metadata_summary(transition=transition)
        if had_condition_values:
            self._append_alert(
                f"{transition}：{previous or '未选择'} → {selected}；人口学与探头固定信息保留。"
            )
        self.statusBar().showMessage(f"{transition}（{previous or '未选择'} → {selected}）。", 8000)

    def _clear_one_trial_metadata(self) -> None:
        probe = self._experiment_metadata.ultrasound_probe
        had_one_trial_values = bool(
            self._experiment_metadata.trial_notes is not None or probe.probe_reapplied is not None
        )
        self._experiment_metadata = self._experiment_metadata.model_copy(
            update={"ultrasound_probe": probe.model_copy(update={"probe_reapplied": None}), "trial_notes": None}
        )
        if self._metadata_identity_key is not None:
            self._experiment_metadata_by_identity[self._metadata_identity_key] = self._experiment_metadata
        transition = "上一 Trial 已结束，一次性备注与'重新贴探头'已清空" if had_one_trial_values else None
        self._render_experiment_metadata_summary(transition=transition)
        if transition:
            self._append_alert(f"{transition}；人口学、探头位置与固定方式仍保留。")
            self.statusBar().showMessage(f"{transition}；下一个 Trial 开始前请重新确认。", 8000)

    @Slot()
    def edit_experiment_metadata(self) -> None:
        if self._configuration_locked:
            return
        dialog = ExperimentMetadataDialog(self._experiment_metadata, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_experiment_metadata(dialog.metadata())

    @Slot()
    def normalize_subject_code(self) -> None:
        raw = self.subject_code_edit.text().strip()
        if raw.isascii() and raw.isdigit() and 1 <= len(raw) <= 3:
            self.subject_code_edit.setText(raw.zfill(3))
        self._update_start_button()

    def _subject_code(self) -> str:
        raw = self.subject_code_edit.text().strip()
        if not raw.isascii() or not raw.isdigit() or not 1 <= len(raw) <= 3:
            raise ValueError("受试者编码必须是三位数字")
        normalized = raw.zfill(3)
        self.subject_code_edit.setText(normalized)
        return normalized

    # ── Legacy Preflight (kept as smoke-test entry point) ───────────────

    @Slot()
    def run_preflight(self) -> None:
        """Legacy preflight — kept for test/smoke compatibility, not exposed as primary connect."""
        if self._worker is not None or self._preflight_worker is not None:
            return
        self._preflight_ready = False
        self._set_preflight_busy(True)
        self._set_trial_state("PREFLIGHT")
        self.statusBar().showMessage("正在独立进程中执行旧版设备预检（测试兼容）…")
        worker: PreflightWorkerHandle | None = None
        try:
            root_text = self.data_root_edit.text().strip()
            if not root_text:
                raise ValueError("数据根目录不能为空")
            root = self.set_data_root(root_text)
            profile_key = self._selected_device_profile_key()
            overrides = self._settings.hardware_device_overrides if profile_key == "hardware" else None
            if self._preflight_worker_factory is simulated_preflight_worker_factory:
                worker = self._preflight_worker_factory(root, profile_key, overrides)
            else:
                worker = self._preflight_worker_factory(root)
            self._preflight_worker = worker
            self._preflight_root = root
            self._preflight_result_handled = False
            self._preflight_empty_exit_polls = 0
            worker.start()
        except Exception:
            details = traceback.format_exc()
            if worker is not None:
                try:
                    worker.terminate(timeout=0.25)
                except Exception:
                    pass
                try:
                    if not self._preflight_worker_is_alive(worker):
                        worker.close()
                except Exception:
                    pass
            self._preflight_worker = None
            self._preflight_root = None
            self._set_preflight_busy(False)
            final_line = next(
                (line for line in reversed(details.splitlines()) if line.strip()),
                "设备预检进程启动失败",
            )
            self._apply_preflight_result(None, error=final_line)
            return
        self._preflight_timer.start()
        self.poll_preflight_worker()

    @Slot()
    def poll_preflight_worker(self) -> None:
        worker = self._preflight_worker
        if worker is None:
            self._preflight_timer.stop()
            return
        if not self._preflight_result_handled:
            try:
                result = worker.poll_result()
            except Exception:
                result = ("failed", traceback.format_exc())
            if result is not None:
                status, payload = result
                self._preflight_result_handled = True
                if status == "completed":
                    self._apply_preflight_result(payload)
                else:
                    self._apply_preflight_result(None, error=str(payload))
        if self._preflight_worker_is_alive(worker):
            self._preflight_empty_exit_polls = 0
            return
        if not self._preflight_result_handled:
            self._preflight_empty_exit_polls += 1
            if self._preflight_empty_exit_polls < 10:
                return
            self._preflight_result_handled = True
            self._apply_preflight_result(
                None,
                error=f"设备预检进程已退出但未返回结果（exitcode={self._preflight_worker_exitcode(worker)}）。",
            )
        try:
            worker.join(timeout=0)
            worker.close()
        except Exception as exc:
            self._append_alert(f"释放预检进程资源时出错：{type(exc).__name__}: {exc}")
        self._preflight_worker = None
        self._preflight_root = None
        self._preflight_timer.stop()
        self._set_preflight_busy(False)

    @staticmethod
    def _preflight_worker_is_alive(worker: PreflightWorkerHandle) -> bool:
        value = worker.is_alive
        return bool(value() if callable(value) else value)

    @staticmethod
    def _preflight_worker_exitcode(worker: PreflightWorkerHandle) -> int | None:
        value = worker.exitcode
        return value() if callable(value) else value

    def _apply_preflight_result(self, raw_result: object | None, *, error: str | None = None) -> None:
        report: CollectorPreflightReport | None = None
        try:
            if isinstance(raw_result, CollectorPreflightReport):
                report = raw_result
                if self._preflight_root is not None and report.data_root.resolve() != self._preflight_root.resolve():
                    raise ValueError("设备预检结果来自不同的数据根目录")
                if report.profile_key != self._selected_device_profile_key():
                    raise ValueError("设备预检结果来自不同的设备配置")
                reported = {modality: item.status for modality, item in report.devices.items()}
            elif isinstance(raw_result, Mapping):
                reported = {str(modality): str(status).strip().upper() for modality, status in raw_result.items()}
            else:
                reported = {}
                if error is None:
                    error = "设备预检进程返回了无效结果"
        except Exception as exc:
            reported = {}
            error = f"{type(exc).__name__}: {exc}"
        if error:
            final_line = next((line for line in reversed(error.splitlines()) if line.strip()), error)
            self._append_alert(f"设备预检失败：{final_line}")
        missing_or_failed: list[str] = []
        for modality in MODALITIES:
            status = reported.get(modality, "MISSING")
            row = self._health_rows[modality]
            modality_item = self.health_table.item(row, HEALTH_COLUMN_MODALITY)
            modality_item.setToolTip(f"设备状态：{status}")
            if report is not None and modality in report.devices:
                result = report.devices[modality]
                modality_item.setToolTip(
                    f"设备状态：{status}\n设备：{result.device_id}\n"
                    f"{result.message}\nchannels={result.channel_count} · raw={result.observed_raw_data}"
                )
                self.health_table.item(row, HEALTH_COLUMN_RATE).setText(
                    "-" if result.actual_rate_hz is None else f"{result.actual_rate_hz:.1f} Hz"
                )
            if modality in CRITICAL_MODALITIES and status != "READY":
                missing_or_failed.append(f"{modality}={status}")
        self._preflight_ready = not missing_or_failed and (report.ready if report is not None else True)
        if self._preflight_ready:
            self._set_trial_state("PREFLIGHT_READY")
            storage = ""
            if report is not None:
                storage = (
                    f" 可用空间 {report.disk_free_bytes / 1024**3:.2f} GiB；"
                    f"落盘探测 {report.measured_write_mib_s:.1f} MiB/s（阈值待真实超声最大速率确定）；"
                    f"耗时 {report.elapsed_s:.2f} s。"
                )
            self.statusBar().showMessage(f"六个必需模态已实际连接/准备/采样，同步上升沿已观测。{storage}", 8000)
        else:
            self._set_trial_state("FAILED")
            detail = "、".join(missing_or_failed) or "预检服务未返回设备状态"
            self._append_alert(f"关键设备未 READY：{detail}")
            self.statusBar().showMessage("设备预检失败；开始采集保持禁用。")
        self._update_start_button()

    # ── Per-modality Preview Connect / Disconnect ───────────────────────

    def _build_single_adapter_factory(self, modality: str) -> AdapterFactory:
        """Build a Windows-spawn-safe factory for exactly one modality."""
        if self._injected_preview_factory is not None:
            return self._injected_preview_factory

        profile_key = self._selected_device_profile_key()
        overrides = (
            self._settings.hardware_device_overrides
            if profile_key == "hardware"
            else {}
        )
        return ProfileModalityAdapterFactory(
            profile_key=profile_key,
            modality=modality,
            overrides=overrides,
        )

    def _get_modality_info(self, modality: str) -> tuple[str, bool]:
        """Return (device_id, simulated) for the given modality."""
        profile_key = self._selected_device_profile_key()
        profile = load_device_profile(profile_key)
        try:
            device = profile.by_modality()[modality]
        except KeyError as exc:
            raise RuntimeError(
                f"profile {profile_key!r} has no {modality!r} device"
            ) from exc
        simulated = profile_key == "simulated" or bool(
            getattr(device, "simulated", False)
        )
        return device.device_id, simulated

    @staticmethod
    def _connection_indicator_spec(
        status: str,
    ) -> tuple[str, str, str, str]:
        normalized = status.strip().upper()
        if (
            normalized in {"FAULT", "FAILED", "ERROR", "UNHEALTHY", "错误", "故障"}
            or any(token in status for token in ("失败", "异常退出"))
        ):
            return "red", "#EF4444", "#B91C1C", "故障"
        if normalized in {"DISCONNECTED", "CLOSED", "未连接", "UNKNOWN"}:
            return "neutral", "#94A3B8", "#64748B", "未连接"
        if (
            normalized in {"CONNECTING", "STOPPING"}
            or any(token in status for token in ("连接中", "断开中", "启动中"))
        ):
            return "blue", "#3B82F6", "#1D4ED8", "连接中"
        if (
            normalized in {"CONNECTED", "PREPARING", "PREVIEW_STARTING"}
            or "等待数据" in status
            or "无数据" in status
        ):
            return "yellow", "#FBBF24", "#D97706", "已连接，等待数据"
        if (
            normalized == "DEGRADED"
            or any(token in status for token in ("数据异常", "数据中断", "降级"))
        ):
            return "orange", "#F97316", "#C2410C", "数据异常"
        if normalized in {
            "READY",
            "RECORDING",
            "HEALTHY",
            "已连接",
            "数据正常",
        }:
            return "green", "#22C55E", "#15803D", "数据正常"
        return "neutral", "#94A3B8", "#64748B", "状态未知"

    @classmethod
    def _style_connection_indicator(cls, label: QLabel, status: str) -> str:
        indicator_state, fill, border, display = cls._connection_indicator_spec(
            status
        )
        label.setText("")
        label.setProperty("indicatorState", indicator_state)
        label.setProperty("indicatorStatus", display)
        label.setStyleSheet(
            f"QLabel {{ background-color:{fill}; border:2px solid {border}; "
            "border-radius:8px; }}"
        )
        return display

    def _set_preview_status(
        self,
        modality: str,
        status: str,
        device_id: str,
        simulated: bool,
        error: str | None = None,
        detail_lines: tuple[str, ...] = (),
    ) -> None:
        """Update the per-row UI status labels."""
        row_key = _MODALITY_ROW_KEY.get(modality, modality)
        if row_key in self._connect_status_labels:
            label = self._connect_status_labels[row_key]
            source = "模拟" if simulated else "真实"
            display_status = self._style_connection_indicator(label, status)
            tooltip_lines = [
                f"状态：{display_status}",
                f"来源：{source}",
            ]
            if device_id:
                tooltip_lines.append(f"设备 ID：{device_id}")
            tooltip_lines.extend(detail_lines)
            if error:
                tooltip_lines.append(f"详情：{error}")
            label.setToolTip("\n".join(tooltip_lines))
            label.setAccessibleName(f"{row_key} 状态：{display_status}")

    # ── XINGYING 远程触发（动捕 Marker + 测力台）─────────────────────────

    def _xingying_linked_enabled(self) -> bool:
        """仅在真实硬件配置下把动捕/测力台当作 XINGYING 远程触发处理。"""
        return self._selected_device_profile_key() == "hardware"

    def _xingying_remote_config(self) -> dict[str, Any]:
        """读取 mocap 设备参数中的远程控制配置。"""
        defaults = {
            "ip": "127.0.0.1",
            "port": 7060,
            "trigger_port": 7061,
            "database_path": (
                "C:/Users/Admin/Desktop/SEU_liangji/software/"
                "Exo_Collection_Calibration_XINGYING"
            ),
        }
        try:
            profile = load_device_profile(self._selected_device_profile_key())
            device = profile.by_modality()["mocap"]
        except (KeyError, RuntimeError):
            return defaults
        params = getattr(device, "parameters", None)
        data = params.model_dump(exclude_none=True) if hasattr(params, "model_dump") else {}
        return {
            "ip": str(data.get("remote_control_ip") or defaults["ip"]),
            "port": int(data.get("remote_control_port") or defaults["port"]),
            "trigger_port": int(
                data.get("remote_trigger_port") or defaults["trigger_port"]
            ),
            "database_path": str(
                data.get("database_path") or defaults["database_path"]
            ),
        }

    def _connect_xingying_remote(self) -> None:
        """连接 XINGYING 远程触发端口（动捕 Marker 与测力台绑定，不读取数据）。"""
        if self._xingying_remote is not None:
            self._append_alert("XINGYING 远程捕获已就绪，无需重复连接。")
            return
        if self._worker is not None:
            self._append_alert("Trial 进行中，无法连接远程捕获。")
            return
        cfg = self._xingying_remote_config()
        remote = XingYingRemoteCapture(ip=cfg["ip"], port=cfg["port"])
        self._xingying_remote = remote
        self._start_xingying_trigger_listener(cfg)
        available = set(load_device_profile(self._selected_device_profile_key()).by_modality())
        for modality in XINGYING_LINKED_MODALITIES:
            if modality not in available:
                continue
            device_id, simulated = self._get_modality_info(modality)
            self._preview_connected_modalities.add(modality)
            self._preview_connection_status[modality] = "已连接"
            self._set_preview_status(
                modality,
                "已连接",
                device_id,
                simulated,
                detail_lines=(
                    f"远程触发已就绪（{remote.ip}:{remote.port}）",
                    "开始采集时将触发 XINGYING 录制 .cap",
                ),
            )
            if self.preview_workspace is not None:
                self.preview_workspace.set_stream_state("xingying", "connected")
        if self._xingying_status_panel is not None:
            self._xingying_status_panel.set_connected(True)
        self._update_connect_button_state()
        self._update_start_button()
        self._append_alert(
            f"XINGYING 远程捕获已就绪（{remote.ip}:{remote.port}），"
            "动捕 Marker 与测力台已绑定。开始采集时将触发 .cap 录制。"
        )
        LOG.info("XINGYING 远程捕获已就绪 ip=%s port=%s", remote.ip, remote.port)

    def _disconnect_xingying_remote(self) -> None:
        """断开 XINGYING 远程触发，同步解除动捕 Marker 与测力台的绑定。"""
        remote = self._xingying_remote
        self._stop_xingying_trigger_listener()
        available = set(load_device_profile(self._selected_device_profile_key()).by_modality())
        for modality in XINGYING_LINKED_MODALITIES:
            if modality not in available:
                continue
            self._preview_connected_modalities.discard(modality)
            self._preview_connection_status[modality] = "未连接"
            device_id, simulated = self._get_modality_info(modality)
            self._set_preview_status(modality, "未连接", device_id, simulated)
            if self.preview_workspace is not None:
                self.preview_workspace.set_stream_state("xingying", "disconnected")
        if self._xingying_status_panel is not None:
            self._xingying_status_panel.set_connected(False)
            self._xingying_status_panel.set_recording(False)
        self._xingying_remote = None
        self._xingying_capture_name = None
        self._update_connect_button_state()
        self._update_start_button()
        if remote is not None:
            self._append_alert("XINGYING 远程捕获已断开。")
            LOG.info("XINGYING 远程捕获已断开")

    def _start_xingying_trigger_listener(self, cfg: dict[str, Any]) -> None:
        """启动 7061「远程触发」监听，接收 XINGYING 起停通知。"""
        if self._xingying_trigger is not None:
            return
        try:
            trigger = XingYingRemoteTrigger(
                ip=str(cfg["ip"]),
                port=int(cfg["trigger_port"]),
                on_trigger=self._on_xingying_trigger,
            )
            trigger.start()
        except Exception as exc:
            self._append_alert(
                f"启动 XINGYING 远程触发监听失败：{type(exc).__name__}: {exc}"
            )
            LOG.error("启动 XINGYING 远程触发监听失败: %s", exc)
            return
        self._xingying_trigger = trigger
        LOG.info(
            "XINGYING 远程触发监听已就绪 ip=%s port=%s", trigger.ip, trigger.port
        )

    def _stop_xingying_trigger_listener(self) -> None:
        """停止并清空 7061 远程触发监听（幂等）。"""
        trigger = self._xingying_trigger
        self._xingying_trigger = None
        if trigger is None:
            return
        try:
            trigger.stop()
        except Exception as exc:
            LOG.warning("停止 XINGYING 远程触发监听时出错: %s", exc)

    def _on_xingying_trigger(
        self,
        kind: str,
        payload: dict[str, Any],
        host_monotonic_ns: int,
        host_utc_ns: int,
    ) -> None:
        """收到 XINGYING 起停通知：记录主机时间戳，录制中转发给 Worker 落盘。"""
        try:
            trigger_kind = XingYingTriggerKind(kind)
        except ValueError:
            LOG.warning("未知 XINGYING 触发类型: %s", kind)
            return
        name = str(payload.get("capture_name") or "")
        display = "开始" if trigger_kind is XingYingTriggerKind.CAPTURE_START else "停止"
        self.xingying_alert_requested.emit(f"收到 XINGYING {display}通知：{name}")
        LOG.info(
            "收到 XINGYING %s name=%s host_monotonic_ns=%d",
            kind,
            name,
            host_monotonic_ns,
        )
        if self._worker is None or not self._worker.is_alive:
            return
        try:
            self._worker.record_xingying_trigger(
                trigger_kind,
                capture_name=name,
                database_path=str(payload.get("database_path") or ""),
                notes=str(payload.get("notes") or ""),
                description=str(payload.get("description") or ""),
                delay=str(payload.get("delay") or ""),
                timecode=str(payload.get("timecode") or ""),
                packet_id=str(payload.get("packet_id") or ""),
                host_monotonic_ns=host_monotonic_ns,
                host_utc_ns=host_utc_ns,
            )
        except Exception as exc:
            self.xingying_alert_requested.emit(
                f"写入 XINGYING 触发事件失败：{type(exc).__name__}: {exc}"
            )
            LOG.error("写入 XINGYING 触发事件失败: %s", exc)

    def _build_xingying_capture_name(self, request: TrialRunRequest) -> str:
        """生成 XINGYING 录制文件名（.cap 前缀，XINGYING 会追加 take 序号）。"""
        subject = str(request.subject_code or "subj").strip() or "subj"
        condition = str(request.condition_code or "cond").strip() or "cond"
        repeat = int(request.repeat_index or 1)
        short_uuid = str(request.trial_uuid).replace("-", "")[:8]
        return f"{subject}_{condition}_r{repeat}_{short_uuid}"

    def _start_xingying_capture(
        self,
        request: TrialRunRequest,
        database_path: Path,
    ) -> None:
        """Trial 开始时触发 XINGYING 把 .cap 录进固定的工程目录。

        XINGYING 加载刚体/人体模板后，``DatabasePath`` 必须与其工作目录（工程
        目录）一致，否则现场会弹「录制失败」。因此 .cap 直接录到操作员指定的
        XINGYING 工程目录（.cap / 标定 / .mars 模型资产全部在此），本系统绝不
        搬移文件；Data/ 下只保留超声/IMU/编码器/肌电等流式模态，动捕+测力台仅
        记录对应的 .cap 文件名（由 7061 触发监听写入 raw/xingying_trigger.jsonl）。
        """
        remote = self._xingying_remote
        if remote is None:
            return
        try:
            database_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._append_alert(f"创建 XINGYING 工程目录失败：{exc}")
            LOG.error("创建 XINGYING 工程目录失败: %s", exc)
            return
        name = self._build_xingying_capture_name(request)
        try:
            remote.capture_start(name, database_path)
        except Exception as exc:
            self._append_alert(f"触发 XINGYING 开始录制失败：{type(exc).__name__}: {exc}")
            LOG.error("XINGYING CaptureStart 失败: %s", exc)
            return
        self._xingying_capture_name = name
        if self._xingying_status_panel is not None:
            self._xingying_status_panel.set_recording(True)
        self._append_alert(f"已触发 XINGYING 录制：{name} → {database_path}")
        LOG.info("XINGYING CaptureStart name=%s path=%s", name, database_path)

    def _stop_xingying_capture(self) -> None:
        """Trial 停止/失败时触发 XINGYING 停止录制（幂等）。"""
        remote = self._xingying_remote
        name = self._xingying_capture_name
        self._xingying_capture_name = None
        if self._xingying_status_panel is not None:
            self._xingying_status_panel.set_recording(False)
        if remote is None or not name:
            return
        # XINGYING 的 7061「捕获--触发」在停止时可能不广播 CaptureStop（实测只
        # 广播 CaptureStart）。在主动发送 CaptureStop 的同时，取主机时钟补记一条
        # stop 锚点，保证每个 Trial 都有完整的 start→stop 区间供对齐。
        host_monotonic_ns = time.perf_counter_ns()
        host_utc_ns = time.time_ns()
        try:
            remote.capture_stop(name)
        except Exception as exc:
            self._append_alert(f"触发 XINGYING 停止录制失败：{type(exc).__name__}: {exc}")
            LOG.error("XINGYING CaptureStop 失败: %s", exc)
            return
        self._append_alert(f"已触发 XINGYING 停止录制：{name}")
        LOG.info("XINGYING CaptureStop name=%s", name)
        worker = self._worker
        if worker is not None and worker.is_alive:
            try:
                worker.record_xingying_trigger(
                    XingYingTriggerKind.CAPTURE_STOP,
                    capture_name=name,
                    database_path=str(
                        self._xingying_remote_config().get("database_path") or ""
                    ),
                    notes=(
                        "host-synthesized CaptureStop fallback "
                        "(no XINGYING broadcast received)"
                    ),
                    description="",
                    delay="",
                    timecode="",
                    packet_id="",
                    host_monotonic_ns=host_monotonic_ns,
                    host_utc_ns=host_utc_ns,
                )
            except Exception as exc:
                self._append_alert(
                    f"补记 XINGYING 停止锚点失败：{type(exc).__name__}: {exc}"
                )
                LOG.error("补记 XINGYING 停止锚点失败: %s", exc)

    def _maybe_start_xingying_capture(self, event: WorkerEvent) -> None:
        """Worker 进入 RECORDING 时触发一次 XINGYING 录制到固定工程目录。

        只在硬件模式 + 已连接远程端口 + 尚未开始录制时触发；``database_path`` 取自
        mocap 设备参数（XINGYING 工程目录），与 Worker 回报的 session 目录无关。
        """
        if str(event.payload.get("state") or "") != "RECORDING":
            return
        if self._xingying_remote is None or self._xingying_capture_name is not None:
            return
        request = self._active_request
        if request is None:
            return
        database_path = str(self._xingying_remote_config().get("database_path") or "")
        if not database_path:
            self._append_alert("XINGYING 工程目录（DatabasePath）未配置，跳过 .cap 录制。")
            LOG.error("XINGYING database_path 未配置")
            return
        self._start_xingying_capture(request, Path(database_path))

    @Slot()
    def _connect_group(self, group: tuple[str, ...]) -> None:
        """Connect every available modality in a device-connection display row."""
        if self._xingying_linked_enabled() and group == XINGYING_LINKED_MODALITIES:
            self._connect_xingying_remote()
            return
        available = set(
            load_device_profile(self._selected_device_profile_key()).by_modality()
        )
        for modality in group:
            if modality not in available:
                continue
            self._connect_modality(modality)

    def _disconnect_group(self, group: tuple[str, ...]) -> None:
        """Disconnect every modality in a device-connection display row."""
        if self._xingying_linked_enabled() and group == XINGYING_LINKED_MODALITIES:
            self._disconnect_xingying_remote()
            return
        for modality in group:
            self._disconnect_modality(modality)

    def _connect_modality(self, modality: str) -> None:
        """Spawn a single-modality preview worker for one modality."""
        if self._xingying_linked_enabled() and modality in XINGYING_LINKED_MODALITIES:
            self._connect_xingying_remote()
            return
        if modality in self._preview_workers:
            self._append_alert(f"{modality} 已有预览连接，请先断开。")
            return
        if self._worker is not None:
            self._append_alert("Trial 进行中，无法连接预览。")
            return

        device_id, simulated = self._get_modality_info(modality)
        adapter_factory = self._build_single_adapter_factory(modality)

        self._set_preview_status(modality, "连接中", device_id, simulated)
        self._preview_connection_status[modality] = "连接中"
        LOG.info("正在连接 %s 预览 (%s, simulated=%s)", modality, device_id, simulated)

        handle = ModalityPreviewProcessHandle(
            adapter_factory=adapter_factory,
            device_id=device_id,
            modality=modality,
            simulated=simulated,
        )
        self._preview_workers[modality] = handle
        try:
            handle.start()
        except Exception as exc:
            self._preview_workers.pop(modality, None)
            self._set_preview_status(modality, f"失败: {exc}", device_id, simulated, error=str(exc))
            self._preview_connection_status[modality] = "错误"
            self._append_alert(f"{modality} 预览启动失败：{type(exc).__name__}: {exc}")
            LOG.error("%s 预览启动失败: %s", modality, exc)
            return

        self._preview_timer.start()
        self._update_connect_button_state()
        self._append_alert(f"正在启动 {modality} 预览（{device_id}，{'模拟' if simulated else '真实'}）…")
        LOG.info("已启动 %s 预览 worker alive=%s", modality, handle.is_alive)

    @Slot()
    def _disconnect_modality(self, modality: str) -> None:
        """Request a non-blocking controlled stop for one preview worker."""
        if self._xingying_linked_enabled() and modality in XINGYING_LINKED_MODALITIES:
            self._disconnect_xingying_remote()
            return
        handle = self._preview_workers.get(modality)
        if handle is None:
            return
        self._preview_connected_modalities.discard(modality)
        self._preview_connection_status[modality] = "断开中"
        self._preview_disconnect_deadlines[modality] = time.monotonic() + 3.0
        LOG.info("正在断开 %s 预览", modality)
        try:
            handle.request_stop()
        except Exception as exc:
            LOG.warning("断开 %s 预览时出错: %s", modality, exc)
        self._set_preview_status(modality, "断开中", handle.device_id, handle.simulated)
        self._preview_timer.start()
        self._update_connect_button_state()
        self._append_alert(f"正在断开 {modality} 预览…")

    @Slot()
    def _start_button_marker(self) -> None:
        """连接按钮标签：启用全局键盘钩子监听逗号键。"""
        if self._button_marker is not None:
            return
        if self._worker is not None:
            self._append_alert("Trial 进行中，无法连接按钮标签。")
            return
        marker = ButtonMarkerListener()
        marker.start()
        self._button_marker = marker
        self._button_poll_timer.start()
        status_label = self._connect_status_labels.get(BUTTON_ROW_KEY)
        if status_label is not None:
            self._style_connection_indicator(status_label, "已连接")
            status_label.setToolTip("状态：已连接，等待按钮")
        self._update_connect_button_state()
        self._append_alert("按钮标签已启用：按下 USB 按钮即记录标记。")
        LOG.info("按钮标签监听已启用")

    @Slot()
    def _stop_button_marker(self) -> None:
        """断开按钮标签：停止全局键盘钩子。"""
        marker = self._button_marker
        self._button_marker = None
        if marker is not None:
            try:
                marker.stop()
            except Exception as exc:
                LOG.warning("停止按钮标签监听时出错: %s", exc)
        self._button_poll_timer.stop()
        status_label = self._connect_status_labels.get(BUTTON_ROW_KEY)
        if status_label is not None:
            self._style_connection_indicator(status_label, "未连接")
            status_label.setToolTip("状态：未连接")
        self._update_connect_button_state()
        self._append_alert("按钮标签已停止。")
        LOG.info("按钮标签监听已停止")

    @Slot()
    def _poll_button_marker(self) -> None:
        """主线程定时器：把钩子线程排队的按钮按下转成标签事件。"""
        marker = self._button_marker
        if marker is None:
            return
        for host_monotonic_ns, host_utc_ns in marker.drain():
            self._capture_prompt_label(
                PromptLabelSource.BUTTON,
                host_monotonic_ns=host_monotonic_ns,
                host_utc_ns=host_utc_ns,
            )

    def _start_start_stop_listener(self) -> None:
        """启动开始/停止 USB 按钮监听（句号键，启动即始终启用）。"""
        if self._button_marker_factory is None:
            return
        marker = self._button_marker_factory(
            vk=START_STOP_VK,
            ignore_shift=True,
        )
        marker.start()
        self._start_stop_button = marker
        self._start_stop_poll_timer.start()
        LOG.info("开始/停止按钮监听已启用（句号键）")

    @Slot()
    def _poll_start_stop_button(self) -> None:
        """主线程定时器：句号键按下 → 切换开始/停止写盘。"""
        marker = self._start_stop_button
        if marker is None:
            return
        if marker.drain():
            self._toggle_write()

    @Slot()
    @Slot()
    def _toggle_connect_all(self) -> None:
        """Toggle between connect-all and disconnect-all."""
        if self._preview_workers or self._xingying_remote is not None or self._button_marker is not None:
            for modality in list(self._preview_workers.keys()):
                self._disconnect_modality(modality)
            if self._xingying_remote is not None:
                self._disconnect_xingying_remote()
            if self._button_marker is not None:
                self._stop_button_marker()
        else:
            available = set(
                load_device_profile(self._selected_device_profile_key()).by_modality()
            )
            for modality in MODALITIES:
                if modality not in available:
                    continue
                if self._xingying_linked_enabled() and modality == "force_plate":
                    # force_plate 由 mocap 绑定连接，避免重复触发告警。
                    continue
                self._connect_modality(modality)

    def _update_connect_button_state(self) -> None:
        """Update connect-all toggle and per-row buttons."""
        has_any_connection = bool(self._preview_workers) or self._xingying_remote is not None
        can_change = not self._configuration_locked and self._worker is None
        if has_any_connection:
            self.connect_all_button.setText("全部断开")
            self.connect_all_button.setStyleSheet(
                "QPushButton { font-weight: 600; padding: 8px; color: #842029; background: #f8d7da; border: 1px solid #f5c2c7; border-radius: 4px; }"
            )
        else:
            self.connect_all_button.setText("全部连接")
            self.connect_all_button.setStyleSheet("")
        self.connect_all_button.setEnabled(can_change)

        available = set(
            load_device_profile(self._selected_device_profile_key()).by_modality()
        )
        for group in CONNECTION_ROWS:
            row_key = group[0] if len(group) == 1 else XINGYING_GROUP_KEY
            connect_button = self._connect_buttons.get(row_key)
            disconnect_button = self._disconnect_buttons.get(row_key)
            if connect_button is None or disconnect_button is None:
                continue
            if row_key == XINGYING_GROUP_KEY:
                if self._xingying_linked_enabled():
                    active = self._xingying_remote is not None
                else:
                    active = any(m in self._preview_workers for m in group)
                group_available = any(m in available for m in group)
            else:
                modality = group[0]
                active = modality in self._preview_workers
                group_available = modality in available
            stopping = any(m in self._preview_disconnect_deadlines for m in group)
            connect_button.setText("连接")
            connect_button.setEnabled(can_change and not active and group_available)
            if not group_available:
                connect_button.setToolTip("该设备仅在真实设备配置中可用")
            disconnect_button.setText("断开中…" if stopping else "断开")
            disconnect_button.setEnabled(can_change and active and not stopping)

        # 按钮标签行：非数据模态，活跃 = 监听器已启动。
        button_connect = self._connect_buttons.get(BUTTON_ROW_KEY)
        button_disconnect = self._disconnect_buttons.get(BUTTON_ROW_KEY)
        if button_connect is not None:
            button_active = self._button_marker is not None
            button_connect.setText("连接")
            button_connect.setEnabled(can_change and not button_active)
            button_connect.setToolTip("连接后按下 USB 按钮即记录标记")
            button_disconnect.setText("断开")
            button_disconnect.setEnabled(can_change and button_active)

        self._update_start_button()

    @Slot()
    def _poll_preview_workers(self) -> None:
        """Poll events from all active preview workers and dispatch to UI handlers."""
        if not self._preview_workers:
            self._preview_timer.stop()
            return
        now = time.monotonic()
        for modality, handle in list(self._preview_workers.items()):
            try:
                events = handle.poll_events(limit=100)
            except Exception as exc:
                self._append_alert(
                    f"读取 {modality} 预览事件失败：{type(exc).__name__}: {exc}"
                )
                events = []
            for event in events:
                try:
                    self._handle_preview_worker_event(event, handle, modality)
                except Exception as exc:
                    self._append_alert(
                        f"处理 {modality} 预览事件失败："
                        f"{type(exc).__name__}: {exc}"
                    )

            # Recording control ACKs update the handle's local active UUID.
            # Draining them is essential for a later Trial to reuse this same
            # persistent preview handle after STOPPED or FAULT.
            drain_control_ack = getattr(handle, "drain_control_ack", None)
            if callable(drain_control_ack):
                try:
                    drain_control_ack()
                except Exception as exc:
                    LOG.warning(
                        "draining %s recording control ACKs failed: %s",
                        modality,
                        exc,
                    )

            if not handle.is_alive and handle.exitcode is not None:
                self._handle_preview_worker_death(modality, handle)
                continue
            deadline = self._preview_disconnect_deadlines.get(modality)
            if deadline is not None and now >= deadline:
                self._append_alert(f"{modality} 预览断开超时，正在强制回收。")
                try:
                    handle.terminate(timeout=0.25)
                except Exception as exc:
                    LOG.error("强制回收 %s 预览失败: %s", modality, exc)
                self._handle_preview_worker_death(modality, handle)

    def _handle_preview_worker_event(self, event: WorkerEvent,
                                      handle: ModalityPreviewHandle,
                                      modality: str) -> None:
        if event.event_type is WorkerEventType.STATE:
            state = str(event.payload.get("state") or "UNKNOWN")
            if state == "READY":
                self._preview_connected_modalities.add(modality)
                self._preview_connection_status[modality] = "已连接"
                if self.preview_workspace is not None:
                    self.preview_workspace.set_stream_state(modality, "connected")
                self._set_preview_status(
                    modality,
                    "数据正常",
                    handle.device_id,
                    handle.simulated,
                    detail_lines=("已收到首批有效数据",),
                )
                self._update_connect_button_state()
                self._update_start_button()
                self._append_alert(
                    f"{modality} ({handle.device_id}) "
                    f"{'模拟' if handle.simulated else '真实'}预览已就绪。"
                )
                LOG.info("%s (%s) preview READY simulated=%s", modality, handle.device_id, handle.simulated)
            elif state == "CONNECTING":
                self._set_preview_status(
                    modality,
                    "连接中",
                    handle.device_id,
                    handle.simulated,
                )
            elif state == "PREVIEW_STARTING":
                self._set_preview_status(
                    modality,
                    "已连接，等待数据",
                    handle.device_id,
                    handle.simulated,
                    detail_lines=("通信已建立，正在等待首批有效数据",),
                )
            elif state in {"CONNECTED", "PREPARING", "RECORDING"}:
                if modality not in self._preview_connected_modalities:
                    self._set_preview_status(
                        modality,
                        "已连接，等待数据",
                        handle.device_id,
                        handle.simulated,
                        detail_lines=("通信已建立，正在等待首批有效数据",),
                    )
            elif state == "STOPPING":
                self._set_preview_status(
                    modality,
                    "断开中",
                    handle.device_id,
                    handle.simulated,
                )
            elif state == "FAULT":
                self._preview_connected_modalities.discard(modality)
                self._preview_connection_status[modality] = "错误"
                self._set_preview_status(
                    modality,
                    "故障",
                    handle.device_id,
                    handle.simulated,
                    error=event.message or "设备报告故障",
                )
                self._update_start_button()
            elif state == "DISCONNECTED":
                self._preview_connected_modalities.discard(modality)
                self._preview_connection_status[modality] = "未连接"
                if self.preview_workspace is not None:
                    self.preview_workspace.set_stream_state(modality, "disconnected")
                self._set_preview_status(
                    modality,
                    "未连接",
                    handle.device_id,
                    handle.simulated,
                )
                self._update_connect_button_state()
                self._update_start_button()
        elif event.event_type is WorkerEventType.FAILED:
            if str(event.payload.get("state") or "").upper() == "FAULT":
                self._handle_recording_branch_fault(event, modality)
                return
            error_msg = event.message or "未知错误"
            full_tb = str(event.payload.get("traceback") or "")
            self._preview_connected_modalities.discard(modality)
            self._preview_connection_status[modality] = "错误"
            if self.preview_workspace is not None:
                self.preview_workspace.set_stream_state(modality, "error")
            self._set_preview_status(modality, "错误", handle.device_id, handle.simulated, error=error_msg)
            self._append_alert(f"{modality} 预览失败：{error_msg}")
            if full_tb:
                LOG.error("%s preview failed:\n%s", modality, full_tb)
            else:
                LOG.error("%s preview failed: %s", modality, error_msg)
            try:
                handle.request_stop()
            except Exception:
                pass
            self._preview_disconnect_deadlines[modality] = time.monotonic() + 1.0
            self._update_connect_button_state()
            self._update_start_button()
        elif event.event_type is WorkerEventType.HEALTH:
            self._handle_preview_health(event, modality)
        elif event.event_type is WorkerEventType.PREVIEW:
            self._handle_preview(event)

    def _handle_recording_branch_fault(
        self,
        event: WorkerEvent,
        modality: str,
    ) -> None:
        """Fail only the Trial recording branch; keep device preview alive."""

        event_trial_uuid = event.trial_uuid or event.payload.get("trial_uuid")
        active_trial_uuid = self._active_trial_uuid
        fault = str(
            event.payload.get("fault")
            or event.message
            or "unknown recording stream fault"
        )
        if (
            active_trial_uuid is None
            or (
                event_trial_uuid is not None
                and str(event_trial_uuid) != active_trial_uuid
            )
        ):
            self._append_alert(
                f"已忽略 {modality} 的过期记录支路 FAULT："
                f"trial={event_trial_uuid or '未知'}，{fault}"
            )
            LOG.warning(
                "ignored stale recording branch fault: modality=%s "
                "event_trial=%s active_trial=%s fault=%s",
                modality,
                event_trial_uuid,
                active_trial_uuid,
                fault,
            )
            return

        if self._recording_branch_fault is None:
            self._recording_branch_fault = f"{modality}: {fault}"
            self._append_alert(
                f"Trial 记录支路故障（{modality}）：{fault}。"
                "已停止全部写盘转发；设备预览保持连接。"
            )
            self._add_timeline_event(2, f"RECORDING FAULT · {modality} · {fault}")
        self._trial_succeeded = False
        self._end_recording_streams()

        worker = self._worker
        if worker is not None and not self._stop_requested:
            try:
                worker.request_stop()
            except Exception as exc:
                self._append_alert(
                    f"记录支路故障后发送 Collector 停止请求失败："
                    f"{type(exc).__name__}: {exc}"
                )
            self._stop_requested = True
            self._stop_requested_at = time.monotonic()
        self.start_button.setEnabled(False)
        self._set_trial_state("FAILED")

    def _handle_preview_health(self, event: WorkerEvent, modality: str) -> None:
        payload = event.payload
        row = self._health_rows.get(modality)
        if row is None:
            return
        status = str(payload.get("status") or "UNKNOWN").upper()
        self.health_table.item(row, HEALTH_COLUMN_MODALITY).setToolTip(
            f"设备状态：{status}"
        )
        sample_count = payload.get("sample_count")
        if sample_count is not None:
            self.health_table.item(row, HEALTH_COLUMN_SAMPLE_COUNT).setText(
                str(int(sample_count))
            )
        rate = payload.get("actual_sample_rate_hz")
        self.health_table.item(row, HEALTH_COLUMN_RATE).setText(
            "-" if rate is None else f"{float(rate):.1f} Hz"
        )
        nominal = payload.get("nominal_sample_rate_hz")
        self.health_table.item(row, HEALTH_COLUMN_NOMINAL_RATE).setText(
            "-" if nominal is None else f"{float(nominal):.1f} Hz"
        )
        dropped = payload.get("dropped_packets")
        self.health_table.item(row, HEALTH_COLUMN_DROPPED).setText(
            "-" if dropped is None else str(int(dropped))
        )
        previous = self._last_health_status.get(modality)
        self._last_health_status[modality] = status
        indicator_status, indicator_reason, data_age_s = (
            self._classify_preview_health(payload)
        )
        handle = self._preview_workers.get(modality)
        device_id = str(
            payload.get("device_id")
            or (handle.device_id if handle is not None else "")
        )
        simulated = bool(
            payload.get(
                "simulated",
                handle.simulated if handle is not None else False,
            )
        )
        health_status = str(
            payload.get("health_status") or status or "UNKNOWN"
        ).upper()
        nominal_rate = payload.get("nominal_sample_rate_hz")
        queue_depth = payload.get("queue_depth")
        queue_capacity = payload.get("queue_capacity")
        details = [
            f"设备健康：{health_status}",
            f"累计样本/帧：{int(sample_count or 0)}",
            (
                "实际/标称速率："
                f"{float(rate):.1f} / {float(nominal_rate):.1f} Hz"
                if rate is not None and nominal_rate is not None
                else "实际/标称速率：尚无完整数据"
            ),
            f"累计丢包：{int(dropped or 0)}",
        ]
        if queue_depth is not None and queue_capacity is not None:
            details.append(
                f"队列：{int(queue_depth)} / {int(queue_capacity)}"
            )
        if data_age_s is not None:
            details.append(f"距最近数据：{data_age_s:.2f} s")
        self._set_preview_status(
            modality,
            indicator_status,
            device_id,
            simulated,
            error=indicator_reason,
            detail_lines=tuple(details),
        )
        if status in {"DEGRADED", "UNHEALTHY", "FAULT"} and status != previous:
            detail = event.message or str(payload.get("message") or "")
            suffix = f"：{detail}" if detail else ""
            self._append_alert(f"{modality} 健康状态 {status}{suffix}")

    @staticmethod
    def _classify_preview_health(
        payload: Mapping[str, Any],
    ) -> tuple[str, str | None, float | None]:
        """Collapse connection, data freshness and health into one lamp state."""

        device_status = str(payload.get("status") or "UNKNOWN").upper()
        health_status = str(
            payload.get("health_status")
            or (
                device_status
                if device_status in {"HEALTHY", "DEGRADED", "UNHEALTHY"}
                else "UNKNOWN"
            )
        ).upper()
        connected = bool(
            payload.get(
                "connected",
                device_status not in {"DISCONNECTED", "CLOSED"},
            )
        )
        try:
            sample_count = max(0, int(payload.get("sample_count") or 0))
        except (TypeError, ValueError):
            sample_count = 0
        try:
            dropped_packets = max(
                0, int(payload.get("dropped_packets") or 0)
            )
        except (TypeError, ValueError):
            dropped_packets = 0

        if (
            device_status in {"FAULT", "FAILED"}
            or health_status == "UNHEALTHY"
        ):
            reason = str(payload.get("message") or "设备报告故障")
            return "故障", reason, None
        if not connected or device_status in {"DISCONNECTED", "CLOSED"}:
            return "未连接", None, None
        if device_status == "CONNECTING":
            return "连接中", None, None
        if sample_count <= 0:
            return "已连接，等待数据", None, None

        data_age_s: float | None = None
        last_data_ns = payload.get("last_data_host_monotonic_ns")
        try:
            if last_data_ns is not None and int(last_data_ns) > 0:
                data_age_s = max(
                    0.0,
                    (time.perf_counter_ns() - int(last_data_ns))
                    / 1_000_000_000,
                )
        except (TypeError, ValueError):
            data_age_s = None
        nominal_rate = payload.get("nominal_sample_rate_hz")
        try:
            stale_after_s = max(
                2.0,
                20.0 / max(float(nominal_rate or 0.0), 1e-9),
            )
        except (TypeError, ValueError):
            stale_after_s = 2.0
        if data_age_s is not None and data_age_s > stale_after_s:
            return (
                "数据中断",
                f"超过 {stale_after_s:.1f} s 未收到新数据",
                data_age_s,
            )

        queue_depth = payload.get("queue_depth")
        queue_capacity = payload.get("queue_capacity")
        try:
            queue_fill = (
                float(queue_depth) / float(queue_capacity)
                if float(queue_capacity) > 0
                else 0.0
            )
        except (TypeError, ValueError):
            queue_fill = 0.0
        if health_status == "DEGRADED":
            return (
                "数据异常",
                str(payload.get("message") or "设备健康状态降级"),
                data_age_s,
            )
        if dropped_packets > 0:
            return (
                "数据异常",
                f"已检测到 {dropped_packets} 个丢包",
                data_age_s,
            )
        if queue_fill >= 0.8:
            return (
                "数据异常",
                f"原始队列占用达到 {queue_fill:.0%}",
                data_age_s,
            )
        return "数据正常", None, data_age_s

    def _handle_preview_worker_death(self, modality: str, handle: ModalityPreviewHandle) -> None:
        requested = modality in self._preview_disconnect_deadlines
        self._preview_connected_modalities.discard(modality)
        previous_status = self._preview_connection_status.get(modality)
        if not requested and previous_status in {"连接中", "已连接"}:
            self._preview_connection_status[modality] = "错误"
            self._set_preview_status(
                modality, f"启动失败 (exitcode={handle.exitcode})", handle.device_id, handle.simulated,
                error="子进程异常退出"
            )
            self._append_alert(
                f"{modality} 预览进程异常退出 (exitcode={handle.exitcode})。"
                f"可能是 SDK 依赖缺失或配置错误。"
            )
            LOG.error("%s preview exitcode=%s", modality, handle.exitcode)
        self._preview_workers.pop(modality, None)
        self._preview_disconnect_deadlines.pop(modality, None)
        try:
            handle.join(timeout=0)
            handle.close()
        except Exception as exc:
            LOG.warning("释放 %s 预览句柄失败: %s", modality, exc)
        if requested and previous_status != "错误":
            self._preview_connection_status[modality] = "未连接"
            self._set_preview_status(
                modality, "未连接", handle.device_id, handle.simulated
            )
            row = self._health_rows.get(modality)
            if row is not None:
                self.health_table.item(row, HEALTH_COLUMN_MODALITY).setToolTip(
                    "设备状态：DISCONNECTED"
                )
                self._last_health_status.pop(modality, None)
            self._append_alert(f"{modality} 预览已断开。")
        self._update_connect_button_state()
        self._update_start_button()
        if not self._preview_workers:
            self._preview_timer.stop()

    # ── Trial Workflow ────────────────────────────────────────────────

    def _refresh_identity_context(self, data_root: Path, project_code: str, subject_code: str) -> None:
        session_key = (str(data_root), project_code, subject_code)
        if session_key != self._session_key:
            self._session_key = session_key
            self._session_uuid = uuid4()

    def build_request(self) -> TrialRunRequest:
        data_root_text = self.data_root_edit.text().strip()
        if not data_root_text:
            raise ValueError("数据根目录不能为空")
        data_root = self.set_data_root(data_root_text)
        project = self.project_combo.currentData()
        if not isinstance(project, dict):
            raise ValueError("请选择有效项目")
        project_code = str(project.get("project_code") or "").strip().upper()
        project_name = str(project.get("project_name") or "").strip()
        if project_code not in SUPPORTED_PROJECT_CODES or not project_name:
            raise ValueError("请选择有效项目")
        subject_code = self._subject_code()
        self._activate_selected_metadata_identity()
        self._handle_metadata_condition_changed()
        operator = DEFAULT_OPERATOR
        condition = self.condition_combo.currentData()
        if not isinstance(condition, dict):
            raise ValueError("请选择有效工况")
        self._refresh_identity_context(data_root, project_code, subject_code)
        payload: dict[str, Any] = {
            "data_root": data_root,
            "device_profile_key": self._selected_device_profile_key(),
            "device_overrides": (
                self._settings.hardware_device_overrides
                if self._selected_device_profile_key() == "hardware" else {}
            ),
            "duration_s": None,
            "session_uuid": self._session_uuid,
            "project_code": project_code,
            "project_name": project_name,
            "subject_code": subject_code,
            "day": self.day_spin.value(),
            "operator": operator,
            "condition_code": str(condition["condition_code"]),
            "condition_name": str(condition["condition_name"]),
            "condition_level": condition.get("condition_level"),
            "condition_parameters": dict(condition.get("parameters", {})),
            "repeat_index": self.repeat_spin.value(),
            "protocol_version": _PROTOCOL.protocol_version,
            "config_version": "1.0.0",
            "experiment_metadata": self._experiment_metadata.model_dump(mode="python"),
        }
        return TrialRunRequest.model_validate(payload)

    @Slot()
    def start_trial(self) -> None:
        if (
            self._worker is not None
            or self._preflight_worker is not None
        ):
            return

        # Build request first (validates input)
        try:
            request = self.build_request()
        except Exception as exc:
            self._append_alert(f"无法构建 Trial 请求：{type(exc).__name__}: {exc}")
            self.statusBar().showMessage("Trial 请求构建失败。")
            return

        # Only require at least one connected modality.
        if not self._preview_connected_modalities:
            self.statusBar().showMessage("请先连接至少一个模态的设备预览。")
            self._update_start_button()
            return

        # Pass enabled modalities to the trial worker so only connected
        # devices are recorded.  在真实硬件模式下，XINGYING 绑定的动捕/测力台
        # 不产生流数据，由远程触发录制 .cap，因此从 Worker 的流模态集合中排除；
        # 模拟模式下动捕仍是正常流模态，保持不变。
        connected = frozenset(self._preview_connected_modalities)
        streaming = frozenset(
            modality
            for modality in connected
            if not (
                self._xingying_linked_enabled()
                and modality in XINGYING_LINKED_MODALITIES
            )
        )
        if not streaming:
            self.statusBar().showMessage(
                "请至少连接一个数据模态（超声/IMU/编码器/EMG）。"
            )
            self._update_start_button()
            return
        request = request.model_copy(update={"enabled_modalities": streaming})

        # The already-running preview processes own the hardware Adapters.
        # Recording attaches to their raw IPC endpoints without stopping or
        # reconnecting a single device.
        self._active_request = request
        self._set_configuration_locked(True)
        self._launch_trial_worker(request)

    def _launch_trial_worker(self, request: TrialRunRequest) -> None:
        """Start a disk consumer and then open each preview recording gate."""
        worker: WorkerHandle | None = None
        try:
            handles: dict[str, ModalityPreviewHandle] = {}
            endpoints: dict[str, RecordingStreamEndpoint] = {}
            for modality in sorted(request.enabled_modalities or ()):
                handle = self._preview_workers.get(modality)
                if handle is None or modality not in self._preview_connected_modalities:
                    raise RuntimeError(f"{modality} preview is not READY")
                discard_backlog = getattr(
                    handle, "discard_recording_backlog", None
                )
                if callable(discard_backlog):
                    discarded = int(discard_backlog() or 0)
                    if discarded:
                        LOG.warning(
                            "discarded %d stale %s recording queue items "
                            "before Trial %s",
                            discarded,
                            modality,
                            request.trial_uuid,
                        )
                endpoint = getattr(handle, "recording_endpoint", None)
                if endpoint is None:
                    raise RuntimeError(
                        f"{modality} preview has no recording endpoint"
                    )
                handles[modality] = handle
                endpoints[modality] = endpoint

            worker = self._create_recording_worker(request, endpoints)
            worker.start()
            trial_uuid = str(request.trial_uuid)
            self._worker = worker
            self._active_trial_uuid = trial_uuid
            self._recording_preview_handles = {}
            self._recording_streams_ended = False
            for modality, handle in handles.items():
                handle.begin_recording(trial_uuid)
                self._recording_preview_handles[modality] = handle
        except Exception as exc:
            if self._recording_preview_handles:
                self._end_recording_streams()
            if worker is not None and self._worker_is_alive(worker):
                try:
                    worker.request_stop()
                except Exception:
                    pass
                try:
                    worker.terminate_for_recovery(timeout=0.5)
                except Exception as cleanup_exc:
                    LOG.error(
                        "terminating Collector after recording attach failure "
                        "failed: %s",
                        cleanup_exc,
                    )
            if worker is not None:
                try:
                    if not self._worker_is_alive(worker):
                        worker.join(timeout=0)
                        worker.close()
                except Exception:
                    pass
            worker_still_alive = (
                worker is not None and self._worker_is_alive(worker)
            )
            self._worker = worker if worker_still_alive else None
            if not worker_still_alive:
                self._active_trial_uuid = None
            self._recording_preview_handles.clear()
            self._recording_streams_ended = False
            if worker_still_alive:
                self._stop_requested = True
                self._stop_requested_at = time.monotonic()
                self._poll_timer.start()
            else:
                self._poll_timer.stop()
            self._set_trial_state("FAILED")
            self._append_alert(f"无法启动 Trial：{type(exc).__name__}: {exc}")
            self.statusBar().showMessage("Trial 启动失败。")
            LOG.error("Trial 启动失败: %s", exc)
            self._set_configuration_locked(worker_still_alive)
            return

        self._terminal_event_received = False
        self._dead_poll_count = 0
        self._stop_requested = False
        self._stop_requested_at = None
        self._forced_stop_alerted = False
        self._close_when_finished = False
        self._trial_succeeded = False
        self._recording_branch_fault = None
        self._reset_trial_telemetry()
        self._set_trial_state("PREPARING")
        self._update_start_button()
        self._poll_timer.start()
        self.trial_started.emit(request)
        self._show_toast("● 开始记录数据", level="INFO")
        self.statusBar().showMessage(f"Trial {request.trial_uuid} 已交给独立 Collector Worker。")
        LOG.info("Trial 已启动: %s", request.trial_uuid)

    def _create_recording_worker(
        self,
        request: TrialRunRequest,
        endpoints: Mapping[str, RecordingStreamEndpoint],
    ) -> WorkerHandle:
        """Call the endpoint-aware factory, retaining old test integrations."""

        try:
            signature = inspect.signature(self._worker_factory)
            signature.bind(request, endpoints)
        except TypeError:
            return self._worker_factory(request)  # type: ignore[call-arg]
        except (ValueError, AttributeError):
            pass
        return self._worker_factory(request, endpoints)

    def _reset_trial_telemetry(self) -> None:
        """Reset Trial-scoped counters without touching live signal curves."""

        # Do NOT clear alerts or real-time preview buffers.  Starting disk
        # recording is not a device-stream boundary and must be visually
        # imperceptible apart from the Trial state controls.
        self._last_health_status.clear()
        self._prompt_label_counts = {
            PromptLabelSource.SUBJECT: 0,
            PromptLabelSource.OPERATOR: 0,
            PromptLabelSource.BUTTON: 0,
        }
        for row in self._health_rows.values():
            self.health_table.item(row, HEALTH_COLUMN_MODALITY).setToolTip("")
            self.health_table.item(row, HEALTH_COLUMN_SAMPLE_COUNT).setText("0")
            self.health_table.item(row, HEALTH_COLUMN_RATE).setText("-")
            self.health_table.item(row, HEALTH_COLUMN_DROPPED).setText("-")
        self._timeline_started_at = time.monotonic()
        self._timeline_x.clear()
        self._timeline_y.clear()
        self._timeline_text.clear()
        self._add_timeline_event(0, "PREPARING")

    def _reset_realtime_preview_curves(self) -> None:
        """Explicitly clear signal displays at a true preview-stream boundary."""

        empty_ultrasound = np.full(ULTRASOUND_PREVIEW_SAMPLES, np.nan, dtype=np.float64)
        for curve in self._us_curves:
            curve.setData(self._us_x, empty_ultrasound)
        self._ultrasound_format_alerted.clear()
        for trace in self._imu_traces.values():
            trace.reset()
        for trace in self._enc_traces.values():
            trace.reset()

    @Slot()
    def request_controlled_stop(self) -> None:
        if self._worker is None or self._stop_requested:
            return
        # Close only the recording gates.  The preview workers and hardware
        # Adapter instances remain alive and continue publishing UI previews.
        self._end_recording_streams()
        self._stop_xingying_capture()
        try:
            self._worker.request_stop()
        except Exception as exc:
            self._append_alert(f"发送停止请求失败：{type(exc).__name__}: {exc}")
            return
        self._stop_requested = True
        self._stop_requested_at = time.monotonic()
        self.start_button.setEnabled(False)
        self._set_trial_state("STOPPING")
        self._append_alert("已发送受控停止请求；正在等待 Writer flush 与 Trial 最终化。预览持续运行。")
        LOG.info("Trial 受控停止请求已发送，记录流转发已停止")

    def _end_recording_streams(self) -> None:
        """Idempotently close Trial forwarding without stopping previews."""

        if self._recording_streams_ended:
            return
        trial_uuid = self._active_trial_uuid
        if trial_uuid is None:
            return
        self._recording_streams_ended = True
        for modality, handle in self._recording_preview_handles.items():
            try:
                handle.end_recording(trial_uuid)
            except Exception as exc:
                self._append_alert(
                    f"停止 {modality} 写盘转发失败："
                    f"{type(exc).__name__}: {exc}"
                )
                LOG.error("end_recording failed for %s: %s", modality, exc)

    @Slot()
    def poll_worker_events(self) -> None:
        worker = self._worker
        if worker is None:
            self._poll_timer.stop()
            return
        try:
            events = worker.poll_events(limit=200)
        except Exception as exc:
            self._mark_failed(f"读取 Worker 事件失败：{type(exc).__name__}: {exc}")
            events = []
        for event in events:
            try:
                self._handle_worker_event(event)
            except Exception as exc:
                self._append_alert(
                    f"已忽略无效 {event.event_type.value} 事件：{type(exc).__name__}: {exc}"
                )
        if self._worker_is_alive(worker):
            self._enforce_controlled_stop_deadline(worker)
        if self._worker_is_alive(worker):
            self._dead_poll_count = 0
            return
        self._dead_poll_count += 1
        if not self._terminal_event_received and self._dead_poll_count < 3:
            return
        if not self._terminal_event_received:
            try:
                trailing_events = worker.poll_events(limit=200)
            except Exception:
                trailing_events = []
            for event in trailing_events:
                self._handle_worker_event(event)
        exitcode = self._worker_exitcode(worker)
        if not self._terminal_event_received:
            self._mark_failed(
                f"Collector Worker 在未发布 COMPLETED/FAILED 事件时退出（exit code {exitcode}）。"
            )
        self._release_worker(worker)

    def _enforce_controlled_stop_deadline(self, worker: WorkerHandle) -> None:
        requested_at = self._stop_requested_at
        if requested_at is None:
            return
        elapsed = time.monotonic() - requested_at
        if elapsed < self._controlled_stop_timeout_s:
            return
        if not self._forced_stop_alerted:
            self._forced_stop_alerted = True
            self._append_alert(
                "受控停止等待超时；正在终止 Collector Worker。未发布的数据包将保持 "
                ".recording，由恢复工作流检查，绝不会伪装为 FINALIZED。"
            )
            self.statusBar().showMessage("Writer/设备停止超时；正在保留 .recording 并执行强制回收。")
        try:
            worker.terminate_for_recovery(timeout=1.0)
        except Exception as exc:
            self._append_alert(f"强制回收 Collector Worker 失败：{type(exc).__name__}: {exc}")
            return
        if self._worker_is_alive(worker):
            return
        self._terminal_event_received = True
        if not self._trial_succeeded:
            self._mark_failed(
                "受控停止超时，Worker 已终止；原始数据保持 .recording，需在 Data Studio 的恢复工作流中审计。"
            )

    @staticmethod
    def _worker_is_alive(worker: WorkerHandle) -> bool:
        value = worker.is_alive
        return bool(value() if callable(value) else value)

    @staticmethod
    def _worker_exitcode(worker: WorkerHandle) -> int | None:
        value = worker.exitcode
        return value() if callable(value) else value

    def _handle_worker_event(self, event: WorkerEvent) -> None:
        expected_trial_uuid = self._active_trial_uuid
        claimed_trial_uuid = event.trial_uuid or event.payload.get("trial_uuid")
        if (
            expected_trial_uuid is not None
            and claimed_trial_uuid is not None
            and str(claimed_trial_uuid) != expected_trial_uuid
        ):
            self._append_alert(
                f"已拒绝不属于当前 Trial 的 Worker 事件：expected={expected_trial_uuid}，"
                f"received={claimed_trial_uuid}，type={event.event_type.value}。"
            )
            return
        if event.event_type is WorkerEventType.STATE:
            state = str(event.payload.get("state") or event.message or "UNKNOWN")
            self._set_trial_state(state)
            self._add_timeline_event(0, state.upper())
            self._maybe_start_xingying_capture(event)
        elif event.event_type is WorkerEventType.SYNC:
            self._handle_sync(event.payload, record_event=True)
        elif event.event_type is WorkerEventType.HEALTH:
            self._handle_health(event)
        elif event.event_type is WorkerEventType.METRIC:
            self._handle_metric(event.payload)
        elif event.event_type is WorkerEventType.PROMPT_LABEL:
            self._handle_prompt_label(event)
        elif event.event_type is WorkerEventType.ALERT:
            message = event.message or "Collector Worker 报告需要关注的事件。"
            self._append_alert(message)
            self._add_timeline_event(2, message)
        elif event.event_type is WorkerEventType.PREVIEW:
            self._handle_preview(event)
        elif event.event_type is WorkerEventType.COMPLETED:
            self._handle_completed(event)
        elif event.event_type is WorkerEventType.FAILED:
            self._terminal_event_received = True
            self._mark_failed(event.message or "Collector Worker 报告未知错误。")

    def _handle_prompt_label(self, event: WorkerEvent) -> None:
        try:
            source = PromptLabelSource(str(event.payload.get("source") or ""))
        except ValueError:
            self._append_alert("Collector Worker 返回了未知的人工标签来源。")
            return
        if source is PromptLabelSource.SUBJECT:
            count_key = "subject_count"
            row_key = "subject_prompt"
        elif source is PromptLabelSource.OPERATOR:
            count_key = "operator_count"
            row_key = "operator_prompt"
        else:
            count_key = "button_count"
            row_key = "button_prompt"
        count = max(0, int(event.payload.get(count_key) or 0))
        self._prompt_label_counts[source] = count
        row = self._health_rows[row_key]
        self.health_table.item(row, HEALTH_COLUMN_SAMPLE_COUNT).setText(str(count))
        self.health_table.item(row, HEALTH_COLUMN_SAMPLE_COUNT).setToolTip(
            f"{source.display_name}累计 {count} 次"
        )
        self._add_timeline_event(1, f"{source.display_name} · {count}")
        LOG.info(
            "Prompt label persisted acknowledgement: source=%s count=%d "
            "host_monotonic_ns=%s",
            source.value,
            count,
            event.payload.get("host_monotonic_ns"),
        )

    def _handle_health(self, event: WorkerEvent) -> None:
        payload = event.payload
        modality = self._normalize_modality(
            event.modality or str(payload.get("modality") or payload.get("device_id") or "")
        )
        if modality not in self._health_rows:
            return
        row = self._health_rows[modality]
        status = str(payload.get("status") or "UNKNOWN").upper()
        self.health_table.item(row, HEALTH_COLUMN_MODALITY).setToolTip(
            f"设备状态：{status}"
        )
        if "sample_count" in payload:
            self.health_table.item(row, HEALTH_COLUMN_SAMPLE_COUNT).setText(
                str(int(payload["sample_count"]))
            )
        rate = payload.get("actual_sample_rate_hz")
        self.health_table.item(row, HEALTH_COLUMN_RATE).setText(
            "-" if rate is None else f"{float(rate):.1f} Hz"
        )
        nominal = payload.get("nominal_sample_rate_hz")
        self.health_table.item(row, HEALTH_COLUMN_NOMINAL_RATE).setText(
            "-" if nominal is None else f"{float(nominal):.1f} Hz"
        )
        dropped = payload.get("dropped_packets")
        self.health_table.item(row, HEALTH_COLUMN_DROPPED).setText(
            "-" if dropped is None else str(int(dropped))
        )
        previous = self._last_health_status.get(modality)
        self._last_health_status[modality] = status
        if status in {"DEGRADED", "UNHEALTHY", "FAULT"} and status != previous:
            detail = event.message or str(payload.get("message") or "")
            suffix = f"：{detail}" if detail else ""
            self._append_alert(f"{modality} 健康状态 {status}{suffix}")
        if status in {"UNHEALTHY", "FAULT"} and modality in CRITICAL_MODALITIES:
            self._preflight_ready = False
            self._set_trial_state("FAILED")
            self._update_start_button()

    def _handle_metric(self, payload: dict[str, Any]) -> None:
        counts = payload.get("modality_counts")
        if isinstance(counts, dict):
            for raw_modality, count in counts.items():
                modality = self._normalize_modality(str(raw_modality))
                if modality in self._health_rows:
                    row = self._health_rows[modality]
                    self.health_table.item(row, HEALTH_COLUMN_SAMPLE_COUNT).setText(
                        str(int(count))
                    )
        if any(key in payload for key in ("status", "quality", "trigger_count",
                                            "first_trigger_host_monotonic_ns", "trigger_time_utc")):
            self._handle_sync(payload, record_event=False)

    def _handle_sync(self, payload: Mapping[str, Any], *, record_event: bool) -> None:
        status = str(payload.get("status") or "WAITING_SYNC").strip().upper()
        quality = str(payload.get("quality") or "WAITING").strip().upper()
        try:
            trigger_count = max(0, int(payload.get("trigger_count") or 0))
        except (TypeError, ValueError):
            trigger_count = 0
        # Missing synchronization is deliberately informational.  Acquisition
        # faults and modality loss still arrive through HEALTH/FAILED events.
        if record_event:
            self._add_timeline_event(1, f"{status} · {quality} · trigger={trigger_count}")

    def _handle_preview(self, event: WorkerEvent) -> None:
        modality = self._normalize_modality(event.modality or "")
        if self.preview_workspace is not None:
            self.preview_workspace.set_stream_state(modality, "live")
        if modality == "ultrasound":
            raw_channels = event.payload.get("channels")
            if not isinstance(raw_channels, (list, tuple)):
                legacy_values = event.payload.get("values")
                raw_channels = [legacy_values] if legacy_values is not None else []
            raw_channel_index = event.payload.get("channel_index")
            channel_index: int | None = None
            if raw_channel_index is not None:
                try:
                    candidate = int(raw_channel_index)
                except (TypeError, ValueError):
                    candidate = -1
                if 0 <= candidate < len(self._us_curves):
                    channel_index = candidate
                    if "ultrasound" not in self._preview_y_ranges:
                        lower, upper = -128.0, 128.0
                        span = upper - lower
                        self._preview_y_ranges["ultrasound"] = (lower, upper)
                        for plot in self._us_plots:
                            plot.setYRange(lower, upper, padding=0.0)
                            plot.setLimits(
                                yMin=lower,
                                yMax=upper,
                                minYRange=span,
                                maxYRange=span,
                            )
                            plot.setMouseEnabled(x=False, y=False)
            raw_metrics = event.payload.get("format_metrics")
            if isinstance(raw_metrics, (list, tuple)):
                for metric_offset, metric in enumerate(raw_metrics[:4]):
                    if not isinstance(metric, Mapping) or not bool(metric.get("all_zero")):
                        continue
                    metric_channel = (
                        channel_index if channel_index is not None else metric_offset
                    )
                    alert_key = (metric_channel, "ALL_ZERO")
                    if alert_key in self._ultrasound_format_alerted:
                        continue
                    message = f"ultrasound 通道 {metric_channel + 1} 当前帧全零；请检查探头、通道和设备连接。"
                    self._append_alert(message)
                    self._add_timeline_event(2, message)
                    self._ultrasound_format_alerted.add(alert_key)
            prepared_channels: list[tuple[int, list[float]]] = []
            for i, raw_channel in enumerate(raw_channels):
                target_index = channel_index if channel_index is not None else i
                if target_index >= len(self._us_curves):
                    break
                values = self._numeric_values(raw_channel)
                if values:
                    prepared_channels.append((target_index, values))
            if prepared_channels:
                targets = [idx for idx, _ in prepared_channels]
                LOG.debug("超声预览更新通道: %s (channel_index=%s)", targets, channel_index)
            self._lock_preview_y_axis("ultrasound", [values for _, values in prepared_channels], self._us_plots)
            for index, values in prepared_channels:
                self._us_curves[index].setData(self._us_x, self._fixed_ultrasound_frame(values))
            return
        if modality == "imu":
            prepared_series: list[tuple[str, list[float]]] = []
            for label, values in self._preview_series(event.payload, IMU_PREVIEW_LABELS):
                numeric = self._numeric_values(values)
                if label in self._imu_traces and numeric:
                    prepared_series.append((label, numeric))
            self._lock_preview_y_axis("imu", [values for _, values in prepared_series],
                                       [trace.plot for trace in self._imu_traces.values()])
            for label, values in prepared_series:
                self._imu_traces[label].append(values)
            return
        if modality == "encoder":
            prepared_series = []
            for label, values in self._preview_series(event.payload, ENCODER_PREVIEW_LABELS):
                numeric = self._numeric_values(values)
                if label in self._enc_traces and numeric:
                    prepared_series.append((label, numeric))
            for label, values in prepared_series:
                self._enc_traces[label].append(values)
            return
        if modality == "emg":
            channels = event.payload.get("channels")
            if not isinstance(channels, (list, tuple)):
                return
            labels = self._emg_preview_labels(event.payload, len(channels))
            if tuple(labels) != tuple(self._emg_traces.keys()):
                self._build_emg_preview(labels)
            for index, raw_values in enumerate(channels):
                values = self._numeric_values(raw_values)
                if values:
                    self._emg_traces[labels[index]].append(values)
            return

    @staticmethod
    def _emg_preview_labels(payload: Mapping[str, Any], count: int) -> list[str]:
        """Resolve per-channel preview labels, falling back to ``emg_XX``.

        Muscle names only become dict keys when they are non-empty and unique;
        otherwise (or when the payload omits ``labels``) we use stable synthetic
        labels so ``_build_emg_preview`` can address each trace deterministically.
        """
        raw = payload.get("labels")
        if isinstance(raw, (list, tuple)) and len(raw) == count:
            labels = [str(item) for item in raw]
            if all(labels) and len(set(labels)) == count:
                return labels
        return [f"emg_{index + 1:02d}" for index in range(count)]

    def _build_emg_preview(self, labels: Sequence[str]) -> None:
        """Rebuild the EMG preview as one window (plot) per configured channel."""
        grid_layout = self._emg_grid_layout
        if grid_layout is None:
            return
        if self._emg_grid_content is not None:
            grid_layout.removeWidget(self._emg_grid_content)
            self._emg_grid_content.deleteLater()
            self._emg_grid_content = None
        self._emg_traces = {}
        container = QWidget()
        layout = QGridLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        columns = 4
        for index, label in enumerate(labels):
            text = str(label)
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            name_label = QLabel(text)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("QLabel { font-weight: 700; color: #374151; }")
            cell_layout.addWidget(name_label)
            plot = HoverDetailsPlotWidget()
            plot.setObjectName(f"emg_ring_{index + 1}")
            plot.setMinimumHeight(110)
            legend = plot.addLegend(offset=(5, 5))
            trace = RingTrace(
                plot,
                _SIGNAL_COLORS[index % len(_SIGNAL_COLORS)],
                text,
                capacity=EMG_PREVIEW_RING_CAPACITY,
                render_stride=4,
            )
            legend.addItem(trace.curve, text)
            plot.setLabel("left", "幅值")
            cell_layout.addWidget(plot, 1)
            layout.addWidget(cell, index // columns, index % columns)
            self._emg_traces[text] = trace
        grid_layout.addWidget(container)
        self._emg_grid_content = container

    def _lock_preview_y_axis(self, modality: str, series: list[list[float]],
                              plots: list["pg.PlotWidget"]) -> None:
        if modality in self._preview_y_ranges or not series:
            return
        values = np.concatenate([np.asarray(channel, dtype=np.float64) for channel in series])
        finite = values[np.isfinite(values)]
        if not finite.size:
            return
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        if modality == "ultrasound" and minimum >= 0:
            lower, upper = 0.0, max(1.0, maximum * 1.1)
        else:
            extent = max(abs(minimum), abs(maximum), 1e-6) * 1.1
            lower, upper = -extent, extent
        span = upper - lower
        self._preview_y_ranges[modality] = (lower, upper)
        for plot in plots:
            plot.setYRange(lower, upper, padding=0)
            plot.setLimits(yMin=lower, yMax=upper, minYRange=span, maxYRange=span)
            plot.setMouseEnabled(x=False, y=False)

    def _fixed_ultrasound_frame(self, values: list[float]) -> np.ndarray:
        source = np.asarray(values, dtype=np.float64)
        if source.size > ULTRASOUND_PREVIEW_SAMPLES:
            indices = np.linspace(0, source.size - 1, ULTRASOUND_PREVIEW_SAMPLES, dtype=np.int64)
            source = source[indices]
        display = np.full(ULTRASOUND_PREVIEW_SAMPLES, np.nan, dtype=np.float64)
        display[: source.size] = source
        return display

    @staticmethod
    def _preview_series(payload: Mapping[str, Any], expected_labels: tuple[str, ...]) -> list[tuple[str, object]]:
        channels = payload.get("channels")
        if isinstance(channels, Mapping):
            return [(label, channels[label]) for label in expected_labels if label in channels]
        if isinstance(channels, (list, tuple)):
            labels = payload.get("labels")
            provided_labels = labels if isinstance(labels, (list, tuple)) else ()
            result: list[tuple[str, object]] = []
            for index, values in enumerate(channels[: len(expected_labels)]):
                candidate = str(provided_labels[index]) if index < len(provided_labels) else expected_labels[index]
                label = candidate if candidate in expected_labels else expected_labels[index]
                result.append((label, values))
            return result
        streams = payload.get("streams")
        if isinstance(streams, (list, tuple)):
            result = []
            for index, stream in enumerate(streams[: len(expected_labels)]):
                if not isinstance(stream, Mapping):
                    continue
                candidate = str(stream.get("label") or expected_labels[index])
                label = candidate if candidate in expected_labels else expected_labels[index]
                result.append((label, stream.get("values")))
            return result
        legacy_values = payload.get("values")
        return [(expected_labels[0], legacy_values)] if legacy_values is not None else []

    def _add_timeline_event(self, category: int, text: str) -> None:
        elapsed = max(0.0, time.monotonic() - self._timeline_started_at)
        self._timeline_x.append(elapsed)
        self._timeline_y.append(float(category))
        self._timeline_text.append(text)

    @staticmethod
    def _numeric_values(value: object) -> list[float]:
        if not isinstance(value, (list, tuple)):
            return []
        converted: list[float] = []
        for item in value:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                converted.append(number)
            if len(converted) >= MAX_PREVIEW_POINTS:
                break
        return converted

    @staticmethod
    def _normalize_modality(value: str) -> str:
        normalized = value.strip().lower()
        if "ultrasound" in normalized:
            return "ultrasound"
        if normalized == "imu" or "imu" in normalized:
            return "imu"
        if "encoder" in normalized:
            return "encoder"
        if "mocap" in normalized or "motion" in normalized or "marker" in normalized:
            return "mocap"
        if "force_plate" in normalized or "forceplate" in normalized:
            return "force_plate"
        if "emg" in normalized or "analog" in normalized:
            return "emg"
        if "sync" in normalized or "pulse" in normalized:
            return "sync_pulse"
        return normalized

    def _handle_completed(self, event: WorkerEvent) -> None:
        self._end_recording_streams()
        self._terminal_event_received = True
        if self._recording_branch_fault is not None:
            self._mark_failed(
                "记录支路已故障，即使 Collector 发布 COMPLETED "
                f"也不能将本 Trial 判为成功：{self._recording_branch_fault}"
            )
            return
        self._trial_succeeded = True
        state = str(event.payload.get("state") or "FINALIZED")
        self._set_trial_state(state)
        self._add_timeline_event(0, state.upper())
        manifest_path = event.payload.get("manifest_path")
        if manifest_path:
            LOG.info("Manifest 已生成: %s", manifest_path)
        else:
            LOG.warning("Collector Worker 已完成，但未返回 Manifest 路径")
        self.start_button.setEnabled(False)
        # XINGYING 的 .cap 保留在其固定工程目录中，本系统只记录对应的 .cap 文件名
        # （由 7061 触发监听写入 raw/xingying_trigger.jsonl + sync_manifest.json），
        # 不做任何搬移。
        self._show_toast("Trial 记录完成", level="SUCCESS")
        self.statusBar().showMessage(event.message or "Trial 数据包已最终化。")

    def _mark_failed(self, message: str) -> None:
        self._end_recording_streams()
        self._stop_xingying_capture()
        self._trial_succeeded = False
        self._set_trial_state("FAILED")
        self.start_button.setEnabled(False)
        self._append_alert(f"FAILED：{message}")
        self._add_timeline_event(2, f"FAILED · {message}")
        self.statusBar().showMessage("Trial 失败；请检查告警信息。")
        LOG.error("Trial FAILED: %s", message)

    def _release_worker(self, worker: WorkerHandle) -> None:
        self._end_recording_streams()
        self._stop_xingying_capture()
        try:
            worker.join(timeout=0)
            worker.close()
        except Exception as exc:
            self._append_alert(f"释放 Worker 资源时出错：{type(exc).__name__}: {exc}")
        self._worker = None
        self._active_trial_uuid = None
        self._active_request = None
        self._recording_preview_handles.clear()
        self._recording_streams_ended = False
        self._stop_requested_at = None
        self._forced_stop_alerted = False
        self._poll_timer.stop()
        self._preflight_ready = False
        for row in self._health_rows.values():
            self.health_table.item(row, HEALTH_COLUMN_MODALITY).setToolTip("")
        # DO NOT clear _preview_connected_modalities — preview workers
        # remain alive and connected across Trials.
        self._set_configuration_locked(False)
        self.start_button.setEnabled(False)
        self._clear_one_trial_metadata()
        self._update_connect_button_state()
        self._update_start_button()
        if self._trial_succeeded:
            self._set_trial_state(
                "PREFLIGHT_READY"
                if self._preview_connected_modalities
                else "IDLE"
            )
            self.statusBar().showMessage(
                "Trial 已最终化；预览连接保持，可立即开始下一个 Trial。",
                8000,
            )
            LOG.info("Trial 成功完成: 预览连接保持中")
        else:
            LOG.warning("Trial 失败: 预览连接保持中，不重连设备")
        self.trial_finished.emit(self._trial_succeeded)
        if self._close_when_finished:
            self._close_when_finished = False
            QTimer.singleShot(0, self.close)

    def _set_configuration_locked(self, locked: bool) -> None:
        self._configuration_locked = locked
        self._refresh_configuration_enabled()

    def _set_preflight_busy(self, busy: bool) -> None:
        self._preflight_busy = busy
        self._refresh_configuration_enabled()

    def _refresh_configuration_enabled(self) -> None:
        enabled = not self._configuration_locked and not self._preflight_busy
        for widget in self._configuration_widgets:
            widget.setEnabled(enabled)
        self._render_device_profile()
        self._update_connect_button_state()
        self._update_start_button()

    @Slot()
    @Slot()
    def _toggle_write(self) -> None:
        """Toggle between start-write and stop-write."""
        if self._worker is not None and self._worker_state in ("RECORDING", "WAITING_SYNC", "PREPARING", "READY"):
            self.request_controlled_stop()
        else:
            self.start_trial()

    def _update_start_button(self) -> None:
        if not hasattr(self, "start_button"):
            return
        trial_active = (
            self._worker is not None
            and self._worker_state in ("RECORDING", "WAITING_SYNC", "PREPARING", "READY", "STOPPING")
        )
        if trial_active:
            self.start_button.setText("停止写盘")
            self.start_button.setStyleSheet(
                "QPushButton { font-weight: 600; padding: 8px; color: #ffffff; background: #dc3545; border: 1px solid #dc3545; border-radius: 4px; }"
            )
            self.start_button.setEnabled(True)
            self.start_button.setToolTip("")
        else:
            self.start_button.setText("开始写盘")
            self.start_button.setStyleSheet(
                "QPushButton { font-weight: 600; padding: 8px; color: #ffffff; background: #0f766e; border: 1px solid #115e59; border-radius: 4px; }"
            )
            subject_valid = bool(
                QRegularExpression(r"^\d{3}$").match(self.subject_code_edit.text().strip()).hasMatch()
            )
            any_connected = bool(self._preview_connected_modalities)
            blockers: list[str] = []
            if not any_connected:
                blockers.append("请先连接至少一个模态的设备预览")
            if not subject_valid:
                blockers.append("受试者编码须为 3 位数字")
            if self._configuration_locked:
                blockers.append("配置已锁定（Trial 进行中）")
            if self._preflight_busy:
                blockers.append("设备预检进行中")
            if self._worker is not None:
                blockers.append("Worker 仍在运行")
            can_start = not blockers
            self.start_button.setEnabled(can_start)
            self.start_button.setToolTip(
                "" if can_start else "无法开始写盘：\n" + "\n".join(f"• {b}" for b in blockers)
            )

    def _set_trial_state(self, state: str) -> None:
        previous = self._worker_state
        normalized = state.strip().upper() or "UNKNOWN"
        self._worker_state = normalized
        if self._elapsed_timer is not None:
            if normalized == "RECORDING" and previous != "RECORDING":
                self._elapsed_timer.start_recording()
            elif previous == "RECORDING" and normalized != "RECORDING":
                self._elapsed_timer.set_recording(False)
        display = {
            "IDLE": "未连接",
            "DISCONNECTED": "未连接",
            "PREFLIGHT_READY": "可采集",
            "PREFLIGHT": "设备预检",
            "PREPARING": "等待同步",
            "READY": "等待同步",
            "WAITING_SYNC": "等待同步",
            "RECORDING": "● 采集中",
            "STOPPING": "保存中",
            "FINALIZING": "保存中",
            "FINALIZED": "可采集",
            "COMPLETED": "可采集",
            "FAILED": "失败",
            "ABORTED": "失败",
            "RECOVERABLE": "失败",
        }.get(normalized, "未连接")
        self.state_label.setText(f"总状态：{display}")
        self.state_label.setToolTip(f"Worker state: {normalized}")
        if display == "失败":
            colors = "background:#f8d7da;color:#842029;border:1px solid #f5c2c7;"
        elif "采集中" in display:
            colors = "background:#dc3545;color:#ffffff;border:1px solid #b02a37;font-size:15px;"
        elif display in {"可采集"}:
            colors = "background:#d1e7dd;color:#0f5132;border:1px solid #badbcc;"
        elif display in {"设备预检", "切换至记录", "等待同步", "保存中"}:
            colors = "background:#fff3cd;color:#664d03;border:1px solid #ffecb5;"
        else:
            colors = "background:#e2e3e5;color:#41464b;border:1px solid #d3d6d8;"
        self.state_label.setStyleSheet(
            f"QLabel {{{colors}padding:6px;border-radius:3px;font-weight:600;}}"
        )

    def _append_alert(self, message: str) -> None:
        upper_message = message.upper()
        error_markers = (
            "失败",
            "错误",
            "异常",
            "超时",
            "FAILED",
            "ERROR",
            "FAULT",
            "掉线",
            "异常退出",
        )
        warning_markers = (
            "警告",
            "降级",
            "丢包",
            "全零",
            "等待",
            "未收到",
            "未 READY",
            "WARNING",
        )
        success_markers = ("成功", "已就绪", "已连接", "记录完成", "SUCCESS")
        if any(marker in upper_message for marker in error_markers):
            level = "ERROR"
        elif any(marker in upper_message for marker in warning_markers):
            level = "WARNING"
        elif any(marker in upper_message for marker in success_markers):
            level = "SUCCESS"
        else:
            level = "INFO"
        if level == "ERROR":
            LOG.error("UI: %s", message)
        elif level == "WARNING":
            LOG.warning("UI: %s", message)
        else:
            LOG.info("UI: %s", message)
        self._show_toast(message, level=level)

    # ── toast overlay ────────────────────────────────────────────────────

    def _show_toast(self, message: str, *, level: str = "INFO") -> None:
        normalized_level = level.strip().upper()
        palettes = {
            "ERROR": ("#FDE8E7", "#7F1D1D", "#D96C68", "⛔", 10_000),
            "WARNING": ("#FFF4D6", "#6B4A00", "#D6A63C", "⚠", 7_000),
            "SUCCESS": ("#E6F4EA", "#1F5D36", "#79B78C", "✓", 4_000),
            "INFO": ("#E8F1FF", "#173B67", "#7AA7D9", "ℹ", 4_000),
        }
        if normalized_level not in palettes:
            normalized_level = "INFO"
        bg, fg, border, icon, timeout_ms = palettes[normalized_level]
        self._toast_label.setProperty("toastLevel", normalized_level)
        self._toast_label.setAccessibleName(
            f"{normalized_level} 通知；点击关闭"
        )
        self._toast_label.setText(f"{icon}  {message}    ×")
        self._toast_label.setStyleSheet(
            "QLabel {"
            f"background-color:{bg};"
            f"color:{fg};"
            f"border:2px solid {border};"
            "border-radius:8px;"
            "font-size:13px;"
            "font-weight:600;"
            "padding:10px 14px;"
            "}"
        )
        self._toast_label.adjustSize()
        self._position_toast()
        self._toast_label.setVisible(True)
        self._toast_label.raise_()
        self._toast_timer.start(timeout_ms)

    @Slot()
    def _hide_toast(self) -> None:
        self._toast_timer.stop()
        self._toast_label.setVisible(False)

    def _position_toast(self) -> None:
        w = self._toast_label.width()
        h = self._toast_label.height()
        parent_w = self.width()
        margin = 16
        available_w = max(0, parent_w - 2 * margin)
        toast_w = min(w, available_w)
        self._toast_label.setGeometry(
            max(margin, (parent_w - toast_w) // 2),
            margin,
            toast_w,
            h,
        )

    @Slot(bool)
    def _set_preview_focus_mode(self, enabled: bool) -> None:
        """Temporarily give the complete content area to preview docks."""

        if enabled:
            if self._preview_focus_previous_sizes is None:
                self._preview_focus_previous_sizes = self._body_splitter.sizes()
            self._controls_scroll.hide()
        else:
            self._controls_scroll.show()
            if self._preview_focus_previous_sizes:
                self._body_splitter.setSizes(self._preview_focus_previous_sizes)
            self._preview_focus_previous_sizes = None
        if self.preview_workspace is not None:
            self.preview_workspace.set_focus_mode(enabled)

    @Slot()
    def _save_preview_workspace_layout(self) -> None:
        workspace = self.preview_workspace
        if workspace is None:
            return
        try:
            self._settings.set_preview_workspace_state(workspace.save_layout())
        except Exception as exc:
            LOG.warning("无法保存预览窗口布局: %s", exc)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_toast_label") and self._toast_label.isVisible():
            self._position_toast()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._save_preview_workspace_layout()
        if self.preview_workspace is not None:
            self.preview_workspace.suspend_layout_tracking()
        preflight_worker = self._preflight_worker
        if preflight_worker is not None:
            self._preflight_timer.stop()
            try:
                preflight_worker.terminate(timeout=0.5)
            except Exception as exc:
                self._append_alert(f"停止设备预检进程失败：{type(exc).__name__}: {exc}")
            if self._preflight_worker_is_alive(preflight_worker):
                if self._close_started_at is None:
                    self._close_started_at = time.monotonic()
                if time.monotonic() - self._close_started_at < 5.0:
                    self.statusBar().showMessage("正在终止设备预检进程，完成后关闭。")
                    event.ignore()
                    QTimer.singleShot(100, self.close)
                    return
            try:
                preflight_worker.join(timeout=0)
                preflight_worker.close()
            except Exception as exc:
                self._append_alert(f"释放设备预检资源失败：{type(exc).__name__}: {exc}")
            self._preflight_worker = None
            self._preflight_root = None
            self._set_preflight_busy(False)

        # ── Handle recording worker first ──
        worker = self._worker
        if worker is not None and self._worker_is_alive(worker):
            # Recording is active: request controlled stop and defer close
            self._close_when_finished = True
            self.request_controlled_stop()
            self.statusBar().showMessage("正在受控停止并最终化 Trial；完成后将自动关闭。")
            event.ignore()
            return
        if worker is not None:
            self._release_worker(worker)

        # ── Stop all preview workers AFTER recording worker is done ──
        self._preview_timer.stop()
        for modality, handle in list(self._preview_workers.items()):
            try:
                handle.request_stop()
                handle.join(timeout=1.0)
                handle.close()
            except Exception:
                try:
                    handle.terminate(timeout=1.0)
                    handle.close()
                except Exception:
                    pass
        self._preview_workers.clear()
        self._preview_connected_modalities.clear()
        LOG.info("关闭窗口：所有预览 worker 已回收")

        # ── Stop button label marker ──
        self._button_poll_timer.stop()
        button_marker = self._button_marker
        self._button_marker = None
        if button_marker is not None:
            try:
                button_marker.stop()
            except Exception as exc:
                LOG.warning("关闭窗口时停止按钮标签监听出错: %s", exc)

        # ── Stop start/stop toggle marker ──
        self._start_stop_poll_timer.stop()
        start_stop_button = self._start_stop_button
        self._start_stop_button = None
        if start_stop_button is not None:
            try:
                start_stop_button.stop()
            except Exception as exc:
                LOG.warning("关闭窗口时停止开始/停止按钮监听出错: %s", exc)

        self._poll_timer.stop()
        self._close_started_at = None
        if self._prompt_event_filter_installed:
            application = QApplication.instance()
            if application is not None:
                application.removeEventFilter(self)
            self._prompt_event_filter_installed = False
        LOG.info("CollectorWindow 已关闭")
        event.accept()
