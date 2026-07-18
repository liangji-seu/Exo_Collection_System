"""Responsive PySide6 shell for the Collector worker process.

Per-modality preview connect/disconnect with independent subprocess workers.
Trial lifecycle: stop previews → start CollectorWorker → restore previews.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QLocale, QRegularExpression, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QDoubleValidator,
    QIntValidator,
    QKeyEvent,
    QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
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
from exo_collection.acquisition.workers import CollectorWorker
from exo_collection.apps.collector.device_preview import (
    AdapterFactory,
    ModalityPreviewHandle,
    ModalityPreviewProcessHandle,
    ProfileModalityAdapterFactory,
)
from exo_collection.apps.collector.device_settings import DEVICE_SETTINGS_DIALOGS
from exo_collection.apps.collector.preflight import (
    CollectorPreflightReport,
    CollectorPreflightWorker,
    run_simulated_preflight,
)
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
from exo_collection.logging_setup import collector_log_path, setup_collector_logging
from exo_collection.orchestration.models import (
    MeasuredConditionMetadata,
    TrialExperimentMetadata,
    TrialRunRequest,
)
from exo_collection.protocols import load_default_protocol
from exo_collection.quality import load_storage_policy

LOG = logging.getLogger("exo_collection.collector.ui")

MODALITIES = ("ultrasound", "imu", "encoder", "sync_pulse")
CRITICAL_MODALITIES = frozenset(MODALITIES)
MAX_PREVIEW_POINTS = 4096
MAX_TIMELINE_EVENTS = 300
SIGNAL_RING_CAPACITY = 1000
ULTRASOUND_PREVIEW_SAMPLES = 512
IMU_PREVIEW_LABELS = ("imu_trunk", "imu_left", "imu_right")
ENCODER_PREVIEW_LABELS = ("left_position", "right_position")
DEFAULT_OPERATOR = "not_recorded"
DEFAULT_CONTROLLED_STOP_TIMEOUT_S = 30.0

PROJECTS: tuple[dict[str, str], ...] = (
    {"project_code": "F", "project_name": "正式"},
    {"project_code": "T", "project_name": "测试"},
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

    def poll_events(self, limit: int = 100) -> list[WorkerEvent]: ...

    def join(self, timeout: float | None = None) -> int | None: ...

    def terminate_for_recovery(self, timeout: float = 5.0) -> int | None: ...

    def close(self) -> None: ...


WorkerFactory = Callable[[TrialRunRequest], WorkerHandle]


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
    """Persistent, non-secret settings for the three supported real devices."""

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
        self.awinda_ids_edit = QLineEdit(
            ", ".join(str(item) for item in imu.get("sensor_ids", ()))
        )
        self.awinda_ids_edit.setPlaceholderText(
            "可留空；或按躯干、左腿、右腿顺序填写 3 个 MTw ID"
        )
        form.addRow("3 个 MTw ID：", self.awinda_ids_edit)

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
            sensor_ids = tuple(
                item.strip()
                for item in self.awinda_ids_edit.text().split(",")
                if item.strip()
            )
            encoder_port = self.encoder_port_edit.text().strip()
            interface_name = str(
                self.ultrasound_interface_combo.currentData() or ""
            ).strip()
            overrides: dict[str, dict[str, Any]] = {
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
            }
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


class RingTrace:
    """Ring-buffer trace backed by a fixed-size numpy array for pyqtgraph."""

    __slots__ = (
        "_buffer", "_capacity", "_count", "_cursor", "_x",
        "curve", "cursor_line", "plot",
    )

    def __init__(
        self, plot: "pg.PlotWidget", pen: str, label: str,
        *, capacity: int = SIGNAL_RING_CAPACITY,
    ) -> None:
        if capacity < 2:
            raise ValueError("ring trace capacity must be at least two")
        self._capacity = int(capacity)
        self._buffer = np.full(self._capacity, np.nan, dtype=np.float64)
        self._x = np.arange(self._capacity, dtype=np.float64)
        self._cursor = 0
        self._count = 0
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

    def _render(self) -> None:
        display = self._buffer.copy()
        if self._count == self._capacity:
            display[self._cursor] = np.nan
        self.curve.setData(self._x, display)
        if self._count:
            self.cursor_line.setPos((self._cursor - 1) % self._capacity)

    def reset(self) -> None:
        self._buffer.fill(np.nan)
        self._cursor = 0
        self._count = 0
        self.curve.setData(self._x, self._buffer)
        self.cursor_line.setPos(0.0)


# ── Preview Worker Factory Helpers ─────────────────────────────────────────

# ── CollectorWindow ────────────────────────────────────────────────────────


class CollectorWindow(QMainWindow):
    """Collect one Trial at a time with per-modality preview workers."""

    trial_started = Signal(object)
    trial_finished = Signal(bool)

    def __init__(
        self,
        data_root: str | Path,
        *,
        settings: SharedAppSettings | None = None,
        worker_factory: WorkerFactory = CollectorWorker,
        preflight_worker_factory: PreflightWorkerFactory = simulated_preflight_worker_factory,
        preview_worker_factory: AdapterFactory | None = None,
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
        self._controlled_stop_timeout_s = float(controlled_stop_timeout_s)
        self._worker: WorkerHandle | None = None
        self._active_trial_uuid: str | None = None
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
        self._missing_trigger_alerted = False

        # Per-modality preview workers
        self._preview_workers: dict[str, ModalityPreviewHandle] = {}
        self._preview_connected_modalities: set[str] = set()
        self._preview_connection_status: dict[str, str] = {
            m: "未连接" for m in MODALITIES
        }
        self._preview_disconnect_deadlines: dict[str, float] = {}
        self._preview_restore_modalities: set[str] = set()
        self._pending_trial_request: TrialRunRequest | None = None
        self._injected_preview_factory = preview_worker_factory

        self._experiment_metadata = TrialExperimentMetadata()
        self._experiment_metadata_by_identity: dict[tuple[str, str], TrialExperimentMetadata] = {}
        self._metadata_identity_key: tuple[str, str] | None = None
        self._metadata_condition_code: str | None = None

        self._session_key: tuple[str, str, str] | None = None
        self._session_uuid = uuid4()

        self._health_rows = {name: index for index, name in enumerate(MODALITIES)}
        self._last_health_status: dict[str, str] = {}
        self._us_plots: list["pg.PlotWidget"] = []
        self._us_curves: list["pg.PlotDataItem"] = []
        self._us_x = np.arange(ULTRASOUND_PREVIEW_SAMPLES, dtype=np.float64)
        self._ultrasound_format_alerted: set[tuple[int, str]] = set()
        self._imu_traces: dict[str, RingTrace] = {}
        self._enc_traces: dict[str, RingTrace] = {}
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
        self.project_combo.currentIndexChanged.connect(self._activate_selected_metadata_identity)
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
        self._set_trial_state("IDLE")
        self._update_start_button()

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

        controls = QWidget()
        controls.setObjectName("controls_content")
        controls.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # ── Trial Settings ──
        metadata_box = QGroupBox("Trial 设置")
        form = QFormLayout(metadata_box)
        root_row = QHBoxLayout()
        self.data_root_edit = QLineEdit(str(data_root))
        self.data_root_edit.setObjectName("data_root")
        self.data_root_edit.textChanged.connect(self._invalidate_preflight)
        root_row.addWidget(self.data_root_edit, 1)
        self.browse_button = QPushButton("选择…")
        self.browse_button.clicked.connect(self.choose_data_root)
        root_row.addWidget(self.browse_button)
        form.addRow("数据根目录：", root_row)

        # Row 1: 项目 + 受试者编码
        row1 = QHBoxLayout()
        self.project_combo = QComboBox()
        self.project_combo.setObjectName("project")
        for project in PROJECTS:
            self.project_combo.addItem(
                f"{project['project_code']} — {project['project_name']}",
                dict(project),
            )
        self.project_combo.setCurrentIndex(1)
        row1.addWidget(QLabel("项目："))
        row1.addWidget(self.project_combo, 3)

        self.subject_code_edit = QLineEdit("001")
        self.subject_code_edit.setObjectName("subject_code")
        self.subject_code_edit.setMaxLength(3)
        self.subject_code_edit.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"\d{3}"), self)
        )
        self.subject_code_edit.editingFinished.connect(self.normalize_subject_code)
        self.subject_code_edit.textChanged.connect(self._update_start_button)
        row1.addWidget(QLabel("受试者编码："))
        row1.addWidget(self.subject_code_edit, 1)
        form.addRow(row1)

        # Row 2: 工况 + 重复轮次
        row2 = QHBoxLayout()
        self.condition_combo = QComboBox()
        self.condition_combo.setObjectName("condition")
        for condition in CONDITIONS:
            self.condition_combo.addItem(
                f"{condition['condition_code']} — {condition['condition_name']}",
                dict(condition),
            )
        self.condition_combo.setCurrentIndex(1)
        row2.addWidget(QLabel("工况："))
        row2.addWidget(self.condition_combo, 3)

        self.repeat_spin = QSpinBox()
        self.repeat_spin.setObjectName("repeat_index")
        self.repeat_spin.setRange(1, 9999)
        self.repeat_spin.setValue(1)
        row2.addWidget(QLabel("重复轮次："))
        row2.addWidget(self.repeat_spin, 1)
        form.addRow(row2)
        controls_layout.addWidget(metadata_box)

        experiment_box = QGroupBox("详细信息")
        experiment_layout = QHBoxLayout(experiment_box)
        self.experiment_metadata_button = QPushButton("填写 / 修改…")
        self.experiment_metadata_button.setObjectName("edit_experiment_metadata")
        self.experiment_metadata_button.clicked.connect(self.edit_experiment_metadata)
        experiment_layout.addWidget(self.experiment_metadata_button)
        self.experiment_metadata_summary = QLabel("未填写；不影响采集")
        self.experiment_metadata_summary.setObjectName("experiment_metadata_summary")
        self.experiment_metadata_summary.setWordWrap(True)
        experiment_layout.addWidget(self.experiment_metadata_summary, 1)
        controls_layout.addWidget(experiment_box)

        # ── Trial buttons ──
        buttons = QHBoxLayout()
        self.connect_all_button = QPushButton("全部连接")
        self.connect_all_button.setObjectName("connect_all")
        self.connect_all_button.clicked.connect(self._toggle_connect_all)
        self.connect_all_button.setMinimumWidth(105)
        buttons.addWidget(self.connect_all_button)
        self.start_button = QPushButton("开始写盘")
        self.start_button.setObjectName("start_trial")
        self.start_button.setStyleSheet(
            "QPushButton { font-weight: 600; padding: 8px; color: #ffffff; background: #0d6efd; border: 1px solid #0d6efd; border-radius: 4px; }"
        )
        self.start_button.clicked.connect(self._toggle_write)
        self.start_button.setMinimumWidth(105)
        buttons.addWidget(self.start_button)
        controls_layout.addLayout(buttons)

        # ── Device Connection Area ──
        connection_box = QGroupBox("设备连接")
        connection_layout = QGridLayout(connection_box)
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
        connection_layout.addWidget(self._device_profile_label, len(MODALITIES) + 1, 0, 1, 3)

        # Per-modality rows
        _modality_labels = {
            "ultrasound": "超声", "imu": "IMU", "encoder": "电机编码器", "sync_pulse": "同步脉冲",
        }
        for row_idx, modality in enumerate(MODALITIES, start=1):
            configure_btn = QPushButton(_modality_labels[modality])
            configure_btn.setObjectName(f"configure_{modality}")
            configure_btn.setProperty("buttonRole", "deviceConfig")
            configure_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            configure_btn.setToolTip(f"设置{_modality_labels[modality]}设备参数（自动保存）")
            configure_btn.clicked.connect(
                lambda _checked=False, selected=modality: self.edit_modality_device_settings(selected)
            )
            connection_layout.addWidget(configure_btn, row_idx, 0)
            self._configure_buttons[modality] = configure_btn

            status_label = QLabel("")
            status_label.setObjectName(f"connect_status_{modality}")
            status_label.setFixedSize(20, 20)
            self._style_connection_indicator(status_label, "未连接")
            status_label.setToolTip("状态：未连接")
            connection_layout.addWidget(
                status_label,
                row_idx,
                1,
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
            self._connect_status_labels[modality] = status_label

            btn_container = QHBoxLayout()
            connect_btn = QPushButton("连接")
            connect_btn.setObjectName(f"connect_{modality}")
            connect_btn.setProperty("buttonRole", "connect")
            disconnect_btn = QPushButton("断开")
            disconnect_btn.setObjectName(f"disconnect_{modality}")
            disconnect_btn.setProperty("buttonRole", "disconnect")

            def _make_connect_handler(m: str):
                return lambda: self._connect_modality(m)
            def _make_disconnect_handler(m: str):
                return lambda: self._disconnect_modality(m)

            connect_btn.clicked.connect(_make_connect_handler(modality))
            disconnect_btn.clicked.connect(_make_disconnect_handler(modality))
            disconnect_btn.setEnabled(False)
            connect_btn.setMinimumWidth(72)
            disconnect_btn.setMinimumWidth(72)

            btn_container.addWidget(connect_btn)
            btn_container.addWidget(disconnect_btn)
            connection_layout.addLayout(btn_container, row_idx, 2)
            self._connect_buttons[modality] = connect_btn
            self._disconnect_buttons[modality] = disconnect_btn

        controls_layout.addWidget(connection_box)

        # ── Health Table ──
        health_box = QGroupBox("设备健康与样本计数")
        health_layout = QVBoxLayout(health_box)
        self.health_table = QTableWidget(len(MODALITIES), 7)
        self.health_table.setObjectName("health_table")
        self.health_table.setHorizontalHeaderLabels(
            ["模态", "健康", "样本/帧", "实际速率", "丢包", "队列", "最近更新"]
        )
        self.health_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.health_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.health_table.setAlternatingRowColors(True)
        self.health_table.verticalHeader().setVisible(False)
        for row, modality in enumerate(MODALITIES):
            self.health_table.setItem(row, 0, QTableWidgetItem(modality))
            self.health_table.setItem(row, 1, QTableWidgetItem("DISCONNECTED"))
            self.health_table.setItem(row, 2, QTableWidgetItem("0"))
            self.health_table.setItem(row, 3, QTableWidgetItem("-"))
            self.health_table.setItem(row, 4, QTableWidgetItem("-"))
            self.health_table.setItem(row, 5, QTableWidgetItem("-"))
            self.health_table.setItem(row, 6, QTableWidgetItem("-"))
        self.health_table.resizeColumnsToContents()
        health_layout.addWidget(self.health_table)
        controls_layout.addWidget(health_box)

        # ── Sync Status ──
        sync_box = QGroupBox("同步状态")
        sync_layout = QGridLayout(sync_box)
        sync_layout.addWidget(QLabel("状态："), 0, 0)
        self.sync_status_label = QLabel("未开始")
        self.sync_status_label.setObjectName("sync_status")
        sync_layout.addWidget(self.sync_status_label, 0, 1)
        sync_layout.addWidget(QLabel("合格触发："), 0, 2)
        self.trigger_count_label = QLabel("0")
        self.trigger_count_label.setObjectName("trigger_count")
        sync_layout.addWidget(self.trigger_count_label, 0, 3)
        sync_layout.addWidget(QLabel("首触发："), 1, 0)
        self.first_trigger_label = QLabel("—")
        self.first_trigger_label.setObjectName("first_trigger")
        self.first_trigger_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sync_layout.addWidget(self.first_trigger_label, 1, 1, 1, 3)
        sync_layout.addWidget(QLabel("质量："), 2, 0)
        self.sync_quality_label = QLabel("—")
        self.sync_quality_label.setObjectName("sync_quality")
        sync_layout.addWidget(self.sync_quality_label, 2, 1, 1, 3)
        controls_layout.addWidget(sync_box)

        # ── Toast overlay for alerts ──
        self._toast_label = QLabel(self)
        self._toast_label.setObjectName("toast")
        self._toast_label.setWordWrap(True)
        self._toast_label.setMaximumWidth(480)
        self._toast_label.setMinimumHeight(36)
        self._toast_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._toast_label.setVisible(False)
        self._toast_label.setContentsMargins(16, 8, 16, 8)
        self._toast_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._hide_toast)
        self._manifest_and_log_row = QHBoxLayout()
        self.manifest_label = QLabel("Manifest：尚未生成")
        self.manifest_label.setObjectName("manifest_path")
        self.manifest_label.setWordWrap(True)
        self.manifest_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._manifest_and_log_row.addWidget(self.manifest_label, 1)
        self.open_log_dir_button = QPushButton("打开日志目录")
        self.open_log_dir_button.setObjectName("open_log_dir")
        self.open_log_dir_button.clicked.connect(self._open_log_directory)
        self._manifest_and_log_row.addWidget(self.open_log_dir_button)
        controls_layout.addLayout(self._manifest_and_log_row)
        controls_scroll.setWidget(controls)
        body.addWidget(controls_scroll)

        # ── Preview Plots ──
        preview_box = QGroupBox("实时预览（固定长度循环显示；不参与原始写盘）")
        preview_layout = QVBoxLayout(preview_box)
        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")

        us_grid = QGroupBox("超声 · 4 通道当前单帧")
        us_grid.setObjectName("ultrasound_grid")
        us_grid_layout = QGridLayout(us_grid)
        us_grid_layout.setContentsMargins(0, 0, 0, 0)
        for i in range(4):
            plot = pg.PlotWidget(title=f"超声通道 {i + 1} · 当前帧")
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
            curve = plot.plot(pen=pg.mkPen("#2457c5", width=1.2))
            curve.setData(self._us_x, np.full(ULTRASOUND_PREVIEW_SAMPLES, np.nan, dtype=np.float64))
            self._us_plots.append(plot)
            self._us_curves.append(curve)
            us_grid_layout.addWidget(plot, i // 2, i % 2)
        preview_layout.addWidget(us_grid, 4)

        imu_grid = QGroupBox("IMU · 3 个传感器 acc_x 循环帧")
        imu_grid.setObjectName("imu_ring_grid")
        imu_layout = QHBoxLayout(imu_grid)
        imu_layout.setContentsMargins(0, 0, 0, 0)
        for index, label in enumerate(IMU_PREVIEW_LABELS):
            plot = pg.PlotWidget()
            plot.setObjectName(f"imu_ring_{label}")
            trace = RingTrace(plot, "#1a936f", f"IMU {index + 1} · {label} · acc_x")
            self._imu_traces[label] = trace
            imu_layout.addWidget(plot, 1)
        preview_layout.addWidget(imu_grid, 2)

        enc_grid = QGroupBox("电机编码器 · 左右位置循环帧")
        enc_grid.setObjectName("encoder_ring_grid")
        enc_layout = QHBoxLayout(enc_grid)
        enc_layout.setContentsMargins(0, 0, 0, 0)
        for label in ENCODER_PREVIEW_LABELS:
            plot = pg.PlotWidget()
            plot.setObjectName(f"encoder_ring_{label}")
            side = "左侧" if label.startswith("left") else "右侧"
            trace = RingTrace(plot, "#d97706", f"{side}电机编码器 · position")
            self._enc_traces[label] = trace
            enc_layout.addWidget(plot, 1)
        preview_layout.addWidget(enc_grid, 2)

        body.addWidget(preview_box)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([630, 1270])
        outer.addWidget(body, 1)

        self.setCentralWidget(central)
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

    def _render_device_profile(self) -> None:
        hardware = self._selected_device_profile_key() == "hardware"

        if hardware:
            self._device_profile_label.setText(
                "真实设备模式：Raw Ethernet 超声 + Xsens MTw IMU + Teensy 编码器。"
                "点击蓝色模态名称可分别设置；参数保存后自动恢复；同步脉冲仍为模拟台架信号。"
            )
            self._device_profile_label.setStyleSheet("color:#842029;font-weight:600;")
        else:
            self._device_profile_label.setText(
                "当前为自动化测试用模拟设备；正常启动并保存任一设备设置后切换为真实设备模式。"
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
        display = {
            "ultrasound": "超声",
            "imu": "IMU",
            "encoder": "电机编码器",
            "sync_pulse": "同步脉冲",
        }[modality]
        self.statusBar().showMessage(
            f"{display}设备设置已保存；下次启动将自动恢复。", 8000
        )
        LOG.info("%s 设备设置已保存并持久化", modality)

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
        for modality, row in self._health_rows.items():
            self.health_table.item(row, 1).setText("UNKNOWN")
            self.health_table.item(row, 1).setToolTip("")
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
        if project_code not in {"F", "T"}:
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
            self.health_table.item(row, 1).setText(status)
            if report is not None and modality in report.devices:
                result = report.devices[modality]
                self.health_table.item(row, 1).setToolTip(
                    f"{result.device_id} · {result.message} · channels={result.channel_count} · raw={result.observed_raw_data}"
                )
                self.health_table.item(row, 3).setText("-" if result.actual_rate_hz is None else f"{result.actual_rate_hz:.1f}")
                self.health_table.item(row, 5).setText(f"0/{result.queue_capacity}")
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
            self.statusBar().showMessage(f"四个必需模态已实际连接/准备/采样，同步上升沿已观测。{storage}", 8000)
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
    def _style_connection_indicator(label: QLabel, status: str) -> None:
        normalized = status.strip().upper()
        if normalized in {"READY", "已连接"}:
            indicator_state, fill, border = "green", "#22C55E", "#15803D"
        elif any(token in status for token in ("连接中", "断开中", "启动中")):
            indicator_state, fill, border = "yellow", "#FBBF24", "#D97706"
        else:
            indicator_state, fill, border = "red", "#EF4444", "#B91C1C"
        label.setText("")
        label.setProperty("indicatorState", indicator_state)
        label.setStyleSheet(
            f"QLabel {{ background-color:{fill}; border:2px solid {border}; "
            "border-radius:10px; }}"
        )

    def _set_preview_status(self, modality: str, status: str, device_id: str,
                            simulated: bool, error: str | None = None) -> None:
        """Update the per-modality UI status labels."""
        if modality in self._connect_status_labels:
            label = self._connect_status_labels[modality]
            source = "模拟" if simulated else "真实"
            self._style_connection_indicator(label, status)
            tooltip_lines = [f"状态：{status}", f"来源：{source}"]
            if device_id:
                tooltip_lines.append(f"设备 ID：{device_id}")
            if error:
                tooltip_lines.append(f"详情：{error}")
            label.setToolTip("\n".join(tooltip_lines))
            label.setAccessibleName(f"{modality} 状态：{status}")

    @Slot()
    def _connect_modality(self, modality: str) -> None:
        """Spawn a single-modality preview worker for one modality."""
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
    @Slot()
    def _toggle_connect_all(self) -> None:
        """Toggle between connect-all and disconnect-all."""
        if self._preview_workers:
            for modality in list(self._preview_workers.keys()):
                self._disconnect_modality(modality)
        else:
            for modality in MODALITIES:
                self._connect_modality(modality)

    def _update_connect_button_state(self) -> None:
        """Update connect-all toggle and per-modality buttons."""
        has_any_connection = bool(self._preview_workers)
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

        for modality in MODALITIES:
            connect_button = self._connect_buttons.get(modality)
            disconnect_button = self._disconnect_buttons.get(modality)
            if connect_button is None or disconnect_button is None:
                continue
            active = modality in self._preview_workers
            stopping = modality in self._preview_disconnect_deadlines
            connect_button.setText("连接")
            connect_button.setEnabled(can_change and not active)
            disconnect_button.setText("断开中…" if stopping else "断开")
            disconnect_button.setEnabled(
                can_change and active and not stopping
            )

        self._update_start_button()

    @Slot()
    def _poll_preview_workers(self) -> None:
        """Poll events from all active preview workers and dispatch to UI handlers."""
        if not self._preview_workers:
            self._preview_timer.stop()
            self._maybe_launch_pending_trial()
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
        self._maybe_launch_pending_trial()

    def _handle_preview_worker_event(self, event: WorkerEvent,
                                      handle: ModalityPreviewHandle,
                                      modality: str) -> None:
        if event.event_type is WorkerEventType.STATE:
            state = str(event.payload.get("state") or "UNKNOWN")
            if state == "READY":
                self._preview_connected_modalities.add(modality)
                self._preview_connection_status[modality] = "已连接"
                self._set_preview_status(modality, "READY", handle.device_id, handle.simulated)
                self._update_connect_button_state()
                self._update_start_button()
                self._append_alert(
                    f"{modality} ({handle.device_id}) "
                    f"{'模拟' if handle.simulated else '真实'}预览已就绪。"
                )
                LOG.info("%s (%s) preview READY simulated=%s", modality, handle.device_id, handle.simulated)
            elif state in ("CONNECTING", "PREVIEW_STARTING"):
                pass
            elif state == "DISCONNECTED":
                self._preview_connected_modalities.discard(modality)
                self._preview_connection_status[modality] = "未连接"
                self._update_connect_button_state()
                self._update_start_button()
        elif event.event_type is WorkerEventType.FAILED:
            error_msg = event.message or "未知错误"
            full_tb = str(event.payload.get("traceback") or "")
            self._preview_connected_modalities.discard(modality)
            self._preview_connection_status[modality] = "错误"
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

    def _handle_preview_health(self, event: WorkerEvent, modality: str) -> None:
        payload = event.payload
        row = self._health_rows.get(modality)
        if row is None:
            return
        status = str(payload.get("status") or "UNKNOWN").upper()
        self.health_table.item(row, 1).setText(status)
        sample_count = payload.get("sample_count")
        if sample_count is not None:
            self.health_table.item(row, 2).setText(str(int(sample_count)))
        rate = payload.get("actual_sample_rate_hz")
        self.health_table.item(row, 3).setText("-" if rate is None else f"{float(rate):.1f} Hz")
        dropped = payload.get("dropped_packets")
        self.health_table.item(row, 4).setText("-" if dropped is None else str(int(dropped)))
        depth = payload.get("queue_depth")
        capacity = payload.get("queue_capacity")
        self.health_table.item(row, 5).setText(f"{int(depth)}/{int(capacity)}" if capacity else "-")
        sampled_at = str(payload.get("sampled_at_utc") or "").strip()
        self.health_table.item(row, 6).setText(sampled_at or "-")
        previous = self._last_health_status.get(modality)
        self._last_health_status[modality] = status
        if status in {"DEGRADED", "UNHEALTHY", "FAULT"} and status != previous:
            detail = event.message or str(payload.get("message") or "")
            suffix = f"：{detail}" if detail else ""
            self._append_alert(f"{modality} 健康状态 {status}{suffix}")

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
                self.health_table.item(row, 1).setText("DISCONNECTED")
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
        if project_code not in {"F", "T"} or not project_name:
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
            or self._pending_trial_request is not None
        ):
            return

        # Build request first (validates input)
        try:
            request = self.build_request()
        except Exception as exc:
            self._append_alert(f"无法构建 Trial 请求：{type(exc).__name__}: {exc}")
            self.statusBar().showMessage("Trial 请求构建失败。")
            return

        # Log modality details before starting
        for modality in MODALITIES:
            device_id, simulated = self._get_modality_info(modality)
            LOG.info(
                "Trial 请求模态: %s adapter=%s device_id=%s simulated=%s",
                modality, "<profile>", device_id, simulated,
            )

        # Check that required modalities are READY (connected via preview)
        required_connected = True
        for modality in CRITICAL_MODALITIES:
            if modality not in self._preview_connected_modalities:
                required_connected = False
                self._append_alert(
                    f"{modality} 尚未连接/就绪。请先点击对应模态的'连接'按钮。"
                )

        if not required_connected:
            self.statusBar().showMessage("请先连接所有必需模态的设备预览。")
            self._update_start_button()
            return

        # Stop all preview workers asynchronously to release exclusive device
        # handles. The GUI timer launches the recording worker only after every
        # preview process has exited.
        self._preview_restore_modalities = set(self._preview_connected_modalities)
        self._pending_trial_request = request
        self._set_configuration_locked(True)
        self._set_trial_state("SWITCHING_TO_RECORD")
        self._stop_all_preview_workers_for_trial()
        self.statusBar().showMessage("正在从预览切换到记录…")
        self._append_alert("正在停止所有预览 Worker；设备释放后将启动 Trial 记录。")
        LOG.info("正在停止预览 workers 并启动 Trial")
        self._preview_timer.start()
        self._maybe_launch_pending_trial()

    def _launch_trial_worker(self, request: TrialRunRequest) -> None:
        """Start the recording worker after preview devices are fully released."""
        worker: WorkerHandle | None = None
        try:
            worker = self._worker_factory(request)
            worker.start()
        except Exception as exc:
            if worker is not None:
                try:
                    if not self._worker_is_alive(worker):
                        worker.join(timeout=0)
                        worker.close()
                except Exception:
                    pass
            self._set_trial_state("FAILED")
            self._append_alert(f"无法启动 Trial：{type(exc).__name__}: {exc}")
            self.statusBar().showMessage("Trial 启动失败。")
            LOG.error("Trial 启动失败: %s", exc)
            self._set_configuration_locked(False)
            self._restore_preview_workers()
            return

        self._worker = worker
        self._active_trial_uuid = str(request.trial_uuid)
        self._terminal_event_received = False
        self._dead_poll_count = 0
        self._stop_requested = False
        self._stop_requested_at = None
        self._forced_stop_alerted = False
        self._close_when_finished = False
        self._trial_succeeded = False
        self._reset_trial_display()
        self.start_button.setEnabled(True)
        self._set_trial_state("PREPARING")
        self._poll_timer.start()
        self.trial_started.emit(request)
        self.statusBar().showMessage(f"Trial {request.trial_uuid} 已交给独立 Collector Worker。")
        LOG.info("Trial 已启动: %s", request.trial_uuid)

    def _stop_all_preview_workers_for_trial(self) -> None:
        """Request all preview workers to stop without blocking the GUI thread."""
        for modality, handle in list(self._preview_workers.items()):
            self._preview_connected_modalities.discard(modality)
            self._preview_disconnect_deadlines[modality] = time.monotonic() + 5.0
            self._preview_connection_status[modality] = "断开中"
            self._set_preview_status(
                modality, "切换至记录中", handle.device_id, handle.simulated
            )
            try:
                handle.request_stop()
            except Exception as exc:
                LOG.warning("停止 %s 预览时出错: %s", modality, exc)
        self._update_connect_button_state()

    def _maybe_launch_pending_trial(self) -> None:
        request = self._pending_trial_request
        if request is None:
            return
        if self._preview_workers:
            return
        self._pending_trial_request = None
        self._append_alert("所有预览设备已释放；现在开始 Trial 原始写盘。")
        self._launch_trial_worker(request)

    def _restore_preview_workers(self) -> None:
        """Attempt to reconnect preview workers for previously connected modalities."""
        prev_connected = [m for m in MODALITIES if m in self._preview_restore_modalities]
        self._preview_restore_modalities.clear()
        if not prev_connected:
            return
        self._append_alert("正在恢复预览连接…")
        LOG.info("正在恢复 %d 个模态的预览连接", len(prev_connected))
        for modality in prev_connected:
            try:
                self._connect_modality(modality)
            except Exception as exc:
                self._append_alert(f"恢复 {modality} 预览失败：{type(exc).__name__}: {exc}")
                LOG.error("恢复 %s 预览失败: %s", modality, exc)

    def _reset_trial_display(self) -> None:
        # Do NOT clear alerts — keep the log history
        self.manifest_label.setText("Manifest：正在采集，尚未最终化")
        self._last_health_status.clear()
        self._missing_trigger_alerted = False
        self.sync_status_label.setText("等待同步")
        self.trigger_count_label.setText("0")
        self.first_trigger_label.setText("—")
        self.sync_quality_label.setText("WAITING")
        self.sync_status_label.setStyleSheet("")
        for modality, row in self._health_rows.items():
            self.health_table.item(row, 1).setText("UNKNOWN")
            self.health_table.item(row, 2).setText("0")
            self.health_table.item(row, 3).setText("-")
            self.health_table.item(row, 4).setText("-")
            self.health_table.item(row, 5).setText("-")
            self.health_table.item(row, 6).setText("-")
        empty_ultrasound = np.full(ULTRASOUND_PREVIEW_SAMPLES, np.nan, dtype=np.float64)
        for curve in self._us_curves:
            curve.setData(self._us_x, empty_ultrasound)
        self._ultrasound_format_alerted.clear()
        for trace in self._imu_traces.values():
            trace.reset()
        for trace in self._enc_traces.values():
            trace.reset()
        self._timeline_started_at = time.monotonic()
        self._timeline_x.clear()
        self._timeline_y.clear()
        self._timeline_text.clear()
        self._add_timeline_event(0, "PREPARING")

    @Slot()
    def request_controlled_stop(self) -> None:
        if self._worker is None or self._stop_requested:
            return
        try:
            self._worker.request_stop()
        except Exception as exc:
            self._append_alert(f"发送停止请求失败：{type(exc).__name__}: {exc}")
            return
        self._stop_requested = True
        self._stop_requested_at = time.monotonic()
        self.start_button.setEnabled(False)
        self._set_trial_state("STOPPING")
        self._append_alert("已发送受控停止请求；正在等待 Writer flush 与 Trial 最终化。")
        LOG.info("Trial 受控停止请求已发送")

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
        elif event.event_type is WorkerEventType.SYNC:
            self._handle_sync(event.payload, record_event=True)
        elif event.event_type is WorkerEventType.HEALTH:
            self._handle_health(event)
        elif event.event_type is WorkerEventType.METRIC:
            self._handle_metric(event.payload)
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

    def _handle_health(self, event: WorkerEvent) -> None:
        payload = event.payload
        modality = self._normalize_modality(
            event.modality or str(payload.get("modality") or payload.get("device_id") or "")
        )
        if modality not in self._health_rows:
            return
        row = self._health_rows[modality]
        status = str(payload.get("status") or "UNKNOWN").upper()
        self.health_table.item(row, 1).setText(status)
        if "sample_count" in payload:
            self.health_table.item(row, 2).setText(str(int(payload["sample_count"])))
        rate = payload.get("actual_sample_rate_hz")
        self.health_table.item(row, 3).setText("-" if rate is None else f"{float(rate):.1f} Hz")
        dropped = payload.get("dropped_packets")
        self.health_table.item(row, 4).setText("-" if dropped is None else str(int(dropped)))
        depth = payload.get("queue_depth")
        capacity = payload.get("queue_capacity")
        self.health_table.item(row, 5).setText(str(int(depth)) if depth is not None and capacity is None else
                                               f"{int(depth)}/{int(capacity)}" if capacity is not None else "-")
        sampled_at = str(payload.get("sampled_at_utc") or "").strip()
        self.health_table.item(row, 6).setText(sampled_at or "-")
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
                    self.health_table.item(row, 2).setText(str(int(count)))
        if "pulse_event_count" in payload:
            row = self._health_rows["sync_pulse"]
            self.health_table.item(row, 2).setToolTip(f"已检测边沿：{int(payload['pulse_event_count'])}")
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
        first_trigger = payload.get("first_trigger_host_monotonic_ns")
        trigger_utc = str(payload.get("trigger_time_utc") or "").strip()
        labels = {
            "WAITING_SYNC": "等待同步触发",
            "TRIGGERED": "已同步",
            "MISSING_TRIGGER": "缺少同步触发",
        }
        self.sync_status_label.setText(labels.get(status, status))
        self.trigger_count_label.setText(str(trigger_count))
        if trigger_utc:
            first_text = trigger_utc
            if first_trigger is not None:
                first_text += f" · host {int(first_trigger)} ns"
        elif first_trigger is not None:
            first_text = f"host {int(first_trigger)} ns"
        else:
            first_text = "—"
        self.first_trigger_label.setText(first_text)
        self.sync_quality_label.setText(quality)
        if status == "MISSING_TRIGGER" or quality == "FAIL":
            self.sync_status_label.setStyleSheet(
                "QLabel { color:#842029; background:#f8d7da; padding:4px; font-weight:700; }"
            )
            if not self._missing_trigger_alerted:
                message = "严重：未检测到合格同步触发，本 Trial 不得作为已同步采集使用。"
                self._append_alert(message)
                self._add_timeline_event(2, message)
                self._missing_trigger_alerted = True
            self._set_trial_state("FAILED")
        elif status == "TRIGGERED" and quality == "PASS":
            self.sync_status_label.setStyleSheet(
                "QLabel { color:#0f5132; background:#d1e7dd; padding:4px; font-weight:700; }"
            )
        else:
            self.sync_status_label.setStyleSheet(
                "QLabel { color:#664d03; background:#fff3cd; padding:4px; font-weight:600; }"
            )
        if record_event:
            self._add_timeline_event(1, f"{status} · {quality} · trigger={trigger_count}")

    def _handle_preview(self, event: WorkerEvent) -> None:
        modality = self._normalize_modality(event.modality or "")
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
            self._lock_preview_y_axis("encoder", [values for _, values in prepared_series],
                                       [trace.plot for trace in self._enc_traces.values()])
            for label, values in prepared_series:
                self._enc_traces[label].append(values)
            return

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
        if "sync" in normalized or "pulse" in normalized:
            return "sync_pulse"
        return normalized

    def _handle_completed(self, event: WorkerEvent) -> None:
        self._terminal_event_received = True
        self._trial_succeeded = True
        state = str(event.payload.get("state") or "FINALIZED")
        self._set_trial_state(state)
        self._add_timeline_event(0, state.upper())
        manifest_path = event.payload.get("manifest_path")
        if manifest_path:
            self.manifest_label.setText(f"Manifest：{manifest_path}")
            LOG.info("Manifest 已生成: %s", manifest_path)
        else:
            self.manifest_label.setText("Manifest：Worker 已完成，但未返回路径")
        self.start_button.setEnabled(False)
        self.statusBar().showMessage(event.message or "Trial 数据包已最终化。")

    def _mark_failed(self, message: str) -> None:
        self._trial_succeeded = False
        self._set_trial_state("FAILED")
        self.start_button.setEnabled(False)
        self._append_alert(f"FAILED：{message}")
        self._add_timeline_event(2, f"FAILED · {message}")
        self.statusBar().showMessage("Trial 失败；请检查告警信息。")
        LOG.error("Trial FAILED: %s", message)

    def _release_worker(self, worker: WorkerHandle) -> None:
        try:
            worker.join(timeout=0)
            worker.close()
        except Exception as exc:
            self._append_alert(f"释放 Worker 资源时出错：{type(exc).__name__}: {exc}")
        self._worker = None
        self._active_trial_uuid = None
        self._stop_requested_at = None
        self._forced_stop_alerted = False
        self._poll_timer.stop()
        self._preflight_ready = False
        for _modality, row in self._health_rows.items():
            self.health_table.item(row, 1).setText("UNKNOWN")
            self.health_table.item(row, 1).setToolTip("")
        # Clear preview connected state — must explicitly reconnect
        self._preview_connected_modalities.clear()
        self._set_configuration_locked(False)
        self.start_button.setEnabled(False)
        self._clear_one_trial_metadata()
        self._update_connect_button_state()
        self._update_start_button()
        if self._trial_succeeded:
            self._set_trial_state("IDLE")
            self.statusBar().showMessage(
                "Trial 已最终化；设备 Worker 已关闭，一次性元数据已清空；"
                "下一个 Trial 前请重新确认记录。",
                8000,
            )
            LOG.info("Trial 成功完成: 尝试恢复预览连接")
        else:
            LOG.warning("Trial 失败: 在记录 Worker 释放后恢复预览")
        if not self._close_when_finished:
            self._restore_preview_workers()
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
        else:
            self.start_button.setText("开始写盘")
            self.start_button.setStyleSheet(
                "QPushButton { font-weight: 600; padding: 8px; color: #ffffff; background: #0d6efd; border: 1px solid #0d6efd; border-radius: 4px; }"
            )
            subject_valid = bool(
                QRegularExpression(r"^\d{3}$").match(self.subject_code_edit.text().strip()).hasMatch()
            )
            all_ready = all(
                m in self._preview_connected_modalities for m in CRITICAL_MODALITIES
            )
            self.start_button.setEnabled(
                all_ready
                and subject_valid
                and not self._configuration_locked
                and not self._preflight_busy
                and self._worker is None
                and self._pending_trial_request is None
            )
            self.start_button.setToolTip(
                "" if all_ready else "请先连接所有必需模态的设备预览（超声、IMU、编码器、同步脉冲）"
            )

    def _set_trial_state(self, state: str) -> None:
        normalized = state.strip().upper() or "UNKNOWN"
        self._worker_state = normalized
        display = {
            "IDLE": "未连接",
            "DISCONNECTED": "未连接",
            "PREFLIGHT_READY": "可采集",
            "PREFLIGHT": "设备预检",
            "SWITCHING_TO_RECORD": "切换至记录",
            "PREPARING": "等待同步",
            "READY": "等待同步",
            "WAITING_SYNC": "等待同步",
            "RECORDING": "采集中",
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
        elif display in {"可采集", "采集中"}:
            colors = "background:#d1e7dd;color:#0f5132;border:1px solid #badbcc;"
        elif display in {"设备预检", "切换至记录", "等待同步", "保存中"}:
            colors = "background:#fff3cd;color:#664d03;border:1px solid #ffecb5;"
        else:
            colors = "background:#e2e3e5;color:#41464b;border:1px solid #d3d6d8;"
        self.state_label.setStyleSheet(
            f"QLabel {{{colors}padding:6px;border-radius:3px;font-weight:600;}}"
        )

    def _append_alert(self, message: str) -> None:
        error_markers = ("失败", "错误", "异常", "超时", "FAILED", "ERROR",
                         "FAULT", "掉线", "断开", "断开连接")
        level = "ERROR" if any(marker in message for marker in error_markers) else "INFO"
        if level == "ERROR":
            LOG.error("UI: %s", message)
        else:
            LOG.info("UI: %s", message)
        self._show_toast(message, level=level)

    # ── toast overlay ────────────────────────────────────────────────────

    def _show_toast(self, message: str, *, level: str = "INFO") -> None:
        if level == "ERROR":
            bg = "#dc3545"; fg = "#ffffff"; icon = "⚠ "
        else:
            bg = "#0d6efd"; fg = "#ffffff"; icon = "ℹ "
        self._toast_label.setText(f"{icon}{message}")
        self._toast_label.setStyleSheet(
            f"QLabel {{ background:{bg}; color:{fg}; border-radius:6px; "
            f"font-size:13px; padding:8px 16px; }}"
        )
        self._toast_label.adjustSize()
        self._position_toast()
        self._toast_label.setVisible(True)
        self._toast_label.raise_()
        # Auto-hide: 8 s for errors, 4 s for info
        self._toast_timer.start(8000 if level == "ERROR" else 4000)

    @Slot()
    def _hide_toast(self) -> None:
        self._toast_label.setVisible(False)

    def _position_toast(self) -> None:
        w = self._toast_label.width()
        h = self._toast_label.height()
        parent_w = self.width()
        margin = 16
        self._toast_label.setGeometry(
            parent_w - w - margin, margin, min(w, parent_w - 2 * margin), h
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_toast_label") and self._toast_label.isVisible():
            self._position_toast()

    @Slot()
    def _open_log_directory(self) -> None:
        log_dir = str(collector_log_path().parent)
        try:
            if sys.platform == "win32":
                os.startfile(log_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", log_dir])
            else:
                subprocess.Popen(["xdg-open", log_dir])
        except Exception as exc:
            QMessageBox.warning(
                self, "无法打开日志目录",
                f"打开日志目录失败：{exc}\n\n路径：{log_dir}\n"
                f"请手动打开该目录查看日志文件。"
            )
            self._append_alert(f"打开日志目录失败：{type(exc).__name__}: {exc}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        # Closing cancels a pending preview-to-record transition.  A timer tick
        # during teardown must never launch a new Trial worker.
        self._pending_trial_request = None
        self._preview_restore_modalities.clear()
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

        # Stop all preview workers
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

        worker = self._worker
        if worker is not None and self._worker_is_alive(worker):
            self._close_when_finished = True
            self.request_controlled_stop()
            self.statusBar().showMessage("正在受控停止并最终化 Trial；完成后将自动关闭。")
            event.ignore()
            return
        if worker is not None:
            self._release_worker(worker)
        self._poll_timer.stop()
        self._close_started_at = None
        LOG.info("CollectorWindow 已关闭")
        event.accept()
