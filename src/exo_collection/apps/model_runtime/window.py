"""Online-testing main window: live subscriptions + model runtime + torque out.

Layout mirrors the collector's split: a left control column (device connection,
model spec load/reload, torque output arming, run status) and a right preview
column (a ``ModalityStatusStrip`` above a dockable ``PreviewWorkspace`` holding
one live panel per modality plus a "模型输出" panel for predicted/commanded
torque).

The data path is identical to the collector's preview workers — each modality
runs in a ``ModalityPreviewProcessHandle`` subprocess that publishes PREVIEW
events for the panels and, once ``begin_recording`` is armed, forwards raw
``SampleBatch``/``FrameBatch`` events into a ``RecordingStreamEndpoint``.  Those
endpoints are registered in a ``SubscriptionHub``, drained by an
``InferenceController`` QThread (fast model hot-swap), and mapped onto a
either a ``MockTorqueOutput`` or an explicitly connected ``SerialTorqueOutput``.
"""

from __future__ import annotations

import uuid
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.apps.collector.device_preview import (
    ModalityPreviewProcessHandle,
    ProfileModalityAdapterFactory,
)
from exo_collection.apps.collector.preview_workspace import PreviewWorkspace
from exo_collection.apps.collector.signal_previews import (
    ENCODER_PREVIEW_LABELS,
    SIGNAL_PREVIEW_PANEL_TITLES,
    SignalPreviewMixin,
)
from exo_collection.apps.collector.status_overview import ModalityStatusStrip
from exo_collection.apps.collector.theme import COLLECTOR_STYLESHEET
from exo_collection.apps.model_runtime.backends import (
    ModelRuntime,
    ModelRuntimeError,
    build_model_runtime,
)
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.gate_compatibility import gate_compatibility_error
from exo_collection.apps.model_runtime.run_coordinator import (
    RunCoordinator, OutputState, _motor_fault_reason,
)
from exo_collection.apps.model_runtime.startup import StartupCoordinator, StartupPhase
from exo_collection.apps.model_runtime.control_widgets import AssistanceControls
from exo_collection.apps.model_runtime.session_log import SessionLog
from exo_collection.apps.model_runtime.recording import OnlineRecording
from exo_collection.apps.model_runtime.diagnostics import Diagnostics
from exo_collection.apps.model_runtime.model_loader import ModelLoader
from exo_collection.apps.model_runtime.preview_panels import (
    ModelOutputPanel,
)
from exo_collection.apps.model_runtime.uncertainty_panel import UncertaintyPanel
from exo_collection.apps.model_runtime.spec import ModelSpec, load_model_spec
from exo_collection.apps.model_runtime.model_bindings import bind_model_inputs
from exo_collection.apps.model_runtime.model_catalog import (
    RETIRED_ONLINE_DEPLOYMENTS, ModelCatalogEntry, discover_local_models, read_model_entry,
)
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput, TorqueCommand, TorqueOutput
from exo_collection.configuration import SharedAppSettings, load_device_profile

MODALITY_DISPLAY_NAMES = {
    "ultrasound": "超声",
    "imu": "IMU",
    "encoder": "电机编码器",
    "mocap": "动捕 Marker",
    "emg": "表面肌电 EMG",
    "force_plate": "测力台",
    "sync_pulse": "同步脉冲",
}

_PREVIEW_POLL_MS = 33
_WATCHDOG_POLL_MS = 100
_INPUT_RECOVERY_DELAY_MS = 50
_INPUT_RECOVERY_WAIT_MS = 100
_INPUT_RECOVERY_DEADLINE_S = 10.0


def _transient_input_error(value: object) -> bool:
    """Return whether an input failure is safe to retry once.

    Configuration, dependency and contract failures must remain visible to the
    operator.  Only transport-ish failures are eligible for the bounded retry.
    """
    text = str(value or "").lower()
    if not text:
        return False
    if any(token in text for token in (
        "configuration", "config", "dependency", "module not found", "contract",
        "unsupported", "permission denied", "access is denied", "invalid profile",
    )):
        return False
    if any(token in text for token in (
        "modulenotfound", "module not found", "importerror", "no module",
        "sdk", "not installed", "dependency", "adapter configuration",
    )):
        return False
    return any(token in text for token in (
        "timeout", "timed out", "i/o error", "ioerror", "serial read",
        "serial write", "usb failed", "usb disconnected", "broken pipe",
        "device disconnected", "connection reset", "read failed", "write failed",
        "queue timeout", "queue full", "eof",
    ))


def _block_state(state: str) -> str:
    normalized = (state or "").strip().upper()
    if normalized in {"READY", "RECORDING", "CONNECTED", "RUNNING"}:
        return "ok"
    if normalized in {"FAULT", "FAILED", "ERROR"}:
        return "bad"
    if normalized in {"CONNECTING", "PREVIEW_STARTING", "STOPPING"}:
        return "transition"
    return "off"


class ModelRuntimeWindow(SignalPreviewMixin, QMainWindow):
    """Top-level window for local real-time model testing."""

    def __init__(self, settings: SharedAppSettings, *, smoke_test: bool = False) -> None:
        super().__init__()
        self.setObjectName("model_runtime_window")
        self.setWindowTitle("在线测试 · 本地模型实时推理")
        self.setStyleSheet(COLLECTOR_STYLESHEET)

        self._settings = settings
        self._gate_mode = str(self._settings._backend.value("model_runtime/gate_mode", "none"))
        if self._gate_mode not in {"none", "ao", "phase_ensemble"}:
            self._gate_mode = "none"
        self._diagnostics = Diagnostics(self._project_root() / "runtime_data/model_runtime_diagnostics")
        self._loader = ModelLoader(self._diagnostics)
        self._loader.completed.connect(self._on_model_loaded, Qt.ConnectionType.QueuedConnection)
        self._load_generation = None
        self._closing = False
        # Fault/recovery lifecycle.  A fatal edge is latched so queued Qt
        # callbacks cannot reopen a stream or produce duplicate dialogs.
        self._fatal_latched = False
        self._fault_incident_generation = 0
        self._recovery_attempts: dict[str, int] = {}
        self._recovery_pending = None
        self._recovery_generation = 0
        self._fault_dialog = None
        self._pending_motor_cleanup = None
        self._model_sha256 = None
        self._preview_counts = {}
        self._preview_latest_ns = {}
        self._worker_health = {}
        self._worker_states = {}
        self._worker_connected_ns = {}
        self._diagnostic_last = time.monotonic()
        self._poll_last = self._diagnostic_last
        self._poll_max_gap_s = 0.0
        self._poll_max_duration_s = 0.0
        self._poll_count = 0
        self._diagnostic_errors = {}
        self._smoke_test = bool(smoke_test)
        self._profile_key = settings.device_profile_key
        self._hardware_overrides = settings.hardware_device_overrides
        self._modalities = self._load_modalities()

        self._handles: dict[str, ModalityPreviewProcessHandle] = {}
        self._registered: set[str] = set()
        self._panels: dict[str, QWidget] = {}
        self.init_signal_previews()
        self._preview_focus_previous_sizes: list[int] | None = None
        self._connect_buttons: dict[str, QPushButton] = {}
        self._connect_dots: dict[str, QLabel] = {}

        self._online_recording = OnlineRecording(self._project_root() / "runtime_data/online_recordings")
        self._last_recording_error = None
        self._hub = SubscriptionHub(buffer_capacity=4096, raw_tap=self._online_recording.raw)
        self._spec: ModelSpec | None = None
        self._model_template_spec: ModelSpec | None = None
        self._runtime: ModelRuntime | None = None
        self._run = RunCoordinator()
        self._startup = StartupCoordinator()
        self._startup_probe_after = 0.0
        self._last_preflight_report = {}
        self._session_log = None
        self._applied_signature = None
        self._last_logged_command_ns = -1
        self._last_feedback_ns = -1
        self._last_motor_fault = None
        self._encoder_input_requested = False
        self._motor_encoder_queue = None
        self._encoder_snapshot_sequence = 0
        self._spec_path: Path | None = None
        self._trial_uuid = str(uuid.uuid4())

        self._build_ui()
        cached_parameters, cache_errors = self._settings.read_model_runtime_parameters()
        self._controls.restore_parameters(cached_parameters)
        self._diagnostics.record("runtime_parameters_restored", parameters=cached_parameters)
        for error in cache_errors:
            self._diagnostics.record("runtime_parameter_cache_invalid", reason=error)
        self._update_log_label()
        self._run_button.setEnabled(False)
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(_PREVIEW_POLL_MS)
        self._preview_timer.timeout.connect(self._poll_previews)
        self._preview_timer.start()

        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.setInterval(_WATCHDOG_POLL_MS)
        self._watchdog_timer.timeout.connect(self._check_watchdog)
        self._watchdog_timer.start()
        self._recovery_timer = QTimer(self)
        self._recovery_timer.setSingleShot(True)
        self._recovery_timer.timeout.connect(self._on_recovery_timer)

    # -- profile / modality discovery -------------------------------------

    def _load_modalities(self) -> tuple[str, ...]:
        try:
            profile = load_device_profile(self._profile_key)
            return tuple(profile.by_modality())
        except Exception:
            return ("ultrasound", "imu", "encoder", "mocap", "emg")

    # -- UI construction ---------------------------------------------------

    def _build_ui(self) -> None:
        body = QSplitter(Qt.Orientation.Horizontal, self)
        self._body_splitter = body
        body.setChildrenCollapsible(False)
        body.addWidget(self._build_control_column())
        body.addWidget(self._build_preview_column())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([430, 1290])
        container = QWidget()
        layout = QVBoxLayout(container)
        steps = QLabel("连接输入 → 加载模型与填写参数 → 运行预测 → 停止")
        layout.addWidget(steps)
        layout.addWidget(body, 1)
        stop_bar = QHBoxLayout()
        self._persistent_command_label = QLabel("辅助未启动 · 电机保持失能")
        self._persistent_command_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        stop_bar.addWidget(self._persistent_command_label, 1)
        stop_button = QPushButton("急停 · 零力矩并失能")
        stop_button.setProperty("buttonRole", "danger")
        stop_button.setMinimumHeight(42)
        stop_button.clicked.connect(self.emergency_stop)
        stop_bar.addWidget(stop_button)
        layout.addLayout(stop_bar)
        self.setCentralWidget(container)

    def _build_control_column(self) -> QWidget:
        scroll = QScrollArea()
        self._controls_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Construct the compatibility controls before the device group so its
        # physical motor panel can be placed with the device connections.
        torque_group = self._build_torque_group()
        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_status_group())
        layout.addWidget(self._fault_label)
        layout.addStretch(1)

        scroll.setWidget(column)
        return scroll

    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("设备连接")
        layout = QVBoxLayout(group)
        profile_row = QHBoxLayout()
        self._profile_combo = QComboBox()
        self._profile_combo.addItem("真实传感器", "hardware")
        self._profile_combo.addItem("模拟传感器（仅 Mock）", "simulated")
        self._profile_combo.setCurrentIndex(max(0, self._profile_combo.findData(self._profile_key)))
        self._profile_combo.currentIndexChanged.connect(self._change_profile)
        profile_row.addWidget(self._profile_combo, 1)
        imu_settings = QPushButton("IMU 设备设置…")
        imu_settings.clicked.connect(self._configure_imu)
        profile_row.addWidget(imu_settings)
        layout.addLayout(profile_row)

        buttons = QHBoxLayout()
        connect_all = QPushButton("连接模型所需输入")
        connect_all.setProperty("buttonRole", "primary")
        connect_all.clicked.connect(self.connect_all)
        disconnect_all = QPushButton("全部断开")
        disconnect_all.setProperty("buttonRole", "disconnect")
        disconnect_all.clicked.connect(self.disconnect_all)
        buttons.addWidget(connect_all)
        buttons.addWidget(disconnect_all)
        layout.addLayout(buttons)

        layout.addWidget(QLabel("输入设备"))
        for modality in self._modalities:
            layout.addLayout(self._build_connect_row(modality))
        # The motor service owns the encoder port and supplies its feedback;
        # never open a second read-only encoder worker for that same port.
        motor_group = self._controls.motor_group
        layout.addWidget(motor_group)
        return group

    def _build_connect_row(self, modality: str) -> QHBoxLayout:
        name = MODALITY_DISPLAY_NAMES.get(modality, modality)
        row = QHBoxLayout()

        dot = QLabel()
        dot.setFixedSize(12, 12)
        dot.setStyleSheet("background:#d3d0c7; border-radius:6px;")
        self._connect_dots[modality] = dot

        label = QLabel(name)
        label.setMinimumWidth(90)

        button = QPushButton("连接")
        button.setProperty("buttonRole", "connect")
        button.clicked.connect(
            lambda _checked=False, m=modality: self._toggle_connect(m)
        )
        self._connect_buttons[modality] = button

        row.addWidget(dot)
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(button)
        return row

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("模型设置")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        path_row = QHBoxLayout()
        # Keep the path API for programmatic callers; the operator chooses a
        # short model name and sees the complete path only in its tooltip.
        self._spec_path_edit = QLineEdit(group)
        self._spec_path_edit.setReadOnly(True)
        self._spec_path_edit.hide()
        self._model_choice = QComboBox()
        catalog = discover_local_models(self._project_root() / "runtime_data/models")
        self._catalog_entries = {str(entry.spec_path): entry for entry in catalog.entries}
        self._model_choice.addItem("选择模型…" if catalog.entries else "暂无可用模型", "")
        for entry in catalog.entries:
            self._model_choice.addItem(entry.name, str(entry.spec_path))
            self._model_choice.setItemData(self._model_choice.count() - 1, entry.tooltip, Qt.ItemDataRole.ToolTipRole)
        for path, reason in catalog.issues:
            self._diagnostics.record("model_catalog_skipped", path=str(path), reason=reason)
        self._diagnostics.record("model_catalog_ready", count=len(catalog.entries))
        if not catalog.entries:
            self._model_choice.setToolTip("本地暂无完整的已部署模型；请检查模型部署目录，详情见诊断日志。")
        self._model_choice.model().item(0).setEnabled(False)
        self._model_choice.setMinimumWidth(0)
        self._model_choice.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._model_choice.currentIndexChanged.connect(self._choose_model_entry)
        path_row.addWidget(self._model_choice, 1)
        layout.addLayout(path_row)

        gate_row = QHBoxLayout()
        gate_row.addWidget(QLabel("助力门控"))
        self._gate_choice = QComboBox(group)
        for label, value in (("不门控", "none"),
                             ("AO24 + RF-full · 原始分数", "ao"),
                             ("七相位 TCN · 0.5 s 因果中值", "phase_ensemble")):
            self._gate_choice.addItem(label, value)
        idx = self._gate_choice.findData(self._gate_mode)
        self._gate_choice.setCurrentIndex(max(0, idx))
        self._gate_choice.currentIndexChanged.connect(self._on_gate_mode_changed)
        gate_row.addWidget(self._gate_choice, 1)
        layout.addLayout(gate_row)

        load_row = QHBoxLayout()
        load_btn = QPushButton("重载模型", group)
        load_btn.clicked.connect(self._load_selected)
        load_btn.hide()
        run_btn = QPushButton("运行预测")
        run_btn.setProperty("buttonRole", "primary")
        run_btn.clicked.connect(self._run_prediction)
        self._model_status = QLabel("未加载")
        self._model_status.setWordWrap(True)
        status_policy = self._model_status.sizePolicy()
        status_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self._model_status.setSizePolicy(status_policy)
        load_row.addWidget(run_btn)
        stop_btn = QPushButton("停止预测")
        stop_btn.clicked.connect(self.emergency_stop)
        load_row.addWidget(stop_btn)
        self._stop_button = stop_btn
        layout.addLayout(load_row)
        layout.addWidget(self._model_status)

        layout.addWidget(self._controls)
        self._load_button = load_btn
        self._run_button = run_btn
        return group

    def _on_gate_mode_changed(self, _index=None):
        if self._run.running or self._startup.active:
            idx = self._gate_choice.findData(self._gate_mode)
            self._gate_choice.blockSignals(True)
            self._gate_choice.setCurrentIndex(max(0, idx))
            self._gate_choice.blockSignals(False)
            return
        self._gate_mode = str(self._gate_choice.currentData() or "none")
        self._applied_signature = None
        self._uncertainty_panel.set_mode(self._gate_mode)
        self._workspace.set_stream_state("uncertainty", "waiting")
        self._settings._backend.setValue("model_runtime/gate_mode", self._gate_mode)
        self._diagnostics.record("gate_mode_selected", mode=self._gate_mode,
                                 version="ao24_rf_full_raw_v1" if self._gate_mode == "ao"
                                 else "gaitphase7_variance_median_0p5s_v1"
                                 if self._gate_mode == "phase_ensemble" else "none")
        self._update_gate_compatibility_status()

    def _build_torque_group(self) -> QGroupBox:
        group = self._controls = AssistanceControls()
        group.ports_scanned.connect(self._record_motor_port_scan)
        self._record_motor_port_scan(group.last_port_scan)
        self._enable_check = group.enable
        self._enable_check.toggled.connect(self._toggle_enable)
        self._torque_label = group.command_label
        group.stop_requested.connect(self.emergency_stop)
        group.motor_requested.connect(self._toggle_motor)
        group.changed.connect(self._controls_changed)
        group.parameter_committed.connect(self._save_runtime_parameter)
        # Compatibility controls remain available to old callers/tests, while
        # the visible workflow is Load -> Run -> Stop.
        group.enable.setVisible(False)
        group.mode.setVisible(False)
        group.motor_group.setVisible(True)
        group.command_label.hide()
        group.preflight_label.hide()
        return group

    def _record_motor_port_scan(self, scan: dict) -> None:
        self._diagnostics.record("motor_port_scan", scan=scan)

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("详情与日志")
        group.setCheckable(True)
        group.setChecked(False)
        outer = QVBoxLayout(group)
        details = QWidget()
        layout = QVBoxLayout(details)
        outer.addWidget(details)
        group.toggled.connect(details.setVisible)
        details.hide()
        self._status_labels: dict[str, QLabel] = {}
        for key, caption in (
            ("backend", "模型后端"),
            ("control_rate", "控制频率"),
            ("window", "特征窗口"),
            ("limit", "扭矩限幅"),
            ("last_predict", "上次预测"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(caption))
            value = QLabel("—")
            value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            value.setWordWrap(True)
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._status_labels[key] = value
            row.addWidget(value, 1)
            layout.addLayout(row)
        self._fault_label = QLabel()
        self._fault_label.setStyleSheet("color:#a53f3f;")
        self._fault_label.setWordWrap(True)
        fault_policy = self._fault_label.sizePolicy()
        fault_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self._fault_label.setSizePolicy(fault_policy)
        layout.addWidget(self._controls.preflight_label)
        layout.addWidget(self._controls.log_label)
        self._controls.preflight_label.show()
        self._controls.log_label.show()
        # Fault summaries stay visible even when technical details are closed.
        group._fault_summary = self._fault_label
        return group

    def _build_preview_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        names = {m: MODALITY_DISPLAY_NAMES.get(m, m) for m in self._modalities}
        self._status_strip = ModalityStatusStrip(names)
        layout.addWidget(self._status_strip)

        self._workspace = PreviewWorkspace()
        panels = self.build_signal_preview_panels()
        for modality in self._modalities:
            title = SIGNAL_PREVIEW_PANEL_TITLES.get(modality, modality)
            panel = panels.get(modality)
            if panel is None:
                continue
            self._panels[modality] = panel
            self._workspace.register_panel(modality, title, panel)
        self._model_output_panel = ModelOutputPanel("模型输出")
        self._workspace.register_panel("model_output", "模型输出", self._model_output_panel,
                                       preferred_height=340)
        self._uncertainty_panel = UncertaintyPanel()
        self._uncertainty_panel.set_mode(self._gate_mode)
        self._workspace.register_panel("uncertainty", "不确定性判定", self._uncertainty_panel,
                                       preferred_height=240)
        self._workspace.reset_default_layout()
        self._workspace.focus_mode_requested.connect(self._set_preview_focus_mode)
        layout.addWidget(self._workspace, 1)
        return column

    def _set_preview_focus_mode(self, enabled: bool) -> None:
        if enabled:
            if self._preview_focus_previous_sizes is None:
                self._preview_focus_previous_sizes = self._body_splitter.sizes()
            self._controls_scroll.hide()
        else:
            self._controls_scroll.show()
            if self._preview_focus_previous_sizes:
                self._body_splitter.setSizes(self._preview_focus_previous_sizes)
            self._preview_focus_previous_sizes = None
        self._workspace.set_focus_mode(enabled)

    def _on_signal_preview_alert(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self._write_log("preview_alert", message)

    # -- device connection -------------------------------------------------

    def _toggle_connect(self, modality: str) -> None:
        if modality == "encoder" and self._motor is not None:
            if self._encoder_input_requested:
                self._disconnect_modality(modality)
            else:
                self._connect_modality(modality)
        elif modality in self._handles:
            self._disconnect_modality(modality)
        else:
            self._connect_modality(modality)

    def connect_all(self) -> None:
        self._reset_fault_lifecycle("explicit_connect_all")
        modalities = [entry.modality for entry in self._spec.features.inputs] if self._spec else ["imu"]
        for modality in dict.fromkeys(modalities):
            if not self._connect_modality(modality, quiet=True) or self._fatal_latched:
                break

    def disconnect_all(self) -> None:
        self.emergency_stop()
        self._retry_pending_motor_cleanup()
        self._encoder_input_requested = False
        for modality in list(self._handles):
            self._disconnect_modality(modality)
        self._disconnect_motor(restore_encoder=False)

    # -- fault lifecycle / bounded input recovery ------------------------

    def _cancel_input_recovery(self, reason: str) -> None:
        self._recovery_generation = getattr(self, "_recovery_generation", 0) + 1
        self._recovery_pending = None
        timer = getattr(self, "_recovery_timer", None)
        if timer is not None:
            timer.stop()
        self._diagnostics.record("input_recovery_cancelled", reason=reason)

    def _reset_fault_lifecycle(self, reason: str) -> None:
        """Reset a latched incident only at an explicit operator edge."""
        self._fatal_latched = False
        self._fault_incident_generation = getattr(self, "_fault_incident_generation", 0) + 1
        self._cancel_input_recovery(reason)
        self._recovery_attempts.clear()

    def _input_recovery_allowed(self, modality: str) -> bool:
        if self._closing or self._fatal_latched or self._motor is not None:
            return False
        if self._run.running and getattr(self._run, "output_state", None) in {
            OutputState.ARM_INTENT, OutputState.WAITING_CONFIRMATION, OutputState.ENABLED,
        }:
            return False
        if getattr(self._online_recording, "active", False):
            return False
        if modality == "encoder" and self._motor is not None:
            return False
        return True

    def _request_input_recovery(self, modality: str, reason: object) -> bool:
        """Schedule at most one retry for a transport-level input fault."""
        if not self._input_recovery_allowed(modality):
            return False
        attempts = self._recovery_attempts.get(modality, 0)
        if attempts >= 1:
            self._diagnostics.record("input_recovery_exhausted", modality=modality,
                                     reason=str(reason))
            return False
        pending = self._recovery_pending
        if pending is not None:
            # Never replace another device's outstanding retry and strand it.
            return pending[0] == modality
        self._recovery_attempts[modality] = attempts + 1
        self._recovery_generation += 1
        generation = self._recovery_generation
        self._recovery_pending = (
            modality, generation, "retry", time.monotonic() + _INPUT_RECOVERY_DEADLINE_S,
        )
        self._diagnostics.record("input_recovery_requested", modality=modality,
                                 generation=generation, reason=str(reason))
        timer = getattr(self, "_recovery_timer", None)
        if timer is None:
            return False
        timer.start(_INPUT_RECOVERY_DELAY_MS)
        return True

    def _on_recovery_timer(self) -> None:
        pending = self._recovery_pending
        if pending is None:
            return
        modality, generation, phase, deadline = pending
        if (generation != self._recovery_generation or self._closing or
                self._fatal_latched):
            return
        if not self._input_recovery_allowed(modality):
            self._cancel_input_recovery("recovery no longer allowed")
            return
        if time.monotonic() >= deadline:
            self._recovery_pending = None
            self._diagnostics.record("input_recovery_exhausted", modality=modality,
                                     reason="connection/wait deadline")
            self._fatal_incident(
                f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 输入恢复超时",
                "输入设备短暂故障在限定时间内未恢复。",
                source="input_recovery_timeout", modality=modality,
            )
            return
        if phase == "retry":
            ok = self._connect_modality(modality, quiet=True)
            if not ok:
                self._recovery_pending = None
                self._diagnostics.record("input_recovery_exhausted", modality=modality,
                                         reason="reconnect failed")
                self._fatal_incident(
                    f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 输入恢复失败",
                    "输入设备重连失败，已停止并关闭当前连接。",
                    source="input_recovery_failed", modality=modality,
                )
                return
            self._recovery_pending = (modality, generation, "wait", deadline)
            self._recovery_timer.start(_INPUT_RECOVERY_WAIT_MS)
            return
        handle = self._handles.get(modality)
        ready = (handle is not None and handle.is_alive and
                 self._worker_states.get(modality) in {"READY", "RUNNING", "CONNECTED", "RECORDING"})
        if ready:
            self._recovery_pending = None
            self._diagnostics.record("input_recovery_succeeded", modality=modality,
                                     generation=generation)
            if not self._run.running and not self._startup.active:
                self.statusBar().showMessage("输入已自动重连；点击运行预测可重新开始。")
            return
        self._recovery_timer.start(_INPUT_RECOVERY_WAIT_MS)

    @staticmethod
    def _safe_operation(diagnostics, kind: str, operation, **fields):
        failures = fields.pop("_failures", None)
        try:
            return operation()
        except Exception as exc:
            diagnostics.record("device_cleanup_failed", stage=kind,
                               exception=exc, **fields)
            if failures is not None:
                failures.append(f"{kind}: {exc}")
            return None

    def _fault_snapshot(self) -> dict[str, object]:
        motor = self._motor
        snapshot = {"runtime": self._runtime_evidence()}
        if motor is not None:
            for name in ("fault_snapshot", "diagnostics"):
                try:
                    value = getattr(motor, name, None)
                    value = value() if callable(value) else value
                except Exception as exc:
                    self._diagnostics.record("fault_snapshot_failed", source=name,
                                             exception=exc)
                    continue
                if value is not None:
                    snapshot[name] = value
        controller = self._controller
        if controller is not None:
            snapshot["controller_fault"] = getattr(controller, "last_fault_details", None)
            snapshot["controller_error"] = getattr(controller, "last_error", None)
        return snapshot

    def _runtime_evidence(self) -> dict[str, object]:
        """Inspect process/thread state before cleanup can destroy the evidence."""
        def read(value, name, default=None):
            try:
                result = getattr(value, name, default)
                return result() if callable(result) else result
            except Exception as exc:
                return {"inspection_error": repr(exc)}
        workers = {}
        for modality, handle in self._handles.items():
            process = getattr(handle, "_process", None)
            workers[modality] = {"pid": read(process, "pid"), "exitcode": read(process, "exitcode"),
                "is_alive": read(handle, "is_alive"),
                "last_poll_error": str(getattr(handle, "last_poll_error", None) or "") or None,
                "last_preview_ns": self._preview_latest_ns.get(modality),
                "health": self._worker_health.get(modality)}
        controller = self._controller
        now = time.monotonic()
        return {"pid": os.getpid(), "host_monotonic_ns": time.perf_counter_ns(),
            "workers": workers, "controller": {"present": controller is not None,
                "is_running": read(controller, "isRunning"),
                "last_error": read(controller, "last_error"),
                "last_predict_ns": read(controller, "last_predict_ns")},
            "ui": {"poll_ticks": self._poll_count,
                   "last_tick_age_s": max(0, now - self._poll_last),
                   "max_tick_gap_s": self._poll_max_gap_s,
                   "max_poll_duration_s": self._poll_max_duration_s}}

    def _incident_message(self, detail: object, snapshot: dict[str, object]) -> str:
        message = self._friendly_fault(str(detail))
        motor = snapshot.get("fault_snapshot") or {}
        code = motor.get("error")
        if code:
            if code not in (253, 254):
                message = "控制板报告电机故障，已归零、失能并断开，请检查电机。"
            message += f"\n控制板故障码：{code}（error={code}）。"
        else:
            message += "\n原因：" + str(detail)[:300]
        transport = motor.get("transport") or snapshot.get("diagnostics", {}).get("transport", {})
        if transport:
            count = transport.get("counts", {}).get("rx_valid_frames", 0)
            gap = transport.get("max_gap_ns", {}).get("rx_valid", 0) / 1e6
            message += f"\n故障现场：电脑已收到 {count} 个有效控制板状态包，最大接收间隔 {gap:.1f} 毫秒。"
            last_rx = transport.get("last_ns", {}).get("rx_valid")
            fault_ns = motor.get("host_monotonic_ns")
            if last_rx is not None and fault_ns is not None:
                message += f"最近状态包在锁存前 {max(0, fault_ns - last_rx) / 1e6:.1f} 毫秒到达。"
            if transport.get("background"):
                names = {"reader": "接收", "sender": "发送"}
                status = [names[name] + ("线程在运行" if value.get("alive") else "线程未运行")
                          for name, value in transport.get("threads", {}).items()]
                message += "\n" + "，".join(status) + "。"
            error = transport.get("last_io_error")
            if error:
                message += "\n串口" + ("读取" if error.get("operation") == "read" else "写入") + "失败，具体系统错误已保存至日志。"
        workers = snapshot.get("runtime", {}).get("workers", {})
        exited = [MODALITY_DISPLAY_NAMES.get(name, name) for name, value in workers.items()
                  if value.get("is_alive") is False]
        if exited:
            message += "\n清理前已退出的输入进程：" + "、".join(exited) + "。"
        elif workers and all(value.get("is_alive") is True for value in workers.values()):
            message += "\n清理前输入进程仍在运行；其他设备随后由本次保护停机统一关闭。"
        elif workers:
            message += "\n部分输入进程的现场状态未能读取，具体检查结果已保存至日志。"
        return message

    def _show_fatal_popup(self, title: str, detail: str) -> None:
        if self._smoke_test:
            return
        current = getattr(self, "_fault_dialog", None)
        if current is not None:
            try:
                if current.isVisible():
                    return
            except RuntimeError:
                self._fault_dialog = None
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle(title)
        dialog.setText(str(detail))
        self._fault_dialog = dialog
        dialog.destroyed.connect(lambda: setattr(self, "_fault_dialog", None))
        dialog.open()

    def _fatal_incident(self, title: str, detail: object, *, source: str = "unknown", modality: str | None = None) -> bool:
        """Latch a serious incident, clean every resource, then notify once."""
        if getattr(self, "_fatal_latched", False):
            self._diagnostics.record("fatal_incident_duplicate", source=source,
                                     title=title, detail=str(detail))
            return False
        self._fatal_latched = True
        self._fault_incident_generation = getattr(self, "_fault_incident_generation", 0) + 1
        snapshot = self._fault_snapshot()
        previously_connected = set(self._handles) | set(self._registered)
        if self._motor is not None:
            previously_connected.add("encoder")
        fault_modalities = {modality} if modality is not None else set()
        if source in {"motor_fault", "output_state_fault", "motor_connect_failed"}:
            fault_modalities.add("encoder")

        # Cancel operator intents and make the first physical action a
        # zero/disable before any controller wait can occur.
        motor = self._motor
        cleanup_failures: list[str] = []
        safe = lambda kind, operation, **fields: self._safe_operation(
            self._diagnostics, kind, operation, _failures=cleanup_failures, **fields)
        safe("disable_before_fatal_wait",
                             lambda: motor.set_enabled(False), modality="motor") if motor is not None else None
        safe("zero_before_fatal_wait",
                             motor.zero, modality="motor") if motor is not None else None
        self._diagnostics.record("fatal_incident", source=source, modality=modality,
                                 title=title, detail=str(detail), snapshot=snapshot)
        self._cancel_input_recovery("fatal_incident")
        self._cancel_startup("fatal_incident")
        self._cancel_model_load("fatal_incident")
        try:
            self._stop_controller()
        except Exception as exc:
            self._diagnostics.record("fatal_controller_stop_failed", exception=exc)
            cleanup_failures.append(f"controller_stop: {exc}")

        # Pop and unregister first so callbacks queued for an old process can
        # never re-register an endpoint while cleanup is in progress.
        handles = list(self._handles.items())
        self._handles.clear()
        self._registered.clear()
        for modality, _handle in handles:
            safe("recording_source_failed",
                                 lambda m=modality: self._online_recording.source_failed(
                                     m, "fatal incident; input connection closed"), modality=modality)
        try:
            available = getattr(self._hub, "modalities", ())
            available = available() if callable(available) else available
            hub_modalities = set(available or ())
        except Exception:
            hub_modalities = set()
        for modality in hub_modalities | {m for m, _ in handles} | {"encoder"}:
            safe("unregister_endpoint",
                                 lambda m=modality: self._hub.unregister(m), modality=modality)
        for modality, handle in handles:
            safe("end_recording",
                                 lambda h=handle: h.end_recording(self._trial_uuid)
                                 if h.recording_active else None, modality=modality)
            safe("stop_worker",
                                 lambda h=handle: h.request_stop(), modality=modality)
            safe("join_worker",
                                 lambda h=handle: h.join(timeout=0.5), modality=modality)
            if getattr(handle, "is_alive", False):
                safe("terminate_worker",
                                     lambda h=handle: h.terminate(timeout=1.0), modality=modality)
            safe("close_worker",
                                 lambda h=handle: h.close(), modality=modality)
        if motor is not None:
            safe("close_motor", lambda: self._close_motor_with_evidence(motor, "fatal"), modality="motor")
            self._run.motor = None
            self._run.output = None
            self._run.output_state = OutputState.DETACHED
            self._run.reason = "电机已断开 · 故障锁定"
        self._motor_encoder_queue = None
        self._encoder_input_requested = False
        for modality in set(self._modalities) | {m for m, _ in handles} | {"encoder"}:
            safe("mark_fault", lambda m=modality:
                                 self._set_connect_state(m, connected=False), modality=modality)
            state, label = (("bad", "故障来源") if modality in fault_modalities else
                            ("off", "因本次故障关闭") if modality in previously_connected else
                            ("off", "未连接"))
            safe("mark_fault_status", lambda m=modality, st=state, text=label:
                                 self._status_strip.set_state(m, st, text), modality=modality)
            safe("mark_fault_stream", lambda m=modality, st=state:
                                 self._workspace.set_stream_state(m, "error" if st == "bad" else "disconnected"), modality=modality)
        safe("mark_model_output_fault", lambda:
             self._workspace.set_stream_state("model_output", "error"))
        self._uncertainty_panel.mark_inactive(fault=True)
        self._workspace.set_stream_state("uncertainty", "error")
        safe("poll_recording_tail", self._poll_recording)
        message = self._friendly_fault(str(detail))
        popup_message = self._incident_message(detail, snapshot)
        self._fault_label.setText(message)
        self._model_status.setText("已停止 · 故障锁定")
        self.statusBar().showMessage(message)
        self._controls.motor_status.setText("电机已断开 · 故障锁定；请人工检查后重新连接。")
        self._model_output_panel.clear_command()
        self._model_output_panel.set_feedback(None)
        self._update_torque_label()
        self._model_output_panel.set_output_status(message)
        if self._pending_motor_cleanup is not None and not cleanup_failures:
            cleanup_failures.append("motor serial release remains unconfirmed")
        if cleanup_failures:
            self._diagnostics.record("fatal_cleanup_incomplete", failures=cleanup_failures)
            popup_message += "\n\n部分连接清理失败，请检查日志并人工确认设备已停止。"
            self._controls.motor_status.setText("已尝试停止并关闭连接；清理异常请查看详情。")
        else:
            popup_message += "\n\n本次输入和输出连接已关闭；请检查后手动重新连接。"
        self._controls.motor_button.setText("连接电机（保持失能）")
        self._show_fatal_popup(title, f"{popup_message}\n\n诊断日志：{self._diagnostics.path}")
        return True

    def _connect_modality(self, modality: str, *, quiet=False) -> bool:
        if not quiet:
            self._reset_fault_lifecycle("explicit_input_connect")
        if self._closing or self._fatal_latched:
            return False
        if modality in self._handles:
            return True
        if modality == "encoder":
            self._encoder_input_requested = True
            if self._motor is not None:
                self._register_motor_encoder()
                return True
        # Collector and online testing share one QSettings namespace.  Reload
        # it at the connection boundary so a Collector save made while this
        # window is open is used for the next USB worker as well.
        self._hardware_overrides = self._settings.hardware_device_overrides
        try:
            profile = load_device_profile(self._profile_key)
            device = profile.by_modality()[modality]
        except Exception as exc:
            if quiet:
                self._fail_startup(f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 输入配置不可用；请检查设备设置。", detail=exc)
            else:
                self._show_error(f"无法加载 {modality} 设备配置", exc)
            return False
        simulated = self._profile_key == "simulated" or bool(
            getattr(device, "simulated", False)
        )
        overrides = (
            self._hardware_overrides
            if self._profile_key == "hardware"
            else {}
        )
        factory = ProfileModalityAdapterFactory(
            profile_key=self._profile_key,
            modality=modality,
            overrides=overrides,
        )
        handle = ModalityPreviewProcessHandle(
            factory,
            device_id=device.device_id,
            modality=modality,
            simulated=simulated,
            health_poll_interval_s=0.1,
            recording_queue_size=2048,
        )
        try:
            handle.start()
        except Exception as exc:
            try:
                handle.close()
            except Exception:
                pass
            self._set_connect_state(modality, connected=False)
            self._status_strip.set_state(modality, "bad", "失败")
            self._workspace.set_stream_state(modality, "error")
            if _transient_input_error(exc) and self._request_input_recovery(modality, exc):
                self._status_strip.set_state(modality, "transition", "自动恢复中")
                self._workspace.set_stream_state(modality, "waiting")
                return True
            self._fatal_incident(
                f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 输入连接失败",
                exc, source="input_connect_failed", modality=modality)
            return False
        self._handles[modality] = handle
        self._worker_connected_ns[modality] = time.perf_counter_ns()
        self._preview_latest_ns.pop(modality, None)
        self._worker_states.pop(modality, None)
        self._worker_health.pop(modality, None)
        self._diagnostics.record("device_connect", modality=modality, profile=self._profile_key,
                                 worker_pid=getattr(getattr(handle, "_process", None), "pid", None))
        self._set_connect_state(modality, connected=False, connecting=True)
        self._status_strip.set_state(modality, "transition", "连接中")
        self._workspace.set_stream_state(modality, "connected")
        self._workspace.show_panel(modality)
        return True

    def _disconnect_modality(self, modality: str, *, handoff: bool = False) -> None:
        self._diagnostics.record("device_disconnect", modality=modality, reason="output_handoff" if handoff else "requested")
        if modality == "encoder" and not handoff:
            self._encoder_input_requested = False
            self._motor_encoder_queue = None
        required = self._spec is not None and any(item.modality == modality for item in self._spec.features.inputs)
        if not handoff or required:
            self.emergency_stop(cancel_recovery=not handoff)
        self._online_recording.source_failed(modality, "input disconnected; raw tail not guaranteed")
        handle = self._handles.pop(modality, None)
        self._registered.discard(modality)
        self._hub.unregister(modality)
        if handle is None:
            self._set_connect_state(modality, connected=False)
            return
        try:
            if handle.recording_active:
                handle.end_recording(self._trial_uuid)
        except Exception as exc:
            self._diagnostics.record("device_end_recording_failed", modality=modality, exception=exc)
        if handle.is_alive:
            handle.request_stop()
            handle.join(timeout=2.0)
        if handle.is_alive:
            handle.terminate(timeout=2.0)
        handle.close()
        self._set_connect_state(modality, connected=False)
        self._status_strip.set_state(modality, "off", "未连接")
        self._workspace.set_stream_state(modality, "disconnected")

    def _set_connect_state(
        self, modality: str, *, connected: bool, connecting: bool = False
    ) -> None:
        dot = self._connect_dots.get(modality)
        button = self._connect_buttons.get(modality)
        if dot is not None:
            color = "#d39e00" if connecting else ("#0f766e" if connected else "#d3d0c7")
            dot.setStyleSheet(
                f"background:{color}; border-radius:6px;"
            )
        if button is not None:
            active = connected or connecting
            button.setText("断开" if active else "连接")
            button.setProperty("buttonRole", "disconnect" if active else "connect")
            button.style().unpolish(button)
            button.style().polish(button)

    # -- preview polling ---------------------------------------------------

    def _poll_previews(self) -> None:
        now = time.monotonic()
        self._poll_max_gap_s = max(self._poll_max_gap_s, now - self._poll_last)
        self._poll_last = now
        self._poll_count += 1
        try:
            self._poll_previews_impl()
            self._advance_startup()
        except Exception as exc:
            # A failing Qt timeout must remain observable and must not silently
            # leave stale green status indicators behind.
            self._record_throttled_exception("preview_poll_failed", exc)
            self._fault_label.setText(f"预览刷新异常：{exc}；详情见诊断日志。")
        if now - self._diagnostic_last >= 1:
            try:
                self._record_progress(now)
            except Exception as exc:
                self._record_throttled_exception("diagnostic_snapshot_failed", exc)
        self._poll_max_duration_s = max(self._poll_max_duration_s, time.monotonic() - now)

    def _record_throttled_exception(self, kind, exc, **fields):
        key = (kind, fields.get("modality"), type(exc).__name__, str(exc))
        if key not in self._diagnostic_errors and len(self._diagnostic_errors) >= 64:
            self._diagnostic_errors.clear()
        now = time.monotonic()
        last, suppressed = self._diagnostic_errors.get(key, (0, 0))
        if now - last >= 5:
            self._diagnostics.record(kind, exception=exc, suppressed=suppressed, **fields)
            self._diagnostic_errors[key] = (now, 0)
        else:
            self._diagnostic_errors[key] = (last, suppressed + 1)

    def _record_progress(self, now):
        streams = {}
        for modality, handle in self._handles.items():
            latest = self._hub.latest(modality)
            endpoint = handle.recording_endpoint
            try:
                depth = endpoint.queue.qsize() if endpoint is not None else None
            except (AttributeError, OSError, NotImplementedError):
                depth = None
            per_device = {}
            if modality in self._hub.modalities:
                descriptor = self._hub.descriptor(modality)
                ids = descriptor.get("metadata", {}).get("device_ids", ())
                for device_id in ids:
                    value = self._hub.latest_for_device(modality, str(device_id))
                    per_device[str(device_id)] = {"last_raw_ns": value[0] if value is not None else None,
                        "age_s": (time.perf_counter_ns() - value[0]) / 1e9 if value is not None else None}
            streams[modality] = {"worker_alive": handle.is_alive,
                "worker_pid": getattr(getattr(handle, "_process", None), "pid", None),
                "worker_exitcode": getattr(getattr(handle, "_process", None), "exitcode", None),
                "preview_events": self._preview_counts.get(modality, 0),
                "last_preview_ns": self._preview_latest_ns.get(modality),
                "raw_buffer_samples": self._hub.count(modality), "raw_queue_depth": depth,
                "last_raw_ns": latest[0] if latest is not None else None,
                "raw_boundary": self._hub.boundary(modality),
                "health": self._worker_health.get(modality), "devices": per_device}
            last_preview = self._preview_latest_ns.get(modality, self._worker_connected_ns.get(modality))
            if last_preview is not None and (time.perf_counter_ns() - last_preview) / 1e9 > 2:
                self._status_strip.set_state(modality, "bad", "预览停滞")
                self._workspace.set_stream_state(modality, "error")
        self._diagnostics.record("heartbeat", poll_ticks=self._poll_count,
            gui_max_gap_s=self._poll_max_gap_s, gui_max_poll_duration_s=self._poll_max_duration_s,
            loading=self._load_generation,
            controller_running=self.controller_running, streams=streams,
            runtime=self._runtime_evidence(),
            motor_transport=(getattr(self._motor, "diagnostics", {}) or {}).get("transport") if self._motor is not None else None)
        self._diagnostic_last = now
        self._poll_max_gap_s = 0.0
        self._poll_max_duration_s = 0.0
        if not self._diagnostics.available:
            self._controls.log_label.setText(f"诊断日志不可用：{self._diagnostics.path}")

    def _poll_previews_impl(self) -> None:
        for modality, handle in list(self._handles.items()):
            if self._fatal_latched:
                return
            poll_error = getattr(handle, "last_poll_error", None)
            if poll_error is not None:
                self._record_throttled_exception("worker_queue_error", poll_error, modality=modality)
                self._status_strip.set_state(modality, "bad", "预览队列异常")
                self._workspace.set_stream_state(modality, "error")
                if _transient_input_error(poll_error):
                    if not self._request_input_recovery(modality, poll_error):
                        self._fatal_incident(f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 输入故障",
                                             poll_error, source="worker_queue_error", modality=modality)
                    else:
                        self._disconnect_modality(modality, handoff=True)
                    continue
                self._fatal_incident(f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 预览故障",
                                     poll_error, source="worker_queue_error", modality=modality)
                return
            try:
                events = handle.poll_events(limit=200)
            except Exception as exc:
                self._record_throttled_exception("worker_poll_failed", exc, modality=modality)
                if _transient_input_error(exc):
                    if not self._request_input_recovery(modality, exc):
                        self._fatal_incident(f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 输入故障",
                                             exc, source="worker_poll_failed", modality=modality)
                    else:
                        self._disconnect_modality(modality, handoff=True)
                else:
                    self._fatal_incident(f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 预览故障",
                                         exc, source="worker_poll_failed", modality=modality)
                continue
            for event in events:
                self._handle_worker_event(modality, event)
                if modality not in self._handles:
                    break
            # A worker can exit before its FAILED event is observed (for
            # example when a USB adapter raises during connect).  Drop the
            # stale handle before looking at its cached endpoint, otherwise a
            # dead stream can be registered and block a retry.
            if modality not in self._handles:
                continue
            if not handle.is_alive:
                self._fatal_incident(f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 预览进程退出",
                                     "输入预览进程已退出，当前连接已关闭。", source="worker_exit", modality=modality)
                continue
            if handle.recording_endpoint is not None and modality not in self._registered:
                self._hub.register_endpoint(modality, handle.recording_endpoint)
                try:
                    handle.begin_recording(self._trial_uuid)
                except Exception as exc:
                    self._fatal_incident(
                        f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 数据流故障",
                        f"无法建立输入数据流：{exc}", source="recording_endpoint", modality=modality)
                    continue
                self._registered.add(modality)
                self._status_strip.set_state(modality, "ok", "实时")
                self._workspace.set_stream_state(modality, "live")
        self._refresh_recording_context()
        if self._controller is None or not self._controller.isRunning():
            self._hub.drain(max_per_stream=256)
        self._poll_motor()
        self._poll_recording()

    def _handle_worker_event(self, modality: str, event: WorkerEvent) -> None:
        if event.event_type is WorkerEventType.STATE:
            state = str(event.payload.get("state") or "")
            if self._worker_states.get(modality) != state:
                self._diagnostics.record("worker_state", modality=modality, payload=event.payload)
                self._worker_states[modality] = state
            self._status_strip.set_state(modality, _block_state(state), state)
            if state == "READY":
                self._set_connect_state(modality, connected=True)
                self._workspace.set_stream_state(modality, "connected")
            elif state in {"DISCONNECTED", "CLOSED"}:
                self._fatal_incident(f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 设备断开",
                                     "输入设备已断开，当前连接已关闭。", source="worker_disconnected", modality=modality)
        elif event.event_type is WorkerEventType.PREVIEW:
            self._preview_counts[modality] = self._preview_counts.get(modality, 0) + 1
            self._preview_latest_ns[modality] = time.perf_counter_ns()
            self.feed_signal_preview(modality, event.payload)
            self._workspace.set_stream_state(modality, "live")
            if modality in self._registered:
                self._status_strip.set_state(modality, "ok", "实时")
        elif event.event_type is WorkerEventType.FAILED:
            self._diagnostics.record("worker_failed", modality=modality, message=event.message, payload=event.payload)
            message = event.message or str(event.payload.get("fault") or "")
            message = message or "预览 worker 故障"
            if _transient_input_error(message) and self._request_input_recovery(modality, message):
                self._diagnostics.record("worker_failure_recovery", modality=modality,
                                         message=message)
                self._disconnect_modality(modality, handoff=True)
                return
            self._fatal_incident(f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 预览 worker 故障",
                                 message, source="worker_failed", modality=modality)
        elif event.event_type is WorkerEventType.HEALTH:
            self._worker_health[modality] = event.payload

    def _remove_failed_modality(
        self, modality: str, message: str, *, failed: bool = True
    ) -> None:
        """Compatibility shim for older callers; fatal failures are global."""
        self._diagnostics.record("device_removed", modality=modality,
                                 reason=message, failed=failed)
        if failed:
            if _transient_input_error(message) and self._request_input_recovery(modality, message):
                self._disconnect_modality(modality, handoff=True)
            else:
                self._fatal_incident(
                    f"{MODALITY_DISPLAY_NAMES.get(modality, modality)} 输入故障",
                    message, source="failed_modality", modality=modality)
        else:
            self._disconnect_modality(modality)

    # -- model load / reload ----------------------------------------------

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[4]

    def _load_bundled_model(self) -> None:
        path = self._project_root() / "runtime_data/models/imu_right_pelvis_torque_100hz_rf94_static_20261017/model_spec.json"
        self.load_model_from_path(path)

    def _choose_model_entry(self) -> None:
        selected = self._model_choice.currentData()
        if selected:
            self.load_model_from_path(Path(selected))

    def _update_gate_compatibility_status(self, spec: ModelSpec | None = None) -> str | None:
        """Show the selected model/gate mismatch as soon as either changes."""
        if spec is None:
            spec = self._spec
        if spec is None:
            path_text = self._spec_path_edit.text().strip()
            if path_text and Path(path_text).is_file():
                try:
                    spec = load_model_spec(path_text)
                except Exception:
                    return None
        if spec is None:
            return None
        reason = gate_compatibility_error(spec, self._gate_mode)
        if reason:
            message = f"门控与模型不兼容：{reason}"
            self._model_status.setText(message)
            self.statusBar().showMessage(message)
        elif self._model_status.text().startswith("门控与模型不兼容："):
            if self._spec is not None:
                message = (f"已加载 · 输入：{', '.join(i.modality for i in self._spec.features.inputs)} · 点击运行")
            elif self._load_generation is not None:
                message = "正在后台加载模型…可点击运行，加载完成后自动准备输入"
            else:
                message = "未加载"
            self._model_status.setText(message)
            self.statusBar().showMessage(message)
        return reason

    def _show_model_choice(self, path: Path) -> None:
        path = Path(path).resolve()
        key = str(path)
        entry = self._catalog_entries.get(key)
        if entry is None:
            try:
                entry = read_model_entry(path, require_files=False)
            except Exception:
                entry = ModelCatalogEntry(path, "模型配置待确认", "正在检查模型配置；详情见诊断日志。")
        self._model_choice.blockSignals(True)
        try:
            index = self._model_choice.findData(key)
            if index < 0:
                self._model_choice.addItem(entry.name, key)
                index = self._model_choice.count() - 1
                # External programmatic models are display-only; the primary
                # selector offers only complete local deployments.
                self._model_choice.model().item(index).setEnabled(False)
            self._model_choice.setItemData(index, entry.tooltip, Qt.ItemDataRole.ToolTipRole)
            self._model_choice.setCurrentIndex(index)
            self._model_choice.setToolTip(entry.tooltip)
        finally:
            self._model_choice.blockSignals(False)

    def _change_profile(self) -> None:
        self._stop_controller()
        self.disconnect_all()
        self._profile_key = str(self._profile_combo.currentData())
        self._applied_signature = None
        self._controls.preflight_label.setText("数据源已切换，请连接输入后运行。")

    def _configure_imu(self) -> None:
        from exo_collection.apps.collector.device_settings import ImuDeviceSettingsDialog
        if "imu" in self._handles:
            self._show_error("请先断开 IMU", "设备设置在下次连接时生效。")
            return
        # Pick up settings saved by Collector while this window remains open.
        self._hardware_overrides = self._settings.hardware_device_overrides
        current = dict(load_device_profile("hardware").by_modality()["imu"].parameters.model_dump())
        previous = dict(self._hardware_overrides.get("imu", {}))
        current.update(previous)
        dialog = ImuDeviceSettingsDialog(current, self)
        dialog.setWindowTitle("在线测试 · Xsens USB 设备设置")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            override = dict(dialog.validated_override)
            try:
                self._hardware_overrides["imu"] = self._settings.set_hardware_device_override(
                    "imu", override
                )
                # Saving an explicit real-device configuration follows
                # Collector's shared-profile convention.  Switching away from
                # simulated inputs uses the existing stop/disconnect path and
                # still does not connect or start hardware.
                self._settings.set_device_profile_key("hardware")
                if self._profile_key != "hardware":
                    self._profile_combo.setCurrentIndex(
                        self._profile_combo.findData("hardware")
                    )
            except Exception as exc:
                self.emergency_stop()
                self._show_error("IMU 设备设置保存失败", exc)
                return
            self._controls_changed()

    def _imu_binding_parameters(self):
        device = load_device_profile(self._profile_key).by_modality()["imu"]
        parameters = device.parameters.model_dump()
        if self._profile_key == "hardware":
            self._hardware_overrides = self._settings.hardware_device_overrides
            parameters.update(self._hardware_overrides.get("imu", {}))
        return parameters

    def _controls_changed(self) -> None:
        self.emergency_stop()
        if self._motor is not None:
            # Reducing a displayed limit takes effect even before Apply.
            self._motor.set_limit(self._controls.limit.value())
        if hasattr(self, "_status_labels"):
            self._status_labels["limit"].setText(
                f"底层 ±{self._motor.torque_limit_nm:g} Nm" if self._motor is not None
                else f"±{self._controls.limit.value():g} Nm"
            )
        self._applied_signature = None
        self._controls.preflight_label.setText("参数已改变；点击运行自动应用并校验。")

    def _save_runtime_parameter(self, name, value):
        try:
            self._settings.set_model_runtime_parameters({name: value})
            self._diagnostics.record("runtime_parameter_saved", parameter=name, value=value,
                                     active_motor_limit_nm=(self._motor.torque_limit_nm
                                                            if self._motor is not None else None))
        except Exception as exc:
            self._diagnostics.record("runtime_parameter_save_failed", parameter=name, exception=exc)
            self.statusBar().showMessage("参数已保留在当前窗口，但保存失败；详情见日志。")

    def _apply_controls(self, *, quiet=False) -> bool:
        if self._spec is None:
            if quiet:
                self._fail_startup("请先选择模型。")
            else:
                self._show_error("未选择模型", "请先选择已部署模型或加载模型接口文件。")
            return False
        try:
            values = self._spec.model_dump()
            if self._model_template_spec is not None:
                values["imu_tcn"] = self._model_template_spec.imu_tcn.model_dump()
            # Operator fields override only editable values; preserve the
            # deployment's motor distribution when applying or rerunning.
            values["assistance"].update(self._controls.assistance_values())
            values["output"]["torque_limit_nm"] = self._controls.limit.value()
            spec = bind_model_inputs(
                ModelSpec.model_validate(values), profile_key=self._profile_key,
                imu_parameters=self._imu_binding_parameters(),
                descriptor=self._hub.descriptor("imu") if "imu" in self._hub.modalities else None,
            )
            if self._install_runtime(spec, base_dir=self._spec_path.parent if self._spec_path else None,
                                     runtime=self._runtime):
                self._applied_signature = self._controls.signature()
                self._preflight(quiet=True)
                self._model_output_panel.clear()
                return True
        except Exception as exc:
            if quiet:
                self._fail_startup(f"模型参数或输入绑定不可用：{exc}", detail=exc)
            else:
                self._show_error("参数应用失败", exc)
        return False

    def _cancel_startup(self, reason):
        if self._startup.active:
            self._gate_choice.setEnabled(True)
            self._diagnostics.record("startup_cancelled", generation=self._startup.generation, reason=reason)
            self._startup.cancel(reason)
            self._run_button.setEnabled(self._spec is not None or bool(self._spec_path_edit.text().strip()))
            self._model_status.setText("启动已取消 · 点击运行可重新准备")

    def _startup_status(self, message):
        self._model_status.setText(message)
        self.statusBar().showMessage(message)
        self._run_button.setEnabled(False)

    def _fail_startup(self, message, *, detail=None, stop=True):
        self._gate_choice.setEnabled(True)
        generation = self._startup.generation
        self._startup.fail(message)
        self._diagnostics.record("startup_failed", generation=generation, reason=message,
                                 detail=str(detail) if detail is not None else None,
                                 preflight=self._last_preflight_report)
        if stop:
            self.emergency_stop()
        self._model_status.setText("启动未完成")
        self._fault_label.setText(message)
        self.statusBar().showMessage(message)
        self._model_output_panel.set_output_status(message)
        self._workspace.set_stream_state("model_output", "error")
        self._uncertainty_panel.mark_inactive(fault=True)
        self._workspace.set_stream_state("uncertainty", "error")

    def _run_prediction(self) -> None:
        """Latch an explicit Run intent; loading/connection/warmup are async."""
        self._reset_fault_lifecycle("explicit_run")
        if self._run.running or self._startup.active:
            return
        if self._spec is None and self._load_generation is None:
            selected_path = self._spec_path_edit.text().strip()
            if selected_path:
                self.load_model_from_path(Path(selected_path))
            else:
                self._model_status.setText("请先从下拉列表选择模型。")
                self.statusBar().showMessage("请先选择模型，再点击运行。")
                return
        generation = self._startup.begin()
        self._gate_choice.setEnabled(False)
        self._startup_probe_after = 0.0
        self._fault_label.clear()
        self._diagnostics.record("startup_requested", generation=generation,
                                 loading=self._load_generation)
        self._advance_startup()

    def _advance_startup(self):
        if not self._startup.active or self._closing:
            return
        generation = self._startup.generation
        try:
            if self._startup.expired:
                captions = {StartupPhase.LOADING: "模型后台加载超时，请重新选择模型。",
                    StartupPhase.CONNECTING: "输入设备连接超时；请检查设备和连接设置后重新运行。",
                    StartupPhase.WARMING: "等待模型输入窗口超时；请检查所需传感器是否持续输出。"}
                self._fail_startup(captions.get(self._startup.phase, "启动等待超时，请重新运行。"))
                return
            if self._load_generation is not None:
                self._startup.wait(generation, StartupPhase.LOADING, timeout_s=60)
                self._startup_status("已请求运行 · 正在后台加载模型…")
                return
            if self._spec is None:
                self._fail_startup("模型尚不可用，请重新选择模型。")
                return
            compatibility_error = gate_compatibility_error(self._spec, self._gate_mode)
            if compatibility_error:
                self._fail_startup(compatibility_error, detail=compatibility_error)
                return
            required = tuple(dict.fromkeys(entry.modality for entry in self._spec.features.inputs))
            self._startup.required = required
            missing = [modality for modality in required if modality not in self._hub.modalities]
            if missing:
                self._startup.wait(generation, StartupPhase.CONNECTING, timeout_s=30)
                for modality in missing:
                    if modality in self._handles or modality in self._startup.attempted:
                        continue
                    self._startup.attempted.add(modality)
                    self._connect_modality(modality, quiet=True)
                    if not self._startup.matches(generation):
                        return
                names = "、".join(MODALITY_DISPLAY_NAMES.get(modality, modality) for modality in missing)
                self._startup_status(f"已请求运行 · 正在连接输入：{names}…")
                return
            self._startup.wait(generation, StartupPhase.WARMING,
                               timeout_s=max(15.0, self._spec.features.window_s + 10.0))
            if self._controller is None or self._applied_signature != self._controls.signature():
                if not self._apply_controls(quiet=True):
                    return
            if not self._startup.matches(generation):
                return
            if time.monotonic() < self._startup_probe_after:
                return
            self._startup_probe_after = time.monotonic() + .1
            if not self._preflight(quiet=True):
                self._startup_status(f"正在预热输入 · 等待完整 {self._spec.features.window_s:g} 秒窗口…")
                return
            # Consume this exact intent before crossing the thread-start edge.
            # Later callbacks/Stop/model changes cannot revive it.
            if not self._startup.finish(generation):
                return
            self._run.start()
            self._diagnostics.record("startup_started", generation=generation, modalities=required)
            self._attempt_motor_arm()
            self._model_status.setText("已加载 · 运行中")
            self._run_button.setEnabled(False)
            self.statusBar().showMessage("推理已运行；电机输出按连接与安全门控决定。")
        except Exception as exc:
            self._fail_startup(f"无法启动模型：{exc}", detail=exc)

    def _preflight(self, _checked=False, *, quiet=False) -> bool:
        if self._controller is None:
            self._controls.preflight_label.setText("未加载模型。")
            return False
        try:
            if not self._controller.isRunning():
                self._hub.drain()
            result = self._controller.preflight()
            self._last_preflight_report = result
            ok = bool(result.get("ready"))
            errors = result.get("errors", [])
            summary = result.get("contract_summary") or self._input_contract_summary()
            message = ("输入校验通过：" if ok else "等待输入：") + summary
            if errors:
                message += "\n" + "; ".join(map(str, errors))
            buffers = result.get("buffer_status", {})
            if buffers:
                message += "\n缓冲：" + "；".join(
                    f"{modality} {status.get('samples', 0)} 个源样本"
                    for modality, status in buffers.items()
                )
            assistance = self._controller.assistance_preflight()
            if not assistance.get("ready"):
                message += "\n仅模型估计；中层不可换算：" + "; ".join(map(str, assistance.get("errors", [])))
            for name in ("right_age_ms", "pelvis_age_ms"):
                if name in result:
                    message += f"\n{name}: {result[name]} ms"
            self._controls.preflight_label.setText(message)
            if not quiet:
                self._write_log("preflight", result)
            return ok
        except Exception as exc:
            self._controls.preflight_label.setText(f"校验未通过：{exc}")
            return False

    def _input_contract_summary(self) -> str:
        spec = self._spec
        if spec is None:
            return "未加载模型"
        inputs = "; ".join(
            f"{item.modality} ({', '.join(item.channels) if item.channels else '描述器全部通道'})"
            for item in spec.features.inputs
        )
        return (f"{inputs}；窗口 {spec.features.window_s:g} s；"
                f"控制率 {spec.features.control_rate_hz:g} Hz；输出 {spec.output.unit}")

    def _require_real_inputs(self) -> None:
        if self._smoke_test or self._profile_key != "hardware":
            raise ValueError("模拟数据或 smoke 模式禁止驱动真实电机。")
        if self._spec is None or not self._spec.imu_tcn.enabled:
            raise ValueError("真实辅助只接受已声明契约的 IMU TCN 模型。")
        if "imu" not in self._registered or "imu" not in self._hub.modalities:
            raise ValueError("请先连接真实 IMU。")
        metadata = self._hub.descriptor("imu").get("metadata", {})
        if metadata.get("simulated") is not False:
            raise ValueError("IMU 描述信息未确认真实数据来源，禁止真实辅助。")
        handle = self._handles.get("imu")
        if handle is None or not handle.is_alive:
            raise ValueError("IMU 采集进程已停止。")

    def _record_motor_close_result(self, motor, stage: str, error=None) -> bool:
        result = getattr(motor, "close_snapshot", None)
        success = error is None or isinstance(result, dict) and result.get("serial_close_succeeded") is True
        self._diagnostics.record("motor_close_result", port=getattr(motor, "port", None),
            stage=stage, succeeded=success, snapshot=result, exception=error)
        if success:
            if self._pending_motor_cleanup is motor:
                self._pending_motor_cleanup = None
        else:
            self._pending_motor_cleanup = motor
        return success

    def _close_motor_with_evidence(self, motor, stage: str) -> None:
        try:
            motor.close()
        except Exception as exc:
            self._record_motor_close_result(motor, stage, exc)
            raise
        self._record_motor_close_result(motor, stage)

    def _retry_pending_motor_cleanup(self) -> bool:
        motor = self._pending_motor_cleanup
        if motor is None:
            return True
        try:
            self._close_motor_with_evidence(motor, "explicit_retry")
            return True
        except Exception:
            message = "电机串口尚未确认释放，禁止建立新连接；请检查USB连接后重试全部断开。"
            self._fault_label.setText(message)
            self._controls.motor_status.setText(message)
            self.statusBar().showMessage(message)
            return False

    def _toggle_motor(self) -> None:
        if self._motor is None:
            if not self._retry_pending_motor_cleanup():
                return
            self._reset_fault_lifecycle("explicit_motor_connect")
        self._diagnostics.record("motor_connection_requested",
            action="disconnect" if self._motor is not None else "connect",
            port=self._controls.port.currentText().strip(), running=self._run.running)
        if self._motor is not None:
            self._disconnect_motor()
            return
        connecting_output = False
        try:
            if self._smoke_test or self._profile_key != "hardware":
                raise ValueError("模拟数据或 smoke 模式禁止连接真实电机。")
            port = self._controls.port.currentText().strip()
            if not port:
                raise ValueError("请选择电机 COM 口。")
            if "encoder" in self._handles:
                self._disconnect_modality("encoder", handoff=True)
            from exo_collection.apps.model_runtime.motor_service import SerialTorqueOutput
            self._refresh_recording_context()
            connecting_output = True
            motor = SerialTorqueOutput(port, torque_limit_nm=self._controls.limit.value(),
                recording_sink=self._online_recording.motor_event)
            try:
                motor.connect()
            except Exception:
                self._close_motor_with_evidence(motor, "connect_failed")
                raise
            try:
                self._run.attach_motor(motor)
            except Exception as exc:
                self._record_motor_close_result(motor, "attach_failed", exc)
                raise
            if self._encoder_input_requested:
                self._register_motor_encoder()
            self._last_motor_fault = None
            self._last_logged_command_ns = -1
            self._last_feedback_ns = -1
            self._controls.motor_button.setText("断开电机")
            self._controls.motor_status.setText(f"已连接 {port}；保持失能。")
            self._workspace.show_panel("encoder")
            self._workspace.set_stream_state("encoder", "connected")
            self._attempt_motor_arm()
            self._diagnostics.record("motor_connected", port=port,
                running=self._run.running, reason=self._run.reason)
        except Exception as exc:
            self._diagnostics.record("motor_connection_failed", exception=exc,
                running=self._run.running)
            if connecting_output:
                self._fatal_incident("电机连接失败", exc, source="motor_connect_failed")
                return
            self._show_error("电机连接失败", exc)
            if not self._run.running:
                self._model_status.setText("预测未运行；请重新运行")
                self._run_button.setEnabled(True)

    def _disconnect_motor(self, *, restore_encoder: bool = True) -> None:
        self._cancel_startup("motor_disconnected")
        if self._motor is None:
            return
        self._diagnostics.record("motor_disconnect", port=getattr(self._motor, "port", None),
            running=self._run.running, fault=self._motor.fault)
        mock = MockTorqueOutput(
            self._spec.output.torque_limit_nm if self._spec is not None else 0.0,
            enable_required=True, zero_on_error=True,
        )
        motor = self._motor
        try:
            self._run.detach_motor(mock)
        except RuntimeError as exc:
            self._record_motor_close_result(motor, "disconnect", exc)
            self._fault_label.setText(str(exc))
            self.statusBar().showMessage(str(exc))
        else:
            self._record_motor_close_result(motor, "disconnect")
        self._motor_encoder_queue = None
        if "encoder" not in self._handles:
            self._hub.unregister("encoder")
            self._registered.discard("encoder")
        self._controls.motor_button.setText("连接电机（保持失能）")
        self._controls.motor_status.setText("电机已断开；继续显示估计和期望辅助（未下发）。")
        self._model_output_panel.clear_command()
        self._model_output_panel.set_feedback(None)
        self._model_output_panel.set_output_status("未连接电机 · 未下发")
        self._update_torque_label()
        self._workspace.set_stream_state("encoder", "disconnected")
        self._status_strip.set_state("encoder", "off", "未连接")
        if self._pending_motor_cleanup is not None:
            self._controls.motor_status.setText("已归零失能；串口释放失败，请点击全部断开重试。")
            self._status_strip.set_state("encoder", "bad", "串口未释放")
        elif restore_encoder and self._encoder_input_requested:
            self._connect_modality("encoder")

    def _register_motor_encoder(self):
        from queue import Queue
        from exo_collection.adapters.encoder.simulated import ENCODER_CHANNELS
        if self._motor_encoder_queue is None:
            self._motor_encoder_queue = Queue(maxsize=16)
            self._encoder_snapshot_sequence = 0
            rate = 1000.0 / _PREVIEW_POLL_MS
            self._hub.register_stream("encoder", self._motor_encoder_queue,
                channels=ENCODER_CHANNELS, sample_shape=(6,), rate_hz=rate,
                device_id="motor_feedback", descriptor={"channels": ENCODER_CHANNELS,
                    "units": ("rad", "rad/s", "N*m", "rad", "rad/s", "N*m"),
                    "sample_shape": (6,), "nominal_rate_hz": rate,
                    "metadata": {"simulated": False, "source": "motor_feedback_ui_snapshot"}})
            self._registered.add("encoder")
        self._set_connect_state("encoder", connected=True)
        self._status_strip.set_state("encoder", "ok", "共享电机反馈")

    def _poll_motor(self) -> None:
        if self._fatal_latched:
            return
        motor = self._motor
        output_state = getattr(self._run, "output_state", None)
        if output_state is OutputState.FAULT:
            self._fatal_incident("电机输出故障", self._run.reason or "输出状态进入故障锁定",
                                 source="output_state_fault")
            return
        if motor is None:
            return
        if motor.fault:
            self._last_motor_fault = motor.fault
            self._diagnostics.record("motor_fault", fault=motor.fault,
                                     snapshot=self._fault_snapshot())
            self._write_log("motor_fault", motor.fault)
            self._fatal_incident("电机底层故障", _motor_fault_reason(motor),
                                 source="motor_fault")
            return
        state = (motor.connected, motor.enabled, motor.fault, self._run.running,
                 str(self._run.output_state), self._run.reason)
        if state != getattr(self, "_last_motor_diagnostic_state", None):
            self._last_motor_diagnostic_state = state
            self._diagnostics.record("motor_output_state", connected=state[0], enabled=state[1],
                fault=state[2], running=state[3], output_state=state[4], reason=state[5])
        feedback = motor.feedback
        age_s = None if feedback is None else (time.perf_counter_ns() - feedback.host_monotonic_ns) / 1e9
        self._model_output_panel.set_feedback(feedback, age_s)
        self._model_output_panel.set_output_status(self._run.reason)
        if feedback is not None:
            unhealthy = getattr(feedback, "error", 0) != 0 or getattr(feedback, "state", 0) not in (0, 2)
            stale = age_s is None or age_s > 0.25 or unhealthy
            self._status_strip.set_state(
                "encoder", "bad" if stale else "ok",
                "控制板状态异常" if unhealthy else ("状态包过期" if stale else "控制板状态包")
            )
            self._workspace.set_stream_state("encoder", "error" if stale else "live")
        self._attempt_motor_arm()
        if getattr(self._run, "output_state", None) is OutputState.FAULT:
            self._fatal_incident("电机输出故障", self._run.reason or "输出状态进入故障锁定",
                                 source="output_state_fault")
            return
        self._controls.motor_status.setText(
            f"{self._run.reason}\n"
            f"{'已连接' if motor.connected else '连接已断开'} ｜ "
            f"{'已使能' if motor.enabled else '失能 / 等待反馈确认'} ｜ "
            f"控制板状态包 {'无' if age_s is None else f'{age_s * 1000:.0f} ms'} ｜底层 ±{motor.torque_limit_nm:g} Nm"
        )
        for command in motor.history:
            if command.host_monotonic_ns > self._last_logged_command_ns:
                self._last_logged_command_ns = command.host_monotonic_ns
                record = self._command_record(command)
                self._write_log("sent_command", record)
                self._online_recording.command(record)
        if motor.last is not None:
            self._model_output_panel.add_command(motor.last)
            self._update_torque_label(motor.last)
        if feedback is not None and feedback.host_monotonic_ns > self._last_feedback_ns:
            self._last_feedback_ns = feedback.host_monotonic_ns
            values = (feedback.left_position_rad, feedback.left_velocity_rad_s,
                      feedback.left_torque_nm, feedback.right_position_rad,
                      feedback.right_velocity_rad_s, feedback.right_torque_nm)
            if self._motor_encoder_queue is not None:
                import numpy as np
                from queue import Full
                from exo_collection.domain.events import SampleBatch
                batch = SampleBatch(device_id="motor_feedback", modality="encoder",
                    clock_domain="host", first_sample_index=self._encoder_snapshot_sequence, sample_count=1,
                    sequence_number=self._encoder_snapshot_sequence, sample_rate_hz=1000.0 / _PREVIEW_POLL_MS,
                    host_monotonic_ns=feedback.host_monotonic_ns,
                    data=np.asarray([values], dtype=np.float32))
                self._encoder_snapshot_sequence += 1
                try:
                    self._motor_encoder_queue.put_nowait(batch)
                except Full:
                    # A bounded subscriber may drop a snapshot; never rewrite
                    # its source timestamp or race the inference queue consumer.
                    pass
            self.feed_signal_preview("encoder", {
                "labels": list(ENCODER_PREVIEW_LABELS),
                "channels": [[value] for value in values],
            })
            self._write_log("feedback_snapshot", feedback)

    def _validate_motor_arm(self) -> None:
        self._require_real_inputs()
        assistance = self._controller.assistance_preflight() if self._controller else {"ready": False, "errors": ["模型未运行"]}
        if not assistance.get("ready"):
            raise ValueError("中层不可换算 / 禁止辅助：" + "; ".join(map(str, assistance.get("errors", []))))
        if self._controls.limit.value() <= 0:
            raise ValueError("底层阈值为 0 Nm；继续仅显示估计。")
        if not self._preflight(quiet=True):
            raise ValueError(self._controls.preflight_label.text())

    def _attempt_motor_arm(self, feedback=None) -> None:
        self._run.poll_output(self._validate_motor_arm)
        if self._motor is not None:
            self._controls.motor_status.setText(self._run.reason)

    def _refresh_recording_context(self):
        streams = {}
        for modality, handle in self._handles.items():
            endpoint = handle.recording_endpoint
            if modality in self._registered and endpoint is not None:
                streams[modality] = {"descriptor": dict(endpoint.descriptor),
                    "configuration_snapshot": dict(endpoint.configuration_snapshot)}
        metadata = {"profile": self._profile_key, "model_sha256": self._model_sha256,
            "model_spec_path": str(self._spec_path) if self._spec_path else None,
            "spec": self._spec.model_dump(mode="json") if self._spec else None,
            "runtime_parameters": self._controls.parameters_snapshot() if hasattr(self._controls, "parameters_snapshot") else {},
            "motor_port": self._controls.port.currentText(),
            "motor_limit_nm": self._controls.limit.value(),
            "gate": {"mode": self._gate_mode, "max_age_ms": 100,
                     "decision_rate_hz": 20, "input_grid_hz": 100,
                     "resume_rule": "keep pregate history; hold zero until delayed sample source reaches new gate epoch"}}
        if self._controller is not None and self._gate_mode != "none":
            metadata["gate"]["provenance"] = self._controller.gate_provenance
        self._online_recording.configure(metadata, streams)

    def _poll_recording(self):
        recording = self._online_recording
        if recording.error and recording.error != self._last_recording_error:
            self._last_recording_error = recording.error
            self._fault_label.setText(f"同步录制不完整：{recording.error}")
            self._diagnostics.record("online_recording_failed", reason=recording.error)
        label = "同步录制中" if recording.active else "同步录制待命"
        if recording.last_path is not None:
            label += f"：{recording.last_path}"
        if recording.error:
            label = f"同步录制不完整：{recording.error}"
        self._controls.log_label.setToolTip(self._controls.log_label.text() + "\n" + label)
        if recording.active:
            self.statusBar().showMessage(label)

    def _update_log_label(self):
        paths = f"{'诊断' if self._diagnostics.available else '诊断日志不可用'}：{self._diagnostics.path}"
        if self._session_log is not None:
            paths += f"\n运行：{self._session_log.path}"
        self._controls.log_label.setText(paths)
        self._controls.log_label.setToolTip(paths)

    def _start_log(self, spec: ModelSpec, base_dir: Path | None) -> None:
        if self._smoke_test:
            return
        metadata = {"spec": spec.model_dump(), "profile": self._profile_key,
                    "diagnostic_log": str(self._diagnostics.path),
                    "output_interface": "serial" if self._motor is not None else "display_only",
                    "right_sign": spec.assistance.right_sign,
                    "motor_distribution": spec.assistance.motor_distribution,
                    "gate": {"mode": self._gate_mode, "max_age_ms": 100,
                             "decision_rate_hz": 20, "input_grid_hz": 100,
                             "resume_rule": "keep pregate history; hold zero until delayed sample source reaches new gate epoch"}}
        if self._model_sha256 is not None:
            metadata["model_sha256"] = self._model_sha256
        self._session_log = SessionLog(self._project_root() / "runtime_data/model_runtime_logs", metadata)
        self._update_log_label()

    def _write_log(self, kind, payload) -> None:
        if self._session_log is None:
            return
        try:
            self._session_log.write(kind, payload)
        except Exception as exc:
            # Detach first: Stop and close must not retry a broken writer or
            # recursively enter emergency_stop through another audit failure.
            broken, self._session_log = self._session_log, None
            self._diagnostics.record("session_log_failed", exception=exc, path=str(broken.path))
            try:
                broken.close()
            except Exception as close_exc:
                self._diagnostics.record("session_log_close_failed", exception=close_exc)
            self.emergency_stop()
            self._fault_label.setText(f"日志写入失败，已失能：{exc}")

    def _on_pipeline_sample(self, payload: dict) -> None:
        if self._fatal_latched or not self._run.running or (self.sender() is not None and self.sender() is not self._controller):
            return
        self._model_output_panel.set_output_status(self._run.reason)
        if self._motor is None:
            self._model_status.setText("推理运行中 · 仅显示预测与辅助" +
                                       self._gate_status_text(payload.get("gate_status")))
            self.statusBar().showMessage("推理运行中 · 电机未连接，未下发。")
        elif payload.get("gate_status") is not None:
            self._model_status.setText("推理运行中" + self._gate_status_text(payload["gate_status"]))
        self._workspace.set_stream_state("model_output", "live")
        self._model_output_panel.add_pipeline(payload)
        gate_status = payload.get("gate_status")
        self._uncertainty_panel.add_pipeline(payload)
        gate_state = getattr(gate_status, "state", None)
        if isinstance(gate_status, dict):
            gate_state = gate_status.get("state")
        self._workspace.set_stream_state(
            "uncertainty", "live" if gate_status is not None and gate_state in {"allow", "rejected", "delay_warming"}
            else "error" if gate_state == "fault" else "waiting"
        )
        self._write_log("pipeline", payload)
        # Input/assistance checks run at explicit Run/arm edges. Re-probing
        # all 40 frames for every GUI plot update blocks the inference lock.

    @staticmethod
    def _gate_status_text(status) -> str:
        if status is None:
            return ""
        name = {"ao": "AO24", "phase_ensemble": "七相位 TCN", "none": "不门控"}.get(status.mode, status.mode)
        parts = [name, {"allow": "放行", "rejected": "拒绝", "warming": "预热",
                        "loading": "载入门控模型", "delay_warming": "等待延时样本", "stale": "过期",
                        "fault": "故障"}.get(status.state, status.state)]
        if status.score is not None and status.threshold is not None:
            parts.append(f"分数 {status.score:.4g} / 阈值 {status.threshold:.4g}")
        if status.age_ms is not None:
            parts.append(f"{status.age_ms:.0f} ms")
        return " · 门控：" + "，".join(parts)

    def _load_selected(self) -> None:
        path = self._spec_path_edit.text().strip()
        if not path:
            self._show_error("未选择模型", "请从下拉列表选择已部署模型。")
            return
        self.load_model_from_path(Path(path))

    def load_model_from_path(self, path: Path) -> None:
        path = Path(path)
        if path.parent.name in RETIRED_ONLINE_DEPLOYMENTS:
            path = (self._project_root() / "runtime_data/models/"
                    / "imu_right_pelvis_torque_100hz_rf94_static_20261017/model_spec.json")
        self._cancel_startup("model_changed")
        try:
            self._stop_controller()
        except Exception as exc:
            self._show_error("旧模型尚未停止", exc)
            return
        self._spec = None
        self._uncertainty_panel.clear()
        self._workspace.set_stream_state("uncertainty", "waiting")
        self._model_template_spec = None
        self._runtime = None
        self._model_sha256 = None
        self._applied_signature = None
        self._load_generation = self._loader.submit(path)
        self._show_model_choice(path)
        self._diagnostics.record("model_load_requested", generation=self._load_generation, path=str(path))
        self._model_status.setText("正在后台加载模型…可点击运行，加载完成后自动准备输入")
        self._run_button.setEnabled(True)
        self._spec_path_edit.setText(str(path))
        self._update_gate_compatibility_status()

    @Slot(object)
    def _on_model_loaded(self, result):
        if self._closing or result.generation != self._load_generation:
            self._diagnostics.record("model_result_ignored", generation=result.generation)
            return
        self._load_generation = None
        if result.error is not None:
            self._model_status.setText("模型加载失败；请重试")
            self._run_button.setEnabled(False)
            if self._startup.active:
                self._fail_startup("模型加载失败，请重新选择模型。", detail=result.error)
            else:
                self._show_error("模型加载失败", result.error)
            self._run_button.setEnabled(bool(self._spec_path_edit.text().strip()))
            return
        self._model_sha256 = result.sha256
        if self._install_runtime(result.spec, base_dir=result.path.parent, runtime=result.runtime):
            self._model_template_spec = result.spec
            self._controls.load_spec(result.spec)
            self._spec_path = result.path
            self._spec_path_edit.setText(str(result.path))
            self._applied_signature = None
            self._diagnostics.record("model_ready", generation=result.generation, backend=result.spec.model.backend)
            self._advance_startup()

    def install_demo_spec(self) -> None:
        """Install a deterministic demo model (no weights) for smoke / first run."""
        self._cancel_startup("model_changed")
        self._cancel_model_load("demo_requested")
        self._model_sha256 = None
        spec = ModelSpec.model_validate(
            {
                "schema_version": "1.0.0",
                "model": {"backend": "demo", "demo": {"amplitude": 1.0, "scale": 10.0}},
                "features": {
                    "control_rate_hz": 50.0,
                    "window_s": 0.2,
                    "inputs": [{"modality": "imu", "channels": None}],
                },
                "output": {
                    "kind": "torque_left_right",
                    "single_output_maps_to": "right_torque",
                    "torque_limit_nm": 5.0,
                },
                "safety": {
                    "enable_required": False,
                    "zero_on_error": True,
                    "watchdog_s": 0.5,
                },
            }
        )
        if self._install_runtime(spec, base_dir=None):
            self._model_template_spec = spec
            self._controls.load_spec(spec)
            self._applied_signature = None
        self._spec_path_edit.setText("（内置 Demo 模型）")

    def _install_runtime(self, spec: ModelSpec, *, base_dir: Path | None,
                         runtime: ModelRuntime | None = None) -> bool:
        try:
            self._stop_controller()
        except RuntimeError as exc:
            self._show_error("旧模型尚未停止", exc)
            return False
        self._uncertainty_panel.clear()
        self._workspace.set_stream_state("uncertainty", "waiting")
        try:
            if runtime is None:
                runtime = build_model_runtime(spec, base_dir=base_dir)
        except ModelRuntimeError as exc:
            self._show_error("模型加载失败", exc)
            return False
        use_serial = self._motor is not None
        torque = self._motor if use_serial else MockTorqueOutput(
            spec.output.torque_limit_nm, enable_required=True, zero_on_error=True,
        )
        if use_serial:
            torque.set_limit(self._controls.limit.value())
        controller = InferenceController(self._hub, spec, runtime, torque, parent=self,
            allow_input_wait_recovery=self._motor is None,
            gate_mode=self._gate_mode,
            gate_recording_callback=self._online_recording.gate_event,
            recording_callback=self._online_recording.prediction)
        controller.diagnostic_event.connect(self._on_runtime_diagnostic, Qt.ConnectionType.QueuedConnection)
        for signal, callback in (
            (controller.prediction, self._on_prediction),
            (controller.torque_commanded, self._on_command),
            (controller.fault, self._on_fault),
            (controller.state_changed, self._on_state),
            (controller.pipeline_sample, self._on_pipeline_sample),
        ):
            # Capture the owner explicitly: Qt can report sender() as None
            # after a stopped QObject is deleted while its event is queued.
            signal.connect(lambda payload, owner=controller, handler=callback:
                           self._dispatch_controller(owner, handler, payload),
                           Qt.ConnectionType.QueuedConnection)
        try:
            self._start_log(spec, base_dir)
        except Exception as exc:
            controller.deleteLater()
            self._show_error("无法创建运行日志", exc)
            return False
        self._spec = spec
        self._runtime = runtime
        self._run.prepare(controller, torque)
        self._gate_choice.setEnabled(False)
        self._model_output_panel.set_output_spec(spec.output,
            active_sides=("right",) if spec.imu_tcn.enabled else ("left", "right"),
            motor_distribution=spec.assistance.motor_distribution)
        self._update_status_labels()
        self._model_status.setText(
            f"已加载 · 输入：{', '.join(i.modality for i in spec.features.inputs)} · 点击运行"
        )
        self._fault_label.setText("")
        self._run_button.setEnabled(True)
        self.statusBar().showMessage("模型已加载，等待运行；尚未启动推理。")
        self._update_gate_compatibility_status(spec)
        return True

    def _stop_controller(self) -> None:
        if not self._run.stop(timeout_ms=3000):
            raise RuntimeError(self._run.reason)
        controller = self._run.release_controller()
        self._gate_choice.setEnabled(True)
        if controller is not None:
            for signal in (controller.prediction, controller.torque_commanded,
                           controller.fault, controller.state_changed, controller.pipeline_sample):
                signal.disconnect()
            controller.deleteLater()
        if self._session_log is not None:
            log, self._session_log = self._session_log, None
            try:
                log.close()
            except Exception as exc:
                self._diagnostics.record("session_log_close_failed", exception=exc)
        self._update_log_label()

    # -- explicit Run / Stop safety --------------------------------------

    def _toggle_enable(self, checked: bool) -> None:
        # Legacy callers use the same explicit workflow; Mock never arms.
        if checked:
            self._run_prediction()
        else:
            self.emergency_stop()

    def _cancel_model_load(self, reason):
        if self._load_generation is not None:
            self._diagnostics.record("model_load_cancelled", generation=self._load_generation, reason=reason)
            self._loader.cancel()
            self._load_generation = None
            self._model_status.setText("模型加载已取消")

    def emergency_stop(self, _checked=False, *, reason=None, cancel_recovery=True) -> None:
        if cancel_recovery:
            self._cancel_input_recovery(reason or "stop_requested")
        self._cancel_startup(reason or "stop_requested")
        self._cancel_model_load("stop_requested")
        self._diagnostics.record("stop", reason=reason or "stop_requested", running=self._run.running)
        try:
            self._stop_controller()
        except RuntimeError as exc:
            self._fault_label.setText(str(exc))
            self.statusBar().showMessage(str(exc))
            self._uncertainty_panel.mark_inactive(fault=True)
            self._workspace.set_stream_state("uncertainty", "error")
            return
        self._run_button.setEnabled(self._spec is not None or bool(self._spec_path_edit.text().strip()))
        self._enable_check.blockSignals(True)
        self._enable_check.setChecked(False)
        self._enable_check.blockSignals(False)
        if self._spec is not None:
            self._model_status.setText("已加载 · 等待运行")
        if self._motor is not None and self._motor.last is not None:
            self._model_output_panel.add_command(self._motor.last)
        else:
            self._model_output_panel.clear_command()
        self._update_torque_label()
        self._workspace.set_stream_state("model_output", "error" if reason else "waiting")
        self._uncertainty_panel.mark_inactive(fault=bool(reason))
        self._workspace.set_stream_state("uncertainty", "error" if reason else "waiting")
        if reason:
            message = self._friendly_fault(reason)
            self._fault_label.setText(message)
            self._model_status.setText("预测已停止 · 输入或推理故障")
            self._model_output_panel.set_output_status(message)
            self.statusBar().showMessage(message)
        else:
            self._model_output_panel.set_output_status("预测已停止 · 零力矩并失能")
            self.statusBar().showMessage("已停止预测：零扭矩 + 失能。")

    def _check_watchdog(self) -> None:
        try:
            self._check_watchdog_impl()
        except Exception as exc:
            self._record_throttled_exception("watchdog_failed", exc)

    def _check_watchdog_impl(self) -> None:
        if self._fatal_latched:
            return
        controller, spec = self._controller, self._spec
        if not self._run.running or controller is None or spec is None:
            return
        if controller.last_error:
            reason = controller.last_error
            self._diagnostics.record("input_fault_stop", reason=reason, details=controller.last_fault_details)
            self._write_log("input_fault", controller.last_fault_details or reason)
            self._fatal_incident("推理或输入故障", reason, source="terminal_watchdog")
            return
        if self._motor is None and getattr(controller, "waiting_for_inputs", False):
            message = "正在等待新鲜、完整的模型输入；数据恢复后继续显示预测。"
            self._model_status.setText(message)
            self.statusBar().showMessage(message)
            self._workspace.set_stream_state("model_output", "waiting")
            self._model_output_panel.set_output_status(message)
            return
        if controller.last_predict_ns is not None:
            stale_s = (time.perf_counter_ns() - controller.last_predict_ns) / 1e9
            if stale_s > spec.safety.watchdog_s:
                self._diagnostics.record("watchdog_stop", stale_s=stale_s)
                self._fatal_incident("预测看门狗超时", f"prediction stale: {stale_s:.2f}s",
                                     source="terminal_watchdog")

    # -- inference signals -------------------------------------------------

    @Slot(object)
    def _on_runtime_diagnostic(self, payload):
        fields = dict(payload)
        kind = fields.pop("kind")
        self._diagnostics.record(kind, **fields)

    def _dispatch_controller(self, owner, handler, payload):
        if not self._fatal_latched and owner is self._controller and self._run.running:
            handler(payload)

    def _on_prediction(self, payload: tuple[int, float, float, Any]) -> None:
        if self._fatal_latched or not self._run.running or (self.sender() is not None and self.sender() is not self._controller):
            return
        now_ns, left_nm, right_nm, _raw = payload
        self._workspace.set_stream_state("model_output", "live")
        self._model_output_panel.add_prediction(now_ns, left_nm, right_nm)
        self._status_labels["last_predict"].setText(f"{now_ns}")

    def _on_command(self, command: TorqueCommand) -> None:
        if self._fatal_latched or not self._run.running or (self.sender() is not None and self.sender() is not self._controller):
            return
        if self._motor is None:
            # Mock frames are an internal safety sink.  They must not be
            # presented as hardware sends; the pipeline panel already shows
            # the desired assistance and explicitly labels it as unissued.
            return
        if self._torque is self._motor:
            return  # Serial commands are displayed from the actual send history.
        self._model_output_panel.add_command(command)
        self._update_torque_label(command)
        self._write_log("mock_command", self._command_record(command))

    def _command_record(self, command):
        # Command provenance was copied into the output frame at its send/cache
        # boundary. Looking up the current UI/controller sample here can bind
        # a later decision to an earlier 200 Hz sender frame.
        context = getattr(command, "gate_context", None) or {}
        return {"command": asdict(command) if is_dataclass(command) else command,
                "host_monotonic_ns": int(command.host_monotonic_ns),
                "gate_mode": context.get("mode"),
                "gate_sequence": context.get("sequence"),
                "gate_source_grid_ns": context.get("source_grid_ns"),
                "gate_generation": context.get("generation"),
                "gate_epoch": context.get("epoch"),
                "gate_pre_left_nm": context.get("pre_left_nm"),
                "gate_pre_right_nm": context.get("pre_right_nm"),
                "gate_post_left_nm": context.get("post_left_nm"),
                "gate_post_right_nm": context.get("post_right_nm"),
                "gate_state": context.get("state")}

    def _on_fault(self, message: str) -> None:
        if (self.sender() is not None and self.sender() is not self._controller):
            return
        self._diagnostics.record("inference_fault", detail=message)
        self._fatal_incident("推理故障", message, source="inference_fault")

    def _on_state(self, state: str) -> None:
        if self._fatal_latched or not self._run.running or (self.sender() is not None and self.sender() is not self._controller):
            return
        names = {"running": "推理运行中", "stopped": "已停止 · 等待运行", "fault": "推理故障"}
        self._model_status.setText(f"状态：{names.get(state, state)}")

    # -- UI helpers --------------------------------------------------------

    def _update_status_labels(self) -> None:
        spec = self._spec
        if spec is None:
            return
        self._status_labels["backend"].setText(spec.model.backend)
        self._status_labels["control_rate"].setText(f"{spec.features.control_rate_hz:.0f} Hz")
        self._status_labels["window"].setText(
            f"{spec.features.window_s:.2f} s × {spec.feature_window_samples} 点"
        )
        self._status_labels["limit"].setText(f"±{spec.output.torque_limit_nm:.1f} Nm")

    def _update_torque_label(self, command: TorqueCommand | None = None) -> None:
        if self._motor is None:
            self._torque_label.setText("估计/期望辅助仅显示 · 未连接电机 · 未下发")
            if hasattr(self, "_persistent_command_label"):
                self._persistent_command_label.setText(self._torque_label.text())
            return
        if command is None:
            command = self._torque.last if self._torque is not None else None
        if command is None:
            self._torque_label.setText("当前指令：左 —  Nm ｜ 右 —  Nm")
        else:
            self._torque_label.setText(
                f"当前指令：左 {command.left_nm:+.2f} Nm ｜ 右 {command.right_nm:+.2f} Nm"
                f"  [{'已使能' if command.enabled else '失能'}]"
            )
        if hasattr(self, "_persistent_command_label"):
            self._persistent_command_label.setText(self._torque_label.text())

    @staticmethod
    def _friendly_fault(message):
        text = str(message)
        if "253" in text or "can" in text.lower() and "feedback" in text.lower():
            return "控制板报告电机 CAN 反馈中断（故障码 253），已归零、失能并断开。现有协议不能确定是哪侧电机或具体原因，请检查电机供电和 CAN 连接。"
        if "254" in text:
            return "底层控制器报告 PC 指令超时，已归零、失能并断开。"
        if "source window expired" in text.lower():
            return "预测所用窗口已过期，已归零并停止；输入恢复后点击“运行预测”，仅重连电机不会重启模型。"
        if "stale" in text.lower():
            return "输入数据不够新鲜，已归零并停止；请检查传感器后点击“运行预测”。"
        if "discontinu" in text.lower() or "gap" in text.lower():
            return "输入数据不连续，预测已停止；请检查传感器后重新运行。"
        return "预测已停止；请展开详情与日志检查原因。"

    def _show_error(self, title: str, detail: object) -> None:
        self._diagnostics.record("ui_error", title=title, detail=str(detail),
                                 exception=detail if isinstance(detail, Exception) else None)
        self.statusBar().showMessage(f"{title}：{detail}")
        if not self._smoke_test:
            QMessageBox.critical(self, title, str(detail))

    # -- public smoke / introspection API ----------------------------------

    @property
    def _controller(self):
        return self._run.controller

    @property
    def _torque(self):
        return self._run.output

    @property
    def _motor(self):
        return self._run.motor

    @_motor.setter
    def _motor(self, motor):
        self._run.motor = motor

    @property
    def torque_history(self) -> tuple[TorqueCommand, ...]:
        return self._torque.history if self._torque is not None else ()

    @property
    def controller_running(self) -> bool:
        return self._controller is not None and self._controller.isRunning()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        self._diagnostics.record("window_close_requested")
        self._cancel_input_recovery("window_close")
        self._cancel_startup("window_close")
        self._cancel_model_load("window_close")
        try:
            self._stop_controller()
        except RuntimeError as exc:
            self._show_error("无法关闭", exc)
            event.ignore()
            return
        self.disconnect_all()
        if self._pending_motor_cleanup is not None:
            self._show_error("无法关闭", "电机串口释放失败，窗口保留清理对象；请检查USB后重试关闭。")
            event.ignore()
            return
        self._closing = True
        self._online_recording.close()
        self._loader.close()
        self._preview_timer.stop()
        self._watchdog_timer.stop()
        self._diagnostics.record("window_closed")
        self._diagnostics.close()
        super().closeEvent(event)


__all__ = ["ModelRuntimeWindow"]
