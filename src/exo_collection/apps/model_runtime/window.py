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
``MockTorqueOutput``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
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
from exo_collection.apps.collector.status_overview import ModalityStatusStrip
from exo_collection.apps.collector.theme import COLLECTOR_STYLESHEET
from exo_collection.apps.model_runtime.backends import (
    ModelRuntime,
    ModelRuntimeError,
    build_model_runtime,
)
from exo_collection.apps.model_runtime.inference import InferenceController
from exo_collection.apps.model_runtime.preview_panels import (
    LivePreviewPanel,
    ModelOutputPanel,
)
from exo_collection.apps.model_runtime.spec import ModelSpec, load_model_spec
from exo_collection.apps.model_runtime.subscription import SubscriptionHub
from exo_collection.apps.model_runtime.torque import MockTorqueOutput, TorqueCommand
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


def _block_state(state: str) -> str:
    normalized = (state or "").strip().upper()
    if normalized in {"READY", "RECORDING", "CONNECTED", "RUNNING"}:
        return "ok"
    if normalized in {"FAULT", "FAILED", "ERROR"}:
        return "bad"
    if normalized in {"CONNECTING", "PREVIEW_STARTING", "STOPPING"}:
        return "transition"
    return "off"


class ModelRuntimeWindow(QMainWindow):
    """Top-level window for local real-time model testing."""

    def __init__(self, settings: SharedAppSettings, *, smoke_test: bool = False) -> None:
        super().__init__()
        self.setObjectName("model_runtime_window")
        self.setWindowTitle("在线测试 · 本地模型实时推理")
        self.setStyleSheet(COLLECTOR_STYLESHEET)

        self._settings = settings
        self._smoke_test = bool(smoke_test)
        self._profile_key = settings.device_profile_key
        self._modalities = self._load_modalities()

        self._handles: dict[str, ModalityPreviewProcessHandle] = {}
        self._registered: set[str] = set()
        self._panels: dict[str, LivePreviewPanel] = {}
        self._connect_buttons: dict[str, QPushButton] = {}
        self._connect_dots: dict[str, QLabel] = {}

        self._hub = SubscriptionHub()
        self._spec: ModelSpec | None = None
        self._runtime: ModelRuntime | None = None
        self._torque: MockTorqueOutput | None = None
        self._controller: InferenceController | None = None
        self._spec_path: Path | None = None
        self._trial_uuid = str(uuid.uuid4())
        self._enabled_wanted = False

        self._build_ui()

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(_PREVIEW_POLL_MS)
        self._preview_timer.timeout.connect(self._poll_previews)
        self._preview_timer.start()

        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.setInterval(_WATCHDOG_POLL_MS)
        self._watchdog_timer.timeout.connect(self._check_watchdog)
        self._watchdog_timer.start()

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
        body.setChildrenCollapsible(False)
        body.addWidget(self._build_control_column())
        body.addWidget(self._build_preview_column())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([430, 1290])
        self.setCentralWidget(body)

    def _build_control_column(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(420)
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_connection_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_torque_group())
        layout.addWidget(self._build_status_group())
        layout.addStretch(1)

        scroll.setWidget(column)
        return scroll

    def _build_connection_group(self) -> QGroupBox:
        group = QGroupBox("设备连接")
        layout = QVBoxLayout(group)

        buttons = QHBoxLayout()
        connect_all = QPushButton("全部连接")
        connect_all.setProperty("buttonRole", "primary")
        connect_all.clicked.connect(self.connect_all)
        disconnect_all = QPushButton("全部断开")
        disconnect_all.setProperty("buttonRole", "disconnect")
        disconnect_all.clicked.connect(self.disconnect_all)
        buttons.addWidget(connect_all)
        buttons.addWidget(disconnect_all)
        layout.addLayout(buttons)

        for modality in self._modalities:
            layout.addLayout(self._build_connect_row(modality))
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
        group = QGroupBox("模型配置")
        layout = QVBoxLayout(group)

        path_row = QHBoxLayout()
        self._spec_path_edit = QLineEdit()
        self._spec_path_edit.setReadOnly(True)
        self._spec_path_edit.setPlaceholderText("选择一个模型 spec JSON（未加载）")
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._choose_spec)
        path_row.addWidget(self._spec_path_edit, 1)
        path_row.addWidget(browse)
        layout.addLayout(path_row)

        load_row = QHBoxLayout()
        load_btn = QPushButton("加载 / 重载模型")
        load_btn.setProperty("buttonRole", "primary")
        load_btn.clicked.connect(self._load_selected)
        self._model_status = QLabel("未加载")
        self._model_status.setWordWrap(True)
        load_row.addWidget(load_btn)
        load_row.addWidget(self._model_status, 1)
        layout.addLayout(load_row)

        self._load_button = load_btn
        return group

    def _build_torque_group(self) -> QGroupBox:
        group = QGroupBox("扭矩输出")
        layout = QVBoxLayout(group)

        self._enable_check = QCheckBox("使能扭矩输出（下发非零扭矩）")
        self._enable_check.setChecked(False)
        self._enable_check.toggled.connect(self._toggle_enable)
        layout.addWidget(self._enable_check)

        self._torque_label = QLabel("当前指令：左 —  Nm ｜ 右 —  Nm")
        layout.addWidget(self._torque_label)

        estop = QPushButton("急停（零扭矩 + 失能）")
        estop.setProperty("buttonRole", "danger")
        estop.clicked.connect(self.emergency_stop)
        layout.addWidget(estop)

        hint = QLabel("当前为 Mock 输出：只记录帧，不打开串口。")
        hint.setStyleSheet("color:#5b6470;")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("运行状态")
        layout = QVBoxLayout(group)
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
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._status_labels[key] = value
            row.addWidget(value, 1)
            layout.addLayout(row)
        self._fault_label = QLabel()
        self._fault_label.setStyleSheet("color:#a53f3f;")
        self._fault_label.setWordWrap(True)
        layout.addWidget(self._fault_label)
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
        for modality in self._modalities:
            title = MODALITY_DISPLAY_NAMES.get(modality, modality)
            panel = LivePreviewPanel(modality, title)
            self._panels[modality] = panel
            self._workspace.register_panel(modality, title, panel)
        self._model_output_panel = ModelOutputPanel("模型输出")
        self._workspace.register_panel("model_output", "模型输出", self._model_output_panel)
        self._workspace.reset_default_layout()
        layout.addWidget(self._workspace, 1)
        return column

    # -- device connection -------------------------------------------------

    def _toggle_connect(self, modality: str) -> None:
        if modality in self._handles:
            self._disconnect_modality(modality)
        else:
            self._connect_modality(modality)

    def connect_all(self) -> None:
        for modality in self._modalities:
            self._connect_modality(modality)

    def disconnect_all(self) -> None:
        for modality in list(self._handles):
            self._disconnect_modality(modality)

    def _connect_modality(self, modality: str) -> None:
        if modality in self._handles:
            return
        try:
            profile = load_device_profile(self._profile_key)
            device = profile.by_modality()[modality]
        except Exception as exc:
            self._show_error(f"无法加载 {modality} 设备配置", exc)
            return
        simulated = self._profile_key == "simulated" or bool(
            getattr(device, "simulated", False)
        )
        overrides = (
            self._settings.hardware_device_overrides
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
        handle.start()
        self._handles[modality] = handle
        self._set_connect_state(modality, connected=True)
        self._status_strip.set_state(modality, "transition", "连接中")
        self._workspace.set_stream_state(modality, "connected")
        self._workspace.show_panel(modality)

    def _disconnect_modality(self, modality: str) -> None:
        handle = self._handles.pop(modality, None)
        self._registered.discard(modality)
        self._hub.unregister(modality)
        if handle is None:
            return
        try:
            if handle.recording_active:
                handle.end_recording(self._trial_uuid)
        except Exception:
            pass
        if handle.is_alive:
            handle.request_stop()
            handle.join(timeout=2.0)
        if handle.is_alive:
            handle.terminate(timeout=2.0)
        handle.close()
        self._set_connect_state(modality, connected=False)
        self._status_strip.set_state(modality, "off", "未连接")
        self._workspace.set_stream_state(modality, "disconnected")

    def _set_connect_state(self, modality: str, *, connected: bool) -> None:
        dot = self._connect_dots.get(modality)
        button = self._connect_buttons.get(modality)
        if dot is not None:
            dot.setStyleSheet(
                f"background:{'#0f766e' if connected else '#d3d0c7'}; border-radius:6px;"
            )
        if button is not None:
            button.setText("断开" if connected else "连接")
            button.setProperty("buttonRole", "disconnect" if connected else "connect")
            button.style().unpolish(button)
            button.style().polish(button)

    # -- preview polling ---------------------------------------------------

    def _poll_previews(self) -> None:
        for modality, handle in list(self._handles.items()):
            try:
                events = handle.poll_events(limit=200)
            except Exception:
                continue
            for event in events:
                self._handle_worker_event(modality, event)
            if handle.recording_endpoint is not None and modality not in self._registered:
                self._hub.register_endpoint(modality, handle.recording_endpoint)
                try:
                    handle.begin_recording(self._trial_uuid)
                except Exception:
                    continue
                self._registered.add(modality)
                self._status_strip.set_state(modality, "ok", "实时")
                self._workspace.set_stream_state(modality, "live")

    def _handle_worker_event(self, modality: str, event: WorkerEvent) -> None:
        if event.event_type is WorkerEventType.STATE:
            state = str(event.payload.get("state") or "")
            self._status_strip.set_state(modality, _block_state(state), state)
            if state == "READY":
                self._workspace.set_stream_state(modality, "connected")
            elif state in {"DISCONNECTED", "CLOSED"}:
                self._set_connect_state(modality, connected=False)
                self._workspace.set_stream_state(modality, "disconnected")
        elif event.event_type is WorkerEventType.PREVIEW:
            panel = self._panels.get(modality)
            if panel is not None:
                panel.feed(event.payload)
        elif event.event_type is WorkerEventType.FAILED:
            message = event.message or str(event.payload.get("fault") or "")
            self._status_strip.set_state(modality, "bad", "故障")
            self._workspace.set_stream_state(modality, "error")
            self._show_error(f"{modality} 预览 worker 故障", message)

    # -- model load / reload ----------------------------------------------

    def _choose_spec(self) -> None:
        start_dir = str(self._spec_path.parent if self._spec_path else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型 spec JSON",
            start_dir,
            "JSON 文件 (*.json);;所有文件 (*.*)",
        )
        if path:
            self._spec_path_edit.setText(path)
            self._load_selected()

    def _load_selected(self) -> None:
        path = self._spec_path_edit.text().strip()
        if not path:
            self._show_error("未选择模型 spec", "请先浏览并选择一个 JSON 文件。")
            return
        self.load_model_from_path(Path(path))

    def load_model_from_path(self, path: Path) -> None:
        try:
            spec = load_model_spec(path)
        except Exception as exc:
            self._show_error("模型 spec 解析失败", exc)
            return
        self._install_runtime(spec, base_dir=path.parent)
        self._spec_path = path
        self._spec_path_edit.setText(str(path))

    def install_demo_spec(self) -> None:
        """Install a deterministic demo model (no weights) for smoke / first run."""
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
        self._install_runtime(spec, base_dir=None)
        self._spec_path_edit.setText("（内置 Demo 模型）")

    def _install_runtime(self, spec: ModelSpec, *, base_dir: Path | None) -> None:
        self._stop_controller()
        try:
            runtime = build_model_runtime(spec, base_dir=base_dir)
        except ModelRuntimeError as exc:
            self._show_error("模型加载失败", exc)
            return
        torque = MockTorqueOutput(
            spec.output.torque_limit_nm,
            enable_required=spec.safety.enable_required,
            zero_on_error=spec.safety.zero_on_error,
        )
        controller = InferenceController(self._hub, spec, runtime, torque, parent=self)
        controller.prediction.connect(self._on_prediction)
        controller.torque_commanded.connect(self._on_command)
        controller.fault.connect(self._on_fault)
        controller.state_changed.connect(self._on_state)
        controller.set_enabled(self._enabled_wanted)
        controller.start()

        self._spec = spec
        self._runtime = runtime
        self._torque = torque
        self._controller = controller
        self._update_status_labels()
        self._model_status.setText(
            f"已加载 · 后端={spec.model.backend} · 输入={[i.modality for i in spec.features.inputs]}"
        )
        self._fault_label.setText("")

    def _stop_controller(self) -> None:
        controller = self._controller
        self._controller = None
        if controller is not None:
            controller.stop_and_wait(timeout_ms=3000)
            controller.deleteLater()

    # -- torque arming / safety -------------------------------------------

    def _toggle_enable(self, checked: bool) -> None:
        self._enabled_wanted = bool(checked)
        if self._controller is not None:
            self._controller.set_enabled(self._enabled_wanted)
        if not self._enabled_wanted and self._torque is not None:
            self._torque.zero()
        self._update_torque_label()

    def emergency_stop(self) -> None:
        if self._torque is not None:
            self._torque.set_enabled(False)
            self._torque.zero()
        self._enabled_wanted = False
        if self._controller is not None:
            self._controller.set_enabled(False)
        self._enable_check.blockSignals(True)
        self._enable_check.setChecked(False)
        self._enable_check.blockSignals(False)
        self._update_torque_label()
        self.statusBar().showMessage("已急停：零扭矩 + 失能。")

    def _check_watchdog(self) -> None:
        controller = self._controller
        spec = self._spec
        if (
            controller is None
            or spec is None
            or not controller.enabled
            or controller.last_predict_ns is None
        ):
            return
        import time

        stale_s = (time.perf_counter_ns() - controller.last_predict_ns) / 1e9
        if stale_s > spec.safety.watchdog_s:
            self._fault_label.setText(f"看门狗触发：{stale_s:.2f}s 无预测，已零扭矩。")
            self.emergency_stop()

    # -- inference signals -------------------------------------------------

    def _on_prediction(self, payload: tuple[int, float, float, Any]) -> None:
        now_ns, left_nm, right_nm, _raw = payload
        self._model_output_panel.add_prediction(now_ns, left_nm, right_nm)
        self._status_labels["last_predict"].setText(f"{now_ns}")

    def _on_command(self, command: TorqueCommand) -> None:
        self._model_output_panel.add_command(command)
        self._update_torque_label(command)

    def _on_fault(self, message: str) -> None:
        self._fault_label.setText(f"推理故障：{message}")
        self.statusBar().showMessage(f"推理故障：{message}")
        if self._torque is not None and self._spec is not None and self._spec.safety.zero_on_error:
            self._enable_check.blockSignals(True)
            self._enable_check.setChecked(False)
            self._enable_check.blockSignals(False)
            self._enabled_wanted = False

    def _on_state(self, state: str) -> None:
        self._model_status.setText(f"状态：{state}")

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
        if command is None:
            command = self._torque.last if self._torque is not None else None
        if command is None:
            self._torque_label.setText("当前指令：左 —  Nm ｜ 右 —  Nm")
        else:
            self._torque_label.setText(
                f"当前指令：左 {command.left_nm:+.2f} Nm ｜ 右 {command.right_nm:+.2f} Nm"
                f"  [{'已使能' if command.enabled else '失能'}]"
            )

    def _show_error(self, title: str, detail: object) -> None:
        self.statusBar().showMessage(f"{title}：{detail}")
        if not self._smoke_test:
            QMessageBox.critical(self, title, str(detail))

    # -- public smoke / introspection API ----------------------------------

    @property
    def torque_history(self) -> tuple[TorqueCommand, ...]:
        return self._torque.history if self._torque is not None else ()

    @property
    def controller_running(self) -> bool:
        return self._controller is not None and self._controller.isRunning()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        self._stop_controller()
        self.disconnect_all()
        super().closeEvent(event)


__all__ = ["ModelRuntimeWindow"]
