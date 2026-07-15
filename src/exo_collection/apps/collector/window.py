"""Responsive PySide6 shell for the Collector worker process."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from exo_collection.acquisition.messages import WorkerEvent, WorkerEventType
from exo_collection.acquisition.workers import CollectorWorker
from exo_collection.configuration import SharedAppSettings
from exo_collection.orchestration.models import TrialRunRequest
from exo_collection.protocols import load_default_protocol


MODALITIES = ("ultrasound", "imu", "encoder", "sync_pulse")
MAX_PREVIEW_POINTS = 4096
MAX_SIGNAL_HISTORY_POINTS = 3000

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

    def close(self) -> None: ...


WorkerFactory = Callable[[TrialRunRequest], WorkerHandle]


class CollectorWindow(QMainWindow):
    """Collect one Trial at a time through a non-blocking worker boundary."""

    trial_started = Signal(object)
    trial_finished = Signal(bool)

    def __init__(
        self,
        data_root: str | Path,
        *,
        settings: SharedAppSettings | None = None,
        worker_factory: WorkerFactory = CollectorWorker,
        poll_interval_ms: int = 50,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if poll_interval_ms <= 0:
            raise ValueError("poll_interval_ms must be positive")
        self._settings = settings if settings is not None else SharedAppSettings()
        self._worker_factory = worker_factory
        self._worker: WorkerHandle | None = None
        self._terminal_event_received = False
        self._dead_poll_count = 0
        self._stop_requested = False
        self._close_when_finished = False
        self._configuration_locked = False

        self._project_key: tuple[str, str] | None = None
        self._subject_key: tuple[UUID, str] | None = None
        self._session_key: tuple[UUID, str] | None = None
        self._project_uuid = uuid4()
        self._subject_uuid = uuid4()
        self._session_uuid = uuid4()

        self._health_rows = {name: index for index, name in enumerate(MODALITIES)}
        self._last_health_status: dict[str, str] = {}
        self._signal_history: dict[str, tuple[list[float], list[float]]] = {
            "imu": ([], []),
            "encoder": ([], []),
        }

        self.setWindowTitle("Exo Collector")
        self.resize(1280, 820)
        self._create_ui(Path(data_root).expanduser().resolve())

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(poll_interval_ms)
        self._poll_timer.timeout.connect(self.poll_worker_events)
        self._set_trial_state("IDLE")

    @property
    def worker(self) -> WorkerHandle | None:
        return self._worker

    @property
    def configuration_locked(self) -> bool:
        return self._configuration_locked

    def _create_ui(self, data_root: Path) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)

        header = QHBoxLayout()
        title = QLabel("Exo Collector · 多模态数据采集")
        title.setStyleSheet("font-size: 19px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch(1)
        self.state_label = QLabel()
        self.state_label.setObjectName("trial_state")
        self.state_label.setMinimumWidth(170)
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.state_label)
        outer.addLayout(header)

        body = QSplitter(Qt.Orientation.Horizontal)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        metadata_box = QGroupBox("Trial 设置")
        form = QFormLayout(metadata_box)
        root_row = QHBoxLayout()
        self.data_root_edit = QLineEdit(str(data_root))
        self.data_root_edit.setObjectName("data_root")
        root_row.addWidget(self.data_root_edit, 1)
        self.browse_button = QPushButton("选择…")
        self.browse_button.clicked.connect(self.choose_data_root)
        root_row.addWidget(self.browse_button)
        form.addRow("数据根目录：", root_row)

        self.project_name_edit = QLineEdit("Exoskeleton Study")
        self.project_name_edit.setObjectName("project_name")
        form.addRow("项目名：", self.project_name_edit)
        self.subject_code_edit = QLineEdit("SIM-001")
        self.subject_code_edit.setObjectName("subject_code")
        form.addRow("受试者编码：", self.subject_code_edit)
        self.operator_edit = QLineEdit("operator")
        self.operator_edit.setObjectName("operator")
        form.addRow("操作者：", self.operator_edit)

        self.condition_combo = QComboBox()
        self.condition_combo.setObjectName("condition")
        for condition in CONDITIONS:
            self.condition_combo.addItem(
                f"{condition['condition_code']} — {condition['condition_name']}",
                dict(condition),
            )
        self.condition_combo.setCurrentIndex(1)
        form.addRow("工况：", self.condition_combo)

        self.repeat_spin = QSpinBox()
        self.repeat_spin.setObjectName("repeat_index")
        self.repeat_spin.setRange(1, 9999)
        self.repeat_spin.setValue(1)
        form.addRow("重复轮次：", self.repeat_spin)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setObjectName("duration_s")
        self.duration_spin.setRange(0.1, 86_400.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setValue(3.0)
        form.addRow("采集时长：", self.duration_spin)
        controls_layout.addWidget(metadata_box)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("开始 Trial")
        self.start_button.setObjectName("start_trial")
        self.start_button.setStyleSheet(
            "QPushButton { font-weight: 600; padding: 8px; }"
        )
        self.start_button.clicked.connect(self.start_trial)
        buttons.addWidget(self.start_button)
        self.stop_button = QPushButton("受控停止")
        self.stop_button.setObjectName("stop_trial")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.request_controlled_stop)
        buttons.addWidget(self.stop_button)
        controls_layout.addLayout(buttons)

        health_box = QGroupBox("设备健康与样本计数")
        health_layout = QVBoxLayout(health_box)
        self.health_table = QTableWidget(len(MODALITIES), 5)
        self.health_table.setObjectName("health_table")
        self.health_table.setHorizontalHeaderLabels(
            ["模态", "健康", "样本/帧", "实际速率", "队列"]
        )
        self.health_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.health_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.health_table.verticalHeader().setVisible(False)
        for row, modality in enumerate(MODALITIES):
            self.health_table.setItem(row, 0, QTableWidgetItem(modality))
            self.health_table.setItem(row, 1, QTableWidgetItem("UNKNOWN"))
            self.health_table.setItem(row, 2, QTableWidgetItem("0"))
            self.health_table.setItem(row, 3, QTableWidgetItem("-"))
            self.health_table.setItem(row, 4, QTableWidgetItem("-"))
        self.health_table.resizeColumnsToContents()
        health_layout.addWidget(self.health_table)
        controls_layout.addWidget(health_box)

        alert_box = QGroupBox("告警与采集消息")
        alert_layout = QVBoxLayout(alert_box)
        self.alerts_edit = QPlainTextEdit()
        self.alerts_edit.setObjectName("alerts")
        self.alerts_edit.setReadOnly(True)
        self.alerts_edit.setMaximumBlockCount(300)
        self.alerts_edit.setPlaceholderText("当前无告警。")
        alert_layout.addWidget(self.alerts_edit)
        controls_layout.addWidget(alert_box, 1)

        self.manifest_label = QLabel("Manifest：尚未生成")
        self.manifest_label.setObjectName("manifest_path")
        self.manifest_label.setWordWrap(True)
        self.manifest_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        controls_layout.addWidget(self.manifest_label)
        body.addWidget(controls)

        preview_box = QGroupBox("实时预览（仅消费降采样 WorkerEvent）")
        preview_layout = QVBoxLayout(preview_box)
        pg.setConfigOptions(antialias=False)
        self.ultrasound_plot = pg.PlotWidget(title="Ultrasound 降采样帧/波形")
        self.ultrasound_plot.setObjectName("ultrasound_preview")
        self.ultrasound_plot.setBackground("w")
        self.ultrasound_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ultrasound_curve = self.ultrasound_plot.plot(
            pen=pg.mkPen("#2457c5", width=1.3)
        )
        preview_layout.addWidget(self.ultrasound_plot, 1)

        self.imu_plot = pg.PlotWidget(title="IMU 实时曲线")
        self.imu_plot.setObjectName("imu_preview")
        self.imu_plot.setBackground("w")
        self.imu_plot.setLabel("bottom", "Trial time", units="s")
        self.imu_plot.showGrid(x=True, y=True, alpha=0.2)
        self.imu_curve = self.imu_plot.plot(pen=pg.mkPen("#1a936f", width=1.4))
        preview_layout.addWidget(self.imu_plot, 1)

        self.encoder_plot = pg.PlotWidget(title="Encoder 实时曲线")
        self.encoder_plot.setObjectName("encoder_preview")
        self.encoder_plot.setBackground("w")
        self.encoder_plot.setLabel("bottom", "Trial time", units="s")
        self.encoder_plot.showGrid(x=True, y=True, alpha=0.2)
        self.encoder_curve = self.encoder_plot.plot(
            pen=pg.mkPen("#d97706", width=1.4)
        )
        preview_layout.addWidget(self.encoder_plot, 1)
        body.addWidget(preview_box)
        body.setSizes([470, 790])
        outer.addWidget(body, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪；原始写盘仅由 Collector Worker 负责。")

        self._configuration_widgets = (
            self.data_root_edit,
            self.browse_button,
            self.project_name_edit,
            self.subject_code_edit,
            self.operator_edit,
            self.condition_combo,
            self.repeat_spin,
            self.duration_spin,
        )

    @Slot()
    def choose_data_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择外骨骼数据根目录",
            self.data_root_edit.text(),
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.set_data_root(selected)

    def set_data_root(self, data_root: str | Path) -> Path:
        normalized = self._settings.set_data_root(data_root)
        self.data_root_edit.setText(str(normalized))
        return normalized

    def _refresh_identity_context(
        self, data_root: Path, project_name: str, subject_code: str, operator: str
    ) -> None:
        project_key = (str(data_root), project_name)
        if project_key != self._project_key:
            self._project_key = project_key
            self._project_uuid = uuid4()
            self._subject_key = None
            self._session_key = None

        subject_key = (self._project_uuid, subject_code)
        if subject_key != self._subject_key:
            self._subject_key = subject_key
            self._subject_uuid = uuid4()
            self._session_key = None

        session_key = (self._subject_uuid, operator)
        if session_key != self._session_key:
            self._session_key = session_key
            self._session_uuid = uuid4()

    def build_request(self) -> TrialRunRequest:
        data_root_text = self.data_root_edit.text().strip()
        if not data_root_text:
            raise ValueError("数据根目录不能为空")
        data_root = self.set_data_root(data_root_text)
        project_name = self.project_name_edit.text().strip()
        subject_code = self.subject_code_edit.text().strip()
        operator = self.operator_edit.text().strip()
        condition = self.condition_combo.currentData()
        if not isinstance(condition, dict):
            raise ValueError("请选择有效工况")

        self._refresh_identity_context(
            data_root, project_name, subject_code, operator
        )
        return TrialRunRequest(
            data_root=data_root,
            duration_s=self.duration_spin.value(),
            project_uuid=self._project_uuid,
            subject_uuid=self._subject_uuid,
            session_uuid=self._session_uuid,
            project_name=project_name,
            subject_code=subject_code,
            operator=operator,
            condition_code=str(condition["condition_code"]),
            condition_name=str(condition["condition_name"]),
            condition_level=condition.get("condition_level"),
            condition_parameters=dict(condition.get("parameters", {})),
            repeat_index=self.repeat_spin.value(),
            protocol_version=_PROTOCOL.protocol_version,
            config_version="1.0.0",
        )

    @Slot()
    def start_trial(self) -> None:
        if self._worker is not None:
            return
        worker: WorkerHandle | None = None
        try:
            request = self.build_request()
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
            return

        self._worker = worker
        self._terminal_event_received = False
        self._dead_poll_count = 0
        self._stop_requested = False
        self._close_when_finished = False
        self._reset_trial_display()
        self._set_configuration_locked(True)
        self.stop_button.setEnabled(True)
        self._set_trial_state("PREPARING")
        self._poll_timer.start()
        self.trial_started.emit(request)
        self.statusBar().showMessage(
            f"Trial {request.trial_uuid} 已交给独立 Collector Worker。"
        )

    def _reset_trial_display(self) -> None:
        self.alerts_edit.clear()
        self.manifest_label.setText("Manifest：正在采集，尚未最终化")
        self._last_health_status.clear()
        for modality, row in self._health_rows.items():
            self.health_table.item(row, 1).setText("UNKNOWN")
            self.health_table.item(row, 2).setText("0")
            self.health_table.item(row, 3).setText("-")
            self.health_table.item(row, 4).setText("-")
        self.ultrasound_curve.setData([], [])
        for modality, curve in (
            ("imu", self.imu_curve),
            ("encoder", self.encoder_curve),
        ):
            self._signal_history[modality] = ([], [])
            curve.setData([], [])

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
        self.stop_button.setEnabled(False)
        self._set_trial_state("STOPPING")
        self._append_alert("已发送受控停止请求；正在等待 Writer flush 与 Trial 最终化。")

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
                    f"已忽略无效 {event.event_type.value} 事件："
                    f"{type(exc).__name__}: {exc}"
                )

        if self._worker_is_alive(worker):
            self._dead_poll_count = 0
            return

        self._dead_poll_count += 1
        # A multiprocessing.Queue feeder can lag process exit very briefly.
        # Give it two GUI ticks, then drain once more before judging the exit.
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
                "Collector Worker 在未发布 COMPLETED/FAILED 事件时退出"
                f"（exit code {exitcode}）。"
            )
        self._release_worker(worker)

    @staticmethod
    def _worker_is_alive(worker: WorkerHandle) -> bool:
        value = worker.is_alive
        return bool(value() if callable(value) else value)

    @staticmethod
    def _worker_exitcode(worker: WorkerHandle) -> int | None:
        value = worker.exitcode
        return value() if callable(value) else value

    def _handle_worker_event(self, event: WorkerEvent) -> None:
        if event.event_type is WorkerEventType.STATE:
            state = str(event.payload.get("state") or event.message or "UNKNOWN")
            self._set_trial_state(state)
        elif event.event_type is WorkerEventType.HEALTH:
            self._handle_health(event)
        elif event.event_type is WorkerEventType.METRIC:
            self._handle_metric(event.payload)
        elif event.event_type is WorkerEventType.ALERT:
            self._append_alert(event.message or "Collector Worker 报告需要关注的事件。")
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
        self.health_table.item(row, 3).setText(
            "-" if rate is None else f"{float(rate):.1f} Hz"
        )
        depth = payload.get("queue_depth")
        capacity = payload.get("queue_capacity")
        if depth is None:
            queue_text = "-"
        elif capacity is None:
            queue_text = str(int(depth))
        else:
            queue_text = f"{int(depth)}/{int(capacity)}"
        self.health_table.item(row, 4).setText(queue_text)

        previous = self._last_health_status.get(modality)
        self._last_health_status[modality] = status
        if status in {"DEGRADED", "UNHEALTHY", "FAULT"} and status != previous:
            detail = event.message or str(payload.get("message") or "")
            suffix = f"：{detail}" if detail else ""
            self._append_alert(f"{modality} 健康状态 {status}{suffix}")

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
            self.health_table.item(row, 2).setToolTip(
                f"已检测边沿：{int(payload['pulse_event_count'])}"
            )

    def _handle_preview(self, event: WorkerEvent) -> None:
        modality = self._normalize_modality(event.modality or "")
        if modality == "ultrasound":
            values = self._numeric_values(event.payload.get("values"))
            if values:
                self.ultrasound_curve.setData(list(range(len(values))), values)
            return
        if modality not in {"imu", "encoder"}:
            return
        values_raw = event.payload.get("values")
        x_raw = event.payload.get("x")
        if not isinstance(values_raw, (list, tuple)):
            return
        if not isinstance(x_raw, (list, tuple)) or len(x_raw) != len(values_raw):
            x_raw = list(range(len(values_raw)))
        pairs: list[tuple[float, float]] = []
        for x_value, y_value in zip(x_raw, values_raw, strict=False):
            try:
                x_number = float(x_value)
                y_number = float(y_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(x_number) and math.isfinite(y_number):
                pairs.append((x_number, y_number))
            if len(pairs) >= MAX_PREVIEW_POINTS:
                break
        if not pairs:
            return
        history_x, history_y = self._signal_history[modality]
        history_x.extend(pair[0] for pair in pairs)
        history_y.extend(pair[1] for pair in pairs)
        if len(history_x) > MAX_SIGNAL_HISTORY_POINTS:
            del history_x[:-MAX_SIGNAL_HISTORY_POINTS]
            del history_y[:-MAX_SIGNAL_HISTORY_POINTS]
        curve = self.imu_curve if modality == "imu" else self.encoder_curve
        curve.setData(history_x, history_y)
        channel = event.payload.get("channel")
        if channel:
            plot = self.imu_plot if modality == "imu" else self.encoder_plot
            plot.setTitle(f"{modality.upper()} 实时曲线 · {channel}")

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
        state = str(event.payload.get("state") or "FINALIZED")
        self._set_trial_state(state)
        manifest_path = event.payload.get("manifest_path")
        if manifest_path:
            self.manifest_label.setText(f"Manifest：{manifest_path}")
        else:
            self.manifest_label.setText("Manifest：Worker 已完成，但未返回路径")
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage(event.message or "Trial 数据包已最终化。")

    def _mark_failed(self, message: str) -> None:
        self._set_trial_state("FAILED")
        self.stop_button.setEnabled(False)
        self._append_alert(f"FAILED：{message}")
        self.statusBar().showMessage("Trial 失败；请检查告警信息。")

    def _release_worker(self, worker: WorkerHandle) -> None:
        try:
            worker.join(timeout=0)
            worker.close()
        except Exception as exc:
            self._append_alert(f"释放 Worker 资源时出错：{type(exc).__name__}: {exc}")
        self._worker = None
        self._poll_timer.stop()
        self._set_configuration_locked(False)
        self.stop_button.setEnabled(False)
        self.trial_finished.emit(self.state_label.text().endswith("FINALIZED"))
        if self._close_when_finished:
            self._close_when_finished = False
            QTimer.singleShot(0, self.close)

    def _set_configuration_locked(self, locked: bool) -> None:
        self._configuration_locked = locked
        for widget in self._configuration_widgets:
            widget.setEnabled(not locked)
        self.start_button.setEnabled(not locked)

    def _set_trial_state(self, state: str) -> None:
        normalized = state.strip().upper() or "UNKNOWN"
        self.state_label.setText(f"Trial: {normalized}")
        if normalized in {"FAILED", "ABORTED"}:
            colors = "background:#f8d7da;color:#842029;border:1px solid #f5c2c7;"
        elif normalized in {"FINALIZED", "COMPLETED"}:
            colors = "background:#d1e7dd;color:#0f5132;border:1px solid #badbcc;"
        elif normalized in {"RECORDING", "PREPARING", "READY", "STOPPING", "FINALIZING"}:
            colors = "background:#fff3cd;color:#664d03;border:1px solid #ffecb5;"
        else:
            colors = "background:#e2e3e5;color:#41464b;border:1px solid #d3d6d8;"
        self.state_label.setStyleSheet(
            f"QLabel {{{colors}padding:6px;border-radius:3px;font-weight:600;}}"
        )

    def _append_alert(self, message: str) -> None:
        self.alerts_edit.appendPlainText(message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        worker = self._worker
        if worker is not None and self._worker_is_alive(worker):
            self._close_when_finished = True
            self.request_controlled_stop()
            self.statusBar().showMessage(
                "正在受控停止并最终化 Trial；完成后将自动关闭。"
            )
            event.ignore()
            return
        if worker is not None:
            self._release_worker(worker)
        self._poll_timer.stop()
        event.accept()
