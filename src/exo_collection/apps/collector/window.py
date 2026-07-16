"""Responsive PySide6 shell for the Collector worker process."""

from __future__ import annotations

import math
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QLocale, QRegularExpression, QTimer, Qt, Signal, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QDoubleValidator,
    QIntValidator,
    QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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
from exo_collection.apps.collector.preflight import (
    CollectorPreflightReport,
    CollectorPreflightWorker,
    run_simulated_preflight,
)
from exo_collection.orchestration.models import (
    MeasuredConditionMetadata,
    TrialExperimentMetadata,
    TrialRunRequest,
)
from exo_collection.protocols import load_default_protocol
from exo_collection.quality import load_storage_policy


MODALITIES = ("ultrasound", "imu", "encoder", "sync_pulse")
CRITICAL_MODALITIES = frozenset(MODALITIES)
MAX_PREVIEW_POINTS = 4096
MAX_SIGNAL_HISTORY_POINTS = 3000
WATERFALL_WINDOW_NS = 8_000_000_000
MAX_WATERFALL_ROWS = 300
MAX_ULTRASOUND_TREND_POINTS = 300
MAX_TIMELINE_EVENTS = 300
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


PreflightWorkerFactory = Callable[[Path], PreflightWorkerHandle]


def simulated_preflight_worker_factory(data_root: Path) -> PreflightWorkerHandle:
    """Build the production spawn boundary for simulator/device preflight."""

    storage_policy = load_storage_policy()
    return CollectorPreflightWorker(
        data_root,
        minimum_free_space_gib=storage_policy.minimum_free_space_gib,
    )


def simulated_profile_preflight(
    data_root: Path,
) -> CollectorPreflightReport:
    """Exercise real simulator lifecycle, sampling, trigger and storage checks."""

    storage_policy = load_storage_policy()
    return run_simulated_preflight(
        data_root,
        minimum_free_space_gib=storage_policy.minimum_free_space_gib,
    )


class ExperimentMetadataDialog(QDialog):
    """Compact editor for optional, structured experimental records."""

    def __init__(
        self,
        metadata: TrialExperimentMetadata,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._validated_metadata: TrialExperimentMetadata | None = None
        self.setWindowTitle("实验记录（可选）")
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
        self,
        object_name: str,
        choices: tuple[tuple[str, object], ...],
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
        for edit, value in zip(
            self.channel_mapping_edits,
            probe.channel_mapping,
            strict=True,
        ):
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
                    "leg_length_cm": self._optional_float(
                        self.leg_length_edit, "腿长"
                    ),
                    "sex": self.sex_combo.currentData(),
                    "age_years": self._optional_int(self.age_edit, "年龄"),
                },
                "ultrasound_probe": {
                    "muscle": self.muscle_edit.text(),
                    "laterality": self.laterality_combo.currentData(),
                    "longitudinal_position": self.position_combo.currentData(),
                    "channel_mapping": [
                        edit.text() for edit in self.channel_mapping_edits
                    ],
                    "fixation_method": self.fixation_edit.text(),
                    "strap_pressure": self.strap_pressure_edit.text(),
                    "probe_reapplied": self.reapplied_combo.currentData(),
                },
                "measured_condition": {
                    "treadmill_speed_mps": self._optional_float(
                        self.speed_edit, "跑台速度"
                    ),
                    "assist_level": self._optional_float(
                        self.assist_edit, "助力等级"
                    ),
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
        preflight_worker_factory: PreflightWorkerFactory = (
            simulated_preflight_worker_factory
        ),
        poll_interval_ms: int = 50,
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
        self._experiment_metadata = TrialExperimentMetadata()
        self._experiment_metadata_by_identity: dict[
            tuple[str, str], TrialExperimentMetadata
        ] = {}
        self._metadata_identity_key: tuple[str, str] | None = None
        self._metadata_condition_code: str | None = None

        self._session_key: tuple[str, str, str] | None = None
        self._session_uuid = uuid4()

        self._health_rows = {name: index for index, name in enumerate(MODALITIES)}
        self._last_health_status: dict[str, str] = {}
        self._signal_history: dict[str, tuple[list[float], list[float]]] = {
            "imu": ([], []),
            "encoder": ([], []),
        }
        self._ultrasound_histories: dict[
            int, deque[tuple[int, np.ndarray]]
        ] = {}
        self._latest_ultrasound_channels: list[list[float]] = []
        self._ultrasound_trend_starts: dict[int, int] = {}
        self._ultrasound_trend_times: dict[int, deque[float]] = {}
        self._ultrasound_peak_depths: dict[int, deque[float]] = {}
        self._ultrasound_peak_strengths: dict[int, deque[float]] = {}
        self._ultrasound_format_metrics: dict[int, dict[str, Any]] = {}
        self._ultrasound_format_alerted: set[tuple[int, str]] = set()
        # Backward-compatible selected-channel aliases used by UI tests and
        # lightweight diagnostics. Switching channels rebinds rather than
        # clearing these histories.
        self._ultrasound_history = self._channel_waterfall_history(0)
        self._ultrasound_trend_x = self._channel_trend_times(0)
        self._ultrasound_peak_depth = self._channel_peak_depths(0)
        self._ultrasound_peak_strength = self._channel_peak_strengths(0)
        self._timeline_started_at = time.monotonic()
        self._timeline_x: deque[float] = deque(maxlen=MAX_TIMELINE_EVENTS)
        self._timeline_y: deque[float] = deque(maxlen=MAX_TIMELINE_EVENTS)

        self.setWindowTitle("Exo Collector")
        self.resize(1280, 820)
        self._create_ui(Path(data_root).expanduser().resolve())
        self.project_combo.currentIndexChanged.connect(
            self._activate_selected_metadata_identity
        )
        self.subject_code_edit.textChanged.connect(
            self._activate_selected_metadata_identity
        )
        self._activate_selected_metadata_identity()
        self.condition_combo.currentIndexChanged.connect(
            self._handle_metadata_condition_changed
        )
        self._metadata_condition_code = self._selected_condition_code()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(poll_interval_ms)
        self._poll_timer.timeout.connect(self.poll_worker_events)
        self._preflight_timer = QTimer(self)
        self._preflight_timer.setInterval(max(20, poll_interval_ms))
        self._preflight_timer.timeout.connect(self.poll_preflight_worker)
        self._set_trial_state("IDLE")
        self._update_start_button()

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
    def overall_status(self) -> str:
        return self.state_label.text().removeprefix("总状态：")

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
        self.data_root_edit.textChanged.connect(self._invalidate_preflight)
        root_row.addWidget(self.data_root_edit, 1)
        self.browse_button = QPushButton("选择…")
        self.browse_button.clicked.connect(self.choose_data_root)
        root_row.addWidget(self.browse_button)
        form.addRow("数据根目录：", root_row)

        self.project_combo = QComboBox()
        self.project_combo.setObjectName("project")
        for project in PROJECTS:
            self.project_combo.addItem(
                f"{project['project_code']} — {project['project_name']}",
                dict(project),
            )
        self.project_combo.setCurrentIndex(1)
        form.addRow("项目：", self.project_combo)

        self.subject_code_edit = QLineEdit("001")
        self.subject_code_edit.setObjectName("subject_code")
        self.subject_code_edit.setMaxLength(3)
        self.subject_code_edit.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"\d{3}"), self)
        )
        self.subject_code_edit.editingFinished.connect(self.normalize_subject_code)
        self.subject_code_edit.textChanged.connect(self._update_start_button)
        form.addRow("受试者编码：", self.subject_code_edit)

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
        controls_layout.addWidget(metadata_box)

        experiment_box = QGroupBox("实验记录（可选）")
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

        buttons = QHBoxLayout()
        self.preflight_button = QPushButton("设备预检 / 连接")
        self.preflight_button.setObjectName("preflight_devices")
        self.preflight_button.clicked.connect(self.run_preflight)
        buttons.addWidget(self.preflight_button)
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
        self.device_profile_label = QLabel(
            "当前设备配置：内置模拟设备（真实厂商 SDK 尚未接入）"
        )
        self.device_profile_label.setObjectName("device_profile")
        self.device_profile_label.setStyleSheet("color:#6c757d;")
        health_layout.addWidget(self.device_profile_label)
        self.health_table = QTableWidget(len(MODALITIES), 7)
        self.health_table.setObjectName("health_table")
        self.health_table.setHorizontalHeaderLabels(
            ["模态", "健康", "样本/帧", "实际速率", "丢包", "队列", "最近更新"]
        )
        self.health_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.health_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
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
        self.first_trigger_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        sync_layout.addWidget(self.first_trigger_label, 1, 1, 1, 3)
        sync_layout.addWidget(QLabel("质量："), 2, 0)
        self.sync_quality_label = QLabel("—")
        self.sync_quality_label.setObjectName("sync_quality")
        sync_layout.addWidget(self.sync_quality_label, 2, 1, 1, 3)
        controls_layout.addWidget(sync_box)

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

        preview_box = QGroupBox("实时预览（共享内存降采样；不参与原始写盘）")
        preview_layout = QVBoxLayout(preview_box)
        preview_splitter = QSplitter(Qt.Orientation.Vertical)
        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")

        ultrasound_box = QGroupBox("Ultrasound A-scan 与最近 8 秒瀑布")
        ultrasound_layout = QVBoxLayout(ultrasound_box)
        ultrasound_controls = QHBoxLayout()
        ultrasound_controls.addWidget(QLabel("预览通道："))
        self.ultrasound_channel_combo = QComboBox()
        self.ultrasound_channel_combo.setObjectName("ultrasound_channel")
        self.ultrasound_channel_combo.addItem("通道 1", 0)
        self.ultrasound_channel_combo.currentIndexChanged.connect(
            self._on_ultrasound_channel_changed
        )
        ultrasound_controls.addWidget(self.ultrasound_channel_combo)
        self.ultrasound_peak_label = QLabel(
            "峰值：等待数据 · 自动阈值待真实设备标定"
        )
        self.ultrasound_peak_label.setObjectName("ultrasound_peak_metrics")
        self.ultrasound_peak_label.setWordWrap(True)
        self.ultrasound_peak_label.setStyleSheet("color:#6c757d;")
        ultrasound_controls.addWidget(self.ultrasound_peak_label, 1)
        ultrasound_layout.addLayout(ultrasound_controls)
        self.ultrasound_plot = pg.PlotWidget(title="A-scan")
        self.ultrasound_plot.setObjectName("ultrasound_preview")
        self.ultrasound_plot.setBackground("w")
        self.ultrasound_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ultrasound_curve = self.ultrasound_plot.plot(
            pen=pg.mkPen("#2457c5", width=1.3)
        )
        ultrasound_layout.addWidget(self.ultrasound_plot, 1)
        peak_trends = QWidget()
        peak_trends_layout = QHBoxLayout(peak_trends)
        peak_trends_layout.setContentsMargins(0, 0, 0, 0)
        self.ultrasound_peak_depth_plot = pg.PlotWidget(title="峰值深度趋势")
        self.ultrasound_peak_depth_plot.setObjectName("ultrasound_peak_depth")
        self.ultrasound_peak_depth_plot.setBackground("w")
        self.ultrasound_peak_depth_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ultrasound_peak_depth_curve = self.ultrasound_peak_depth_plot.plot(
            pen=pg.mkPen("#0d6efd", width=1.2)
        )
        peak_trends_layout.addWidget(self.ultrasound_peak_depth_plot, 1)
        self.ultrasound_peak_strength_plot = pg.PlotWidget(title="峰值强度趋势")
        self.ultrasound_peak_strength_plot.setObjectName("ultrasound_peak_strength")
        self.ultrasound_peak_strength_plot.setBackground("w")
        self.ultrasound_peak_strength_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ultrasound_peak_strength_curve = self.ultrasound_peak_strength_plot.plot(
            pen=pg.mkPen("#fd7e14", width=1.2)
        )
        peak_trends_layout.addWidget(self.ultrasound_peak_strength_plot, 1)
        peak_trends.setMaximumHeight(150)
        ultrasound_layout.addWidget(peak_trends)
        self.ultrasound_waterfall_plot = pg.PlotWidget(title="灰度瀑布 · 最近 8 秒")
        self.ultrasound_waterfall_plot.setObjectName("ultrasound_waterfall")
        self.ultrasound_waterfall_plot.setBackground("#111111")
        self.ultrasound_waterfall_plot.setLabel("left", "时间")
        self.ultrasound_waterfall_plot.setLabel("bottom", "A-scan sample")
        self.ultrasound_waterfall_image = pg.ImageItem(axisOrder="row-major")
        self.ultrasound_waterfall_plot.addItem(self.ultrasound_waterfall_image)
        ultrasound_layout.addWidget(self.ultrasound_waterfall_plot, 1)
        preview_splitter.addWidget(ultrasound_box)

        signals_box = QWidget()
        signals_layout = QHBoxLayout(signals_box)
        signals_layout.setContentsMargins(0, 0, 0, 0)
        self.imu_plot = pg.PlotWidget(title="IMU 实时曲线")
        self.imu_plot.setObjectName("imu_preview")
        self.imu_plot.setBackground("w")
        self.imu_plot.setLabel("bottom", "Trial time", units="s")
        self.imu_plot.showGrid(x=True, y=True, alpha=0.2)
        self.imu_curve = self.imu_plot.plot(pen=pg.mkPen("#1a936f", width=1.4))
        signals_layout.addWidget(self.imu_plot, 1)

        self.encoder_plot = pg.PlotWidget(title="Encoder 实时曲线")
        self.encoder_plot.setObjectName("encoder_preview")
        self.encoder_plot.setBackground("w")
        self.encoder_plot.setLabel("bottom", "Trial time", units="s")
        self.encoder_plot.showGrid(x=True, y=True, alpha=0.2)
        self.encoder_curve = self.encoder_plot.plot(
            pen=pg.mkPen("#d97706", width=1.4)
        )
        signals_layout.addWidget(self.encoder_plot, 1)
        preview_splitter.addWidget(signals_box)

        timeline_box = QWidget()
        timeline_layout = QVBoxLayout(timeline_box)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        self.timeline_plot = pg.PlotWidget(title="同步 / 事件时间线")
        self.timeline_plot.setObjectName("event_timeline")
        self.timeline_plot.setBackground("w")
        self.timeline_plot.setLabel("bottom", "UI elapsed", units="s")
        self.timeline_plot.getAxis("left").setTicks(
            [[(0, "状态"), (1, "同步"), (2, "告警")]]
        )
        self.timeline_plot.setYRange(-0.5, 2.5, padding=0)
        self.timeline_curve = self.timeline_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush="#6f42c1",
        )
        timeline_layout.addWidget(self.timeline_plot, 1)
        self.timeline_last_event_label = QLabel("尚无事件")
        self.timeline_last_event_label.setObjectName("timeline_last_event")
        timeline_layout.addWidget(self.timeline_last_event_label)
        preview_splitter.addWidget(timeline_box)
        preview_splitter.setSizes([410, 230, 160])
        preview_layout.addWidget(preview_splitter, 1)
        body.addWidget(preview_box)
        body.setSizes([470, 790])
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
            self.preflight_button,
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

    @Slot()
    def _invalidate_preflight(self) -> None:
        """A changed storage target or completed Worker requires a fresh probe."""

        if self._worker is not None or not self._preflight_ready:
            return
        self._preflight_ready = False
        for modality, row in self._health_rows.items():
            self.health_table.item(row, 1).setText("UNKNOWN")
            self.health_table.item(row, 1).setToolTip("")
        self._set_trial_state("IDLE")
        self.statusBar().showMessage("配置或存储目标已变化，请重新执行设备预检。")
        self._update_start_button()

    @property
    def experiment_metadata(self) -> TrialExperimentMetadata:
        return self._experiment_metadata

    def set_experiment_metadata(
        self,
        metadata: TrialExperimentMetadata | Mapping[str, Any],
    ) -> None:
        self._experiment_metadata = TrialExperimentMetadata.model_validate(metadata)
        if self._metadata_identity_key is not None:
            self._experiment_metadata_by_identity[self._metadata_identity_key] = (
                self._experiment_metadata
            )
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
        identity = (
            "未识别受试者"
            if self._metadata_identity_key is None
            else f"{self._metadata_identity_key[0]}/{self._metadata_identity_key[1]}"
        )
        if value_count:
            text = f"{identity} 已填写 {value_count} 项；同一受试者后续 Trial 默认沿用"
        else:
            text = f"{identity} 未填写；不影响采集"
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
                if previous_key is not None
                and self._experiment_metadata_value_count(previous_metadata)
                else None
            )
            if transition and hasattr(self, "alerts_edit"):
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
            update={
                "measured_condition": MeasuredConditionMetadata(),
                "trial_notes": None,
            }
        )
        for identity, cached in tuple(self._experiment_metadata_by_identity.items()):
            self._experiment_metadata_by_identity[identity] = cached.model_copy(
                update={
                    "measured_condition": MeasuredConditionMetadata(),
                    "trial_notes": None,
                }
            )
        if self._metadata_identity_key is not None:
            self._experiment_metadata_by_identity[self._metadata_identity_key] = (
                self._experiment_metadata
            )
        transition = "工况已切换，实测工况与 Trial 备注已清空"
        self._render_experiment_metadata_summary(transition=transition)
        if had_condition_values and hasattr(self, "alerts_edit"):
            self._append_alert(
                f"{transition}：{previous or '未选择'} → {selected}；人口学与探头固定信息保留。"
            )
        self.statusBar().showMessage(
            f"{transition}（{previous or '未选择'} → {selected}）。",
            8000,
        )

    def _clear_one_trial_metadata(self) -> None:
        probe = self._experiment_metadata.ultrasound_probe
        had_one_trial_values = bool(
            self._experiment_metadata.trial_notes is not None
            or probe.probe_reapplied is not None
        )
        self._experiment_metadata = self._experiment_metadata.model_copy(
            update={
                "ultrasound_probe": probe.model_copy(
                    update={"probe_reapplied": None}
                ),
                "trial_notes": None,
            }
        )
        if self._metadata_identity_key is not None:
            self._experiment_metadata_by_identity[self._metadata_identity_key] = (
                self._experiment_metadata
            )
        transition = (
            "上一 Trial 已结束，一次性备注与‘重新贴探头’已清空"
            if had_one_trial_values
            else None
        )
        self._render_experiment_metadata_summary(transition=transition)
        if transition:
            self._append_alert(f"{transition}；人口学、探头位置与固定方式仍保留。")
            self.statusBar().showMessage(
                f"{transition}；下一个 Trial 开始前请重新确认。",
                8000,
            )

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

    @Slot()
    def run_preflight(self) -> None:
        if self._worker is not None or self._preflight_worker is not None:
            return
        self._preflight_ready = False
        self._set_preflight_busy(True)
        self._set_trial_state("PREFLIGHT")
        self.statusBar().showMessage(
            "正在独立进程中连接、准备并短时采样四个模态，同时检查写入与磁盘空间…"
        )
        worker: PreflightWorkerHandle | None = None
        try:
            root_text = self.data_root_edit.text().strip()
            if not root_text:
                raise ValueError("数据根目录不能为空")
            root = self.set_data_root(root_text)
            worker = self._preflight_worker_factory(root)
            self._preflight_worker = worker
            self._preflight_root = root
            self._preflight_result_handled = False
            self._preflight_empty_exit_polls = 0
            worker.start()
        except Exception:
            details = traceback.format_exc()
            # Factories and spawn may fail before a PID exists. Each production
            # worker also performs its own partial-start cleanup; this fallback
            # keeps injected implementations honest.
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
        # Non-blocking poll only. This lets deterministic test/injected workers
        # complete immediately without ever executing device work on the GUI.
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
            # A multiprocessing.Queue feeder can trail process exit briefly.
            self._preflight_empty_exit_polls += 1
            if self._preflight_empty_exit_polls < 10:
                return
            self._preflight_result_handled = True
            self._apply_preflight_result(
                None,
                error=(
                    "设备预检进程已退出但未返回结果"
                    f"（exitcode={self._preflight_worker_exitcode(worker)}）。"
                ),
            )
        try:
            worker.join(timeout=0)
            worker.close()
        except Exception as exc:
            self._append_alert(
                f"释放预检进程资源时出错：{type(exc).__name__}: {exc}"
            )
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

    def _apply_preflight_result(
        self,
        raw_result: object | None,
        *,
        error: str | None = None,
    ) -> None:
        report: CollectorPreflightReport | None = None
        try:
            if isinstance(raw_result, CollectorPreflightReport):
                report = raw_result
                if (
                    self._preflight_root is not None
                    and report.data_root.resolve() != self._preflight_root.resolve()
                ):
                    raise ValueError("设备预检结果来自不同的数据根目录")
                reported = {
                    modality: item.status
                    for modality, item in report.devices.items()
                }
            elif isinstance(raw_result, Mapping):
                reported = {
                    str(modality): str(status).strip().upper()
                    for modality, status in raw_result.items()
                }
            else:
                reported = {}
                if error is None:
                    error = "设备预检进程返回了无效结果"
        except Exception as exc:
            reported = {}
            error = f"{type(exc).__name__}: {exc}"

        if error:
            final_line = next(
                (line for line in reversed(error.splitlines()) if line.strip()),
                error,
            )
            self._append_alert(f"设备预检失败：{final_line}")

        missing_or_failed: list[str] = []
        for modality in MODALITIES:
            status = reported.get(modality, "MISSING")
            row = self._health_rows[modality]
            self.health_table.item(row, 1).setText(status)
            if report is not None and modality in report.devices:
                result = report.devices[modality]
                self.health_table.item(row, 1).setToolTip(
                    f"{result.device_id} · {result.message} · "
                    f"channels={result.channel_count} · raw={result.observed_raw_data}"
                )
                self.health_table.item(row, 3).setText(
                    "-"
                    if result.actual_rate_hz is None
                    else f"{result.actual_rate_hz:.1f}"
                )
                self.health_table.item(row, 5).setText(
                    f"0/{result.queue_capacity}"
                )
            if modality in CRITICAL_MODALITIES and status != "READY":
                missing_or_failed.append(f"{modality}={status}")

        self._preflight_ready = not missing_or_failed and (
            report.ready if report is not None else True
        )
        if self._preflight_ready:
            self._set_trial_state("PREFLIGHT_READY")
            storage = ""
            if report is not None:
                storage = (
                    f" 可用空间 {report.disk_free_bytes / 1024**3:.2f} GiB；"
                    f"落盘探测 {report.measured_write_mib_s:.1f} MiB/s"
                    "（阈值待真实超声最大速率确定）；"
                    f"耗时 {report.elapsed_s:.2f} s。"
                )
            self.statusBar().showMessage(
                f"四个必需模态已实际连接/准备/采样，同步上升沿已观测。{storage}",
                8000,
            )
        else:
            self._set_trial_state("FAILED")
            detail = "、".join(missing_or_failed) or "预检服务未返回设备状态"
            self._append_alert(f"关键设备未 READY：{detail}")
            self.statusBar().showMessage("设备预检失败；开始采集保持禁用。")
        self._update_start_button()

    def _refresh_identity_context(
        self, data_root: Path, project_code: str, subject_code: str
    ) -> None:
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
            "experiment_metadata": self._experiment_metadata.model_dump(
                mode="python"
            ),
        }
        return TrialRunRequest.model_validate(payload)

    @Slot()
    def start_trial(self) -> None:
        if self._worker is not None or self._preflight_worker is not None:
            return
        if not self._preflight_ready:
            self._append_alert("开始已阻止：请先完成设备预检，确保所有关键设备 READY。")
            self.statusBar().showMessage("请先执行设备预检 / 连接。")
            self._update_start_button()
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
        self._active_trial_uuid = str(request.trial_uuid)
        self._terminal_event_received = False
        self._dead_poll_count = 0
        self._stop_requested = False
        self._stop_requested_at = None
        self._forced_stop_alerted = False
        self._close_when_finished = False
        self._trial_succeeded = False
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
        self.ultrasound_curve.setData([], [])
        self._latest_ultrasound_channels.clear()
        self._ultrasound_histories.clear()
        self._ultrasound_trend_starts.clear()
        self._ultrasound_trend_times.clear()
        self._ultrasound_peak_depths.clear()
        self._ultrasound_peak_strengths.clear()
        self._ultrasound_format_metrics.clear()
        self._ultrasound_format_alerted.clear()
        self._bind_selected_ultrasound_history(
            max(0, self.ultrasound_channel_combo.currentIndex())
        )
        self.ultrasound_waterfall_image.clear()
        self._clear_ultrasound_trends()
        for modality, curve in (
            ("imu", self.imu_curve),
            ("encoder", self.encoder_curve),
        ):
            self._signal_history[modality] = ([], [])
            curve.setData([], [])
        self._timeline_started_at = time.monotonic()
        self._timeline_x.clear()
        self._timeline_y.clear()
        self.timeline_curve.setData([], [])
        self.timeline_last_event_label.setText("Trial 已创建，等待同步")
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
            self._enforce_controlled_stop_deadline(worker)
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
            self.statusBar().showMessage(
                "Writer/设备停止超时；正在保留 .recording 并执行强制回收。"
            )
        try:
            worker.terminate_for_recovery(timeout=1.0)
        except Exception as exc:
            self._append_alert(
                f"强制回收 Collector Worker 失败：{type(exc).__name__}: {exc}"
            )
            return
        if self._worker_is_alive(worker):
            return
        self._terminal_event_received = True
        if not self._trial_succeeded:
            self._mark_failed(
                "受控停止超时，Worker 已终止；原始数据保持 .recording，需在 "
                "Data Studio 的恢复工作流中审计。"
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
                "已拒绝不属于当前 Trial 的 Worker 事件："
                f"expected={expected_trial_uuid}，received={claimed_trial_uuid}，"
                f"type={event.event_type.value}。"
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
        self.health_table.item(row, 3).setText(
            "-" if rate is None else f"{float(rate):.1f} Hz"
        )
        dropped = payload.get("dropped_packets")
        self.health_table.item(row, 4).setText(
            "-" if dropped is None else str(int(dropped))
        )
        depth = payload.get("queue_depth")
        capacity = payload.get("queue_capacity")
        if depth is None:
            queue_text = "-"
        elif capacity is None:
            queue_text = str(int(depth))
        else:
            queue_text = f"{int(depth)}/{int(capacity)}"
        self.health_table.item(row, 5).setText(queue_text)
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
            self.health_table.item(row, 2).setToolTip(
                f"已检测边沿：{int(payload['pulse_event_count'])}"
            )
        if any(
            key in payload
            for key in (
                "status",
                "quality",
                "trigger_count",
                "first_trigger_host_monotonic_ns",
                "trigger_time_utc",
            )
        ):
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
            channels: list[list[float]] = []
            raw_channels = event.payload.get("channels")
            if isinstance(raw_channels, (list, tuple)):
                channels = [
                    converted
                    for raw_channel in raw_channels
                    if (converted := self._numeric_values(raw_channel))
                ]
            if channels:
                raw_metrics = event.payload.get("format_metrics")
                if isinstance(raw_metrics, (list, tuple)):
                    for metric_index, metric in enumerate(raw_metrics):
                        if metric_index >= len(channels) or not isinstance(metric, dict):
                            continue
                        self._ultrasound_format_metrics[metric_index] = dict(metric)
                        if bool(metric.get("all_zero")):
                            alert_key = (metric_index, "ALL_ZERO")
                            if alert_key not in self._ultrasound_format_alerted:
                                message = (
                                    f"ultrasound 通道 {metric_index + 1} 当前帧全零；"
                                    "请检查探头、通道和设备连接。"
                                )
                                self._append_alert(message)
                                self._add_timeline_event(2, message)
                                self._ultrasound_format_alerted.add(alert_key)
                self._latest_ultrasound_channels = channels
                self._set_ultrasound_channel_count(len(channels))
                timestamp_ns = self._preview_timestamp_ns(event.payload)
                for channel_index, channel_values in enumerate(channels):
                    self._record_ultrasound_channel(
                        channel_index,
                        channel_values,
                        timestamp_ns,
                    )
                selected_index = min(
                    self.ultrasound_channel_combo.currentIndex(),
                    len(channels) - 1,
                )
                values = channels[selected_index]
            else:
                values = self._numeric_values(event.payload.get("values"))
                selected_index = max(0, self.ultrasound_channel_combo.currentIndex())
                if values:
                    self._record_ultrasound_channel(
                        selected_index,
                        values,
                        self._preview_timestamp_ns(event.payload),
                    )
            if values:
                self.ultrasound_curve.setData(list(range(len(values))), values)
                self._render_ultrasound_channel(selected_index)
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
    def _preview_timestamp_ns(payload: Mapping[str, Any]) -> int:
        try:
            return int(payload.get("host_monotonic_ns") or time.monotonic_ns())
        except (TypeError, ValueError):
            return time.monotonic_ns()

    def _channel_waterfall_history(
        self, channel_index: int
    ) -> deque[tuple[int, np.ndarray]]:
        return self._ultrasound_histories.setdefault(channel_index, deque())

    def _channel_trend_times(self, channel_index: int) -> deque[float]:
        return self._ultrasound_trend_times.setdefault(
            channel_index,
            deque(maxlen=MAX_ULTRASOUND_TREND_POINTS),
        )

    def _channel_peak_depths(self, channel_index: int) -> deque[float]:
        return self._ultrasound_peak_depths.setdefault(
            channel_index,
            deque(maxlen=MAX_ULTRASOUND_TREND_POINTS),
        )

    def _channel_peak_strengths(self, channel_index: int) -> deque[float]:
        return self._ultrasound_peak_strengths.setdefault(
            channel_index,
            deque(maxlen=MAX_ULTRASOUND_TREND_POINTS),
        )

    def _bind_selected_ultrasound_history(self, channel_index: int) -> None:
        self._ultrasound_history = self._channel_waterfall_history(channel_index)
        self._ultrasound_trend_x = self._channel_trend_times(channel_index)
        self._ultrasound_peak_depth = self._channel_peak_depths(channel_index)
        self._ultrasound_peak_strength = self._channel_peak_strengths(channel_index)

    def _record_ultrasound_channel(
        self,
        channel_index: int,
        values: list[float],
        timestamp_ns: int,
    ) -> None:
        history = self._channel_waterfall_history(channel_index)
        row = np.asarray(values, dtype=np.float32)
        if history and history[-1][1].size != row.size:
            history.clear()
        history.append((timestamp_ns, row))
        cutoff = timestamp_ns - WATERFALL_WINDOW_NS
        while history and (
            history[0][0] < cutoff
            or len(history) > MAX_WATERFALL_ROWS
        ):
            history.popleft()
        peak_index = int(np.argmax(np.abs(row)))
        peak_strength = float(row[peak_index])
        start_ns = self._ultrasound_trend_starts.setdefault(
            channel_index, timestamp_ns
        )
        elapsed_s = max(
            0.0,
            (timestamp_ns - start_ns) / 1_000_000_000,
        )
        self._channel_trend_times(channel_index).append(elapsed_s)
        self._channel_peak_depths(channel_index).append(float(peak_index))
        self._channel_peak_strengths(channel_index).append(peak_strength)

    def _render_ultrasound_channel(self, channel_index: int) -> None:
        self._bind_selected_ultrasound_history(channel_index)
        if self._ultrasound_history:
            image = np.stack(
                [item[1] for item in self._ultrasound_history], axis=0
            )
            self.ultrasound_waterfall_image.setImage(image, autoLevels=True)
        else:
            self.ultrasound_waterfall_image.clear()
        self.ultrasound_peak_depth_curve.setData(
            list(self._ultrasound_trend_x),
            list(self._ultrasound_peak_depth),
        )
        self.ultrasound_peak_strength_curve.setData(
            list(self._ultrasound_trend_x),
            list(self._ultrasound_peak_strength),
        )
        if self._ultrasound_peak_depth and self._ultrasound_peak_strength:
            peak_index = int(self._ultrasound_peak_depth[-1])
            peak_strength = float(self._ultrasound_peak_strength[-1])
            metric = self._ultrasound_format_metrics.get(channel_index, {})
            zero_fraction = metric.get("zero_fraction")
            full_scale_fraction = metric.get("full_scale_fraction")
            format_text = "格式指标：等待"
            if isinstance(zero_fraction, (int, float)):
                format_text = f"零值 {float(zero_fraction):.2%}"
                if isinstance(full_scale_fraction, (int, float)):
                    format_text += f" / 满量程 {float(full_scale_fraction):.2%}"
            self.ultrasound_peak_label.setText(
                f"通道 {channel_index + 1} · 峰值深度索引 {peak_index} · "
                f"峰值强度 {peak_strength:.3g} · {format_text} · "
                "信号弱/边界/滑移：UNASSESSED（待真实设备标定）"
            )
        else:
            self.ultrasound_peak_label.setText(
                f"通道 {channel_index + 1} · 等待数据 · 自动阈值待真实设备标定"
            )

    def _set_ultrasound_channel_count(self, channel_count: int) -> None:
        if channel_count <= 0 or self.ultrasound_channel_combo.count() == channel_count:
            return
        previous = self.ultrasound_channel_combo.currentIndex()
        self.ultrasound_channel_combo.blockSignals(True)
        self.ultrasound_channel_combo.clear()
        for index in range(channel_count):
            self.ultrasound_channel_combo.addItem(f"通道 {index + 1}", index)
        self.ultrasound_channel_combo.setCurrentIndex(min(previous, channel_count - 1))
        self.ultrasound_channel_combo.blockSignals(False)

    @Slot(int)
    def _on_ultrasound_channel_changed(self, index: int) -> None:
        if 0 <= index < len(self._latest_ultrasound_channels):
            values = self._latest_ultrasound_channels[index]
            self.ultrasound_curve.setData(list(range(len(values))), values)
            self.ultrasound_plot.setTitle(f"A-scan · 通道 {index + 1}")
        self._render_ultrasound_channel(max(index, 0))

    def _clear_ultrasound_trends(self) -> None:
        self._ultrasound_trend_x.clear()
        self._ultrasound_peak_depth.clear()
        self._ultrasound_peak_strength.clear()
        self.ultrasound_peak_depth_curve.setData([], [])
        self.ultrasound_peak_strength_curve.setData([], [])
        self.ultrasound_peak_label.setText(
            "峰值：等待数据 · 自动阈值待真实设备标定"
        )

    def _add_timeline_event(self, category: int, text: str) -> None:
        elapsed = max(0.0, time.monotonic() - self._timeline_started_at)
        self._timeline_x.append(elapsed)
        self._timeline_y.append(float(category))
        self.timeline_curve.setData(list(self._timeline_x), list(self._timeline_y))
        self.timeline_last_event_label.setText(text)

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
        else:
            self.manifest_label.setText("Manifest：Worker 已完成，但未返回路径")
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage(event.message or "Trial 数据包已最终化。")

    def _mark_failed(self, message: str) -> None:
        self._trial_succeeded = False
        self._set_trial_state("FAILED")
        self.stop_button.setEnabled(False)
        self._append_alert(f"FAILED：{message}")
        self._add_timeline_event(2, f"FAILED · {message}")
        self.statusBar().showMessage("Trial 失败；请检查告警信息。")

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
        self._set_configuration_locked(False)
        self.stop_button.setEnabled(False)
        self._clear_one_trial_metadata()
        if self._trial_succeeded:
            self._set_trial_state("IDLE")
            self.statusBar().showMessage(
                "Trial 已最终化；设备 Worker 已关闭，一次性元数据已清空；"
                "下一个 Trial 前请重新预检并确认记录。",
                8000,
            )
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
        self._update_start_button()

    @Slot()
    def _update_start_button(self) -> None:
        if not hasattr(self, "start_button"):
            return
        subject_valid = bool(
            QRegularExpression(r"^\d{3}$").match(
                self.subject_code_edit.text().strip()
            ).hasMatch()
        )
        self.start_button.setEnabled(
            self._preflight_ready
            and subject_valid
            and not self._configuration_locked
            and not self._preflight_busy
            and self._worker is None
        )

    def _set_trial_state(self, state: str) -> None:
        normalized = state.strip().upper() or "UNKNOWN"
        self._worker_state = normalized
        display = {
            "IDLE": "未连接",
            "DISCONNECTED": "未连接",
            "PREFLIGHT_READY": "可采集",
            "PREFLIGHT": "设备预检",
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
        elif display in {"设备预检", "等待同步", "保存中"}:
            colors = "background:#fff3cd;color:#664d03;border:1px solid #ffecb5;"
        else:
            colors = "background:#e2e3e5;color:#41464b;border:1px solid #d3d6d8;"
        self.state_label.setStyleSheet(
            f"QLabel {{{colors}padding:6px;border-radius:3px;font-weight:600;}}"
        )

    def _append_alert(self, message: str) -> None:
        self.alerts_edit.appendPlainText(message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        preflight_worker = self._preflight_worker
        if preflight_worker is not None:
            self._preflight_timer.stop()
            try:
                preflight_worker.terminate(timeout=0.5)
            except Exception as exc:
                self._append_alert(
                    f"停止设备预检进程失败：{type(exc).__name__}: {exc}"
                )
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
                self._append_alert(
                    f"释放设备预检资源失败：{type(exc).__name__}: {exc}"
                )
            self._preflight_worker = None
            self._preflight_root = None
            self._set_preflight_busy(False)
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
        self._close_started_at = None
        event.accept()
