"""PySide6 main window for local Trial browsing and basic statistics."""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

from PySide6.QtCore import QDate, QModelIndex, QObject, QRect, QRunnable, QSize, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QBrush, QCloseEvent, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from exo_collection.configuration import SharedAppSettings
from exo_collection.external import ExternalImportRequest, ExternalImportResult
from exo_collection.storage.activity import AcquisitionActivity, read_activity

from .external_import_dialog import ExternalImportDialog
from .external_import_worker import ExternalImportWorker
from .local_dialogs import (
    ChecksumDialog,
    FullStatisticsDialog,
    PlaybackDialog,
    QualityAuditDialog,
)
from .local_tools import (
    ChecksumReport,
    FullStatistics,
    QualityAudit,
    TrialPlayback,
    compute_full_statistics,
    load_quality_audit,
)
from .management import (
    AnnexScanResult,
    AnnexValidationStatus,
    InventoryExportResult,
    ManagementIndex,
    ManagementRefreshResult,
    ManagementSummaryResult,
    TrialFilter,
    TrialManagementRecord,
    filter_trial_records,
)
from .management_dialog import ManagementSummaryDialog
from .process_workers import DataStudioProcessWorker, ProcessOperation
from .quality_reviews import append_quality_review
from .recovery_dialog import RecoveryDialog
from .service import DataStudioSnapshot
from .credential_store import load_password
from .upload import (
    BatchOfflineUploadResult,
    HostKeyInfo,
    OfflineUploadResult,
    OfflineUploadRequest,
    RemoteStatusSyncResult,
    RemoteTrialStatus,
    UploadOperation,
    UploadWorkerEventType,
    UploadWorkerHandle,
)
from .upload_dialog import OfflineUploadDialog, UploadProgressDialog


_TYPE_LABELS = {
    "project": "项目",
    "subject": "受试者",
    # Catalog keeps these compatibility type names, while the visible tree
    # mirrors the current on-disk project/subject/condition/session layout.
    "session": "工况",
    "trial": "Session",
    "modality": "模态数据集",
    "supporting_files": "辅助资料",
    "artifact": "文件",
    "external_annex": "External Annex",
    "external_artifact": "External File",
}

_MODALITY_LABELS = {
    "ultrasound": "超声",
    "imu": "IMU",
    "encoder": "电机编码器",
    "sync_pulse": "同步脉冲",
}


class _ModalityBadgeDelegate(QStyledItemDelegate):
    """Paint separated rounded modality-count badges in the Trial tree."""

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        node_type = str(
            index.siblingAtColumn(1).data(Qt.ItemDataRole.UserRole) or ""
        )
        base = QStyleOptionViewItem(option)
        base.text = ""
        super().paint(painter, base, index)
        if node_type != "trial" or not text:
            return
        count_text = text.split()[0]
        try:
            has_modalities = int(count_text) > 0
        except ValueError:
            has_modalities = False
        width = min(54, max(42, option.rect.width() - 12))
        height = min(18, max(12, option.rect.height() - 6))
        badge = QRect(option.rect)
        badge.setLeft(option.rect.center().x() - width // 2)
        badge.setWidth(width)
        badge.setTop(option.rect.center().y() - height // 2)
        badge.setHeight(height)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2f6fed" if has_modalities else "#d7dde7"))
        painter.drawRoundedRect(badge, 7, 7)
        painter.setPen(QColor("#ffffff" if has_modalities else "#52606d"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()


class _LocalToolSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class LocalToolTask(QRunnable):
    """Execute a disk-heavy read-only tool outside the GUI thread."""

    def __init__(self, operation: Callable[[], object]) -> None:
        super().__init__()
        self.operation = operation
        self.signals = _LocalToolSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.operation()
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.completed.emit(result)


@dataclass(slots=True)
class _ProcessTaskContext:
    worker: Any
    name: str
    completed: Callable[[object], None]
    kind: str = "local_tool"
    handled: bool = False
    empty_exit_polls: int = 0


@dataclass(slots=True)
class _UploadTaskContext:
    worker: UploadWorkerHandle
    progress_dialog: UploadProgressDialog
    silent: bool = False
    terminal_handled: bool = False
    cancel_requested: bool = False
    empty_exit_polls: int = 0


class DataStudioWindow(QMainWindow):
    """Catalog-backed Data Studio shell for the first runnable milestone."""

    refresh_finished = Signal(bool)
    local_tool_finished = Signal(str, bool)
    upload_finished = Signal(bool)
    management_refresh_finished = Signal(bool)

    def __init__(
        self,
        data_root: str | Path,
        *,
        settings: SharedAppSettings | None = None,
        autostart_refresh: bool = True,
        upload_worker_factory: Callable[[], UploadWorkerHandle] | None = None,
        external_import_worker_factory: (
            Callable[[ExternalImportRequest], ExternalImportWorker] | None
        ) = None,
        process_worker_factory: Callable[..., Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings if settings is not None else SharedAppSettings()
        self._data_root = Path(data_root).expanduser().resolve()
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)
        # Catalog/Alembic/SQLAlchemy work runs in a spawned process.  Keeping
        # the handle here makes refresh ownership explicit without executing
        # C-extension-backed database code in a Qt QRunnable.
        self._active_task: Any | None = None
        self._local_tasks: dict[int, LocalToolTask] = {}
        self._process_tasks: dict[int, _ProcessTaskContext] = {}
        self._upload_worker_factory = upload_worker_factory or UploadWorkerHandle
        self._external_import_worker_factory = (
            external_import_worker_factory or ExternalImportWorker
        )
        self._process_worker_factory = process_worker_factory or DataStudioProcessWorker
        self._active_upload: _UploadTaskContext | None = None
        self._result_dialogs: list[QDialog] = []
        self._closing = False
        self._shutdown_retry_pending = False
        self._close_started_at: float | None = None
        self._statistics: dict[str, Any] = {}
        self._lightweight_mode = False
        self._catalog_tree: list[dict[str, Any]] = []
        self._remote_status_by_manifest: dict[str, tuple[RemoteTrialStatus, str]] = {}
        self._automatic_remote_sync_pending = autostart_refresh
        self._management_index: ManagementIndex | None = None
        self._annex_scan: AnnexScanResult | None = None
        self._filtered_records: tuple[TrialManagementRecord, ...] = ()
        self._populating_filters = False
        self._catalog_summary_text = "尚未刷新。"

        self.setWindowTitle("Exo Data Studio")
        self.resize(1120, 720)
        self._create_actions()
        self._create_ui()

        # Activity detection is a tiny lock-file read and is intentionally
        # immediate; the heavier catalog work starts on the worker pool.
        self._apply_activity(read_activity(self._data_root))
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(1000)
        self._activity_timer.timeout.connect(self._poll_activity)
        self._activity_timer.start()
        self._process_timer = QTimer(self)
        self._process_timer.setInterval(100)
        self._process_timer.timeout.connect(self._poll_process_tools)
        self._process_timer.start()
        if autostart_refresh:
            self.refresh_catalog()

    @property
    def data_root(self) -> Path:
        return self._data_root

    @property
    def statistics(self) -> dict[str, Any]:
        return dict(self._statistics)

    @property
    def lightweight_mode(self) -> bool:
        return self._lightweight_mode

    @property
    def refresh_in_progress(self) -> bool:
        return self._active_task is not None

    def _create_actions(self) -> None:
        self.playback_action = QAction("离线回放", self)
        self.full_statistics_action = QAction("全盘统计", self)
        self.checksum_action = QAction("SHA-256 校验", self)
        self.quality_action = QAction("质量审核", self)
        self.external_import_action = QAction("导入外部模态", self)
        self.recovery_action = QAction("Trial 恢复", self)
        self.upload_action = QAction("人工 SSH/SCP 上传", self)
        self.management_summary_action = QAction("管理摘要", self)
        self.export_inventory_action = QAction("导出筛选清单", self)
        self._restricted_actions = (
            self.playback_action,
            self.full_statistics_action,
            self.checksum_action,
            self.quality_action,
            self.external_import_action,
            self.recovery_action,
            self.upload_action,
        )
        self._management_actions = (
            self.management_summary_action,
            self.export_inventory_action,
        )
        self.playback_action.triggered.connect(self.playback_selected_trial)
        self.full_statistics_action.triggered.connect(self.run_full_statistics)
        self.checksum_action.triggered.connect(self.verify_selected_trial)
        self.quality_action.triggered.connect(self.audit_selected_trial)
        self.external_import_action.triggered.connect(self.import_external_modality)
        self.recovery_action.triggered.connect(self.open_recovery_workflow)
        self.upload_action.triggered.connect(self.upload_selected_trial)
        self.management_summary_action.triggered.connect(self.show_management_summary)
        self.export_inventory_action.triggered.connect(self.export_filtered_inventory)

        toolbar = self.addToolBar("数据工具")
        toolbar.setObjectName("data_tools_toolbar")
        toolbar.setMovable(False)
        for action in self._restricted_actions:
            toolbar.addAction(action)
        toolbar.addSeparator()
        for action in self._management_actions:
            toolbar.addAction(action)

    def _create_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("数据根目录："))
        self.data_root_edit = QLineEdit(str(self._data_root))
        self.data_root_edit.setReadOnly(True)
        self.data_root_edit.setObjectName("data_root")
        root_row.addWidget(self.data_root_edit, 1)
        self.browse_button = QPushButton("选择…")
        self.browse_button.clicked.connect(self.choose_data_root)
        root_row.addWidget(self.browse_button)
        self.refresh_button = QPushButton("刷新 Catalog")
        self.refresh_button.clicked.connect(self.refresh_catalog)
        root_row.addWidget(self.refresh_button)
        self.quick_upload_button = QPushButton("上传所选")
        self.quick_upload_button.setObjectName("quick_upload_selected")
        self.quick_upload_button.clicked.connect(self.upload_selected_trial)
        root_row.addWidget(self.quick_upload_button)
        self.remote_sync_button = QPushButton("同步云端状态")
        self.remote_sync_button.setObjectName("sync_remote_status")
        self.remote_sync_button.clicked.connect(self.sync_remote_status)
        root_row.addWidget(self.remote_sync_button)
        self.remote_settings_button = QPushButton("SSH/SCP 设置…")
        self.remote_settings_button.setObjectName("configure_remote_upload")
        self.remote_settings_button.clicked.connect(self.configure_remote_upload)
        root_row.addWidget(self.remote_settings_button)
        outer.addLayout(root_row)

        self.remote_status_legend = QLabel(
            "Trial 状态灯：● 绿色 已上传   ● 灰色 未上传/尚未同步   "
            "● 橙色 索引待补建   ● 紫色 内容冲突   ·   蓝色徽标表示模态数量"
        )
        self.remote_status_legend.setStyleSheet("QLabel { color: #46566b; }")
        self.remote_status_legend.setToolTip(
            "Trial 云端状态说明：\n"
            "● 绿色：数据已上传，且远端索引与本地验证记录一致\n"
            "● 灰色：未上传，或尚未执行云端状态同步\n"
            "● 橙色：云端已有目录，但 .exo 索引/本地验证缓存尚未对上\n"
            "● 紫色：本地与云端的内容指纹冲突\n"
            "上传并逐文件校验成功后，Trial 会立即显示为绿色。"
        )
        outer.addWidget(self.remote_status_legend)

        self.activity_banner = QLabel()
        self.activity_banner.setObjectName("activity_banner")
        self.activity_banner.setWordWrap(True)
        self.activity_banner.setFrameShape(QFrame.Shape.StyledPanel)
        self.activity_banner.setMargin(8)
        outer.addWidget(self.activity_banner)

        self.filter_group = QGroupBox("Trial 筛选（来自 Manifest 管理索引）")
        self.filter_group.setObjectName("management_filter_group")
        filters = QGridLayout(self.filter_group)
        self.project_filter = QComboBox()
        self.project_filter.setObjectName("filter_project")
        self.subject_filter = QComboBox()
        self.subject_filter.setObjectName("filter_subject")
        self.session_filter = QComboBox()
        self.session_filter.setObjectName("filter_session")
        self.condition_filter = QComboBox()
        self.condition_filter.setObjectName("filter_condition")
        self.quality_filter = QComboBox()
        self.quality_filter.setObjectName("filter_quality")
        for column, (label, widget) in enumerate(
            (
                ("项目：", self.project_filter),
                ("受试者：", self.subject_filter),
                ("Session：", self.session_filter),
                ("工况：", self.condition_filter),
                ("质量：", self.quality_filter),
            )
        ):
            filters.addWidget(QLabel(label), 0, column * 2)
            filters.addWidget(widget, 0, column * 2 + 1)

        self.start_date_enabled = QCheckBox("起始日期")
        self.start_date_enabled.setObjectName("filter_start_enabled")
        self.start_date_edit = QDateEdit(QDate.currentDate())
        self.start_date_edit.setObjectName("filter_start_date")
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_enabled = QCheckBox("结束日期")
        self.end_date_enabled.setObjectName("filter_end_enabled")
        self.end_date_edit = QDateEdit(QDate.currentDate())
        self.end_date_edit.setObjectName("filter_end_date")
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.text_filter = QLineEdit()
        self.text_filter.setObjectName("filter_text")
        self.text_filter.setClearButtonEnabled(True)
        self.text_filter.setPlaceholderText("搜索 UUID、名称、工况或 Manifest 路径")
        filters.addWidget(self.start_date_enabled, 1, 0)
        filters.addWidget(self.start_date_edit, 1, 1)
        filters.addWidget(self.end_date_enabled, 1, 2)
        filters.addWidget(self.end_date_edit, 1, 3)
        filters.addWidget(QLabel("文本："), 1, 4)
        filters.addWidget(self.text_filter, 1, 5, 1, 3)

        self.clear_filters_button = QPushButton("清除筛选")
        self.clear_filters_button.setObjectName("clear_management_filters")
        self.summary_button = QPushButton("管理摘要")
        self.summary_button.setObjectName("management_summary_button")
        self.export_button = QPushButton("导出当前清单")
        self.export_button.setObjectName("management_export_button")
        self.filter_result_label = QLabel("管理索引尚未加载。")
        self.filter_result_label.setObjectName("filter_result_summary")
        filters.addWidget(self.filter_result_label, 2, 0, 1, 5)
        filters.addWidget(self.clear_filters_button, 2, 5)
        filters.addWidget(self.summary_button, 2, 6)
        filters.addWidget(self.export_button, 2, 7, 1, 2)
        outer.addWidget(self.filter_group)

        self._filter_inputs = (
            self.project_filter,
            self.subject_filter,
            self.session_filter,
            self.condition_filter,
            self.quality_filter,
            self.start_date_enabled,
            self.start_date_edit,
            self.end_date_enabled,
            self.end_date_edit,
            self.text_filter,
            self.clear_filters_button,
        )
        for combo in (
            self.project_filter,
            self.subject_filter,
            self.session_filter,
            self.condition_filter,
            self.quality_filter,
        ):
            combo.currentIndexChanged.connect(self._apply_management_filters)
        self.start_date_enabled.toggled.connect(self._apply_management_filters)
        self.end_date_enabled.toggled.connect(self._apply_management_filters)
        self.start_date_edit.dateChanged.connect(self._apply_management_filters)
        self.end_date_edit.dateChanged.connect(self._apply_management_filters)
        self.text_filter.textChanged.connect(self._apply_management_filters)
        self.clear_filters_button.clicked.connect(self.clear_management_filters)
        self.summary_button.clicked.connect(lambda: self.management_summary_action.trigger())
        self.export_button.clicked.connect(lambda: self.export_inventory_action.trigger())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setObjectName("catalog_tree")
        self.tree_widget.setHeaderLabels(["名称", "类型", "模态", "详情"])
        self.tree_widget.setIconSize(QSize(16, 16))
        self.tree_widget.setItemDelegateForColumn(
            2, _ModalityBadgeDelegate(self.tree_widget)
        )
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setUniformRowHeights(True)
        self.tree_widget.setStyleSheet(
            "QTreeWidget#catalog_tree::item { min-height: 24px; }"
        )
        self.tree_widget.header().setStretchLastSection(True)
        self.tree_widget.setColumnWidth(0, 350)
        self.tree_widget.setColumnWidth(1, 90)
        self.tree_widget.setColumnWidth(2, 70)
        splitter.addWidget(self.tree_widget)

        statistics_panel = QGroupBox("基础统计（来自 SQLite Catalog）")
        statistics_layout = QVBoxLayout(statistics_panel)
        cards = QGridLayout()
        self.trial_count_label = QLabel("Trial 总数：0")
        self.finalized_count_label = QLabel("已最终化：0")
        self.duration_label = QLabel("总时长：0.00 s")
        cards.addWidget(self.trial_count_label, 0, 0)
        cards.addWidget(self.finalized_count_label, 0, 1)
        cards.addWidget(self.duration_label, 1, 0, 1, 2)
        statistics_layout.addLayout(cards)
        statistics_layout.addWidget(QLabel("按工况："))
        self.condition_table = QTableWidget(0, 3)
        self.condition_table.setObjectName("condition_statistics")
        self.condition_table.setHorizontalHeaderLabels(["工况代码", "Trial 数", "时长 (s)"])
        self.condition_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.condition_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.condition_table.verticalHeader().setVisible(False)
        self.condition_table.horizontalHeader().setStretchLastSection(True)
        statistics_layout.addWidget(self.condition_table, 1)
        splitter.addWidget(statistics_panel)
        splitter.setSizes([680, 400])
        outer.addWidget(splitter, 1)

        self.scan_summary_label = QLabel("尚未刷新。")
        self.scan_summary_label.setObjectName("scan_summary")
        outer.addWidget(self.scan_summary_label)
        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

    @Slot()
    def choose_data_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择外骨骼数据根目录",
            str(self._data_root),
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.set_data_root(selected)

    def set_data_root(self, data_root: str | Path, *, refresh: bool = True) -> None:
        if (
            self.refresh_in_progress
            or self._active_upload is not None
            or self._process_tasks
            or self._local_tasks
        ):
            raise RuntimeError(
                "Cannot change data root while a Catalog or background task is running"
            )
        self._data_root = self._settings.set_data_root(data_root)
        self.data_root_edit.setText(str(self._data_root))
        self.tree_widget.clear()
        self._catalog_tree = []
        self._remote_status_by_manifest.clear()
        self._automatic_remote_sync_pending = refresh
        self._management_index = None
        self._annex_scan = None
        self._filtered_records = ()
        self._reset_filter_options()
        self._statistics = {}
        self._render_statistics({})
        self._apply_activity(read_activity(self._data_root))
        if refresh:
            self.refresh_catalog()

    @Slot()
    def refresh_catalog(self) -> None:
        if self.refresh_in_progress or self._closing:
            return
        self._apply_activity(read_activity(self._data_root))
        self.refresh_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.scan_summary_label.setText("正在后台迁移 Catalog 并扫描已发布 Manifest…")
        self.statusBar().showMessage("正在刷新；界面保持可响应。")

        try:
            worker = self._process_worker_factory(
                "catalog_refresh",
                data_root=str(self._data_root),
            )
        except Exception:
            self._refresh_failed(traceback.format_exc())
            return
        self._active_task = worker
        self._register_process_worker(
            "Catalog 迁移与 Manifest 索引",
            worker,
            self._refresh_succeeded,
            kind="catalog_refresh",
        )

    @Slot(object)
    def _refresh_succeeded(self, snapshot: object) -> None:
        if self._closing:
            return
        if not isinstance(snapshot, DataStudioSnapshot):
            self._refresh_failed("Catalog worker returned an invalid snapshot")
            return
        self._catalog_tree = deepcopy(snapshot.tree)
        self._management_index = None
        self._annex_scan = None
        self._filtered_records = ()
        self._reset_filter_options()
        self._render_tree(self._catalog_tree)
        self._statistics = dict(snapshot.statistics)
        self._render_statistics(self._statistics)
        self._apply_activity(snapshot.acquisition_activity)
        report = snapshot.scan_report
        summary = f"已索引 {report.indexed} 份正式 Manifest"
        if report.failures:
            summary += f"；{len(report.failures)} 份失败（详情见日志）"
        self._catalog_summary_text = summary
        if snapshot.lightweight_mode:
            self.scan_summary_label.setText(
                summary + "；轻量模式下已暂停管理索引和 annex 完整性扫描。"
            )
            self.filter_result_label.setText("采集期间管理筛选已暂停。")
        else:
            self.scan_summary_label.setText(
                summary + "；正在独立进程加载管理索引并校验 annex…"
            )
            self._start_process_tool(
                "管理索引与 annex 校验",
                "management_refresh",
                self._management_refresh_succeeded,
                kind="management_refresh",
                snapshot=snapshot,
            )
        self.statusBar().showMessage("Catalog 刷新完成。", 5000)
        self._finish_refresh(True)
        if self._automatic_remote_sync_pending:
            QTimer.singleShot(0, self._start_automatic_remote_sync)

    @Slot(str)
    def _refresh_failed(self, details: str) -> None:
        if self._closing:
            return
        final_line = next(
            (line for line in reversed(details.splitlines()) if line.strip()),
            details,
        )
        self.scan_summary_label.setText(f"Catalog 刷新失败：{final_line}")
        self.statusBar().showMessage("Catalog 刷新失败。")
        self._apply_activity(read_activity(self._data_root))
        self._finish_refresh(False)

    def _finish_refresh(self, succeeded: bool) -> None:
        self._active_task = None
        enabled = (
            self._active_upload is None
            and not self._process_tasks
            and not self._local_tasks
        )
        self.refresh_button.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.refresh_finished.emit(succeeded)

    def _management_refresh_succeeded(self, result: object) -> None:
        if not isinstance(result, ManagementRefreshResult):
            raise TypeError("management worker returned an invalid result")
        if result.index.data_root != self._data_root:
            raise ValueError("management worker returned a different data root")
        self._management_index = result.index
        self._annex_scan = result.annex_scan
        self._catalog_tree = self._attach_annex_nodes(
            self._catalog_tree,
            result.annex_scan,
        )
        self._populate_filter_options(result.index.records)
        self._apply_management_filters()
        invalid_annexes = sum(
            item.validation_status is AnnexValidationStatus.INVALID
            for item in result.annex_scan.annexes
        )
        failure_count = (
            len(result.index.catalog_scan_failures)
            + len(result.index.manifest_failures)
            + len(result.annex_scan.scan_failures)
        )
        self.scan_summary_label.setText(
            f"{self._catalog_summary_text}；管理 Trial {len(result.index.records)} 份；"
            f"annex {len(result.annex_scan.annexes)} 份"
            f"（无效 {invalid_annexes}）"
            + (f"；另有 {failure_count} 项扫描失败。" if failure_count else "。")
        )
        self._apply_activity(read_activity(self._data_root))
        self.management_refresh_finished.emit(True)

    def _management_refresh_failed(self, details: str) -> None:
        self._management_index = None
        self._annex_scan = None
        self._filtered_records = ()
        final_line = next(
            (line for line in reversed(details.splitlines()) if line.strip()),
            details,
        )
        self.filter_result_label.setText(f"管理索引加载失败：{final_line}")
        self.scan_summary_label.setText(
            f"{self._catalog_summary_text}；管理索引/annex 校验未完成。"
        )
        self._apply_activity(read_activity(self._data_root))
        self.management_refresh_finished.emit(False)

    def _reset_filter_options(self) -> None:
        if not hasattr(self, "project_filter"):
            return
        self._populating_filters = True
        try:
            for combo in (
                self.project_filter,
                self.subject_filter,
                self.session_filter,
                self.condition_filter,
                self.quality_filter,
            ):
                combo.clear()
                combo.addItem("全部", None)
            self.start_date_enabled.setChecked(False)
            self.end_date_enabled.setChecked(False)
            self.text_filter.clear()
            self.filter_result_label.setText("管理索引尚未加载。")
        finally:
            self._populating_filters = False

    def _populate_filter_options(
        self,
        records: tuple[TrialManagementRecord, ...],
    ) -> None:
        self._reset_filter_options()
        self._populating_filters = True
        try:
            projects = {
                record.project_uuid: (
                    " · ".join(
                        value
                        for value in (record.project_code, record.project_name)
                        if value
                    )
                    or record.project_uuid
                )
                for record in records
            }
            subjects = {
                record.subject_uuid: (
                    f"{record.project_code or record.project_uuid[:8]} / "
                    f"{record.subject_code or record.subject_uuid[:8]}"
                )
                for record in records
            }
            sessions = {
                record.session_uuid: (
                    f"{record.subject_code or record.subject_uuid[:8]} / "
                    f"{record.session_uuid[:8]}"
                )
                for record in records
            }
            conditions = {
                record.condition_code: (
                    f"{record.condition_code} · {record.condition_name}"
                )
                for record in records
            }
            qualities = {record.effective_quality_grade for record in records}
            for value, label in sorted(projects.items(), key=lambda item: item[1]):
                self.project_filter.addItem(label, value)
            for value, label in sorted(subjects.items(), key=lambda item: item[1]):
                self.subject_filter.addItem(label, value)
            for value, label in sorted(sessions.items(), key=lambda item: item[1]):
                self.session_filter.addItem(label, value)
            for value, label in sorted(conditions.items()):
                self.condition_filter.addItem(label, value)
            quality_order = {"A": 0, "B": 1, "C": 2, "INVALID": 3, "UNASSESSED": 4}
            for quality in sorted(
                qualities,
                key=lambda value: (quality_order.get(value, 99), value),
            ):
                self.quality_filter.addItem(quality, quality)
            if records:
                minimum = min(record.started_date for record in records)
                maximum = max(record.started_date for record in records)
                self.start_date_edit.setDate(
                    QDate(minimum.year, minimum.month, minimum.day)
                )
                self.end_date_edit.setDate(
                    QDate(maximum.year, maximum.month, maximum.day)
                )
        finally:
            self._populating_filters = False

    @Slot()
    def clear_management_filters(self, *_args: object) -> None:
        if self._management_index is None:
            return
        self._populating_filters = True
        try:
            for combo in (
                self.project_filter,
                self.subject_filter,
                self.session_filter,
                self.condition_filter,
                self.quality_filter,
            ):
                combo.setCurrentIndex(0)
            self.start_date_enabled.setChecked(False)
            self.end_date_enabled.setChecked(False)
            self.text_filter.clear()
        finally:
            self._populating_filters = False
        self._apply_management_filters()

    def _current_trial_filter(self) -> TrialFilter:
        def selected(combo: QComboBox) -> tuple[str, ...]:
            value = combo.currentData()
            return (str(value),) if value else ()

        start_qdate = self.start_date_edit.date()
        end_qdate = self.end_date_edit.date()
        return TrialFilter(
            projects=selected(self.project_filter),
            subjects=selected(self.subject_filter),
            sessions=selected(self.session_filter),
            conditions=selected(self.condition_filter),
            qualities=selected(self.quality_filter),
            start_date=(
                date(start_qdate.year(), start_qdate.month(), start_qdate.day())
                if self.start_date_enabled.isChecked()
                else None
            ),
            end_date=(
                date(end_qdate.year(), end_qdate.month(), end_qdate.day())
                if self.end_date_enabled.isChecked()
                else None
            ),
            text=self.text_filter.text(),
        )

    @Slot()
    def _apply_management_filters(self, *_args: object) -> None:
        if self._populating_filters or self._management_index is None:
            return
        try:
            criteria = self._current_trial_filter()
        except Exception as exc:
            self._filtered_records = ()
            self.filter_result_label.setText(f"筛选条件无效：{exc}")
            self.filter_result_label.setStyleSheet("color: #b42318;")
            self._apply_activity(read_activity(self._data_root))
            return
        self.filter_result_label.setStyleSheet("")
        self._filtered_records = filter_trial_records(
            self._management_index.records,
            criteria,
        )
        allowed = {record.trial_uuid for record in self._filtered_records}
        self._render_tree(self._filter_catalog_tree(self._catalog_tree, allowed))
        total = len(self._management_index.records)
        total_bytes = sum(record.artifact_total_bytes for record in self._filtered_records)
        total_duration = sum(record.duration_s for record in self._filtered_records)
        self.filter_result_label.setText(
            f"当前显示 {len(self._filtered_records)} / {total} 个 Trial · "
            f"{total_duration:.2f} s · Artifact {total_bytes:,} B"
        )
        self._apply_activity(read_activity(self._data_root))

    @staticmethod
    def _filter_catalog_tree(
        tree: list[dict[str, Any]],
        allowed_trial_uuids: set[str],
    ) -> list[dict[str, Any]]:
        def retained(node: dict[str, Any]) -> dict[str, Any] | None:
            node_type = str(node.get("type", ""))
            if node_type == "trial":
                return deepcopy(node) if str(node.get("uuid")) in allowed_trial_uuids else None
            children = [
                child_result
                for child in node.get("children", [])
                if isinstance(child, dict)
                and (child_result := retained(child)) is not None
            ]
            if node_type in {"project", "subject", "session"} and not children:
                return None
            result = {key: deepcopy(value) for key, value in node.items() if key != "children"}
            result["children"] = children
            return result

        return [
            result
            for node in tree
            if (result := retained(node)) is not None
        ]

    @staticmethod
    def _attach_annex_nodes(
        tree: list[dict[str, Any]],
        annex_scan: AnnexScanResult,
    ) -> list[dict[str, Any]]:
        grouped = annex_scan.by_trial_uuid()
        result = deepcopy(tree)

        def visit(node: dict[str, Any]) -> None:
            if str(node.get("type")) == "trial":
                trial_uuid = str(node.get("uuid"))
                for annex in grouped.get(trial_uuid, ()):
                    integrity = annex.validation_status.value
                    annex_node: dict[str, Any] = {
                        "type": "external_annex",
                        "uuid": annex.annex_uuid,
                        "label": (
                            f"{annex.modality_label or annex.modality or 'external'} · "
                            f"{(annex.annex_uuid or annex.annex_directory.name)[:8]}"
                        ),
                        "validation_status": integrity,
                        "mapping_quality": annex.mapping_quality,
                        "mapping_anchor_count": annex.mapping_anchor_count,
                        "mapping_offset_only": annex.mapping_offset_only,
                        "file_count": annex.file_count,
                        "size_bytes": annex.total_bytes,
                        "annex_manifest_path": str(annex.annex_manifest_path),
                        "errors": list(annex.errors),
                        "children": [
                            {
                                "type": "external_artifact",
                                "uuid": artifact.artifact_uuid,
                                "label": artifact.relative_path,
                                "role": artifact.role,
                                "media_type": artifact.media_type,
                                "size_bytes": artifact.size_bytes,
                                "sha256": artifact.sha256,
                                "validation_status": integrity,
                                "children": [],
                            }
                            for artifact in annex.files
                        ],
                    }
                    node.setdefault("children", []).append(annex_node)
            for child in node.get("children", []):
                if isinstance(child, dict) and child.get("type") != "external_annex":
                    visit(child)

        for root_node in result:
            visit(root_node)
        return result

    def _selected_finalized_manifest_path(self) -> Path | None:
        item = self.tree_widget.currentItem()
        if item is None or item.data(1, Qt.ItemDataRole.UserRole) != "trial":
            QMessageBox.information(self, "请选择 Trial", "请先在数据树中选中一个 Trial。")
            return None
        state = str(item.data(0, Qt.ItemDataRole.UserRole + 2) or "")
        if state != "FINALIZED":
            QMessageBox.warning(
                self,
                "Trial 未最终化",
                f"只允许处理 FINALIZED Trial，当前状态为 {state or '未知'}。",
            )
            return None
        raw_path = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if not raw_path:
            QMessageBox.warning(self, "Manifest 缺失", "Catalog 中没有该 Trial 的 Manifest 路径。")
            return None
        path = Path(str(raw_path))
        if any(
            part.endswith(".recording") or part.endswith(".partial")
            for part in path.parts
        ):
            QMessageBox.critical(
                self,
                "已拒绝读取",
                "Data Studio 不会打开 .recording/.partial 数据。",
            )
            return None
        return path

    def _manifest_paths_below_item(self, root: QTreeWidgetItem) -> tuple[Path, ...]:
        paths: list[Path] = []

        def visit(item: QTreeWidgetItem) -> None:
            node_type = str(item.data(1, Qt.ItemDataRole.UserRole) or "")
            if node_type == "trial":
                state = str(item.data(0, Qt.ItemDataRole.UserRole + 2) or "")
                raw_path = item.data(0, Qt.ItemDataRole.UserRole + 1)
                if state == "FINALIZED" and raw_path:
                    path = Path(str(raw_path)).expanduser().resolve()
                    if not any(
                        part.endswith(".recording") or part.endswith(".partial")
                        for part in path.parts
                    ):
                        paths.append(path)
                return
            for index in range(item.childCount()):
                visit(item.child(index))

        visit(root)
        return tuple(dict.fromkeys(paths))

    def _selected_finalized_manifest_paths(self) -> tuple[Path, ...]:
        item = self.tree_widget.currentItem()
        if item is None:
            QMessageBox.information(
                self, "请选择上传范围", "请在数据树中选择项目、受试者、工况、Session 或 Trial。"
            )
            return ()
        node_type = str(item.data(1, Qt.ItemDataRole.UserRole) or "")
        if node_type not in {"project", "subject", "session", "trial"}:
            QMessageBox.information(
                self, "请选择目录层级", "文件或模态节点不能作为上传范围，请选择其所属 Session 或上级目录。"
            )
            return ()
        paths = self._manifest_paths_below_item(item)
        if not paths:
            QMessageBox.warning(self, "没有可上传数据", "所选层级下没有 FINALIZED Trial。")
        return paths

    def _all_finalized_manifest_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        root = self.tree_widget.invisibleRootItem()
        for index in range(root.childCount()):
            paths.extend(self._manifest_paths_below_item(root.child(index)))
        return tuple(dict.fromkeys(paths))

    def _start_local_tool(
        self,
        name: str,
        operation: Callable[[], object],
        completed: Callable[[object], None],
    ) -> None:
        self._apply_activity(read_activity(self._data_root))
        if self._lightweight_mode:
            self.statusBar().showMessage("采集期间已禁用该工具。", 5000)
            return
        task = LocalToolTask(operation)
        task_id = id(task)
        self._local_tasks[task_id] = task
        task.signals.completed.connect(
            lambda result, key=task_id: self._local_tool_succeeded(
                key, name, result, completed
            )
        )
        task.signals.failed.connect(
            lambda details, key=task_id: self._local_tool_failed(key, name, details)
        )
        self.statusBar().showMessage(f"正在后台执行{name}；界面保持可响应。")
        self._thread_pool.start(task)
        self._apply_activity(read_activity(self._data_root))

    def _start_process_tool(
        self,
        name: str,
        operation: ProcessOperation,
        completed: Callable[[object], None],
        *,
        kind: str = "local_tool",
        **keyword_arguments: object,
    ) -> None:
        self._apply_activity(read_activity(self._data_root))
        if self._lightweight_mode:
            self.statusBar().showMessage("采集期间已禁用该工具。", 5000)
            return
        try:
            worker = self._process_worker_factory(operation, **keyword_arguments)
        except Exception:
            self._local_tool_failed(
                id(keyword_arguments),
                name,
                traceback.format_exc(),
                emit_finished=kind != "management_refresh",
            )
            if kind == "management_refresh":
                self._management_refresh_failed(traceback.format_exc())
            return
        self._register_process_worker(name, worker, completed, kind=kind)

    def _register_process_worker(
        self,
        name: str,
        worker: Any,
        completed: Callable[[object], None],
        *,
        kind: str = "local_tool",
    ) -> None:
        """Register any spawn worker implementing the local polling contract."""

        context = _ProcessTaskContext(
            worker=worker,
            name=name,
            completed=completed,
            kind=kind,
        )
        task_id = id(worker)
        self._process_tasks[task_id] = context
        try:
            worker.start()
        except Exception:
            self._process_tasks.pop(task_id, None)
            details = traceback.format_exc()
            if kind == "catalog_refresh":
                self._refresh_failed(details)
            elif kind == "management_refresh":
                self._management_refresh_failed(details)
            else:
                self._local_tool_failed(task_id, name, details)
            return
        self.statusBar().showMessage(
            f"已启动独立进程执行{name}；GUI 只轮询结果。"
        )
        self._apply_activity(read_activity(self._data_root))

    @Slot()
    def _poll_process_tools(self) -> None:
        self._poll_upload_worker()
        for task_id, context in list(self._process_tasks.items()):
            worker = context.worker
            if not context.handled:
                result = worker.poll_result()
                if result is not None:
                    status, payload = result
                    context.handled = True
                    if status == "completed":
                        if context.kind == "catalog_refresh":
                            try:
                                context.completed(payload)
                            except Exception:
                                self._refresh_failed(traceback.format_exc())
                        else:
                            self._local_tool_succeeded(
                                task_id,
                                context.name,
                                payload,
                                context.completed,
                                remove_task=False,
                                emit_finished=context.kind != "management_refresh",
                            )
                    else:
                        if context.kind == "catalog_refresh":
                            self._refresh_failed(str(payload))
                        elif context.kind == "management_refresh":
                            self._management_refresh_failed(str(payload))
                        else:
                            self._local_tool_failed(
                                task_id,
                                context.name,
                                str(payload),
                                remove_task=False,
                            )
                elif worker.exitcode is not None:
                    # Queue feeder delivery can lag process exit very briefly.
                    context.empty_exit_polls += 1
                    queue_grace_polls = (
                        50
                        if context.kind
                        in {
                            "catalog_refresh",
                            "management_refresh",
                            "management_summary",
                        }
                        else 10
                    )
                    if context.empty_exit_polls >= queue_grace_polls:
                        context.handled = True
                        details = (
                            f"独立进程已退出（exitcode={worker.exitcode}），但未返回结果。"
                        )
                        if context.kind == "catalog_refresh":
                            self._refresh_failed(details)
                        elif context.kind == "management_refresh":
                            self._management_refresh_failed(details)
                        else:
                            self._local_tool_failed(
                                task_id,
                                context.name,
                                details,
                                remove_task=False,
                            )
            if context.handled and not worker.is_alive:
                worker.join(0)
                worker.close()
                self._process_tasks.pop(task_id, None)
                self._apply_activity(read_activity(self._data_root))

    def _poll_upload_worker(self) -> None:
        context = self._active_upload
        if context is None:
            return
        worker = context.worker
        if not context.terminal_handled:
            for event in worker.poll_events():
                if event.event_type is UploadWorkerEventType.PROGRESS:
                    if event.progress is not None:
                        context.progress_dialog.update_progress(event.progress)
                elif event.event_type is UploadWorkerEventType.HOST_KEY_REQUIRED:
                    if event.host_key is None:
                        context.terminal_handled = True
                        self._upload_failed(
                            "INVALID_HOST_KEY_EVENT",
                            "上传 Worker 未返回主机指纹。",
                            silent=context.silent,
                        )
                        self._cancel_active_upload()
                    elif context.silent:
                        _log.warning(
                            "启动自动云端状态同步需要首次确认 SSH 主机指纹，"
                            "已留待用户手动同步。"
                        )
                        self.statusBar().showMessage(
                            "自动同步需要首次确认 SSH 主机指纹；请手动点击‘同步云端状态’。",
                            10000,
                        )
                        self._cancel_active_upload()
                    else:
                        self._confirm_host_key(event.host_key)
                elif event.event_type is UploadWorkerEventType.COMPLETED:
                    context.terminal_handled = True
                    context.progress_dialog.mark_finished()
                    context.progress_dialog.close()
                    if isinstance(event.result, RemoteStatusSyncResult):
                        self._remote_sync_succeeded(event.result, silent=context.silent)
                    elif isinstance(event.result, (OfflineUploadResult, BatchOfflineUploadResult)):
                        self._upload_succeeded(event.result)
                    else:
                        self._upload_failed(
                            "INVALID_RESULT",
                            "远程 Worker 返回了无效结果。",
                            silent=context.silent,
                        )
                elif event.event_type is UploadWorkerEventType.FAILED:
                    context.terminal_handled = True
                    context.progress_dialog.mark_finished()
                    context.progress_dialog.close()
                    self._upload_failed(
                        event.error_code,
                        event.message,
                        silent=context.silent,
                    )

            if not context.terminal_handled and worker.exitcode is not None:
                context.empty_exit_polls += 1
                if context.empty_exit_polls >= 10:
                    context.terminal_handled = True
                    context.progress_dialog.mark_finished()
                    context.progress_dialog.close()
                    self._upload_failed(
                        "WORKER_EXITED",
                        f"上传进程已退出（exitcode={worker.exitcode}），但未返回终态事件。",
                        silent=context.silent,
                    )

        if context.terminal_handled and not worker.is_alive:
            worker.join(0)
            worker.close()
            self._active_upload = None
            self._apply_activity(read_activity(self._data_root))

    @Slot()
    def playback_selected_trial(self) -> None:
        _log.info("=== 离线回放请求开始 ===")
        manifest_path = self._selected_finalized_manifest_path()
        if manifest_path is None:
            _log.warning("回放取消：未选中有效 Trial")
            return
        _log.info("选中 Manifest: %s", manifest_path)
        _log.info("数据根目录: %s", self._data_root)
        self._start_process_tool(
            "离线回放读取",
            "playback",
            self._show_playback,
            manifest_path=str(manifest_path),
            data_root=str(self._data_root),
        )
        _log.info("回放子进程已启动，等待结果…")

    @Slot()
    def run_full_statistics(self) -> None:
        self._start_local_tool(
            "全盘统计",
            lambda: compute_full_statistics(self._data_root),
            self._show_full_statistics,
        )

    @Slot()
    def show_management_summary(self) -> None:
        self._apply_activity(read_activity(self._data_root))
        if self._lightweight_mode:
            QMessageBox.warning(
                self,
                "采集期间禁止管理扫描",
                "Collector 正在采集，工况覆盖和状态证据扫描已暂停。",
            )
            return
        if self._management_index is None:
            QMessageBox.information(
                self,
                "管理索引尚未就绪",
                "请先刷新 Catalog，并等待管理索引与 annex 校验完成。",
            )
            return
        if any(
            context.kind == "management_summary"
            for context in self._process_tasks.values()
        ):
            return
        self._start_process_tool(
            "管理摘要",
            "management_summary",
            self._show_management_summary,
            kind="management_summary",
            data_root=str(self._data_root),
        )

    @Slot()
    def export_filtered_inventory(self) -> None:
        self._apply_activity(read_activity(self._data_root))
        if self._lightweight_mode:
            QMessageBox.warning(
                self,
                "采集期间禁止导出",
                "Collector 正在采集，Manifest 清单导出已暂停。",
            )
            return
        if self._management_index is None:
            QMessageBox.information(
                self,
                "管理索引尚未就绪",
                "请先刷新 Catalog，并等待管理索引完成。",
            )
            return
        if not self._filtered_records:
            QMessageBox.information(self, "没有可导出的 Trial", "当前筛选结果为空。")
            return
        if any(
            context.kind == "management_export"
            for context in self._process_tasks.values()
        ):
            return
        default_name = (
            self._data_root
            / "exports"
            / f"manifest_inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "选择清单文件名前缀（将同时生成 CSV 和 JSON）",
            str(default_name),
            "Manifest 清单 (*.csv *.json);;所有文件 (*)",
        )
        if not selected:
            return
        stem = Path(selected).expanduser().resolve()
        if stem.suffix.casefold() in {".csv", ".json"}:
            stem = stem.with_suffix("")
        csv_path = stem.with_suffix(".csv")
        json_path = stem.with_suffix(".json")
        overwrite = False
        if csv_path.exists() or json_path.exists():
            answer = QMessageBox.question(
                self,
                "清单已存在",
                f"以下目标至少一个已存在：\n{csv_path}\n{json_path}\n\n覆盖两份清单吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        self._start_process_tool(
            "Manifest 清单导出",
            "management_export",
            self._show_inventory_export,
            kind="management_export",
            data_root=str(self._data_root),
            records=self._filtered_records,
            destination_stem=str(stem),
            overwrite=overwrite,
        )

    @Slot()
    def verify_selected_trial(self) -> None:
        manifest_path = self._selected_finalized_manifest_path()
        if manifest_path is None:
            return
        self._start_process_tool(
            "SHA-256 校验",
            "checksum",
            self._show_checksum_report,
            manifest_path=str(manifest_path),
            data_root=str(self._data_root),
        )

    @Slot()
    def audit_selected_trial(self) -> None:
        manifest_path = self._selected_finalized_manifest_path()
        if manifest_path is None:
            return
        self._start_local_tool(
            "质量审核",
            lambda: load_quality_audit(
                manifest_path,
                data_root=self._data_root,
            ),
            self._show_quality_audit,
        )

    @Slot()
    def import_external_modality(self) -> None:
        """Collect generic external-file timing inputs and start an annex worker."""

        self._apply_activity(read_activity(self._data_root))
        if self._lightweight_mode:
            QMessageBox.warning(
                self,
                "采集期间禁止导入",
                "Collector 正在采集，外部模态导入已在轻量模式中禁用。",
            )
            return
        if any(
            context.kind == "external_import"
            for context in self._process_tasks.values()
        ):
            QMessageBox.information(
                self,
                "外部导入进行中",
                "当前已有一个外部模态附录正在生成，请完成后再导入。",
            )
            return
        manifest_path = self._selected_finalized_manifest_path()
        if manifest_path is None:
            return

        dialog = ExternalImportDialog(manifest_path, self)
        request: ExternalImportRequest | None = None
        while dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                request = dialog.take_request(self._data_root)
            except (TypeError, ValueError) as exc:
                QMessageBox.warning(self, "外部导入参数无效", str(exc))
                continue
            break
        dialog.deleteLater()
        if request is None:
            return
        if read_activity(self._data_root) is not None:
            QMessageBox.warning(
                self,
                "采集已开始",
                "Collector 已开始采集，本次外部模态导入未启动。",
            )
            return
        try:
            worker = self._external_import_worker_factory(request)
        except Exception:
            self._local_tool_failed(
                id(request),
                "外部模态导入",
                traceback.format_exc(),
            )
            return
        self._register_process_worker(
            "外部模态导入",
            worker,
            self._show_external_import_result,
            kind="external_import",
        )

    @Slot()
    def open_recovery_workflow(self) -> None:
        """Open evidence-gated recovery; its own service performs spawned scans."""

        self._apply_activity(read_activity(self._data_root))
        if self._lightweight_mode:
            QMessageBox.warning(
                self,
                "采集期间禁止恢复",
                "Collector 正在采集，不会扫描或处置 .recording 数据包。",
            )
            return
        dialog = RecoveryDialog(self._data_root, self)
        dialog.finished.connect(self._recovery_dialog_finished)
        self._show_result_dialog(dialog)

    @Slot(int)
    def _recovery_dialog_finished(self, _result: int) -> None:
        if not self._closing and not self.refresh_in_progress:
            self.refresh_catalog()

    @Slot()
    def upload_selected_trial(self) -> None:
        """Upload every finalized Trial below the selected tree level."""

        self._apply_activity(read_activity(self._data_root))
        if self._lightweight_mode:
            QMessageBox.warning(
                self,
                "采集期间禁止上传",
                "Collector 正在采集，Data Studio 已进入轻量模式。",
            )
            return
        if self._active_upload is not None:
            QMessageBox.information(self, "上传进行中", "当前已有一个 Trial 正在上传。")
            return
        manifest_paths = self._selected_finalized_manifest_paths()
        if not manifest_paths:
            return
        self._start_remote_operation(manifest_paths, status_only=False)

    @Slot()
    def configure_remote_upload(self) -> None:
        """Open endpoint settings explicitly, then upload the selected scope."""

        manifest_paths = self._selected_finalized_manifest_paths()
        if manifest_paths:
            self._start_remote_operation(
                manifest_paths, status_only=False, force_dialog=True
            )

    @Slot()
    def sync_remote_status(self) -> None:
        """Read-only compare every local finalized Trial with remote data/."""

        self._apply_activity(read_activity(self._data_root))
        if self._lightweight_mode:
            QMessageBox.warning(self, "采集期间禁止同步", "Collector 正在采集，云端状态同步未启动。")
            return
        if self._active_upload is not None:
            QMessageBox.information(self, "远程任务进行中", "请等待当前远程任务结束。")
            return
        manifest_paths = self._all_finalized_manifest_paths()
        if not manifest_paths:
            QMessageBox.warning(self, "没有可同步数据", "请先刷新 Catalog；当前没有 FINALIZED Trial。")
            return
        self._start_remote_operation(manifest_paths, status_only=True)

    @Slot()
    def _start_automatic_remote_sync(self) -> None:
        """Silently sync cloud state once after the initial Catalog scan."""

        if not self._automatic_remote_sync_pending or self._closing:
            return
        self._automatic_remote_sync_pending = False
        self._apply_activity(read_activity(self._data_root))
        if self._lightweight_mode or self._active_upload is not None:
            _log.info(
                "启动自动云端状态同步已跳过：lightweight=%s active_remote=%s",
                self._lightweight_mode,
                self._active_upload is not None,
            )
            return
        manifest_paths = self._all_finalized_manifest_paths()
        if not manifest_paths:
            _log.info("启动自动云端状态同步已跳过：没有 FINALIZED Trial。")
            return
        if self._saved_remote_request(
            manifest_paths, status_only=True, quiet=True
        ) is None:
            _log.info("启动自动云端状态同步已跳过：未保存完整凭据。")
            self.statusBar().showMessage(
                "尚未保存完整 SSH/SCP 凭据，本次未自动同步云端状态。",
                8000,
            )
            return
        _log.info("启动后自动同步云端状态：Trial 数=%d", len(manifest_paths))
        self._start_remote_operation(
            manifest_paths,
            status_only=True,
            silent=True,
        )

    def _saved_remote_request(
        self,
        manifest_paths: tuple[Path, ...],
        *,
        status_only: bool,
        quiet: bool = False,
    ) -> OfflineUploadRequest | None:
        endpoint = self._settings.upload_endpoint
        host = str(endpoint.get("host", "")).strip()
        username = str(endpoint.get("username", "")).strip()
        remote_workdir = str(endpoint.get("remote_workdir", "")).strip()
        if not host or not username or not remote_workdir:
            return None
        authentication = str(endpoint.get("authentication", "PASSWORD"))
        password: str | None = None
        private_key_path: Path | None = None
        if authentication == "PRIVATE_KEY":
            raw_key = str(endpoint.get("private_key_path", "")).strip()
            if not raw_key:
                return None
            private_key_path = Path(raw_key)
        else:
            try:
                password = load_password(host, int(endpoint.get("port", 22)), username)
            except RuntimeError as exc:
                if not quiet:
                    QMessageBox.warning(self, "无法读取已保存密码", str(exc))
                else:
                    _log.warning("启动自动同步无法读取已保存密码：%s", exc)
                return None
            if not password:
                return None
        try:
            return OfflineUploadRequest(
                dataset_root=self._data_root,
                manifest_path=manifest_paths[0],
                additional_manifest_paths=manifest_paths[1:],
                operation=(
                    UploadOperation.SYNC_REMOTE_STATUS
                    if status_only
                    else UploadOperation.UPLOAD
                ),
                host=host,
                port=int(endpoint.get("port", 22)),
                username=username,
                remote_workdir=remote_workdir,
                password=password,
                private_key_path=private_key_path,
            )
        except (TypeError, ValueError):
            return None

    def _start_remote_operation(
        self,
        manifest_paths: tuple[Path, ...],
        *,
        status_only: bool,
        force_dialog: bool = False,
        silent: bool = False,
    ) -> None:
        """Collect ephemeral credentials and start one isolated remote worker."""

        request = None if force_dialog else self._saved_remote_request(
            manifest_paths, status_only=status_only, quiet=silent
        )
        if request is None:
            if silent:
                return
            dialog = OfflineUploadDialog(
                manifest_paths,
                self,
                status_only=status_only,
                settings=self._settings,
            )
            while dialog.exec() == QDialog.DialogCode.Accepted:
                try:
                    request = dialog.take_request(self._data_root)
                except (TypeError, ValueError, RuntimeError) as exc:
                    QMessageBox.warning(self, "远程连接参数无效", str(exc))
                    continue
                break
            else:
                dialog.deleteLater()
                return
            dialog.deleteLater()

        # Re-check immediately before process creation; selection and dialog
        # entry may have taken long enough for Collector to become active.
        if read_activity(self._data_root) is not None:
            if not silent:
                QMessageBox.warning(
                    self,
                    "采集已开始",
                    "Collector 已开始采集，本次上传未启动。",
                )
            else:
                _log.info("启动自动云端状态同步已取消：Collector 开始采集。")
            return

        worker = self._upload_worker_factory()
        progress_dialog = UploadProgressDialog(self)
        if status_only:
            progress_dialog.setWindowTitle("同步云端状态")
            progress_dialog.cancel_button.setText("取消状态同步")
        context = _UploadTaskContext(
            worker=worker,
            progress_dialog=progress_dialog,
            silent=silent,
        )
        progress_dialog.cancel_requested.connect(self._cancel_active_upload)
        self._active_upload = context
        self._apply_activity(None)
        if not silent:
            progress_dialog.show()
            progress_dialog.raise_()
        try:
            worker.start(request)
        except Exception as exc:
            self._active_upload = None
            progress_dialog.mark_finished()
            progress_dialog.close()
            worker.terminate_for_shutdown()
            if not worker.is_alive:
                try:
                    worker.close()
                except Exception:
                    pass
            if not silent:
                QMessageBox.critical(self, "远程任务启动失败", str(exc))
            else:
                _log.exception("启动自动云端状态同步失败")
                self.statusBar().showMessage(
                    "自动云端状态同步启动失败；可稍后手动重试。",
                    8000,
                )
            self._apply_activity(read_activity(self._data_root))
            self.upload_finished.emit(False)
            return
        # The request (and credentials) is intentionally not stored by the
        # window. UploadWorkerHandle has already sent it through its memory pipe.
        request = None
        self.statusBar().showMessage(
            "已启动只读云端状态同步进程。" if status_only else
            f"已启动批量 SSH/SCP 上传进程（{len(manifest_paths)} 个 Trial）。"
        )

    @Slot()
    def _cancel_active_upload(self) -> None:
        context = self._active_upload
        if context is None or context.cancel_requested:
            return
        context.cancel_requested = True
        context.progress_dialog.mark_cancelling()
        context.worker.request_cancel()

    def _confirm_host_key(self, host_key: HostKeyInfo) -> None:
        context = self._active_upload
        if context is None:
            return
        context.progress_dialog.waiting_for_host_key()
        answer = QMessageBox.warning(
            self,
            "首次 SSH 主机指纹确认",
            "这是本系统首次连接该 SSH 主机。\n\n"
            f"主机：{host_key.lookup_hostname}\n"
            f"算法：{host_key.algorithm}\n"
            f"SHA-256 指纹：{host_key.sha256_fingerprint}\n\n"
            "请先通过独立渠道与服务器管理员核对。确认信任并保存该指纹吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                context.worker.trust_host_key(host_key)
            except Exception as exc:
                QMessageBox.critical(self, "指纹确认失败", str(exc))
                self._cancel_active_upload()
        else:
            self._cancel_active_upload()

    def _upload_succeeded(
        self, result: OfflineUploadResult | BatchOfflineUploadResult
    ) -> None:
        # The uploader has already verified every remote file and atomically
        # updated both sync indexes. Reflect that terminal result immediately;
        # otherwise the tree keeps showing its stale pre-upload (often orange)
        # state until the operator performs another remote-status scan.
        uploaded_results = (
            result.results if isinstance(result, BatchOfflineUploadResult) else (result,)
        )
        self._mark_uploaded_results_verified(uploaded_results)
        if isinstance(result, BatchOfflineUploadResult):
            QMessageBox.information(
                self,
                "批量上传与校验完成",
                f"Trial：{result.trial_count} 个\n"
                f"文件：{result.file_count} 个，{result.total_bytes:,} B\n\n"
                "本地 data/ 相对目录已完整保留；云端额外内容未删除。",
            )
            self.statusBar().showMessage("批量 SSH/SCP 上传并逐文件校验完成。", 8000)
            self.upload_finished.emit(True)
            self._schedule_catalog_refresh()
            return
        QMessageBox.information(
            self,
            "上传与校验完成",
            f"Trial：{result.trial_uuid}\n"
            f"远程目录：{result.remote_trial_directory}\n"
            f"文件：{result.file_count} 个，{result.total_bytes:,} B\n"
            f"审计记录：{result.audit_record_path}",
        )
        self.statusBar().showMessage("人工 SSH/SCP 上传并逐文件校验完成。", 8000)
        self.upload_finished.emit(True)
        self._schedule_catalog_refresh()

    def _mark_uploaded_results_verified(
        self, results: tuple[OfflineUploadResult, ...]
    ) -> None:
        uploaded_by_uuid = {str(item.trial_uuid): item for item in results}

        def visit(nodes: list[dict[str, Any]]) -> None:
            for node in nodes:
                if str(node.get("type", "")) == "trial":
                    uploaded = uploaded_by_uuid.get(str(node.get("uuid", "")))
                    raw_manifest = node.get("manifest_path")
                    if uploaded is not None and raw_manifest:
                        manifest_key = str(
                            Path(str(raw_manifest)).expanduser().resolve()
                        )
                        self._remote_status_by_manifest[manifest_key] = (
                            RemoteTrialStatus.UPLOADED,
                            "本次上传已完成逐文件 SHA-256 校验，"
                            "远端 .exo 索引与本地验证缓存已更新。",
                        )
                children = node.get("children", [])
                if isinstance(children, list):
                    visit(children)

        visit(self._catalog_tree)
        visible_tree = self._catalog_tree
        if self._management_index is not None:
            visible_tree = self._filter_catalog_tree(
                self._catalog_tree,
                {record.trial_uuid for record in self._filtered_records},
            )
        self._render_tree(visible_tree)

    def _remote_sync_succeeded(
        self, result: RemoteStatusSyncResult, *, silent: bool = False
    ) -> None:
        self._remote_status_by_manifest = {
            str(record.manifest_path.expanduser().resolve()): (record.status, record.detail)
            for record in result.records
        }
        visible_tree = self._catalog_tree
        if self._management_index is not None:
            visible_tree = self._filter_catalog_tree(
                self._catalog_tree,
                {record.trial_uuid for record in self._filtered_records},
            )
        self._render_tree(visible_tree)
        counts = {status: 0 for status in RemoteTrialStatus}
        for record in result.records:
            counts[record.status] += 1
        if not silent:
            QMessageBox.information(
                self,
                "云端状态同步完成",
                f"已核对 {len(result.records)} 个 Trial。\n"
                f"已上传：{counts[RemoteTrialStatus.UPLOADED]}\n"
                f"未上传：{counts[RemoteTrialStatus.NOT_UPLOADED]}\n"
                f"部分缺失：{counts[RemoteTrialStatus.PARTIAL]}\n"
                f"内容冲突：{counts[RemoteTrialStatus.CONFLICT]}",
            )
        self.statusBar().showMessage(
            "启动自动云端状态同步完成。"
            if silent
            else "云端 data/ 状态同步完成。",
            8000,
        )
        _log.info(
            "云端状态同步完成：automatic=%s uploaded=%d missing=%d partial=%d conflict=%d",
            silent,
            counts[RemoteTrialStatus.UPLOADED],
            counts[RemoteTrialStatus.NOT_UPLOADED],
            counts[RemoteTrialStatus.PARTIAL],
            counts[RemoteTrialStatus.CONFLICT],
        )
        self.upload_finished.emit(True)

    def _upload_failed(
        self,
        error_code: str | None,
        message: str | None,
        *,
        silent: bool = False,
    ) -> None:
        safe_message = message or "上传 Worker 未返回错误详情。"
        if not silent:
            QMessageBox.critical(
                self,
                "人工 SSH/SCP 上传失败",
                f"错误代码：{error_code or 'UNKNOWN'}\n{safe_message}\n\n"
                "可在确认 Collector 未采集、远程空间与网络后重新执行上传。",
            )
        self.statusBar().showMessage(
            "自动云端状态同步失败；本地数据仍可正常使用。"
            if silent
            else "人工 SSH/SCP 上传失败，可重试。",
            8000,
        )
        if silent:
            _log.warning(
                "启动自动云端状态同步失败：code=%s message=%s",
                error_code or "UNKNOWN",
                safe_message,
            )
        self.upload_finished.emit(False)

    def _local_tool_succeeded(
        self,
        task_id: int,
        name: str,
        result: object,
        completed: Callable[[object], None],
        *,
        remove_task: bool = True,
        emit_finished: bool = True,
    ) -> None:
        if remove_task:
            self._local_tasks.pop(task_id, None)
            self._apply_activity(read_activity(self._data_root))
        if self._closing:
            return
        try:
            completed(result)
        except Exception:
            self._local_tool_failed(
                task_id,
                name,
                traceback.format_exc(),
                remove_task=remove_task,
                emit_finished=emit_finished,
            )
            return
        self.statusBar().showMessage(f"{name}完成。", 5000)
        if emit_finished:
            self.local_tool_finished.emit(name, True)

    def _local_tool_failed(
        self,
        task_id: int,
        name: str,
        details: str,
        *,
        remove_task: bool = True,
        emit_finished: bool = True,
    ) -> None:
        if remove_task:
            self._local_tasks.pop(task_id, None)
            self._apply_activity(read_activity(self._data_root))
        if self._closing:
            return
        final_line = next(
            (line for line in reversed(details.splitlines()) if line.strip()), details
        )
        self.statusBar().showMessage(f"{name}失败。", 5000)
        QMessageBox.critical(self, f"{name}失败", final_line)
        if emit_finished:
            self.local_tool_finished.emit(name, False)

    def _show_result_dialog(self, dialog: QDialog) -> None:
        _log.info("准备显示对话框: %s", dialog.objectName())
        self._result_dialogs.append(dialog)
        dialog.finished.connect(
            lambda _result, current=dialog: self._forget_result_dialog(current)
        )
        try:
            dialog.show()
        except Exception:
            _log.exception("dialog.show() 崩溃")
            raise
        _log.info("对话框已显示")
        dialog.raise_()
        dialog.activateWindow()

    def _forget_result_dialog(self, dialog: QDialog) -> None:
        if dialog in self._result_dialogs:
            self._result_dialogs.remove(dialog)
        dialog.deleteLater()

    def _show_playback(self, result: object) -> None:
        _log.info("回放数据已就绪，创建 PlaybackDialog…")
        if not isinstance(result, TrialPlayback):
            _log.error("回放 worker 返回了无效结果：%s", type(result))
            raise TypeError("playback worker returned an invalid result")
        _log.info("Trial UUID: %s, US: %s, IMU: %s, Encoder: %s",
                  result.trial_uuid,
                  result.ultrasound is not None,
                  result.imu is not None,
                  result.encoder is not None)
        try:
            dialog = PlaybackDialog(result, self)
        except Exception:
            _log.exception("创建 PlaybackDialog 失败")
            raise
        _log.info("PlaybackDialog 创建成功，显示中…")
        self._show_result_dialog(dialog)

    def _show_full_statistics(self, result: object) -> None:
        if not isinstance(result, FullStatistics):
            raise TypeError("statistics worker returned an invalid result")
        self._show_result_dialog(FullStatisticsDialog(result, self))

    def _show_management_summary(self, result: object) -> None:
        if not isinstance(result, ManagementSummaryResult):
            raise TypeError("management summary worker returned an invalid result")
        self._show_result_dialog(ManagementSummaryDialog(result, self))

    def _show_inventory_export(self, result: object) -> None:
        if not isinstance(result, InventoryExportResult):
            raise TypeError("inventory export worker returned an invalid result")
        QMessageBox.information(
            self,
            "Manifest 清单导出完成",
            f"已导出 {result.record_count} 个 Trial。\n\n"
            f"CSV：{result.csv_path}\n"
            f"JSON：{result.json_path}\n\n"
            "原始 Trial 数据未被修改。",
        )

    def _show_checksum_report(self, result: object) -> None:
        if not isinstance(result, ChecksumReport):
            raise TypeError("checksum worker returned an invalid result")
        self._show_result_dialog(ChecksumDialog(result, self))

    def _show_quality_audit(self, result: object) -> None:
        if not isinstance(result, QualityAudit):
            raise TypeError("quality worker returned an invalid result")
        review_root = self._data_root
        manifest_path = result.manifest_path

        def submit_review(grade: str, reviewer: str, reason: str) -> QualityAudit:
            append_quality_review(
                review_root,
                manifest_path,
                reviewed_grade=grade,
                reviewer=reviewer,
                reason=reason,
            )
            self._schedule_catalog_refresh()
            return load_quality_audit(manifest_path, data_root=review_root)

        self._show_result_dialog(
            QualityAuditDialog(result, self, review_submit=submit_review)
        )

    def _show_external_import_result(self, result: object) -> None:
        if not isinstance(result, ExternalImportResult):
            raise TypeError("external import worker returned an invalid result")
        mapping_kind = "单脉冲偏移" if result.offset_only else "仿射时钟映射"
        QMessageBox.information(
            self,
            "外部模态导入完成",
            f"Trial：{result.trial_uuid}\n"
            f"附录目录：{result.annex_directory}\n"
            f"同步质量：{result.quality}\n"
            f"映射：{mapping_kind}，{result.anchor_count} 个脉冲\n\n"
            "已有 FINALIZED Trial、Manifest 和原始数据均未改写。",
        )
        self._schedule_catalog_refresh()

    def _schedule_catalog_refresh(self) -> None:
        if self._closing:
            return

        def refresh_when_idle() -> None:
            if self._closing:
                return
            if (
                self.refresh_in_progress
                or self._active_upload is not None
                or self._process_tasks
                or self._local_tasks
            ):
                QTimer.singleShot(100, refresh_when_idle)
                return
            self.refresh_catalog()

        QTimer.singleShot(0, refresh_when_idle)

    def _apply_activity(self, activity: AcquisitionActivity | None) -> None:
        self._lightweight_mode = activity is not None
        for dialog in tuple(self._result_dialogs):
            if isinstance(dialog, RecoveryDialog):
                dialog.set_acquisition_activity(activity)
        external_import_busy = any(
            context.kind == "external_import"
            for context in self._process_tasks.values()
        )
        management_refresh_busy = any(
            context.kind == "management_refresh"
            for context in self._process_tasks.values()
        )
        management_summary_busy = any(
            context.kind == "management_summary"
            for context in self._process_tasks.values()
        )
        management_export_busy = any(
            context.kind == "management_export"
            for context in self._process_tasks.values()
        )
        if activity is not None and self._active_upload is not None:
            self._cancel_active_upload()
        if activity is None:
            self.activity_banner.setText(
                "完整模式：当前数据根目录未检测到 Collector 活动采集。"
            )
            self.activity_banner.setStyleSheet(
                "QLabel { background: #e9f7ef; color: #155724; border: 1px solid #a9dfbf; }"
            )
        elif activity.pid <= 0 or activity.hostname == "unreadable-lock":
            self.activity_banner.setText(
                "轻量模式：活动锁不可安全解析，已保守进入轻量模式。"
                "已暂停回放、统计、校验、质控审核、外部导入、恢复、管理扫描、"
                "清单导出和上传；Catalog 浏览仍可用。"
            )
            self.activity_banner.setStyleSheet(
                "QLabel { background: #fff3cd; color: #664d03; border: 1px solid #ffecb5; }"
            )
        else:
            trial = f"，Trial {activity.trial_uuid}" if activity.trial_uuid else ""
            self.activity_banner.setText(
                "轻量模式：检测到 Collector 正在采集"
                f"（PID {activity.pid}{trial}）。已暂停回放、统计、校验、质控审核、"
                "外部导入、恢复、管理扫描、清单导出和上传；Catalog 浏览仍可用。"
            )
            self.activity_banner.setStyleSheet(
                "QLabel { background: #fff3cd; color: #664d03; border: 1px solid #ffecb5; }"
            )
        for action in self._restricted_actions:
            enabled = not self._lightweight_mode and self._active_upload is None
            if action is self.external_import_action and external_import_busy:
                enabled = False
            action.setEnabled(enabled)
            if self._lightweight_mode:
                action.setToolTip("采集活动期间已禁用（轻量模式）")
            else:
                tooltips = {
                    self.playback_action: "选中 FINALIZED Trial 后进行降采样离线回放",
                    self.full_statistics_action: "后台刷新 Manifest/Catalog 并统计全部数据",
                    self.checksum_action: "选中 FINALIZED Trial 后重算 SHA-256",
                    self.quality_action: "读取已发布质控报告、设备与同步摘要",
                    self.external_import_action: (
                        "将测力台/动捕/其他文件导入为独立校验附录，不改写 Trial"
                    ),
                    self.recovery_action: (
                        "只读扫描 .recording，并执行证据约束的恢复/中止"
                    ),
                    self.upload_action: "上传所选层级下的全部 FINALIZED Trial，并保留 data/ 目录结构",
                }
                action.setToolTip(tooltips[action])
        management_ready = self._management_index is not None
        self.management_summary_action.setEnabled(
            not self._lightweight_mode
            and self._active_upload is None
            and management_ready
            and not management_refresh_busy
            and not management_summary_busy
        )
        self.export_inventory_action.setEnabled(
            not self._lightweight_mode
            and self._active_upload is None
            and management_ready
            and bool(self._filtered_records)
            and not management_refresh_busy
            and not management_export_busy
        )
        self.management_summary_action.setToolTip(
            "后台统计每受试者工况覆盖，并验证恢复/质检/上传状态"
        )
        self.export_inventory_action.setToolTip(
            "把当前筛选的 Manifest 清单原子导出为 CSV 与 JSON"
        )
        filters_enabled = (
            not self._lightweight_mode
            and management_ready
            and not management_refresh_busy
        )
        for widget in self._filter_inputs:
            widget.setEnabled(filters_enabled)
        self.summary_button.setEnabled(self.management_summary_action.isEnabled())
        self.export_button.setEnabled(self.export_inventory_action.isEnabled())
        self.upload_action.setEnabled(
            not self._lightweight_mode and self._active_upload is None
        )
        root_controls_enabled = (
            self._active_upload is None
            and not self.refresh_in_progress
            and not self._process_tasks
            and not self._local_tasks
        )
        self.browse_button.setEnabled(root_controls_enabled)
        self.refresh_button.setEnabled(root_controls_enabled)
        self.quick_upload_button.setEnabled(
            root_controls_enabled and not self._lightweight_mode and bool(self._catalog_tree)
        )
        self.remote_sync_button.setEnabled(
            root_controls_enabled and not self._lightweight_mode and bool(self._catalog_tree)
        )
        self.remote_settings_button.setEnabled(
            root_controls_enabled and not self._lightweight_mode and bool(self._catalog_tree)
        )

    @Slot()
    def _poll_activity(self) -> None:
        """Follow Collector lock changes even when the Catalog is not refreshed."""

        if not self._closing:
            self._apply_activity(read_activity(self._data_root))

    def _render_tree(self, tree: list[dict[str, Any]]) -> None:
        self.tree_widget.clear()
        for node in tree:
            self.tree_widget.addTopLevelItem(self._make_tree_item(node))
        self.tree_widget.expandToDepth(2)

    def _make_tree_item(self, node: dict[str, Any]) -> QTreeWidgetItem:
        node_type = str(node.get("type", ""))
        details = self._node_details(node_type, node)
        item = QTreeWidgetItem(
            [
                str(node.get("label", "")),
                _TYPE_LABELS.get(node_type, node_type),
                "",
                details,
            ]
        )
        item.setData(0, Qt.ItemDataRole.UserRole, node.get("uuid"))
        item.setData(1, Qt.ItemDataRole.UserRole, node_type)
        if node_type == "trial":
            item.setData(
                0,
                Qt.ItemDataRole.UserRole + 1,
                node.get("manifest_path"),
            )
            item.setData(0, Qt.ItemDataRole.UserRole + 2, node.get("state"))
            modality_count = int(node.get("modality_count") or 0)
            item.setText(2, f"{modality_count} 个")
            item.setTextAlignment(2, Qt.AlignmentFlag.AlignCenter)
            raw_manifest = node.get("manifest_path")
            remote_state = (
                self._remote_status_by_manifest.get(
                    str(Path(str(raw_manifest)).expanduser().resolve())
                )
                if raw_manifest
                else None
            )
            status = remote_state[0] if remote_state is not None else None
            remote_detail = remote_state[1] if remote_state is not None else "尚未同步云端状态"
            colors = {
                RemoteTrialStatus.UPLOADED: "#20a35a",
                RemoteTrialStatus.NOT_UPLOADED: "#9aa3ad",
                RemoteTrialStatus.PARTIAL: "#e39a22",
                RemoteTrialStatus.CONFLICT: "#8e44ad",
                None: "#9aa3ad",
            }
            row_backgrounds = {
                RemoteTrialStatus.UPLOADED: "#e3f5e9",
                RemoteTrialStatus.NOT_UPLOADED: "#f1f3f5",
                RemoteTrialStatus.PARTIAL: "#fff0d6",
                RemoteTrialStatus.CONFLICT: "#f3e5f5",
                None: "#f1f3f5",
            }
            labels = {
                RemoteTrialStatus.UPLOADED: "已上传",
                RemoteTrialStatus.NOT_UPLOADED: "未上传",
                RemoteTrialStatus.PARTIAL: "索引待补建",
                RemoteTrialStatus.CONFLICT: "内容冲突",
                None: "尚未同步",
            }
            item.setIcon(0, self._status_light_icon(colors[status]))
            tooltip = (
                f"当前状态：{labels[status]}\n"
                f"判定原因：{remote_detail}\n\n"
                "颜色说明：绿=已上传，灰=未上传/未同步，"
                "橙=索引待补建，紫=内容冲突"
            )
            for column in range(4):
                item.setToolTip(column, tooltip)
                item.setBackground(column, QBrush(QColor(row_backgrounds[status])))
            item.setText(3, f"{details} · 云端：{labels[status]}")
        elif node_type == "external_annex":
            item.setData(
                0,
                Qt.ItemDataRole.UserRole + 1,
                node.get("annex_manifest_path"),
            )
        errors = node.get("errors")
        if isinstance(errors, list) and errors:
            item.setToolTip(3, "\n".join(str(error) for error in errors))
        children = node.get("children", [])
        if node_type == "trial":
            children = self._group_trial_artifacts(children)
        for child in children:
            if isinstance(child, dict):
                item.addChild(self._make_tree_item(child))
        return item

    @staticmethod
    def _status_light_icon(color: str) -> QIcon:
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor("#5f6b76"), 1))
        painter.setBrush(QBrush(QColor(color)))
        painter.drawEllipse(2, 2, 12, 12)
        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _group_trial_artifacts(children: object) -> list[dict[str, Any]]:
        """Group Manifest artifacts for display without changing their identity.

        The Catalog remains a flat UUID-linked artifact index.  Group nodes are
        presentation-only and therefore cannot accidentally become a source of
        truth for playback, upload, recovery, or integrity checks.
        """

        if not isinstance(children, list):
            return []
        grouped: dict[str, list[dict[str, Any]]] = {}
        passthrough: list[dict[str, Any]] = []
        for child in children:
            if not isinstance(child, dict):
                continue
            if child.get("type") != "artifact":
                passthrough.append(child)
                continue
            modality = str(child.get("modality") or "unknown")
            grouped.setdefault(modality, []).append(child)

        result: list[dict[str, Any]] = []
        for modality, artifacts in sorted(
            grouped.items(), key=lambda item: (item[0] == "trial", item[0])
        ):
            supporting = modality == "trial"
            result.append(
                {
                    "type": "supporting_files" if supporting else "modality",
                    "uuid": None,
                    "label": (
                        "系统资料"
                        if supporting
                        else _MODALITY_LABELS.get(modality, modality)
                    ),
                    "modality": modality,
                    "artifact_count": len(artifacts),
                    "size_bytes": sum(
                        int(artifact.get("size_bytes") or 0)
                        for artifact in artifacts
                    ),
                    "children": artifacts,
                }
            )
        result.extend(passthrough)
        return result

    @staticmethod
    def _node_details(node_type: str, node: dict[str, Any]) -> str:
        if node_type == "trial":
            duration = float(node.get("duration_s") or 0.0)
            quality = node.get("quality_grade") or "-"
            return f"{duration:.2f} s | 质量 {quality}"
        if node_type in {"modality", "supporting_files"}:
            count = int(node.get("artifact_count") or 0)
            size = int(node.get("size_bytes") or 0)
            return f"{count} 个文件 | {size:,} B"
        if node_type == "artifact":
            size = int(node.get("size_bytes") or 0)
            modality = node.get("modality") or "-"
            return f"{modality} | {size:,} B"
        if node_type == "external_annex":
            size = int(node.get("size_bytes") or 0)
            integrity = node.get("validation_status") or "UNKNOWN"
            quality = node.get("mapping_quality") or "-"
            anchors = int(node.get("mapping_anchor_count") or 0)
            return (
                f"完整性 {integrity} | 映射 {quality} / {anchors} 脉冲 | "
                f"{size:,} B"
            )
        if node_type == "external_artifact":
            size = int(node.get("size_bytes") or 0)
            role = node.get("role") or "external"
            media_type = node.get("media_type") or "application/octet-stream"
            integrity = node.get("validation_status") or "UNKNOWN"
            return f"{role} | {media_type} | {size:,} B | {integrity}"
        return ""

    def _render_statistics(self, statistics: dict[str, Any]) -> None:
        trial_count = int(statistics.get("trial_count") or 0)
        finalized_count = int(statistics.get("finalized_count") or 0)
        duration = float(statistics.get("total_duration_s") or 0.0)
        self.trial_count_label.setText(f"Trial 总数：{trial_count}")
        self.finalized_count_label.setText(f"已最终化：{finalized_count}")
        self.duration_label.setText(f"总时长：{duration:.2f} s")

        by_condition = statistics.get("by_condition") or {}
        rows = sorted(by_condition.items()) if isinstance(by_condition, dict) else []
        self.condition_table.setRowCount(len(rows))
        for row, (condition_code, values) in enumerate(rows):
            values = values if isinstance(values, dict) else {}
            count = int(values.get("trial_count") or 0)
            condition_duration = float(values.get("duration_s") or 0.0)
            self.condition_table.setItem(row, 0, QTableWidgetItem(str(condition_code)))
            self.condition_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.condition_table.setItem(row, 2, QTableWidgetItem(f"{condition_duration:.2f}"))
        self.condition_table.resizeColumnsToContents()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if not self._closing:
            self._closing = True
            self._activity_timer.stop()
            self._process_timer.stop()
            # Cancel QRunnables that have not started. A running QRunnable
            # cannot safely be killed, so the window remains alive and retries
            # close without blocking the GUI until it finishes.
            self._thread_pool.clear()
            for dialog in list(self._result_dialogs):
                dialog.close()

        pending_shutdown = False
        if self._active_upload is not None:
            upload_context = self._active_upload
            try:
                upload_context.worker.request_cancel()
                upload_context.worker.join(0.25)
                if upload_context.worker.is_alive:
                    upload_context.worker.terminate_for_shutdown()
                upload_context.worker.join(0)
            except Exception as exc:
                self.statusBar().showMessage(
                    f"关闭上传进程失败，正在重试：{type(exc).__name__}: {exc}"
                )
            if upload_context.worker.is_alive:
                pending_shutdown = True
            else:
                try:
                    upload_context.worker.close()
                except Exception as exc:
                    self.statusBar().showMessage(
                        f"释放上传进程失败，正在重试：{type(exc).__name__}: {exc}"
                    )
                    pending_shutdown = True
                else:
                    upload_context.progress_dialog.mark_finished()
                    upload_context.progress_dialog.close()
                    self._active_upload = None

        for task_id, context in list(self._process_tasks.items()):
            worker = context.worker
            try:
                worker.terminate(timeout=0.5)
            except Exception as exc:
                self.statusBar().showMessage(
                    f"关闭 {context.name} 进程失败，正在重试："
                    f"{type(exc).__name__}: {exc}"
                )
            if worker.is_alive:
                pending_shutdown = True
                continue
            try:
                worker.join(0)
                worker.close()
            except Exception as exc:
                self.statusBar().showMessage(
                    f"释放 {context.name} 进程失败，正在重试："
                    f"{type(exc).__name__}: {exc}"
                )
                pending_shutdown = True
            else:
                self._process_tasks.pop(task_id, None)

        threads_done = self._thread_pool.waitForDone(0)
        if pending_shutdown or not threads_done:
            event.ignore()
            if self._close_started_at is None:
                self._close_started_at = time.monotonic()
            # A running QRunnable cannot be terminated safely.  Never destroy
            # its owning QObject merely because a wall-clock deadline elapsed:
            # the task may still emit a signal or execute Python/C-extension
            # code.  Keep the window alive and retry asynchronously until all
            # threads and child processes have actually stopped.
            if not self._shutdown_retry_pending:
                self._shutdown_retry_pending = True

                def retry_close() -> None:
                    self._shutdown_retry_pending = False
                    self.close()

                QTimer.singleShot(100, retry_close)
            return
        self._active_task = None
        self._local_tasks.clear()
        self._close_started_at = None
        event.accept()
