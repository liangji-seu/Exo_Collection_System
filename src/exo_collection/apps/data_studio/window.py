"""PySide6 main window for local Trial browsing and basic statistics."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from exo_collection.storage.activity import AcquisitionActivity, read_activity

from .service import DataStudioSnapshot, load_catalog_snapshot


_TYPE_LABELS = {
    "project": "Project",
    "subject": "Subject",
    "session": "Session",
    "trial": "Trial",
    "artifact": "Artifact",
}


class _RefreshSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class CatalogRefreshTask(QRunnable):
    """Run migration and Manifest indexing outside the GUI thread."""

    def __init__(self, data_root: Path) -> None:
        super().__init__()
        self.data_root = data_root
        self.signals = _RefreshSignals()

    @Slot()
    def run(self) -> None:
        try:
            snapshot = load_catalog_snapshot(self.data_root)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.completed.emit(snapshot)


class DataStudioWindow(QMainWindow):
    """Catalog-backed Data Studio shell for the first runnable milestone."""

    refresh_finished = Signal(bool)

    def __init__(
        self,
        data_root: str | Path,
        *,
        autostart_refresh: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._data_root = Path(data_root).expanduser().resolve()
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)
        self._active_task: CatalogRefreshTask | None = None
        self._closing = False
        self._statistics: dict[str, Any] = {}
        self._lightweight_mode = False

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
        self.playback_action = QAction("离线回放（占位）", self)
        self.full_statistics_action = QAction("全盘统计（占位）", self)
        self.checksum_action = QAction("SHA-256 校验（占位）", self)
        self.upload_action = QAction("人工 SSH/SCP 上传（占位）", self)
        self._restricted_actions = (
            self.playback_action,
            self.full_statistics_action,
            self.checksum_action,
            self.upload_action,
        )
        for action in self._restricted_actions:
            action.triggered.connect(
                lambda _checked=False, name=action.text(): self.statusBar().showMessage(
                    f"{name} 将在后续里程碑实现。", 5000
                )
            )

        toolbar = self.addToolBar("数据工具")
        toolbar.setObjectName("data_tools_toolbar")
        toolbar.setMovable(False)
        for action in self._restricted_actions:
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
        outer.addLayout(root_row)

        self.activity_banner = QLabel()
        self.activity_banner.setObjectName("activity_banner")
        self.activity_banner.setWordWrap(True)
        self.activity_banner.setFrameShape(QFrame.Shape.StyledPanel)
        self.activity_banner.setMargin(8)
        outer.addWidget(self.activity_banner)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setObjectName("catalog_tree")
        self.tree_widget.setHeaderLabels(["名称", "类型", "详情"])
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setUniformRowHeights(True)
        self.tree_widget.header().setStretchLastSection(True)
        self.tree_widget.setColumnWidth(0, 370)
        self.tree_widget.setColumnWidth(1, 90)
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
        if self.refresh_in_progress:
            raise RuntimeError("Cannot change data root while Catalog refresh is running")
        self._data_root = Path(data_root).expanduser().resolve()
        self.data_root_edit.setText(str(self._data_root))
        self.tree_widget.clear()
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

        task = CatalogRefreshTask(self._data_root)
        self._active_task = task
        task.signals.completed.connect(self._refresh_succeeded)
        task.signals.failed.connect(self._refresh_failed)
        self._thread_pool.start(task)

    @Slot(object)
    def _refresh_succeeded(self, snapshot: object) -> None:
        if self._closing:
            return
        if not isinstance(snapshot, DataStudioSnapshot):
            self._refresh_failed("Catalog worker returned an invalid snapshot")
            return
        self._render_tree(snapshot.tree)
        self._statistics = dict(snapshot.statistics)
        self._render_statistics(self._statistics)
        self._apply_activity(snapshot.acquisition_activity)
        report = snapshot.scan_report
        summary = f"已索引 {report.indexed} 份正式 Manifest"
        if report.failures:
            summary += f"；{len(report.failures)} 份失败（详情见日志）"
        self.scan_summary_label.setText(summary + "。")
        self.statusBar().showMessage("Catalog 刷新完成。", 5000)
        self._finish_refresh(True)

    @Slot(str)
    def _refresh_failed(self, details: str) -> None:
        if self._closing:
            return
        final_line = next((line for line in reversed(details.splitlines()) if line.strip()), details)
        self.scan_summary_label.setText(f"Catalog 刷新失败：{final_line}")
        self.statusBar().showMessage("Catalog 刷新失败。")
        self._apply_activity(read_activity(self._data_root))
        self._finish_refresh(False)

    def _finish_refresh(self, succeeded: bool) -> None:
        self._active_task = None
        self.refresh_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.refresh_finished.emit(succeeded)

    def _apply_activity(self, activity: AcquisitionActivity | None) -> None:
        self._lightweight_mode = activity is not None
        if activity is None:
            self.activity_banner.setText(
                "完整模式：当前数据根目录未检测到 Collector 活动采集。"
            )
            self.activity_banner.setStyleSheet(
                "QLabel { background: #e9f7ef; color: #155724; border: 1px solid #a9dfbf; }"
            )
        else:
            trial = f"，Trial {activity.trial_uuid}" if activity.trial_uuid else ""
            self.activity_banner.setText(
                "轻量模式：检测到 Collector 正在采集"
                f"（PID {activity.pid}{trial}）。已暂停大型回放、全盘统计、"
                "SHA-256 重算和上传；Catalog 浏览仍可用。"
            )
            self.activity_banner.setStyleSheet(
                "QLabel { background: #fff3cd; color: #664d03; border: 1px solid #ffecb5; }"
            )
        for action in self._restricted_actions:
            action.setEnabled(not self._lightweight_mode)
            if self._lightweight_mode:
                action.setToolTip("采集活动期间已禁用（轻量模式）")
            else:
                action.setToolTip("第一个里程碑的功能占位")

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
                details,
            ]
        )
        item.setData(0, Qt.ItemDataRole.UserRole, node.get("uuid"))
        item.setData(1, Qt.ItemDataRole.UserRole, node_type)
        for child in node.get("children", []):
            if isinstance(child, dict):
                item.addChild(self._make_tree_item(child))
        return item

    @staticmethod
    def _node_details(node_type: str, node: dict[str, Any]) -> str:
        if node_type == "trial":
            duration = float(node.get("duration_s") or 0.0)
            quality = node.get("quality_grade") or "-"
            return f"{duration:.2f} s | 质量 {quality}"
        if node_type == "artifact":
            size = int(node.get("size_bytes") or 0)
            modality = node.get("modality") or "-"
            return f"{modality} | {size:,} B"
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
        self._closing = True
        self._activity_timer.stop()
        # A refresh is metadata-only and expected to finish quickly.  Waiting
        # here prevents Qt from destroying signal objects while the runnable is
        # still completing during application shutdown/smoke tests.
        self._thread_pool.waitForDone(5000)
        event.accept()
