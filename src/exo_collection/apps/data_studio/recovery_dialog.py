"""Data Studio dialog for explicit, evidence-gated Trial recovery."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import UUID

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from exo_collection.domain.states import TrialState
from exo_collection.storage.activity import AcquisitionActivity, read_activity
from exo_collection.storage.recovery_manager import RecoveryAction, TrialRecoveryReport

from .recovery_service import RecoveryBackgroundService, RecoveryOperation


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, TrialState):
        return value.value
    if isinstance(value, RecoveryAction):
        return value.value
    raise TypeError(type(value).__name__)


class RecoveryDialog(QDialog):
    """Show read-only evidence and run only decisions enabled by that evidence.

    Constructing the dialog schedules a startup scan.  The same ``rescan``
    method is the manual refresh interface; all work executes in a spawned
    process and is polled by a short Qt timer.
    """

    def __init__(self, dataset_root: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self._service = RecoveryBackgroundService()
        self._closing = False
        self._reports: tuple[TrialRecoveryReport, ...] = ()
        self._activity_blocked = False
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_service)
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(500)
        self._activity_timer.timeout.connect(self._poll_activity)

        self.setWindowTitle("Trial 恢复与人工处置")
        self.resize(1120, 680)
        outer = QVBoxLayout(self)

        explanation = QLabel(
            "系统只读扫描 .recording 数据包。只有 Manifest、全部 Artifact、SHA-256、"
            "文件覆盖范围和关闭状态均能证明完整时，才允许恢复为 FINALIZED；"
            "ABORTED 会保留全部原始数据并写入不可覆盖的审计记录。"
        )
        explanation.setWordWrap(True)
        outer.addWidget(explanation)

        root_label = QLabel(f"数据根目录：{self.dataset_root}")
        root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(root_label)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, 8)
        self.table.setObjectName("recovery_trial_table")
        self.table.setHorizontalHeaderLabels(
            [
                "Trial UUID",
                "状态",
                "超声块",
                "HDF5",
                ".partial",
                "可安全修复",
                "可最终化",
                "目录",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        splitter.addWidget(self.table)

        self.details = QPlainTextEdit()
        self.details.setObjectName("recovery_evidence_details")
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("选择一个 Trial 查看完整只读恢复证据。")
        splitter.addWidget(self.details)
        splitter.setSizes([380, 220])
        outer.addWidget(splitter, 1)

        action_row = QHBoxLayout()
        self.rescan_button = QPushButton("重新扫描")
        self.rescan_button.setObjectName("rescan_recoverable_trials")
        self.rescan_button.clicked.connect(self.rescan)
        action_row.addWidget(self.rescan_button)

        self.repair_button = QPushButton("安全修复尾块")
        self.repair_button.setObjectName("repair_recoverable_trial")
        self.repair_button.clicked.connect(self._repair_selected)
        action_row.addWidget(self.repair_button)

        self.finalize_button = QPushButton("确认恢复为 FINALIZED")
        self.finalize_button.setObjectName("finalize_prepared_trial")
        self.finalize_button.clicked.connect(self._finalize_selected)
        action_row.addWidget(self.finalize_button)

        self.abort_button = QPushButton("保留数据并标记 ABORTED")
        self.abort_button.setObjectName("abort_recoverable_trial")
        self.abort_button.clicked.connect(self._abort_selected)
        action_row.addWidget(self.abort_button)
        action_row.addStretch(1)
        outer.addLayout(action_row)

        self.status_label = QLabel("尚未扫描")
        self.status_label.setObjectName("recovery_status")
        outer.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._selection_changed()
        self._activity_timer.start()
        self._poll_activity()
        QTimer.singleShot(0, self.rescan)

    def rescan(self) -> None:
        """Start a non-blocking startup/manual discovery pass."""

        self._poll_activity()
        if self._closing or self._service.busy or self._activity_blocked:
            return
        self._start_operation("scan", lambda: self._service.start_scan(self.dataset_root))

    def _start_operation(self, operation: RecoveryOperation, starter: Any) -> None:
        self._poll_activity()
        if self._activity_blocked:
            QMessageBox.warning(
                self,
                "采集期间禁止恢复",
                "检测到 Collector 活动锁；恢复扫描与处置已切换为只读禁用状态。",
            )
            return
        try:
            starter()
        except Exception as exc:
            QMessageBox.critical(self, "无法启动恢复任务", str(exc))
            return
        self._set_busy(True)
        labels = {
            "scan": "正在后台扫描 .recording…",
            "repair": "正在安全修复超声尾块…",
            "finalize": "正在验证并原子发布 Trial…",
            "abort": "正在记录证据并原子标记 ABORTED…",
        }
        self.status_label.setText(labels[operation])
        self._poll_timer.start()

    def _poll_service(self) -> None:
        result = self._service.poll()
        if result is None:
            return
        self._poll_timer.stop()
        status, operation, payload = result
        try:
            self._service.finish()
        except Exception as exc:
            status = "failed"
            payload = f"{payload}\nWorker cleanup failed: {exc}"
        self._poll_activity()
        self._set_busy(False)
        if status != "completed":
            self.status_label.setText(f"{operation} 失败")
            QMessageBox.critical(self, "恢复任务失败", str(payload))
            return
        if operation == "scan":
            self._reports = tuple(payload)  # type: ignore[arg-type]
            self._render_reports()
            self.status_label.setText(
                f"扫描完成：发现 {len(self._reports)} 个 .recording Trial"
            )
            return
        self.status_label.setText(f"{operation} 完成，正在重新扫描…")
        QMessageBox.information(self, "恢复任务完成", self._operation_summary(operation, payload))
        self.rescan()

    @staticmethod
    def _operation_summary(operation: RecoveryOperation, payload: object) -> str:
        if operation == "repair":
            return "安全尾块修复已完成；Trial 仍保持 RECOVERABLE，不会自动最终化。"
        destination = getattr(payload, "destination_directory", None)
        if operation == "finalize":
            return f"已通过完整性证明并原子发布：\n{destination}"
        return f"已保留全部原始数据并标记 ABORTED：\n{destination}"

    def _render_reports(self) -> None:
        self.table.setRowCount(len(self._reports))
        for row, report in enumerate(self._reports):
            if report.active_collection:
                state = "采集中（禁止读取）"
            elif report.can_finalize:
                state = "RECOVERABLE（发布前崩溃）"
            else:
                state = "RECOVERABLE"
            ultrasound = report.ultrasound
            ultrasound_text = (
                "未检查"
                if report.active_collection
                else (
                    "无"
                    if ultrasound is None
                    else f"{ultrasound.complete_block_count} / "
                    f"{'完整' if ultrasound.is_clean else ultrasound.error_kind or '异常'}"
                )
            )
            hdf5_text = (
                "未检查"
                if report.active_collection
                else f"{sum(item.readable and item.closed_cleanly for item in report.hdf5_files)}"
                f"/{len(report.hdf5_files)} 完整"
            )
            values = (
                str(report.trial_uuid or "目录名不是 UUID"),
                state,
                ultrasound_text,
                hdf5_text,
                str(len(report.partial_files)),
                "是" if report.can_repair else "否",
                "是" if report.can_finalize else "否",
                str(report.recording_directory),
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        if self._reports:
            self.table.selectRow(0)
        else:
            self.details.setPlainText("未发现需要恢复的 .recording Trial。")
        self._selection_changed()

    def _selected_report(self) -> TrialRecoveryReport | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        return self._reports[row] if 0 <= row < len(self._reports) else None

    def _selection_changed(self) -> None:
        report = self._selected_report()
        idle = not self._service.busy and not self._activity_blocked
        self.repair_button.setEnabled(bool(idle and report and report.can_repair))
        self.finalize_button.setEnabled(bool(idle and report and report.can_finalize))
        self.abort_button.setEnabled(bool(idle and report and report.can_abort))
        if report is None:
            return
        self.details.setPlainText(
            json.dumps(
                asdict(report),
                default=_json_default,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )

    def _set_busy(self, busy: bool) -> None:
        interactive = not busy and not self._activity_blocked
        self.rescan_button.setEnabled(interactive)
        self.table.setEnabled(interactive)
        if busy:
            self.repair_button.setEnabled(False)
            self.finalize_button.setEnabled(False)
            self.abort_button.setEnabled(False)
        else:
            self._selection_changed()

    def set_acquisition_activity(
        self,
        activity: AcquisitionActivity | None,
    ) -> None:
        """Immediately revoke recovery controls when Collector becomes active."""

        # Recovery mutation workers briefly own the same exclusive dataset lock
        # to prevent Collector startup. Do not mistake that lock for a competing
        # Collector; the mutual exclusion already protects the running action.
        if self._service.busy:
            return
        blocked = activity is not None
        if blocked == self._activity_blocked:
            return
        self._activity_blocked = blocked
        if blocked:
            if activity is not None and (
                activity.pid <= 0 or activity.hostname == "unreadable-lock"
            ):
                detail = "活动锁不可安全解析，已保守禁用恢复操作。"
            else:
                assert activity is not None
                detail = (
                    "检测到 Collector 正在采集"
                    f"（PID {activity.pid}），恢复操作已切换为只读禁用状态。"
                )
            self.status_label.setText(detail)
        else:
            self.status_label.setText("Collector 活动已结束；可重新扫描恢复数据包。")
        self._set_busy(False)

    def _poll_activity(self) -> None:
        if not self._service.busy:
            self.set_acquisition_activity(read_activity(self.dataset_root))

    def _repair_selected(self) -> None:
        report = self._selected_report()
        if report is None or not report.can_repair:
            return
        response = QMessageBox.question(
            self,
            "确认安全尾块修复",
            "只会截去客观可证明不完整的最后尾块并重建索引。完整块和 HDF5 不会改写；"
            "修复后仍为 RECOVERABLE。是否继续？",
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        self._start_operation(
            "repair",
            lambda: self._service.start_repair(report.recording_directory),
        )

    def _finalize_selected(self) -> None:
        report = self._selected_report()
        if report is None or not report.can_finalize:
            return
        response = QMessageBox.question(
            self,
            "人工确认 FINALIZED",
            "当前包已通过 Manifest、Artifact、SHA-256、文件覆盖和关闭状态的完整证明。"
            "此操作不改写任何包内文件，只原子移除 .recording 后缀。是否确认？",
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        self._start_operation(
            "finalize",
            lambda: self._service.start_finalize(report.recording_directory),
        )

    def _abort_selected(self) -> None:
        report = self._selected_report()
        if report is None or not report.can_abort:
            return
        reason, accepted = QInputDialog.getMultiLineText(
            self,
            "ABORTED 原因",
            "请输入人工判定原因（必填，将写入 append-only 审计）：",
        )
        if not accepted or not reason.strip():
            return
        response = QMessageBox.question(
            self,
            "确认标记 ABORTED",
            "系统会先记录所有现有文件的大小和 SHA-256，再原子改名为 .aborted。"
            "任何原始文件都不会修改或删除。是否确认？",
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        self._start_operation(
            "abort",
            lambda: self._service.start_abort(
                report.recording_directory,
                reason=reason.strip(),
            ),
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._closing = True
        self._poll_timer.stop()
        self._activity_timer.stop()
        # ``busy`` becomes false as soon as the child exits, even though its
        # queued result and Windows process/queue handles still require finish.
        # Always clean any started operation, including that narrow exited-but-
        # not-yet-polled window.
        if self._service.operation is not None:
            self._service.cancel()
        event.accept()


__all__ = ["RecoveryDialog"]
