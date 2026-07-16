"""Data Studio dialog for protocol coverage and local dataset state."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .management import (
    ConditionCompletionStatus,
    ManagementSummaryResult,
    PackageState,
)


_COVERAGE_LABELS = {
    ConditionCompletionStatus.COMPLETED: "已完成",
    ConditionCompletionStatus.ATTEMPTED_NO_VALID_TRIAL: "已尝试、无有效 Trial",
    ConditionCompletionStatus.MISSING: "从未尝试",
}


def _indices(values: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in values) if values else "—"


class ManagementSummaryDialog(QDialog):
    """Read-only management report built entirely by a background worker."""

    def __init__(
        self,
        result: ManagementSummaryResult,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result = result
        self.setObjectName("management_summary_dialog")
        self.setWindowTitle("Data Studio · 管理摘要")
        self.resize(1060, 650)

        layout = QVBoxLayout(self)
        states = result.dataset_states
        invalid_aborted = sum(
            item.state is PackageState.ABORTED_UNVERIFIED for item in states.aborted
        )
        self.state_summary_label = QLabel(
            f"FINALIZED：{states.finalized_count}    "
            f"待恢复：{states.pending_recovery_count}    "
            f"ABORTED：{states.aborted_count}（证据异常 {invalid_aborted}）    "
            f"待质检：{states.pending_quality_count}    "
            f"待上传：{states.pending_upload_count}"
        )
        self.state_summary_label.setObjectName("management_state_summary")
        self.state_summary_label.setWordWrap(True)
        layout.addWidget(self.state_summary_label)

        tabs = QTabWidget()
        tabs.setObjectName("management_summary_tabs")
        tabs.addTab(self._coverage_tab(), "受试者工况覆盖")
        tabs.addTab(self._state_tab(), "状态与待办")
        layout.addWidget(tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _coverage_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        table = QTableWidget()
        table.setObjectName("management_coverage_table")
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels(
            [
                "项目",
                "受试者",
                "覆盖率",
                "工况",
                "状态",
                "尝试数",
                "FINALIZED",
                "有效数",
                "重复轮次",
                "有效轮次",
            ]
        )
        rows = [
            (subject, condition)
            for subject in self.result.subject_coverage
            for condition in subject.conditions
        ]
        table.setRowCount(len(rows))
        for row, (subject, condition) in enumerate(rows):
            values = (
                subject.project_code or subject.project_uuid,
                subject.subject_code or subject.subject_uuid,
                f"{subject.coverage_fraction:.0%}",
                f"{condition.condition_code} · {condition.condition_name}",
                _COVERAGE_LABELS[condition.status],
                str(condition.trial_count),
                str(condition.finalized_trial_count),
                str(condition.valid_trial_count),
                _indices(condition.repeat_indices),
                _indices(condition.valid_repeat_indices),
            )
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        return panel

    def _state_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        table = QTableWidget()
        table.setObjectName("management_state_table")
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["类别", "Trial UUID", "证据/状态", "路径或说明"])
        states = self.result.dataset_states
        rows: list[tuple[str, str, str, str]] = []
        rows.extend(
            (
                "待恢复",
                item.trial_uuid or "—",
                "需恢复检查",
                str(item.path),
            )
            for item in states.pending_recovery
        )
        rows.extend(
            (
                "ABORTED",
                item.trial_uuid or "—",
                "证据通过" if item.evidence_verified else "证据异常",
                str(item.path) if item.message is None else f"{item.path} · {item.message}",
            )
            for item in states.aborted
        )
        rows.extend(
            ("待质检", trial_uuid, "无有效人工审核 sidecar", "—")
            for trial_uuid in states.pending_quality_trial_uuids
        )
        rows.extend(
            ("待上传", trial_uuid, "无 VERIFIED 上传审计", "—")
            for trial_uuid in states.pending_upload_trial_uuids
        )
        rows.extend(
            ("Sidecar 异常", trial_uuid, "校验失败", "请在 Trial 详情中复核")
            for trial_uuid in states.sidecar_error_trial_uuids
        )
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        return panel


__all__ = ["ManagementSummaryDialog"]
