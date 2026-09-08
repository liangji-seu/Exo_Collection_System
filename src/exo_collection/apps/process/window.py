"""Exo Process 主窗口：受试者 + 静态标定选择 → 批量解算状态树 → 批量解算。

只负责组合与信号接线；解算逻辑在 :mod:`exo_collection.apps.process.batch`。所有原始
数据保持只读，解算产出的力矩真值 CSV 直接写进各 session 目录。
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from exo_collection.apps.calculate._pipeline import pipeline_root
from exo_collection.apps.calculate.discovery import (
    discover_sessions,
    recommend_static_for_subject,
)
from exo_collection.apps.calculate.models import SessionRecord
from exo_collection.apps.calculate.opensim_env import (
    discover_opensim_python,
    pick_default_opensim_python,
)
from exo_collection.apps.collector.theme import COLLECTOR_STYLESHEET
from exo_collection.apps.process.batch import (
    BatchWorker,
    SessionSolveState,
    session_solve_status,
)
from exo_collection.configuration import SharedAppSettings

_log = logging.getLogger(__name__)

# 状态配色（与 Data Studio 的同步数据列一致）。
_GREEN = ("#e3f5e9", "#20a35a")
_RED = ("#fdecea", "#b42318")
_GRAY = ("#f1f3f5", "#9aa3ad")
_BLUE = ("#e8f0fe", "#1a73e8")
_ORANGE = ("#fef7e0", "#b26a00")


class ProcessWindow(QMainWindow):
    """Exo Process 主窗口。"""

    def __init__(self, data_root: Path, settings: SharedAppSettings) -> None:
        super().__init__()
        self._settings = settings
        self._data_root = Path(data_root)
        self._thread_pool = QThreadPool.globalInstance()

        self._sessions: list[SessionRecord] = []
        self._subject_sessions: list[SessionRecord] = []
        self._static_candidates: list[SessionRecord] = []
        self._opensim_python: Path | None = None
        self._generic_model: Path | None = None
        self._batch_worker: BatchWorker | None = None
        self._item_by_session_name: dict[str, QTreeWidgetItem] = {}

        self.setWindowTitle("Exo Process —— 批量解算")
        self.setStyleSheet(COLLECTOR_STYLESHEET)
        self.resize(1200, 800)

        central = QWidget()
        root = QVBoxLayout(central)

        self._build_top_bar(root)
        self._build_tree(root)
        self._build_bottom(root)
        self.setCentralWidget(central)

        self._resolve_opensim_env()
        self._reload_sessions()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_top_bar(self, root: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel("受试者："))
        self._subject_combo = QComboBox()
        self._subject_combo.setMinimumWidth(140)
        self._subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        row.addWidget(self._subject_combo)

        static_row = QHBoxLayout()
        static_row.addWidget(QLabel("静态标定："))
        self._static_combo = QComboBox()
        self._static_combo.setMinimumWidth(320)
        static_row.addWidget(self._static_combo, 1)
        self._cross_subject_check = QCheckBox("显示其他受试者静态标定")
        self._cross_subject_check.setToolTip("用于手动选择其他受试者编号下的标定；仅影响静态模型来源。")
        self._cross_subject_check.toggled.connect(
            lambda _: self._rebuild_static(
                self._subject_combo.currentText(), preserve_selection=True
            )
        )
        static_row.addWidget(self._cross_subject_check)

        row.addSpacing(16)
        self._env_label = QLabel("OpenSim：未配置")
        row.addWidget(self._env_label)

        row.addStretch(1)
        self._overwrite_check = QCheckBox("覆盖已解算")
        self._overwrite_check.setToolTip("勾选后，已解算的 session 也会重新解算并覆盖 ground_truth.csv。")
        row.addWidget(self._overwrite_check)

        self._batch_button = QPushButton("批量解算")
        self._batch_button.clicked.connect(self._on_batch_clicked)
        row.addWidget(self._batch_button)

        self._cancel_button = QPushButton("取消")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        row.addWidget(self._cancel_button)
        root.addLayout(row)
        root.addLayout(static_row)

    def _build_tree(self, root: QVBoxLayout) -> None:
        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["名称", "齐全状态", "解算状态"])
        self._tree.setColumnWidth(0, 520)
        self._tree.setColumnWidth(1, 140)
        self._tree.setColumnWidth(2, 220)
        self._tree.setAlternatingRowColors(True)
        root.addWidget(self._tree, 1)

    def _build_bottom(self, root: QVBoxLayout) -> None:
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(5000)
        root.addWidget(self._log, 1)

    # ------------------------------------------------------------------
    # OpenSim 环境
    # ------------------------------------------------------------------
    def _resolve_opensim_env(self) -> None:
        persisted = self._settings.opensim_python_executable
        if persisted is not None and Path(persisted).is_file():
            self._opensim_python = Path(persisted)
            self._env_label.setText(f"OpenSim：{self._opensim_python}")
        else:
            self._env_label.setText("OpenSim：正在自动发现…")
            try:
                discovered = discover_opensim_python()
                chosen = pick_default_opensim_python(discovered)
            except Exception as exc:  # noqa: BLE001
                _log.warning("发现 OpenSim 环境失败：%s", exc)
                chosen = None
            if chosen is None:
                self._opensim_python = None
                self._env_label.setText("OpenSim：未找到（批量解算不可用）")
            else:
                self._opensim_python = chosen.executable
                self._env_label.setText(f"OpenSim：{self._opensim_python}")

        generic = pipeline_root() / "data" / "models" / "gait2392" / "gait2392_simbody.osim"
        self._generic_model = generic if generic.is_file() else None

    # ------------------------------------------------------------------
    # 受试者 / 静态 / 树
    # ------------------------------------------------------------------
    def _reload_sessions(self) -> None:
        try:
            self._sessions = discover_sessions(self._data_root)
        except Exception as exc:  # noqa: BLE001
            _log.exception("扫描 session 失败")
            QMessageBox.warning(self, "扫描失败", f"扫描数据根失败：\n{exc}")
            self._sessions = []

        subjects: list[str] = []
        seen: set[str] = set()
        for record in self._sessions:
            if record.subject_code and record.subject_code not in seen:
                seen.add(record.subject_code)
                subjects.append(record.subject_code)

        self._subject_combo.blockSignals(True)
        self._subject_combo.clear()
        self._subject_combo.addItems(subjects)
        self._subject_combo.blockSignals(False)

        if subjects:
            self._on_subject_changed(0)
        else:
            self._tree.clear()
            self._static_combo.clear()

    def _on_subject_changed(self, index: int) -> None:
        code = self._subject_combo.itemText(index)
        if not code:
            return
        self._subject_sessions = [
            s for s in self._sessions if s.subject_code == code
        ]
        self._rebuild_tree()
        self._rebuild_static(code)

    def _rebuild_static(self, subject_code: str, *, preserve_selection: bool = False) -> None:
        previous = self._selected_static() if preserve_selection else None
        self._static_candidates = [
            s
            for s in self._sessions
            if s.is_stand and s.files.c3d_path is not None
            and (s.subject_code == subject_code or self._cross_subject_check.isChecked())
        ]
        self._static_combo.blockSignals(True)
        self._static_combo.clear()
        for record in self._static_candidates:
            label = f"{record.subject_code} · {record.condition_code} · {record.session_name}"
            self._static_combo.addItem(label, record)
            self._static_combo.setItemData(
                self._static_combo.count() - 1, str(record.files.c3d_path), Qt.ItemDataRole.ToolTipRole
            )
        self._static_combo.setCurrentIndex(-1)
        self._static_combo.blockSignals(False)

        recommended = recommend_static_for_subject(subject_code, self._sessions)
        if previous is not None and any(
            s.manifest_path == previous.manifest_path for s in self._static_candidates
        ):
            recommended = previous
        if recommended is not None:
            for i in range(self._static_combo.count()):
                record = self._static_combo.itemData(i)
                if record is not None and record.manifest_path == recommended.manifest_path:
                    self._static_combo.setCurrentIndex(i)
                    break

    def _rebuild_tree(self) -> None:
        self._tree.clear()
        self._item_by_session_name.clear()
        # 按工况分组（保持数据树结构）。
        groups: dict[str, list[SessionRecord]] = {}
        for record in self._subject_sessions:
            key = record.condition_code or record.condition_name or "(未命名工况)"
            groups.setdefault(key, []).append(record)

        for condition in sorted(groups):
            top = QTreeWidgetItem([condition, "", ""])
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._tree.addTopLevelItem(top)
            for record in groups[condition]:
                child = QTreeWidgetItem([record.session_name, "", ""])
                child.setData(0, Qt.ItemDataRole.UserRole, record)
                top.addChild(child)
                self._item_by_session_name[record.session_name] = child
                self._apply_state(child, record, session_solve_status(record))
            top.setExpanded(True)

    def _apply_state(
        self, item: QTreeWidgetItem, record: SessionRecord, state: SessionSolveState
    ) -> None:
        complete_text, complete_bg, complete_fg = self._completeness(record)
        self._paint_cell(item, 1, complete_text, complete_bg, complete_fg)

        solve_text, solve_bg, solve_fg = self._solve_text(state)
        self._paint_cell(item, 2, solve_text, solve_bg, solve_fg)

    @staticmethod
    def _completeness(record: SessionRecord) -> tuple[str, str, str]:
        files = record.files
        missing = []
        if files.c3d_path is None:
            missing.append("c3d")
        if files.txt_path is None:
            missing.append("txt")
        if missing:
            return f"缺 {'、'.join(missing)}", *_RED
        return "齐全", *_GREEN

    @staticmethod
    def _solve_text(state: SessionSolveState) -> tuple[str, str, str]:
        return {
            SessionSolveState.INCOMPLETE: ("不齐全", *_GRAY),
            SessionSolveState.MISSING_MODALITY: ("缺 mocap.h5/imu.h5", *_ORANGE),
            SessionSolveState.UNSOLVED: ("未解算", *_BLUE),
            SessionSolveState.SOLVED: ("已解算 · QC 未知", *_ORANGE),
            SessionSolveState.RUNNING: ("解算中…", *_BLUE),
            SessionSolveState.DONE: ("已解算 · QC 未知", *_ORANGE),
            SessionSolveState.QC_PASS: ("已解算 · QC PASS", *_GREEN),
            SessionSolveState.QC_WARN: ("已解算 · QC WARN", *_ORANGE),
            SessionSolveState.QC_FAIL: ("已解算 · QC FAIL", *_RED),
            SessionSolveState.FAILED: ("失败", *_RED),
            SessionSolveState.SYNC_FAILED: ("同步待复核 / 失败", *_ORANGE),
            SessionSolveState.CANCELLED: ("已取消", *_GRAY),
        }[state]

    @staticmethod
    def _paint_cell(
        item: QTreeWidgetItem, column: int, text: str, bg: str, fg: str
    ) -> None:
        item.setText(column, text)
        item.setTextAlignment(column, Qt.AlignmentFlag.AlignCenter)
        item.setBackground(column, QBrush(QColor(bg)))
        item.setForeground(column, QBrush(QColor(fg)))

    # ------------------------------------------------------------------
    # 批量解算
    # ------------------------------------------------------------------
    def _selected_static(self) -> SessionRecord | None:
        data = self._static_combo.currentData()
        return data if isinstance(data, SessionRecord) else None

    def _on_batch_clicked(self) -> None:
        if self._batch_worker is not None:
            return
        static = self._selected_static()
        if static is None:
            QMessageBox.warning(self, "缺少静态标定", "请先选择静态标定 session。")
            return
        if self._opensim_python is None:
            QMessageBox.warning(self, "缺少 OpenSim 环境", "未找到 OpenSim 子环境，无法解算。")
            return
        if self._generic_model is None:
            QMessageBox.warning(self, "缺少通用模型", "找不到 gait2392 通用模型。")
            return

        overwrite = self._overwrite_check.isChecked()
        targets: list[SessionRecord] = []
        for record in self._subject_sessions:
            state = session_solve_status(record)
            if state is SessionSolveState.UNSOLVED:
                targets.append(record)
            elif overwrite and state in {
                SessionSolveState.SOLVED, SessionSolveState.QC_PASS,
                SessionSolveState.QC_WARN, SessionSolveState.QC_FAIL,
            }:
                targets.append(record)
        if not targets:
            QMessageBox.information(
                self,
                "没有待解算 session",
                "当前受试者没有需要解算的 session（或都已解算，可勾选「覆盖已解算」）。",
            )
            return

        self._batch_worker = BatchWorker(
            targets,
            static,
            self._opensim_python,
            self._generic_model,
        )
        self._batch_worker.signals.session_started.connect(self._on_session_started)
        self._batch_worker.signals.session_finished.connect(self._on_session_finished)
        self._batch_worker.signals.session_failed.connect(self._on_session_failed)
        self._batch_worker.signals.progress.connect(self._append_log)
        self._batch_worker.signals.all_done.connect(self._on_all_done)

        self._progress.setRange(0, len(targets))
        self._progress.setValue(0)
        self._log.clear()
        self._set_running(True)
        self._append_log(
            f"开始批量解算 {len(targets)} 个 session（动态受试者 {self._subject_combo.currentText()}；"
            f"静态来源 {static.subject_code} / {static.condition_code} / {static.session_name}）。"
        )
        self._append_log(f"静态 C3D：{static.files.c3d_path}")
        self._thread_pool.start(self._batch_worker)

    def _set_running(self, running: bool) -> None:
        self._subject_combo.setEnabled(not running)
        self._static_combo.setEnabled(not running)
        self._cross_subject_check.setEnabled(not running)
        self._overwrite_check.setEnabled(not running)
        self._batch_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)

    def _on_cancel_clicked(self) -> None:
        if self._batch_worker is not None:
            self._batch_worker.cancel()
            self._cancel_button.setEnabled(False)
            self._append_log("已请求取消，等待当前 session 结束…")

    def _on_session_started(self, name: str) -> None:
        item = self._item_by_session_name.get(name)
        if item is not None:
            record = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(record, SessionRecord):
                self._apply_state(item, record, SessionSolveState.RUNNING)
        self._append_log(f"开始解算：{name}")

    def _on_session_finished(self, name: str, state_value: str, out_path: str) -> None:
        item = self._item_by_session_name.get(name)
        if item is not None:
            record = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(record, SessionRecord):
                self._apply_state(item, record, SessionSolveState(state_value))
        self._progress.setValue(self._progress.value() + 1)
        self._append_log(f"完成：{name} → {out_path}")

    def _on_session_failed(self, name: str, state_value: str, message: str) -> None:
        item = self._item_by_session_name.get(name)
        if item is not None:
            record = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(record, SessionRecord):
                self._apply_state(item, record, SessionSolveState(state_value))
        self._progress.setValue(self._progress.value() + 1)
        self._append_log(f"{name}：{message}")

    def _on_all_done(self, total: int, ok: int, failed: int) -> None:
        self._set_running(False)
        self._batch_worker = None
        self._append_log(f"批量解算结束：计算完成 {ok}，失败/待复核 {failed}，计划 {total}。质量结论见各行 QC。")
        self.statusBar().showMessage(f"计算完成 {ok} / 失败或待复核 {failed}；质量结论见 QC")

    def _append_log(self, line: str) -> None:
        self._log.appendPlainText(line)
        cursor = self._log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._log.setTextCursor(cursor)
