"""Session 选择页：受试者 → 动态工况 → 静态标定（自动绑定）。

操作流程按「选受试者 → 自动绑定该受试者最近的 STAND 静态试次 → 选动态工况
反解」设计：选中受试者后自动推荐静态标定 Session，动态工况下拉只列该受试者
的非静态工况；用户仍可手动改静态标定（若有多个 STAND 试次）。

只负责「展示 + 用户选择」，不读 C3D / 大文件；输入检查由 ``check_inputs``
（只读扫描）在需要时调用。数据发现复用 ``discovery`` 模块。
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from exo_collection.apps.calculate.discovery import (
    discover_sessions,
    recommend_static_for_subject,
)
from exo_collection.apps.calculate.models import SessionRecord

_log = logging.getLogger(__name__)


def _format_condition_parameters(record: SessionRecord) -> str:
    params = record.condition_parameters or {}
    pieces = []
    for key in ("speed", "slope", "load", "速度", "坡度", "负载"):
        if key in params:
            pieces.append(f"{key}={params[key]}")
    return " ".join(pieces)


def _short_date(started_at_utc: str) -> str:
    # ISO 字符串形如 2026-09-02T14:58:32(+00:00)；只取日期部分。
    return started_at_utc[:10] if started_at_utc else ""


def _dynamic_label(record: SessionRecord) -> str:
    return (
        f"{record.condition_code} {_format_condition_parameters(record)} · "
        f"r{record.repeat_index} · {_short_date(record.started_at_utc)}"
    ).strip()


def _static_label(record: SessionRecord) -> str:
    return f"{record.condition_code} · r{record.repeat_index} · {_short_date(record.started_at_utc)}"


class SessionSelector(QWidget):
    """受试者 / 动态工况 / 静态标定三下拉选择面板。"""

    dynamic_selected = Signal(object)   # SessionRecord | None
    static_selected = Signal(object)    # SessionRecord | None
    check_requested = Signal(object, object)  # (dynamic, static)

    def __init__(self, data_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data_root = Path(data_root)
        self._sessions: list[SessionRecord] = []
        self._dynamic: SessionRecord | None = None
        self._static: SessionRecord | None = None

        self._subject_combo = QComboBox()
        self._subject_combo.currentIndexChanged.connect(self._on_subject_changed)

        self._dynamic_combo = QComboBox()
        self._dynamic_combo.currentIndexChanged.connect(self._on_dynamic_changed)

        self._static_combo = QComboBox()
        self._static_combo.currentIndexChanged.connect(self._on_static_changed)

        self._check_button = QPushButton("检查输入")
        self._check_button.clicked.connect(self._request_check)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("受试者：", self._subject_combo)
        form.addRow("动态工况：", self._dynamic_combo)
        form.addRow("静态标定：", self._static_combo)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        buttons.addWidget(self._check_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addWidget(self._status_label)
        layout.addStretch(1)

        self.refresh()

    # ------------------------------------------------------------------
    # 数据根 / 发现
    # ------------------------------------------------------------------
    def set_data_root(self, data_root: str | Path) -> None:
        self._data_root = Path(data_root)
        self.refresh()

    def refresh(self) -> None:
        """重新扫描数据根并填充下拉（发现是轻量 manifest 读取，可同步）。"""
        try:
            self._sessions = discover_sessions(self._data_root)
        except Exception as exc:  # noqa: BLE001
            _log.exception("Session 发现失败")
            self._sessions = []
        self._populate_subject_combo()
        self._apply_subject()

    def _populate_subject_combo(self) -> None:
        subjects = sorted({s.subject_code for s in self._sessions if s.subject_code})
        self._subject_combo.blockSignals(True)
        self._subject_combo.clear()
        for code in subjects:
            self._subject_combo.addItem(code, code)
        self._subject_combo.blockSignals(False)

    def _populate_combo(
        self,
        combo: QComboBox,
        records: list[SessionRecord],
        label_func,
        selected: SessionRecord | None = None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        for record in records:
            combo.addItem(label_func(record), record)
        if selected is not None:
            self._select_record(combo, selected)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    @staticmethod
    def _select_record(combo: QComboBox, record: SessionRecord) -> None:
        # 不用 findData：PySide6 对任意 Python 对象的 QVariant 相等比较不可靠，
        # 这里按 itemData() 逐项用 Python ``==`` 匹配。
        for index in range(combo.count()):
            if combo.itemData(index) == record:
                combo.setCurrentIndex(index)
                return

    # ------------------------------------------------------------------
    # 受试者 → 动态工况 + 静态标定（自动绑定）
    # ------------------------------------------------------------------
    def _apply_subject(self) -> None:
        code = self._subject_combo.currentData()
        subject_sessions = [s for s in self._sessions if s.subject_code == code]

        dynamics = sorted(
            [s for s in subject_sessions if not s.is_stand],
            key=lambda s: (s.condition_code, s.repeat_index, s.started_at_utc),
        )
        statics = sorted(
            [s for s in subject_sessions if s.is_stand],
            key=lambda s: s.started_at_utc,
            reverse=True,
        )
        recommended = recommend_static_for_subject(code, self._sessions)

        self._populate_combo(self._dynamic_combo, dynamics, _dynamic_label)
        self._populate_combo(
            self._static_combo, statics, _static_label, selected=recommended
        )

        # 填充期间信号被 block，这里显式同步当前选择并 emit 一次。
        self._sync_dynamic()
        self._sync_static()
        self._update_status()

    def _sync_dynamic(self) -> None:
        self._set_dynamic(self._dynamic_combo.currentData())

    def _sync_static(self) -> None:
        self._set_static(self._static_combo.currentData())

    def _on_subject_changed(self, _index: int) -> None:
        self._apply_subject()

    def _on_dynamic_changed(self, _index: int) -> None:
        self._sync_dynamic()

    def _on_static_changed(self, _index: int) -> None:
        self._sync_static()

    # ------------------------------------------------------------------
    def _set_dynamic(self, record: SessionRecord | None) -> None:
        self._dynamic = record
        self.dynamic_selected.emit(record)
        self._update_status()

    def _set_static(self, record: SessionRecord | None) -> None:
        self._static = record
        self.static_selected.emit(record)
        self._update_status()

    def _update_status(self) -> None:
        if not self._sessions:
            self._status_label.setText("数据根目录下未发现已最终化的 Session。")
            return
        lines = []
        if self._dynamic is not None:
            missing = self._dynamic.files.missing()
            completeness = "齐全" if not missing else "缺 " + "、".join(missing)
            lines.append(f"动态：{self._dynamic.subject_and_condition}（输入 {completeness}）")
        else:
            lines.append("动态：该受试者没有非静态工况。")
        if self._static is not None:
            lines.append(f"静态标定：{self._static.subject_and_condition}")
        else:
            lines.append("静态标定：该受试者未找到 STAND 试次（无法缩放模型）。")
        self._status_label.setText("\n".join(lines))

    def _request_check(self) -> None:
        self.check_requested.emit(self._dynamic, self._static)

    # ------------------------------------------------------------------
    def current_dynamic(self) -> SessionRecord | None:
        return self._dynamic

    def current_static(self) -> SessionRecord | None:
        return self._static


__all__ = ["SessionSelector"]
