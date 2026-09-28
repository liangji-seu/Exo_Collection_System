"""Minimal session picker for the ultrasound feature viewer.

The calculate session selector intentionally carries calculate-only concepts
such as static calibration, input checks, and discard state.  The ultrasound
viewer needs a smaller, read-only picker that can be imported by both its
standalone launcher and its in-window "选择 session" action.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from exo_collection.apps.calculate.discovery import discover_sessions, distinct_days
from exo_collection.apps.calculate.models import SessionRecord
from exo_collection.configuration import SharedAppSettings


_ULTRASOUND_FILES = ("ultrasound.bin", "ultrasound.meta.json", "ultrasound.idx")


def discover_us_sessions(data_root: str | Path) -> list[SessionRecord]:
    """Return finalized sessions that contain the three US reader files."""

    records = discover_sessions(data_root)
    return [
        record
        for record in records
        if all((record.session_dir / name).is_file() for name in _ULTRASOUND_FILES)
    ]


def _condition_label(record: SessionRecord) -> str:
    parameters = record.condition_parameters or {}
    pieces = []
    for key in ("speed", "slope", "load", "速度", "坡度", "负载"):
        if key in parameters:
            pieces.append(f"{key}={parameters[key]}")
    suffix = f" ({' '.join(pieces)})" if pieces else ""
    return f"{record.condition_code}{suffix}"


def _session_label(record: SessionRecord) -> str:
    return f"r{record.repeat_index} · {record.session_name}"


class USSessionSelectorDialog(QDialog):
    """Subject → day → condition → session picker for read-only US viewing."""

    def __init__(
        self,
        data_root: str | Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._data_root = Path(data_root).expanduser().resolve()
        self._sessions: list[SessionRecord] = []
        self._subject_sessions: list[SessionRecord] = []
        self._day_sessions: list[SessionRecord] = []
        self._condition_sessions: list[SessionRecord] = []

        self.setWindowTitle("选择超声 session")
        self.setModal(True)
        self.resize(720, 220)

        self._subject_combo = QComboBox()
        self._day_combo = QComboBox()
        self._condition_combo = QComboBox()
        self._session_combo = QComboBox()
        self._subject_combo.currentIndexChanged.connect(self._apply_subject)
        self._day_combo.currentIndexChanged.connect(self._apply_day)
        self._condition_combo.currentIndexChanged.connect(self._apply_condition)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)

        self._data_root_edit = QLineEdit(str(self._data_root))
        self._data_root_edit.setReadOnly(True)
        self._choose_root_button = QPushButton("选择…")
        self._choose_root_button.clicked.connect(self._choose_data_root)
        self._refresh_button = QPushButton("刷新")
        self._refresh_button.clicked.connect(self._refresh)

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form = QFormLayout()
        root_row = QHBoxLayout()
        root_row.addWidget(self._data_root_edit, 1)
        root_row.addWidget(self._choose_root_button)
        root_row.addWidget(self._refresh_button)
        form.addRow("数据根目录：", root_row)
        form.addRow("受试者：", self._subject_combo)
        form.addRow("日期：", self._day_combo)
        form.addRow("工况：", self._condition_combo)
        form.addRow("Session：", self._session_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Open).setText("打开")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        self._open_button = buttons.button(QDialogButtonBox.StandardButton.Open)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._status)
        layout.addWidget(buttons)

        self._refresh()

    @property
    def data_root(self) -> Path:
        return self._data_root

    def selected_session(self) -> SessionRecord | None:
        value = self._session_combo.currentData()
        return value if isinstance(value, SessionRecord) else None

    def selected_path(self) -> Path | None:
        selected = self.selected_session()
        return selected.session_dir if selected is not None else None

    def _refresh(self) -> None:
        try:
            self._sessions = discover_us_sessions(self._data_root)
        except Exception as exc:  # noqa: BLE001 - GUI boundary
            self._sessions = []
            self._status.setText(f"扫描数据根目录失败：{exc}")
        self._subject_combo.blockSignals(True)
        self._subject_combo.clear()
        for subject in sorted({record.subject_code for record in self._sessions}):
            self._subject_combo.addItem(subject, subject)
        self._subject_combo.blockSignals(False)
        self._apply_subject()

    def _apply_subject(self, _index: int = -1) -> None:
        subject = self._subject_combo.currentData()
        self._subject_sessions = [
            record for record in self._sessions if record.subject_code == subject
        ]
        days = distinct_days(self._subject_sessions)
        self._day_combo.blockSignals(True)
        self._day_combo.clear()
        for day in days:
            self._day_combo.addItem(
                f"d{day}" if day is not None else "未分日", day
            )
        self._day_combo.blockSignals(False)
        self._apply_day()

    def _apply_day(self, _index: int = -1) -> None:
        day = self._day_combo.currentData()
        self._day_sessions = [
            record for record in self._subject_sessions if record.day == day
        ]
        condition_records: dict[str, SessionRecord] = {}
        for record in self._day_sessions:
            condition_records.setdefault(record.condition_code, record)
        self._condition_combo.blockSignals(True)
        self._condition_combo.clear()
        for condition, record in sorted(
            condition_records.items(), key=lambda item: item[0].casefold()
        ):
            self._condition_combo.addItem(_condition_label(record), condition)
        self._condition_combo.blockSignals(False)
        self._apply_condition()

    def _apply_condition(self, _index: int = -1) -> None:
        condition = self._condition_combo.currentData()
        self._condition_sessions = sorted(
            [
                record
                for record in self._day_sessions
                if record.condition_code == condition
            ],
            key=lambda record: (record.repeat_index, record.started_at_utc),
        )
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        for record in self._condition_sessions:
            self._session_combo.addItem(_session_label(record), record)
        self._session_combo.blockSignals(False)
        self._update_status()

    def _on_session_changed(self, _index: int) -> None:
        self._update_status()

    def _choose_data_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "选择数据根目录", str(self._data_root)
        )
        if not selected:
            return
        try:
            self._data_root = SharedAppSettings().set_data_root(selected)
        except Exception as exc:  # noqa: BLE001 - GUI boundary
            self._status.setText(f"保存数据根目录失败：{exc}")
            return
        self._data_root_edit.setText(str(self._data_root))
        self._refresh()

    def _update_status(self) -> None:
        selected = self.selected_session()
        enabled = selected is not None
        self._open_button.setEnabled(enabled)
        if not self._sessions:
            self._status.setText(
                "共享数据根目录下未发现可查看的超声 session。"
                "需要 ultrasound.bin、ultrasound.meta.json 和 ultrasound.idx。"
            )
        elif selected is not None:
            self._status.setText(str(selected.session_dir))
        else:
            self._status.setText("请选择一个超声 session。")

    def _accept_selection(self) -> None:
        if self.selected_session() is not None:
            self.accept()


def select_us_session(
    data_root: str | Path,
    parent: QWidget | None = None,
) -> Path | None:
    """Show the selector in the caller's existing QApplication and return a path."""

    dialog = USSessionSelectorDialog(data_root, parent)
    return dialog.selected_path() if dialog.exec() == QDialog.DialogCode.Accepted else None


__all__ = ["USSessionSelectorDialog", "discover_us_sessions", "select_us_session"]
