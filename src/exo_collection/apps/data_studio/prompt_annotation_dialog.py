"""按键打标语义命名标注对话框。

列出当前 Trial 的全部按键打标事件（按时间排序），让用户为每个事件选择
``start`` / ``end`` / ``nan``（无效）或不标注；「保存」把结论写回
``prompt_annotations.json`` 边车，不碰原始 ``raw/prompt_labels.jsonl``。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from exo_collection.domain.prompt_annotations import (
    AddedPromptLabelEvent,
    PromptName,
    write_added_prompt_labels,
    write_prompt_annotations,
)
from exo_collection.domain.prompt_labels import PromptLabelSource

from .local_tools import PromptLabelPlaybackEvent

# (显示文案, 存值)：存值为 None 表示「未标注」，否则为 PromptName.value 字符串。
_CHOICES: tuple[tuple[str, str | None], ...] = (("未标注", None),) + tuple(
    (name.display_name, name.value) for name in PromptName
)


class PromptAnnotationDialog(QDialog):
    """按键打标事件的语义命名编辑器（模态）。"""

    def __init__(
        self,
        session_dir: Path,
        trial_uuid: str,
        events: tuple[PromptLabelPlaybackEvent, ...],
        parent: QWidget | None = None,
        highlight_event: PromptLabelPlaybackEvent | None = None,
    ) -> None:
        super().__init__(parent)
        self._session_dir = Path(session_dir)
        self._trial_uuid = trial_uuid
        self._events = tuple(sorted(events, key=lambda e: (e.time_s, e.sequence)))
        self._result_events: tuple[PromptLabelPlaybackEvent, ...] | None = None

        self.setObjectName("prompt_annotation_dialog")
        self.setWindowTitle("标注打标 · 语义命名")
        self.resize(720, 520)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "为每个按键事件指定语义名称：开始 / 结束 / 无效，或意图前·后·外的 start / end。\n"
            "保存后写入边车文件，不会改写原始打标数据。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._table = QTableWidget(len(self._events), 5)
        self._table.setObjectName("prompt_annotation_table")
        self._table.setHorizontalHeaderLabels(["序号", "时间 t(s)", "来源", "按键", "名称"])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setColumnWidth(0, 64)
        self._table.setColumnWidth(1, 120)
        self._table.setColumnWidth(2, 140)
        self._table.setColumnWidth(3, 64)
        self._table.setColumnWidth(4, 120)

        self._combos: list[QComboBox] = []
        for row, event in enumerate(self._events):
            sequence_item = QTableWidgetItem(
                "补录" if event.added else str(event.sequence)
            )
            time_item = QTableWidgetItem(f"{event.time_s:.3f}")
            source_item = QTableWidgetItem(event.source.display_name)
            key_item = QTableWidgetItem(event.key)
            combo = QComboBox()
            for label, value in _CHOICES:
                combo.addItem(label, value)
            combo.setCurrentIndex(self._choice_index(event.name))
            combo.currentIndexChanged.connect(self._update_counts)
            self._combos.append(combo)

            self._table.setItem(row, 0, sequence_item)
            self._table.setItem(row, 1, time_item)
            self._table.setItem(row, 2, source_item)
            self._table.setItem(row, 3, key_item)
            self._table.setCellWidget(row, 4, combo)
        layout.addWidget(self._table, 1)

        footer = QHBoxLayout()
        self._count_label = QLabel()
        footer.addWidget(self._count_label)
        footer.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        footer.addWidget(buttons)
        layout.addLayout(footer)

        self._update_counts()
        if highlight_event is not None:
            self._highlight_event(highlight_event)

    @staticmethod
    def _choice_index(name: str | None) -> int:
        for index, (_label, value) in enumerate(_CHOICES):
            if value == name:
                return index
        return 0

    def _update_counts(self) -> None:
        counts: dict[str, int] = {}
        for combo in self._combos:
            value = combo.currentData()
            if value is not None:
                counts[value] = counts.get(value, 0) + 1
        parts = [
            f"{name.display_name}={counts[name.value]}"
            for name in PromptName
            if counts.get(name.value)
        ]
        self._count_label.setText(
            f"{' · '.join(parts) or '未标注'} / 共 {len(self._combos)} 个事件"
        )

    def _highlight_event(self, target: PromptLabelPlaybackEvent) -> None:
        for row, event in enumerate(self._events):
            if event is target:
                self._table.selectRow(row)
                self._table.scrollToItem(self._table.item(row, 0))
                return

    def _on_save(self) -> None:
        raw_names: dict[int, PromptName] = {}
        added_models: list[AddedPromptLabelEvent] = []
        rebuilt: list[PromptLabelPlaybackEvent] = []
        for event, combo in zip(self._events, self._combos):
            value = combo.currentData()  # str | None
            name = str(value) if value is not None else None
            if event.added:
                added_models.append(
                    AddedPromptLabelEvent(
                        time_s=event.time_s,
                        source=event.source,
                        name=PromptName(value) if value is not None else None,
                    )
                )
            elif value is not None:
                raw_names[event.sequence] = PromptName(value)
            rebuilt.append(replace(event, name=name))
        write_prompt_annotations(self._session_dir, self._trial_uuid, raw_names)
        write_added_prompt_labels(self._session_dir, self._trial_uuid, added_models)
        self._result_events = tuple(rebuilt)
        self.accept()

    def result_events(self) -> tuple[PromptLabelPlaybackEvent, ...] | None:
        """保存后返回更新过的完整事件列表（raw + 补录，含新 ``name``）；取消返回 ``None``。"""
        return self._result_events


class AddPromptLabelDialog(QDialog):
    """在当前时间位置补录一条按键打标事件（采集时漏打）。"""

    def __init__(self, time_s: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source: PromptLabelSource | None = None

        self.setObjectName("add_prompt_label_dialog")
        self.setWindowTitle("增加按键打标")
        layout = QVBoxLayout(self)
        hint = QLabel(f"在 t = {time_s:.3f} s 处补录一条按键打标（未标注）：")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QHBoxLayout()
        form.addWidget(QLabel("来源："))
        self._source_combo = QComboBox()
        self._source_combo.setObjectName("add_prompt_label_source")
        for source in (
            PromptLabelSource.SUBJECT,
            PromptLabelSource.OPERATOR,
            PromptLabelSource.BUTTON,
        ):
            # StrEnum 是 str 子类，直接塞 Qt 会被压成普通 str；存 value 字符串再转回。
            self._source_combo.addItem(source.display_name, source.value)
        form.addWidget(self._source_combo)
        form.addStretch(1)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("增加")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        self._source = PromptLabelSource(self._source_combo.currentData())
        self.accept()

    def selected_source(self) -> PromptLabelSource | None:
        """用户选择的来源；取消返回 ``None``。"""
        return self._source


__all__ = ["AddPromptLabelDialog", "PromptAnnotationDialog"]
