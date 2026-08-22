"""「数据查看」标签页：只读检查一个 FINALIZED Trial 的各模态数据内容。

这个控件不自己读盘——它只负责渲染 :func:`~exo_collection.apps.data_studio
.local_tools.inspect_trial_artifacts` 的结果。读取由 ``DataStudioWindow``
通过后台线程触发，完成后回调 :meth:`DataViewWidget.show_result`。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .local_tools import (
    ArtifactInspection,
    Hdf5Inspection,
    JsonlInspection,
    TrialInspection,
    UltrasoundInspection,
)

_log = logging.getLogger(__name__)


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:  # NaN
            return "NaN"
        if value in (float("inf"), float("-inf")):
            return "+∞" if value > 0 else "-∞"
        return f"{value:.6g}"
    return str(value)


def _kind_label(kind: str) -> str:
    return {
        "hdf5": "HDF5",
        "ultrasound": "超声 .bin",
        "jsonl": "JSONL",
        "other": "其它",
    }.get(kind, kind)


class DataViewWidget(QWidget):
    """Render the per-artifact inspection result of one Trial."""

    def __init__(
        self,
        request_inspection: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._request_inspection = request_inspection
        self._artifacts_by_item: dict[int, ArtifactInspection] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        header = QHBoxLayout()
        self.load_button = QPushButton("读取所选 Trial")
        self.load_button.clicked.connect(self._request_inspection)
        header.addWidget(self.load_button)
        self.summary_label = QLabel("尚未加载。先在「数据管理」树中选中一个 FINALIZED Trial，再点击读取。")
        self.summary_label.setWordWrap(True)
        header.addWidget(self.summary_label, 1)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["模态 / 文件", "类型", "大小"])
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setUniformRowHeights(True)
        self.tree_widget.header().setStretchLastSection(False)
        self.tree_widget.setColumnWidth(0, 320)
        self.tree_widget.currentItemChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.tree_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.detail_container = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_container)
        self.detail_layout.addWidget(QLabel("请在左侧选择一个 Artifact。"))
        self.detail_layout.addStretch(1)
        scroll.setWidget(self.detail_container)
        splitter.addWidget(scroll)
        splitter.setSizes([360, 640])

        outer.addWidget(splitter, 1)

    def show_result(self, result: TrialInspection) -> None:
        self.tree_widget.clear()
        self._artifacts_by_item.clear()
        self._clear_detail()

        self.summary_label.setText(
            f"Trial {result.trial_uuid} · 工况 {result.condition_code} · "
            f"{result.artifact_count} 个 Artifact"
        )

        by_modality: dict[str, list[ArtifactInspection]] = {}
        for artifact in result.artifacts:
            by_modality.setdefault(artifact.modality, []).append(artifact)

        for modality in sorted(by_modality):
            artifacts = by_modality[modality]
            parent = QTreeWidgetItem(
                [f"{modality}  ({len(artifacts)})", "", ""]
            )
            self.tree_widget.addTopLevelItem(parent)
            for artifact in artifacts:
                child = QTreeWidgetItem(
                    [
                        artifact.relative_path,
                        _kind_label(artifact.kind),
                        _format_bytes(artifact.size_bytes),
                    ]
                )
                self._artifacts_by_item[id(child)] = artifact
                parent.addChild(child)
            parent.setExpanded(True)

    def _on_selection_changed(self, current: QTreeWidgetItem, previous: QTreeWidgetItem) -> None:
        if current is None:
            return
        artifact = self._artifacts_by_item.get(id(current))
        if artifact is None:
            self._clear_detail()
            self.detail_layout.addWidget(QLabel("请在左侧选择一个 Artifact。"))
            return
        self._render_artifact(artifact)

    def _render_artifact(self, artifact: ArtifactInspection) -> None:
        self._clear_detail()
        self._add_title(artifact.relative_path)
        if artifact.message:
            self.detail_layout.addWidget(QLabel(f"⚠ {artifact.message}"))
        if artifact.hdf5 is not None:
            self._render_hdf5(artifact.hdf5)
        elif artifact.ultrasound is not None:
            self._render_ultrasound(artifact.ultrasound)
        elif artifact.jsonl is not None:
            self._render_jsonl(artifact.jsonl)
        else:
            self.detail_layout.addWidget(QLabel("该 Artifact 无内建预览。"))
        self.detail_layout.addStretch(1)

    def _render_hdf5(self, insp: Hdf5Inspection) -> None:
        overview = self._add_group("概览")
        rows = [
            ("采样点数", f"{insp.sample_count}"),
            ("dtype", insp.dtype),
            ("sample_shape", str(insp.sample_shape)),
            ("标称采样率 (Hz)", _format_value(insp.nominal_rate_hz)),
            ("正常关闭", "是" if insp.closed_cleanly else "否（数据可能不完整）"),
            ("不连续点", f"{insp.discontinuity_count}"),
            ("事件记录", f"{insp.event_count}"),
            ("文件大小", _format_bytes(insp.size_bytes)),
        ]
        for label, value in rows:
            line = QHBoxLayout()
            key = QLabel(f"{label}：")
            key.setMinimumWidth(140)
            line.addWidget(key)
            line.addWidget(QLabel(value))
            overview.addLayout(line)

        if insp.stats:
            group = self._add_group("通道统计")
            table = QTableWidget(len(insp.stats), 6)
            table.setHorizontalHeaderLabels(["通道", "单位", "最小值", "最大值", "均值", "标准差"])
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.verticalHeader().setVisible(False)
            for row, stat in enumerate(insp.stats):
                values = (
                    stat.channel,
                    stat.unit,
                    _format_value(stat.min),
                    _format_value(stat.max),
                    _format_value(stat.mean),
                    _format_value(stat.std),
                )
                for col, value in enumerate(values):
                    table.setItem(row, col, QTableWidgetItem(value))
            table.resizeColumnsToContents()
            group.addWidget(table)

        if insp.preview_rows:
            group = self._add_group("数据预览（前几行）")
            table = QTableWidget(len(insp.preview_rows), len(insp.preview_columns))
            table.setHorizontalHeaderLabels(list(insp.preview_columns))
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.verticalHeader().setVisible(False)
            for row, values in enumerate(insp.preview_rows):
                for col, value in enumerate(values):
                    table.setItem(row, col, QTableWidgetItem(_format_value(value)))
            group.addWidget(table)

        group = self._add_group("结构")
        for line in insp.structure:
            group.addWidget(QLabel(line))

        group = self._add_group("元数据")
        meta_rows = [
            ("channels", ", ".join(insp.channels)),
            ("units", ", ".join(insp.units)),
            ("device", _compact_json(insp.device_metadata)),
            ("trial", _compact_json(insp.trial_metadata)),
            ("clock_model", _compact_json(insp.clock_model)),
        ]
        for label, value in meta_rows:
            line = QHBoxLayout()
            key = QLabel(f"{label}：")
            key.setMinimumWidth(140)
            line.addWidget(key)
            line.addWidget(QLabel(value))
            group.addLayout(line)

    def _render_ultrasound(self, insp: UltrasoundInspection) -> None:
        overview = self._add_group("概览")
        rows = [
            ("数据块数", f"{insp.block_count}"),
            ("文件大小", _format_bytes(insp.size_bytes)),
            ("companion", f"{insp.meta_path} / {insp.index_path}"),
        ]
        for label, value in rows:
            line = QHBoxLayout()
            key = QLabel(f"{label}：")
            key.setMinimumWidth(140)
            line.addWidget(key)
            line.addWidget(QLabel(value))
            overview.addLayout(line)

        if insp.metadata:
            group = self._add_group(".meta 元数据")
            table = QTableWidget(len(insp.metadata), 2)
            table.setHorizontalHeaderLabels(["键", "值"])
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.verticalHeader().setVisible(False)
            for row, (key, value) in enumerate(sorted(insp.metadata.items())):
                table.setItem(row, 0, QTableWidgetItem(str(key)))
                table.setItem(row, 1, QTableWidgetItem(_compact_json(value)))
            table.resizeColumnsToContents()
            group.addWidget(table)

    def _render_jsonl(self, insp: JsonlInspection) -> None:
        self._add_title_small(f"事件数：{insp.event_count}")
        if insp.preview:
            group = self._add_group("前几行")
            for index, payload in enumerate(insp.preview):
                group.addWidget(QLabel(f"{index + 1}. {_compact_json(payload)}"))

    def _add_title(self, text: str) -> None:
        label = QLabel(text)
        label.setStyleSheet("QLabel { font-weight: bold; font-size: 14px; }")
        label.setWordWrap(True)
        self.detail_layout.addWidget(label)

    def _add_title_small(self, text: str) -> None:
        self.detail_layout.addWidget(QLabel(text))

    def _add_group(self, title: str) -> QVBoxLayout:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        self.detail_layout.addWidget(group)
        return layout

    def _clear_detail(self) -> None:
        while self.detail_layout.count():
            item = self.detail_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)
