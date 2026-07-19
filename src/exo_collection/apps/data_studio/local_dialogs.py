"""Result dialogs for Data Studio's read-only local tools."""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .local_tools import (
    ChecksumReport,
    FullStatistics,
    QualityAudit,
    SignalPlayback,
    TrialPlayback,
)

_log = logging.getLogger(__name__)


_PLOT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)

def _safe_colormap() -> np.ndarray | None:
    """Return a viridis-ish lookup table that works across pyqtgraph versions."""
    try:
        import pyqtgraph as _pg
        cmap = _pg.colormap.get("viridis")
        return cmap.getLookupTable(nPts=256, alpha=False)
    except Exception:
        return None  # fall back to grayscale


def _empty_tab(message: str) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    label = QLabel(message)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    return widget


# Playback widgets and dialog are defined below the result-only dialogs.

class FullStatisticsDialog(QDialog):
    def __init__(
        self, statistics: FullStatistics, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.statistics = statistics
        self.setObjectName("full_statistics_dialog")
        self.setWindowTitle("全盘统计")
        self.resize(760, 620)
        layout = QVBoxLayout(self)
        cards = QGridLayout()
        values = (
            ("项目", statistics.projects),
            ("受试者", statistics.subjects),
            ("Session", statistics.sessions),
            ("Trial", statistics.trials),
            ("已最终化", statistics.finalized_trials),
            ("总时长", f"{statistics.total_duration_s:.2f} s"),
            ("Artifact", statistics.artifact_count),
            ("总数据量", f"{statistics.artifact_bytes / (1024 ** 2):.2f} MiB"),
        )
        for index, (name, value) in enumerate(values):
            cards.addWidget(QLabel(f"{name}：{value}"), index // 2, index % 2)
        layout.addLayout(cards)

        table = QTableWidget(0, 4)
        table.setObjectName("full_statistics_table")
        table.setHorizontalHeaderLabels(["分组", "名称", "数量", "数据量 / 时长"])
        rows: list[tuple[str, str, str, str]] = []
        for condition, values_by_condition in statistics.by_condition.items():
            rows.append(
                (
                    "工况",
                    condition,
                    str(int(values_by_condition.get("trial_count") or 0)),
                    f"{float(values_by_condition.get('duration_s') or 0.0):.2f} s",
                )
            )
        for quality, count in statistics.by_quality.items():
            rows.append(("质量", quality, str(count), "-"))
        for modality, values_by_modality in statistics.by_modality.items():
            rows.append(
                (
                    "模态",
                    modality,
                    str(values_by_modality["artifact_count"]),
                    f"{values_by_modality['size_bytes'] / (1024 ** 2):.2f} MiB",
                )
            )
        table.setRowCount(len(rows))
        for row, values_for_row in enumerate(rows):
            for column, value in enumerate(values_for_row):
                table.setItem(row, column, QTableWidgetItem(value))
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ChecksumDialog(QDialog):
    def __init__(self, report: ChecksumReport, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.report = report
        self.setObjectName("checksum_dialog")
        self.setWindowTitle(f"SHA-256 校验 · {report.trial_uuid[:8]}")
        self.resize(980, 650)
        layout = QVBoxLayout(self)
        summary = QLabel(
            "全部通过" if report.passed else "校验未通过：请查看红色条目"
        )
        summary.setObjectName("checksum_summary")
        summary.setStyleSheet(
            "QLabel { padding: 8px; font-weight: 600; "
            + (
                "background: #e9f7ef; color: #155724; }"
                if report.passed
                else "background: #f8d7da; color: #842029; }"
            )
        )
        layout.addWidget(summary)
        table = QTableWidget(len(report.items), 5)
        table.setObjectName("checksum_results")
        table.setHorizontalHeaderLabels(
            ["结果", "相对路径", "大小", "实际 SHA-256", "说明"]
        )
        for row, item in enumerate(report.items):
            values = (
                "PASS" if item.passed else "FAIL",
                item.relative_path,
                f"{item.size_bytes:,} B" if item.size_bytes is not None else "-",
                item.actual_sha256 or "-",
                item.message,
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if not item.passed:
                    cell.setBackground(pg.mkColor("#f8d7da"))
                table.setItem(row, column, cell)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _mapping_table(
    rows: tuple[dict[str, object], ...], columns: tuple[tuple[str, str], ...]
) -> QTableWidget:
    table = QTableWidget(len(rows), len(columns))
    table.setHorizontalHeaderLabels([label for _key, label in columns])
    for row_index, row in enumerate(rows):
        for column_index, (key, _label) in enumerate(columns):
            table.setItem(
                row_index,
                column_index,
                QTableWidgetItem(str(row.get(key, "") or "-")),
            )
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    return table


class QualityAuditDialog(QDialog):
    """Automated evidence plus an append-only human review form."""

    def __init__(
        self,
        audit: QualityAudit,
        parent: QWidget | None = None,
        *,
        review_submit: Callable[[str, str, str], QualityAudit] | None = None,
    ) -> None:
        super().__init__(parent)
        self.audit = audit
        self._review_submit = review_submit
        self.setObjectName("quality_audit_dialog")
        self.setWindowTitle(f"质量审核 · {audit.trial_uuid[:8]}")
        self.resize(980, 700)
        layout = QVBoxLayout(self)
        self.summary_label = QLabel()
        self.summary_label.setObjectName("quality_summary")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "QLabel { padding: 8px; font-weight: 600; background: #eef4fb; "
            "color: #16324f; border: 1px solid #bdd3ea; }"
        )
        self._render_review_summary()
        layout.addWidget(self.summary_label)
        layout.addWidget(QLabel(f"质控算法：{audit.algorithm_version or '-'}"))
        tabs = QTabWidget()
        issue_rows = tuple(
            {
                "severity": item.get("severity", ""),
                "code": item.get("code", ""),
                "modality": item.get("modality", ""),
                "message": item.get("message", ""),
            }
            for item in audit.issues
        )
        issue_table = _mapping_table(
            issue_rows,
            (
                ("severity", "级别"),
                ("code", "代码"),
                ("modality", "模态"),
                ("message", "说明"),
            ),
        )
        issue_table.setObjectName("quality_issues")
        tabs.addTab(issue_table, f"问题 ({len(issue_rows)})")
        device_table = _mapping_table(
            tuple(dict(row) for row in audit.devices),
            (
                ("modality", "模态"),
                ("health_status", "健康"),
                ("actual_sample_rate_hz", "实际采样率"),
                ("persisted_item_count", "已保存"),
                ("dropped_item_count", "丢包/丢帧"),
                ("sequence_gap_count", "序号缺口"),
                ("fault", "故障"),
            ),
        )
        device_table.setObjectName("quality_devices")
        tabs.addTab(device_table, f"设备 ({len(audit.devices)})")
        sync_table = _mapping_table(
            tuple(dict(row) for row in audit.sync_checks),
            (
                ("status", "状态"),
                ("quality", "质量"),
                ("trigger_count", "Trigger 数"),
                ("pulse_event_count", "脉冲数"),
                ("pretrigger_duration_s", "预触发 (s)"),
                ("formal_duration_s", "正式时长 (s)"),
                ("source_device", "来源"),
                ("confidence", "置信度"),
            ),
        )
        sync_table.setObjectName("quality_sync")
        tabs.addTab(sync_table, f"同步 ({len(audit.sync_checks)})")
        warnings = QPlainTextEdit()
        warnings.setObjectName("quality_warnings")
        warnings.setReadOnly(True)
        warnings.setPlainText(audit.warnings_text or "未发布 warnings.txt。")
        tabs.addTab(warnings, "Warnings")
        metrics = QPlainTextEdit()
        metrics.setReadOnly(True)
        metrics.setPlainText(
            "\n".join(
                f"{key}: {value}" for key, value in sorted(audit.soft_metrics.items())
            )
            or "无软质控指标。"
        )
        tabs.addTab(metrics, "US 软质控指标")
        layout.addWidget(tabs, 1)

        review_box = QGroupBox("追加人工审核（不改写原 Manifest）")
        review_form = QFormLayout(review_box)
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setObjectName("quality_reviewer")
        self.reviewer_edit.setMaxLength(120)
        self.reviewer_edit.setPlaceholderText("填写去标识化的审核人编码")
        review_form.addRow("审核人编码：", self.reviewer_edit)
        self.review_grade_combo = QComboBox()
        self.review_grade_combo.setObjectName("quality_review_grade")
        for grade, label in (
            ("A", "A — 通过"),
            ("B", "B — 有警告但可用"),
            ("C", "C — 需谨慎使用"),
            ("INVALID", "INVALID — 不可用"),
        ):
            self.review_grade_combo.addItem(label, grade)
        review_form.addRow("人工等级：", self.review_grade_combo)
        self.review_reason_edit = QPlainTextEdit()
        self.review_reason_edit.setObjectName("quality_review_reason")
        self.review_reason_edit.setMaximumHeight(82)
        self.review_reason_edit.setPlaceholderText("必填：记录信号、同步、动作或实验现场判断依据。")
        review_form.addRow("审核理由：", self.review_reason_edit)
        self.save_review_button = QPushButton("追加审核记录")
        self.save_review_button.setObjectName("quality_review_save")
        self.save_review_button.setEnabled(review_submit is not None)
        self.save_review_button.clicked.connect(self._save_review)
        review_form.addRow("", self.save_review_button)
        if review_submit is None:
            self.save_review_button.setToolTip("当前只读打开；需由 Data Studio 绑定审核存储服务。")
        layout.addWidget(review_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _render_review_summary(self) -> None:
        reviewed = self.audit.reviewed_grade or "未人工复核"
        detail = ""
        if self.audit.reviewed_grade:
            detail = (
                f" · 审核人：{self.audit.reviewed_by or '-'}"
                f" · 审核数：{self.audit.review_count}"
            )
        self.summary_label.setText(
            f"计算等级：{self.audit.computed_grade} · 人工等级：{reviewed}{detail} · "
            f"必需 Artifact：{'PASS' if self.audit.required_artifacts_complete else 'FAIL'} · "
            f"完整性：{'PASS' if self.audit.integrity_checks_passed else 'FAIL'}"
        )
        tooltip_parts = []
        if self.audit.reviewed_at_utc:
            tooltip_parts.append(f"时间：{self.audit.reviewed_at_utc}")
        if self.audit.review_reason:
            tooltip_parts.append(f"理由：{self.audit.review_reason}")
        self.summary_label.setToolTip("\n".join(tooltip_parts))

    def _save_review(self) -> None:
        if self._review_submit is None:
            return
        reviewer = self.reviewer_edit.text().strip()
        reason = self.review_reason_edit.toPlainText().strip()
        grade = str(self.review_grade_combo.currentData() or "")
        if not reviewer or not reason:
            QMessageBox.warning(self, "审核信息不完整", "审核人编码和审核理由均为必填。")
            return
        try:
            updated = self._review_submit(grade, reviewer, reason)
        except Exception as exc:
            QMessageBox.critical(self, "审核记录未保存", str(exc))
            return
        self.audit = updated
        self._render_review_summary()
        self.review_reason_edit.clear()
        QMessageBox.information(
            self,
            "审核记录已追加",
            "记录已写入 Manifest SHA-256 锚定的追加式审计链；原 Trial 未被改写。",
        )


class _SweepWaterfallPlot(pg.PlotWidget):
    """One ultrasound channel rendered as a fixed cyclic time/depth image."""

    def __init__(
        self,
        title: str,
        time_s: np.ndarray,
        depth_by_time: np.ndarray,
        window_s: float,
    ) -> None:
        super().__init__()
        self._times = np.asarray(time_s, dtype=np.float64)
        # Keep a view of the bounded playback array.  The same source is used
        # by both the combined and modality-specific tabs, so converting every
        # plot to a private float32 array would unnecessarily duplicate tens of
        # megabytes of ultrasound data.
        self._data = np.asarray(depth_by_time)
        self._window_s = float(window_s)
        self._columns = max(96, min(800, int(self._times.size or 96)))
        self.setTitle(title)
        self.setLabel("bottom", "循环时间", units="s")
        self.setLabel("left", "深度点")
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.setXRange(0.0, self._window_s, padding=0.0)
        self.setYRange(0.0, 999.0, padding=0.0)
        self.getViewBox().setMouseEnabled(x=False, y=False)
        # ``_canvas`` is indexed as [depth, time-column].  PyQtGraph defaults
        # to column-major image coordinates, which interprets the first axis
        # as X and turns each A-scan into a horizontal stripe.  Row-major makes
        # columns advance left-to-right in time while rows remain depth 0..999.
        self.image = pg.ImageItem(axisOrder="row-major")
        lookup = _safe_colormap()
        if lookup is not None:
            self.image.setLookupTable(lookup)
        self.addItem(self.image)
        self.invertY(True)
        flattened = self._data.reshape(-1)
        level_stride = max(1, flattened.size // 200_000)
        level_sample = flattened[::level_stride]
        finite = level_sample[np.isfinite(level_sample)]
        if finite.size:
            low, high = np.percentile(finite, (1.0, 99.0))
            self._levels = (float(low), float(high if high > low else low + 1.0))
        else:
            self._levels = (0.0, 1.0)
        self.cursor = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#e11d48", width=2.0),
        )
        self.cursor.setZValue(100)
        self.addItem(self.cursor)
        depth = self._data.shape[0] if self._data.ndim == 2 else 0
        self._canvas = np.full(
            (max(1, depth), self._columns), np.nan, dtype=np.float32
        )
        self._last_cycle_start: float | None = None
        self._last_current: float | None = None

    def update_time(self, current_s: float, cycle_start_s: float) -> None:
        phase = min(max(float(current_s - cycle_start_s), 0.0), self._window_s)
        depth = self._data.shape[0] if self._data.ndim == 2 else 0
        continuing = (
            self._last_cycle_start == cycle_start_s
            and self._last_current is not None
            and current_s >= self._last_current
        )
        if not continuing:
            self._canvas.fill(np.nan)
        lower_bound = self._last_current if continuing else cycle_start_s
        image_changed = not continuing
        if depth and self._times.size:
            new_indices = np.flatnonzero(
                (self._times > lower_bound if continuing else self._times >= lower_bound)
                & (self._times <= current_s)
            )
            # Also repaint the sample held at ``lower_bound``.  A sampled
            # waterfall represents each A-scan until the next A-scan arrives;
            # writing only one pixel column per frame creates comb-like black
            # gaps, especially after bounded/downsampled loading.
            held_index = int(np.searchsorted(self._times, lower_bound, side="right") - 1)
            indices = new_indices.tolist()
            if held_index >= 0:
                indices.insert(0, held_index)
            for source_index in dict.fromkeys(indices):
                sample_time = float(self._times[source_index])
                paint_start = max(float(lower_bound), float(cycle_start_s), sample_time)
                if paint_start > current_s:
                    continue
                next_time = (
                    float(self._times[source_index + 1])
                    if source_index + 1 < self._times.size
                    else float(current_s)
                )
                paint_end = min(float(current_s), max(paint_start, next_time))
                start_column = int(
                    np.floor(
                        (paint_start - cycle_start_s)
                        / self._window_s
                        * (self._columns - 1)
                    )
                )
                end_column = int(
                    np.floor(
                        (paint_end - cycle_start_s)
                        / self._window_s
                        * (self._columns - 1)
                    )
                )
                start_column = int(np.clip(start_column, 0, self._columns - 1))
                end_column = int(np.clip(end_column, start_column, self._columns - 1))
                self._canvas[:, start_column : end_column + 1] = self._data[
                    :, source_index : source_index + 1
                ]
                image_changed = True
        if image_changed:
            self.image.setImage(self._canvas, autoLevels=False, levels=self._levels)
            self.image.setRect(
                QRectF(0.0, 0.0, self._window_s, float(max(1, depth)))
            )
        self._last_cycle_start = cycle_start_s
        self._last_current = current_s
        self.cursor.setPos(phase)


class _UltrasoundCurrentFramePlot(pg.PlotWidget):
    """Narrow depth-oriented A-scan view synchronized to playback time."""

    def __init__(
        self,
        time_s: np.ndarray,
        waterfall: np.ndarray,
        channels: tuple[str, ...],
        *,
        object_name: str,
    ) -> None:
        super().__init__()
        self.setObjectName(object_name)
        self._times = np.asarray(time_s, dtype=np.float64)
        self._frames = np.asarray(waterfall)
        self._curves: list[pg.PlotDataItem] = []
        self._last_source_index: int | None = None
        self.setTitle("当前帧 A-scan")
        self.setLabel("bottom", "信号幅值")
        self.setLabel("left", "深度点")
        self.setMinimumWidth(180)
        self.setMaximumWidth(320)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.getViewBox().setMouseEnabled(x=False, y=False)
        self.setYRange(0.0, 999.0, padding=0.0)
        self.invertY(True)
        legend = self.addLegend(offset=(4, 4))
        legend.setBrush(pg.mkBrush(0, 0, 0, 150))

        channel_count = (
            min(4, int(self._frames.shape[0])) if self._frames.ndim == 3 else 0
        )
        for channel in range(channel_count):
            label = channels[channel] if channel < len(channels) else f"ch_{channel + 1}"
            self._curves.append(
                self.plot(
                    [],
                    [],
                    name=label,
                    pen=pg.mkPen(
                        _PLOT_COLORS[channel % len(_PLOT_COLORS)], width=1.2
                    ),
                )
            )

        flattened = self._frames.reshape(-1) if self._frames.size else np.empty(0)
        stride = max(1, flattened.size // 200_000) if flattened.size else 1
        sample = flattened[::stride]
        finite = sample[np.isfinite(sample)]
        if finite.size:
            low, high = np.percentile(finite, (0.5, 99.5))
            span = max(float(high - low), 1.0)
            self.setXRange(
                float(low - 0.05 * span),
                float(high + 0.05 * span),
                padding=0.0,
            )
        else:
            self.setXRange(0.0, 1.0, padding=0.0)

    def update_time(self, current_s: float, _cycle_start_s: float) -> None:
        if not self._times.size or not self._curves:
            return
        source_index = int(np.searchsorted(self._times, current_s, side="right") - 1)
        if source_index < 0:
            for curve in self._curves:
                curve.setData([], [])
            self._last_source_index = None
            return
        source_index = min(source_index, int(self._times.size - 1))
        if source_index == self._last_source_index:
            return
        depth_count = min(1000, int(self._frames.shape[2]))
        depth = np.arange(depth_count, dtype=np.float32)
        for channel, curve in enumerate(self._curves):
            curve.setData(self._frames[channel, source_index, :depth_count], depth)
        self._last_source_index = source_index


class _SweepSignalPlot(pg.PlotWidget):
    """Line plot that clears at each fixed-window sweep boundary."""

    def __init__(
        self,
        title: str,
        series: SignalPlayback,
        indices: tuple[int, ...],
        window_s: float,
    ) -> None:
        super().__init__()
        self._times = np.asarray(series.time_s, dtype=np.float64)
        self._values = np.asarray(series.values)
        self._window_s = float(window_s)
        self._curves: list[tuple[int, pg.PlotDataItem]] = []
        self.setTitle(title)
        self.setLabel("bottom", "循环时间", units="s")
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.setXRange(0.0, self._window_s, padding=0.0)
        self.getViewBox().setMouseEnabled(x=False, y=False)
        finite_parts: list[np.ndarray] = []
        for color_index, index in enumerate(indices):
            if index >= self._values.shape[1]:
                continue
            label = series.channels[index] if index < len(series.channels) else f"ch_{index + 1}"
            curve = self.plot(
                [],
                [],
                name=label,
                pen=pg.mkPen(_PLOT_COLORS[color_index % len(_PLOT_COLORS)], width=1.1),
            )
            self._curves.append((index, curve))
            values = np.asarray(self._values[:, index], dtype=float)
            finite_parts.append(values[np.isfinite(values)])
        finite_parts = [part for part in finite_parts if part.size]
        finite = np.concatenate(finite_parts) if finite_parts else np.empty(0)
        if finite.size:
            low, high = np.percentile(finite, (0.5, 99.5))
            span = max(float(high - low), 1e-6)
            self.setYRange(float(low - 0.08 * span), float(high + 0.08 * span), padding=0.0)
        else:
            self.setYRange(-1.0, 1.0, padding=0.0)
        self.cursor = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#e11d48", width=2.0),
        )
        self.cursor.setZValue(100)
        self.addItem(self.cursor)

    def update_time(self, current_s: float, cycle_start_s: float) -> None:
        phase = min(max(float(current_s - cycle_start_s), 0.0), self._window_s)
        mask = (self._times >= cycle_start_s) & (self._times <= current_s)
        x_values = self._times[mask] - cycle_start_s
        for index, curve in self._curves:
            curve.setData(x_values, self._values[mask, index])
        self.cursor.setPos(phase)


def _measurement_kind(label: str) -> str | None:
    lowered = label.casefold()
    if "acc" in lowered:
        return "acc"
    if "gyr" in lowered or "gyro" in lowered:
        return "gyr"
    if "mag" in lowered:
        return "mag"
    return None


def _imu_sensor_groups(
    series: SignalPlayback,
) -> list[tuple[str, dict[str, tuple[int, ...]]]]:
    """Recover physical-IMU and measurement groups without inventing IDs."""

    grouped: dict[str, dict[str, list[int]]] = {}
    order: list[str] = []
    for index, raw_label in enumerate(series.channels):
        label = str(raw_label)
        if ":" in label:
            raw_sensor, measurement_label = label.split(":", 1)
            try:
                ordinal = int(raw_sensor) - 1
            except ValueError:
                ordinal = -1
            sensor = (
                series.sensor_labels[ordinal]
                if 0 <= ordinal < len(series.sensor_labels)
                else f"IMU {raw_sensor}"
            )
        else:
            measurement_label = label
            sensor = series.sensor_labels[0] if len(series.sensor_labels) == 1 else "IMU 1"
            for candidate in series.sensor_labels:
                prefix = f"{candidate}_"
                if label.casefold().startswith(prefix.casefold()):
                    sensor = candidate
                    measurement_label = label[len(prefix):]
                    break
        kind = _measurement_kind(measurement_label)
        if kind is None:
            continue
        if sensor not in grouped:
            grouped[sensor] = {"acc": [], "mag": [], "gyr": []}
            order.append(sensor)
        grouped[sensor][kind].append(index)
    return [
        (sensor, {kind: tuple(indices) for kind, indices in grouped[sensor].items()})
        for sensor in order[:3]
    ]


def _encoder_side_groups(series: SignalPlayback) -> list[tuple[str, tuple[int, ...]]]:
    """Group published encoder channels by explicit left/right prefixes."""

    result: list[tuple[str, tuple[int, ...]]] = []
    used: set[int] = set()
    for side, title in (("left", "左侧电机编码器"), ("right", "右侧电机编码器")):
        indices = tuple(
            index
            for index, channel in enumerate(series.channels)
            if str(channel).casefold().startswith(f"{side}_")
        )
        if indices:
            result.append((title, indices))
            used.update(indices)
    # Older two-channel files may not publish a side prefix.  Preserve their
    # actual channel labels as independent windows rather than assigning a
    # fabricated left/right identity.
    for index, channel in enumerate(series.channels):
        if index not in used and len(result) < 2:
            result.append((f"电机编码器 · {channel}", (index,)))
    return result[:2]


class PlaybackDialog(QDialog):
    """Non-blocking, fixed-window cyclic playback for every recorded modality."""

    _WINDOW_SECONDS = 10.0

    def __init__(self, playback: TrialPlayback, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.playback = playback
        self.setObjectName("trial_playback_dialog")
        self.setWindowTitle(f"离线回放 · {playback.condition_code} · {playback.trial_uuid[:8]}")
        self.resize(1500, 940)
        self._playing = False
        self._last_tick = perf_counter()
        self._time_min, self._time_max = self._playback_bounds(playback)
        self._current_time = self._time_min
        total_span = max(self._time_max - self._time_min, 1e-6)
        self._window_s = min(self._WINDOW_SECONDS, max(1.0, total_span))
        self._sweep_plots: list[object] = []
        _log.info(
            "创建离线回放: trial=%s, time=[%.3f, %.3f]s, window=%.3fs, "
            "ultrasound=%s, imu=%s, encoder=%s",
            playback.trial_uuid,
            self._time_min,
            self._time_max,
            self._window_s,
            playback.ultrasound is not None,
            playback.imu is not None,
            playback.encoder is not None,
        )

        layout = QVBoxLayout(self)
        banner = QLabel(
            f"Trial {playback.trial_uuid} · 固定 {self._window_s:g} s 循环窗口 · "
            "红线表示当前回放位置"
        )
        banner.setObjectName("playback_banner")
        layout.addWidget(banner)

        # Keep playback controls above the plot area.  When a smaller display
        # cannot satisfy every plot's size hint, Qt may clip the bottom of a
        # dialog; a bottom toolbar then becomes unreachable even though the
        # playback window itself opened successfully.
        control_bar = QWidget(self)
        control_bar.setObjectName("playback_control_bar")
        control_bar.setStyleSheet(
            "QWidget#playback_control_bar {"
            " background: #e8f1ff; border: 1px solid #9bb8df;"
            " border-radius: 6px; }"
            "QPushButton#playback_play_pause {"
            " background: #2563d9; color: white; font-weight: 700;"
            " min-width: 108px; min-height: 32px; border-radius: 5px; }"
            "QPushButton#playback_play_pause:hover { background: #1d4fb3; }"
        )
        controls = QHBoxLayout(control_bar)
        controls.setContentsMargins(8, 6, 8, 6)
        self.play_button = QPushButton("▶ 播放")
        self.play_button.setObjectName("playback_play_pause")
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setObjectName("playback_timeline")
        self.timeline_slider.setRange(0, 10_000)
        self.timeline_slider.valueChanged.connect(self._slider_changed)
        controls.addWidget(self.timeline_slider, 1)
        self.time_label = QLabel()
        self.time_label.setObjectName("playback_time")
        self.time_label.setMinimumWidth(210)
        controls.addWidget(self.time_label)
        controls.addWidget(QLabel("速度："))
        self.speed_combo = QComboBox()
        self.speed_combo.setObjectName("playback_speed")
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)
        controls.addWidget(self.speed_combo)
        layout.addWidget(control_bar)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("playback_tabs")
        layout.addWidget(self.tabs, 1)
        self._build_all_tab(playback)
        self._build_ultrasound_tab(playback)
        self._build_imu_tab(playback)
        self._build_encoder_tab(playback)
        self._timer = QTimer(self)
        # 20 FPS is ample for visual review and leaves the GUI thread enough
        # time to paint four images plus eleven signal plots reliably.
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance_playback)
        self.finished.connect(lambda _result: self._timer.stop())
        self.set_playback_time(self._time_min)

    def _build_all_tab(self, playback: TrialPlayback) -> None:
        """Build one compact dashboard containing every recorded modality."""

        tab = QWidget()
        tab.setObjectName("playback_all_tab")
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        ultrasound_box = QGroupBox("超声 · 4 通道瀑布图")
        ultrasound_box.setObjectName("playback_all_ultrasound")
        ultrasound_layout = QHBoxLayout(ultrasound_box)
        ultrasound_layout.setContentsMargins(3, 3, 3, 3)
        ultrasound_layout.setSpacing(3)
        waterfall_panel = QWidget(ultrasound_box)
        ultrasound_grid = QGridLayout(waterfall_panel)
        ultrasound_grid.setContentsMargins(3, 3, 3, 3)
        ultrasound_grid.setSpacing(3)
        us = playback.ultrasound
        channel_count = (
            min(4, int(us.waterfall.shape[0]))
            if us is not None and us.waterfall.size
            else 0
        )
        for channel in range(4):
            if us is not None and channel < channel_count:
                label = (
                    us.channels[channel]
                    if channel < len(us.channels)
                    else f"ch_{channel + 1}"
                )
                plot = _SweepWaterfallPlot(
                    f"通道 {channel + 1} · {label}",
                    us.time_s,
                    np.asarray(us.waterfall[channel]).T,
                    self._window_s,
                )
                self._sweep_plots.append(plot)
                ultrasound_grid.addWidget(plot, channel // 2, channel % 2)
            else:
                ultrasound_grid.addWidget(
                    _empty_tab(f"超声通道 {channel + 1}：数据缺失"),
                    channel // 2,
                    channel % 2,
                )
        ultrasound_layout.addWidget(waterfall_panel, 1)
        if us is not None and channel_count:
            current_frame = _UltrasoundCurrentFramePlot(
                us.time_s,
                us.waterfall,
                us.channels,
                object_name="playback_all_current_ultrasound",
            )
            self._sweep_plots.append(current_frame)
            ultrasound_layout.addWidget(current_frame)
        else:
            current_missing = _empty_tab("当前帧超声数据缺失")
            current_missing.setMaximumWidth(240)
            ultrasound_layout.addWidget(current_missing)
        outer.addWidget(ultrasound_box, 4)

        imu_box = QGroupBox("IMU · 3 设备 × 3 传感器")
        imu_box.setObjectName("playback_all_imu")
        imu_grid = QGridLayout(imu_box)
        imu_grid.setContentsMargins(3, 3, 3, 3)
        imu_grid.setSpacing(3)
        imu = playback.imu
        imu_groups = (
            _imu_sensor_groups(imu) if imu is not None and imu.time_s.size else []
        )
        sensor_labels = list(imu.sensor_labels[:3]) if imu is not None else []
        for sensor_slot in range(3):
            if sensor_slot < len(imu_groups):
                sensor, kinds = imu_groups[sensor_slot]
            else:
                sensor = (
                    sensor_labels[sensor_slot]
                    if sensor_slot < len(sensor_labels)
                    else f"IMU {sensor_slot + 1}"
                )
                kinds = {}
            for row, (kind, title) in enumerate(
                (("acc", "加速度计"), ("mag", "磁力计"), ("gyr", "陀螺仪"))
            ):
                indices = kinds.get(kind, ())
                if imu is not None and indices:
                    plot = _SweepSignalPlot(
                        f"{sensor} · {title}", imu, indices, self._window_s
                    )
                    self._sweep_plots.append(plot)
                    imu_grid.addWidget(plot, row, sensor_slot)
                else:
                    imu_grid.addWidget(
                        _empty_tab(f"{sensor} · {title}：数据缺失"),
                        row,
                        sensor_slot,
                    )
        outer.addWidget(imu_box, 3)

        encoder_box = QGroupBox("电机编码器 · 左右通道")
        encoder_box.setObjectName("playback_all_encoder")
        encoder_layout = QHBoxLayout(encoder_box)
        encoder_layout.setContentsMargins(3, 3, 3, 3)
        encoder_layout.setSpacing(3)
        encoder = playback.encoder
        encoder_groups = (
            _encoder_side_groups(encoder)
            if encoder is not None and encoder.time_s.size
            else []
        )
        for side_slot in range(2):
            if encoder is not None and side_slot < len(encoder_groups):
                title, indices = encoder_groups[side_slot]
                plot = _SweepSignalPlot(title, encoder, indices, self._window_s)
                self._sweep_plots.append(plot)
                encoder_layout.addWidget(plot, 1)
            else:
                encoder_layout.addWidget(
                    _empty_tab(f"电机编码器 {side_slot + 1}：数据缺失"), 1
                )
        outer.addWidget(encoder_box, 2)

        self.tabs.addTab(tab, "全部")

    def _build_ultrasound_tab(self, playback: TrialPlayback) -> None:
        tab = QWidget()
        tab_layout = QHBoxLayout(tab)
        waterfall_panel = QWidget(tab)
        grid = QGridLayout(waterfall_panel)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        tab_layout.addWidget(waterfall_panel, 1)
        us = playback.ultrasound
        # Compatibility object for automation written for the old selector;
        # all four channels are now visible at once, so it remains hidden.
        selector = QComboBox(tab)
        selector.setObjectName("playback_ultrasound_channel")
        selector.setVisible(False)
        if us is None or not us.waterfall.size:
            grid.addWidget(_empty_tab("该 Trial 没有可回放的超声数据。"), 0, 0, 2, 2)
            _log.warning("Trial %s 没有超声回放数据", playback.trial_uuid)
        else:
            channel_count = min(4, int(us.waterfall.shape[0]))
            for channel in range(4):
                if channel < channel_count:
                    label = us.channels[channel] if channel < len(us.channels) else f"ch_{channel + 1}"
                    selector.addItem(label)
                    plot = _SweepWaterfallPlot(
                        f"超声通道 {channel + 1} · {label}",
                        us.time_s,
                        np.asarray(us.waterfall[channel]).T,
                        self._window_s,
                    )
                    self._sweep_plots.append(plot)
                    grid.addWidget(plot, channel // 2, channel % 2)
                else:
                    missing = _empty_tab(f"超声通道 {channel + 1}：数据缺失")
                    grid.addWidget(missing, channel // 2, channel % 2)
                    _log.warning("Trial %s 缺少超声通道 %d", playback.trial_uuid, channel + 1)
            if not us.device_synchronized:
                notice = QLabel(
                    f"独立通道包按到达序号组合，不代表设备同步；原始包数 "
                    f"{us.source_packet_count}。"
                )
                notice.setObjectName("playback_ultrasound_alignment_notice")
                notice.setStyleSheet("color: #9a3412; background: #fff7ed; padding: 4px;")
                grid.addWidget(notice, 2, 0, 1, 2)
            current_frame = _UltrasoundCurrentFramePlot(
                us.time_s,
                us.waterfall,
                us.channels,
                object_name="playback_ultrasound_current_frame",
            )
            self._sweep_plots.append(current_frame)
            tab_layout.addWidget(current_frame)
        self.tabs.addTab(tab, "超声 · 4 通道瀑布图")

    def _build_imu_tab(self, playback: TrialPlayback) -> None:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        imu = playback.imu
        groups = _imu_sensor_groups(imu) if imu is not None and imu.time_s.size else []
        _log.info("IMU 回放分组: %s", [(name, kinds) for name, kinds in groups])
        labels = list(imu.sensor_labels[:3]) if imu is not None else []
        for slot in range(3):
            if slot < len(groups):
                sensor, kinds = groups[slot]
            else:
                sensor = labels[slot] if slot < len(labels) else f"IMU {slot + 1}"
                kinds = {}
            group_box = QGroupBox(sensor)
            group_layout = QVBoxLayout(group_box)
            for kind, title in (("acc", "加速度计"), ("mag", "磁力计"), ("gyr", "陀螺仪")):
                indices = kinds.get(kind, ())
                if imu is not None and indices:
                    plot = _SweepSignalPlot(title, imu, indices, self._window_s)
                    self._sweep_plots.append(plot)
                    group_layout.addWidget(plot)
                else:
                    group_layout.addWidget(_empty_tab(f"{title}：数据缺失"))
                    _log.warning("Trial %s %s 缺少%s数据", playback.trial_uuid, sensor, title)
            outer.addWidget(group_box, 1)
        self.tabs.addTab(tab, "IMU · 3 设备 × 3 传感器")

    def _build_encoder_tab(self, playback: TrialPlayback) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        encoder = playback.encoder
        groups = (
            _encoder_side_groups(encoder)
            if encoder is not None and encoder.time_s.size
            else []
        )
        _log.info("编码器回放分组: %s", groups)
        for channel in range(2):
            if encoder is not None and channel < len(groups):
                title, indices = groups[channel]
                plot = _SweepSignalPlot(
                    title,
                    encoder,
                    indices,
                    self._window_s,
                )
                self._sweep_plots.append(plot)
                layout.addWidget(plot, 1)
            else:
                layout.addWidget(_empty_tab(f"电机编码器 {channel + 1}：数据缺失"), 1)
                _log.warning("Trial %s 缺少电机编码器 %d", playback.trial_uuid, channel + 1)
        self.tabs.addTab(tab, "电机编码器 · 2 通道")

    @staticmethod
    def _playback_bounds(playback: TrialPlayback) -> tuple[float, float]:
        arrays: list[np.ndarray] = []
        if playback.ultrasound is not None:
            arrays.append(np.asarray(playback.ultrasound.time_s, dtype=float))
        for series in (playback.imu, playback.encoder, playback.sync):
            if series is not None:
                arrays.append(np.asarray(series.time_s, dtype=float))
        finite = [array[np.isfinite(array)] for array in arrays if array.size]
        if not finite:
            return 0.0, 1.0
        minimum = min(float(array.min()) for array in finite)
        maximum = max(float(array.max()) for array in finite)
        return (minimum, maximum if maximum > minimum else minimum + 1.0)

    def set_playback_time(self, value: float) -> None:
        bounded = min(max(float(value), self._time_min), self._time_max)
        self._current_time = bounded
        total_span = self._time_max - self._time_min
        slider_value = int(round((bounded - self._time_min) / total_span * 10_000))
        with QSignalBlocker(self.timeline_slider):
            self.timeline_slider.setValue(slider_value)
        cycle_index = int((bounded - self._time_min) // self._window_s)
        cycle_start = self._time_min + cycle_index * self._window_s
        if bounded >= self._time_max and bounded == cycle_start:
            cycle_start = max(self._time_min, cycle_start - self._window_s)
        for plot in self._sweep_plots:
            plot.update_time(bounded, cycle_start)
        self.time_label.setText(f"t={bounded:.3f} s / {self._time_max:.3f} s")

    def _slider_changed(self, slider_value: int) -> None:
        fraction = float(slider_value) / 10_000.0
        self.set_playback_time(self._time_min + fraction * (self._time_max - self._time_min))
        self._last_tick = perf_counter()

    def toggle_playback(self) -> None:
        self._playing = not self._playing
        if self._playing:
            if self._current_time >= self._time_max:
                self.set_playback_time(self._time_min)
            self._last_tick = perf_counter()
            self._timer.start()
            self.play_button.setText("⏸ 暂停")
            _log.info("开始离线回放: trial=%s, t=%.3fs", self.playback.trial_uuid, self._current_time)
        else:
            self._timer.stop()
            self.play_button.setText("▶ 播放")
            _log.info("暂停离线回放: trial=%s, t=%.3fs", self.playback.trial_uuid, self._current_time)

    def _advance_playback(self) -> None:
        now = perf_counter()
        elapsed = max(0.0, now - self._last_tick)
        self._last_tick = now
        target = self._current_time + elapsed * float(self.speed_combo.currentData() or 1.0)
        if target >= self._time_max:
            self.set_playback_time(self._time_max)
            self._playing = False
            self._timer.stop()
            self.play_button.setText("▶ 播放")
            _log.info("离线回放结束: trial=%s", self.playback.trial_uuid)
            return
        self.set_playback_time(target)


__all__ = [
    "ChecksumDialog",
    "FullStatisticsDialog",
    "PlaybackDialog",
    "QualityAuditDialog",
]
