"""Result dialogs for Data Studio's read-only local tools."""

from __future__ import annotations

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


def _empty_tab(message: str) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    label = QLabel(message)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    return widget


def _signal_tab(
    series: SignalPlayback | None,
    *,
    y_label: str,
    trigger_times_s: np.ndarray | None = None,
    cursor_lines: list[pg.InfiniteLine] | None = None,
) -> QWidget:
    if series is None or series.time_s.size == 0:
        return _empty_tab("该 Trial 没有可回放的已发布数据。")
    plot = pg.PlotWidget()
    plot.setBackground("w")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel("bottom", "相对正式 t0 时间", units="s")
    plot.setLabel("left", y_label)
    plot.addLegend(offset=(10, 10))
    channel_count = min(int(series.values.shape[1]), 12)
    for index in range(channel_count):
        name = series.channels[index] if index < len(series.channels) else f"ch_{index + 1}"
        unit = series.units[index] if index < len(series.units) else ""
        legend = f"{name} [{unit}]" if unit else name
        plot.plot(
            series.time_s,
            series.values[:, index],
            pen=pg.mkPen(_PLOT_COLORS[index % len(_PLOT_COLORS)], width=1.2),
            name=legend,
        )
    for trigger_time in np.asarray(
        trigger_times_s if trigger_times_s is not None else [], dtype=float
    ):
        plot.addItem(
            pg.InfiniteLine(
                pos=float(trigger_time),
                angle=90,
                pen=pg.mkPen("#C00000", width=1.5, style=Qt.PenStyle.DashLine),
                label="trigger",
            )
        )
    cursor = pg.InfiniteLine(
        pos=0.0,
        angle=90,
        movable=False,
        pen=pg.mkPen("#202020", width=1.4),
    )
    cursor.setZValue(20)
    plot.addItem(cursor)
    if cursor_lines is not None:
        cursor_lines.append(cursor)
    return plot


class _UltrasoundPlaybackWidget(QWidget):
    """Time-addressable ultrasound waterfall and A-scan panel."""

    def __init__(self, playback: TrialPlayback) -> None:
        super().__init__()
        ultrasound = playback.ultrasound
        if ultrasound is None or ultrasound.waterfall.size == 0:
            raise ValueError("ultrasound playback data is empty")
        self.ultrasound = ultrasound
        self._times = np.asarray(ultrasound.time_s, dtype=np.float64)
        self._selected_frame = -1

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("瀑布通道："))
        self.channel_combo = QComboBox()
        self.channel_combo.setObjectName("playback_ultrasound_channel")
        for index in range(int(ultrasound.waterfall.shape[0])):
            label = (
                ultrasound.channels[index]
                if index < len(ultrasound.channels)
                else f"ch_{index + 1}"
            )
            self.channel_combo.addItem(label, index)
        self.channel_combo.currentIndexChanged.connect(self._set_waterfall_channel)
        controls.addWidget(self.channel_combo)
        self.frame_label = QLabel()
        self.frame_label.setObjectName("playback_ultrasound_frame")
        controls.addWidget(self.frame_label, 1)
        layout.addLayout(controls)

        graphics = pg.GraphicsLayoutWidget()
        graphics.setBackground("w")
        self.waterfall_plot = graphics.addPlot(
            row=0, col=0, title="A-mode 灰度瀑布（有界降采样）"
        )
        self.waterfall_plot.setLabel("bottom", "相对正式 t0 时间", units="s")
        self.waterfall_plot.setLabel("left", "深度点")
        self.image = pg.ImageItem()
        self.waterfall_plot.addItem(self.image)
        self.waterfall_plot.invertY(True)
        self.cursor = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#202020", width=1.4),
        )
        self.cursor.setZValue(20)
        self.waterfall_plot.addItem(self.cursor)

        self.waveform_plot = graphics.addPlot(
            row=1, col=0, title="当前时标 A-scan（四通道）"
        )
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.2)
        self.waveform_plot.setLabel("bottom", "深度点")
        self.waveform_plot.setLabel("left", "幅值")
        self.waveform_plot.addLegend(offset=(10, 10))
        self.waveform_curves: list[pg.PlotDataItem] = []
        for channel_index in range(min(int(ultrasound.waterfall.shape[0]), 8)):
            label = (
                ultrasound.channels[channel_index]
                if channel_index < len(ultrasound.channels)
                else f"ch_{channel_index + 1}"
            )
            curve = self.waveform_plot.plot(
                pen=pg.mkPen(
                    _PLOT_COLORS[channel_index % len(_PLOT_COLORS)], width=1.2
                ),
                name=label,
            )
            self.waveform_curves.append(curve)
        layout.addWidget(graphics, 1)
        self._set_waterfall_channel(0)
        self.set_time(float(self._times[0]) if self._times.size else 0.0)

    def _set_waterfall_channel(self, combo_index: int) -> None:
        channel = self.channel_combo.itemData(combo_index)
        channel_index = int(channel if channel is not None else max(combo_index, 0))
        data = np.asarray(self.ultrasound.waterfall[channel_index]).T
        self.image.setImage(data, autoLevels=True)
        if self._times.size:
            start = float(self._times[0])
            end = float(self._times[-1])
            width = max(end - start, 1e-9)
            self.image.setRect(
                QRectF(start, 0.0, width, float(max(1, data.shape[0])))
            )

    def set_time(self, value: float) -> None:
        self.cursor.setPos(float(value))
        if not self._times.size:
            return
        insertion = int(np.searchsorted(self._times, value, side="left"))
        if insertion <= 0:
            frame_index = 0
        elif insertion >= self._times.size:
            frame_index = int(self._times.size - 1)
        else:
            before = insertion - 1
            frame_index = (
                before
                if abs(value - float(self._times[before]))
                <= abs(float(self._times[insertion]) - value)
                else insertion
            )
        if frame_index == self._selected_frame:
            return
        self._selected_frame = frame_index
        for channel_index, curve in enumerate(self.waveform_curves):
            curve.setData(
                np.asarray(self.ultrasound.waterfall[channel_index, frame_index])
            )
        self.frame_label.setText(
            f"抽样帧 {frame_index + 1}/{self._times.size} · "
            f"t={float(self._times[frame_index]):.3f} s"
        )


class PlaybackDialog(QDialog):
    """Interactive, shared-cursor playback of one immutable Trial snapshot."""

    def __init__(self, playback: TrialPlayback, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.playback = playback
        self.setObjectName("trial_playback_dialog")
        self.setWindowTitle(f"离线回放 · {playback.condition_code} · {playback.trial_uuid[:8]}")
        self.resize(1180, 820)
        self._cursor_lines: list[pg.InfiniteLine] = []
        self._ultrasound_widget: _UltrasoundPlaybackWidget | None = None
        self._playing = False
        self._last_tick = perf_counter()
        self._time_min, self._time_max = self._playback_bounds(playback)
        self._current_time = self._time_min

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Trial {playback.trial_uuid} · 正式 t0 = "
                f"{playback.formal_t0_host_monotonic_ns} ns · 图形为有界降采样视图"
            )
        )
        controls = QHBoxLayout()
        self.play_button = QPushButton("播放")
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
        self.time_label.setMinimumWidth(170)
        controls.addWidget(self.time_label)
        controls.addWidget(QLabel("速度："))
        self.speed_combo = QComboBox()
        self.speed_combo.setObjectName("playback_speed")
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)
        controls.addWidget(self.speed_combo)
        layout.addLayout(controls)

        tabs = QTabWidget()
        tabs.setObjectName("playback_tabs")
        if playback.ultrasound is None or playback.ultrasound.waterfall.size == 0:
            ultrasound_tab: QWidget = _empty_tab("该 Trial 没有可回放的超声数据。")
        else:
            self._ultrasound_widget = _UltrasoundPlaybackWidget(playback)
            ultrasound_tab = self._ultrasound_widget
        tabs.addTab(ultrasound_tab, "Ultrasound")
        tabs.addTab(
            _signal_tab(
                playback.imu,
                y_label="IMU",
                cursor_lines=self._cursor_lines,
            ),
            "IMU",
        )
        tabs.addTab(
            _signal_tab(
                playback.encoder,
                y_label="Encoder",
                cursor_lines=self._cursor_lines,
            ),
            "Encoder",
        )
        tabs.addTab(
            _signal_tab(
                playback.sync,
                y_label="Sync pulse",
                trigger_times_s=playback.sync_trigger_times_s,
                cursor_lines=self._cursor_lines,
            ),
            "Sync / trigger",
        )
        layout.addWidget(tabs, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance_playback)
        self.finished.connect(lambda _result: self._timer.stop())
        self.set_playback_time(self._time_min)

    @staticmethod
    def _playback_bounds(playback: TrialPlayback) -> tuple[float, float]:
        arrays: list[np.ndarray] = []
        if playback.ultrasound is not None:
            arrays.append(np.asarray(playback.ultrasound.time_s, dtype=float))
        for series in (playback.imu, playback.encoder, playback.sync):
            if series is not None:
                arrays.append(np.asarray(series.time_s, dtype=float))
        arrays.append(np.asarray(playback.sync_trigger_times_s, dtype=float))
        finite = [array[np.isfinite(array)] for array in arrays if array.size]
        if not finite:
            return 0.0, 1.0
        minimum = min(float(array.min()) for array in finite)
        maximum = max(float(array.max()) for array in finite)
        if maximum <= minimum:
            maximum = minimum + 1.0
        return minimum, maximum

    def set_playback_time(self, value: float) -> None:
        bounded = min(max(float(value), self._time_min), self._time_max)
        self._current_time = bounded
        span = self._time_max - self._time_min
        slider_value = int(round((bounded - self._time_min) / span * 10_000))
        with QSignalBlocker(self.timeline_slider):
            self.timeline_slider.setValue(slider_value)
        self.time_label.setText(
            f"t={bounded:.3f} s / {self._time_max:.3f} s"
        )
        for cursor in self._cursor_lines:
            cursor.setPos(bounded)
        if self._ultrasound_widget is not None:
            self._ultrasound_widget.set_time(bounded)

    def _slider_changed(self, slider_value: int) -> None:
        fraction = float(slider_value) / 10_000.0
        self.set_playback_time(
            self._time_min + fraction * (self._time_max - self._time_min)
        )
        self._last_tick = perf_counter()

    def toggle_playback(self) -> None:
        self._playing = not self._playing
        if self._playing:
            if self._current_time >= self._time_max:
                self.set_playback_time(self._time_min)
            self._last_tick = perf_counter()
            self._timer.start()
            self.play_button.setText("暂停")
        else:
            self._timer.stop()
            self.play_button.setText("播放")

    def _advance_playback(self) -> None:
        now = perf_counter()
        elapsed = max(0.0, now - self._last_tick)
        self._last_tick = now
        speed = float(self.speed_combo.currentData() or 1.0)
        target = self._current_time + elapsed * speed
        if target >= self._time_max:
            self.set_playback_time(self._time_max)
            self._playing = False
            self._timer.stop()
            self.play_button.setText("播放")
            return
        self.set_playback_time(target)


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


__all__ = [
    "ChecksumDialog",
    "FullStatisticsDialog",
    "PlaybackDialog",
    "QualityAuditDialog",
]
