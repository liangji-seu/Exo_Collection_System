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


# ---------------------------------------------------------------------------
# Shared waterfall plot (one ultrasound channel)
# ---------------------------------------------------------------------------

class _WaterfallPlotWidget(pg.PlotWidget):
    """Single-channel A-mode waterfall with a movable time cursor."""

    def __init__(
        self,
        title: str,
        time_s: np.ndarray,
        data_2d: np.ndarray,  # (frames, depth)
    ) -> None:
        super().__init__()
        self._times = np.asarray(time_s, dtype=np.float64)
        self._data = np.asarray(data_2d, dtype=np.float32)

        self.setTitle(title)
        self.setLabel("bottom", "时间", units="s")
        self.setLabel("left", "深度点")
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.getViewBox().setMouseEnabled(x=False, y=False)

        _t0 = float(self._times[0]) if self._times.size else 0.0
        _t1 = float(self._times[-1]) if self._times.size else 1.0

        self.image = pg.ImageItem()
        lut = _safe_colormap()
        if lut is not None:
            self.image.setLookupTable(lut)
        self.addItem(self.image)
        self.invertY(True)

        self.cursor = pg.InfiniteLine(
            pos=_t0, angle=90, movable=False,
            pen=pg.mkPen("#E04040", width=2.0),
        )
        self.cursor.setZValue(100)
        self.addItem(self.cursor)

        if self._times.size and self._data.size:
            # self._data is (depth, frames): axis 0 = Y (depth), axis 1 = X (time)
            self.image.setImage(
                self._data,
                autoLevels=True,
                rect=QRectF(
                    _t0, 0.0,
                    max(_t1 - _t0, 1e-9),
                    float(max(1, self._data.shape[0])),
                ),
            )


# ---------------------------------------------------------------------------
# Shared signal plot (one IMU / encoder group)
# ---------------------------------------------------------------------------

class _SignalPlotWidget(pg.PlotWidget):
    """Multi-channel line plot with a movable time cursor."""

    def __init__(
        self,
        title: str,
        series: SignalPlayback,
        channel_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        self._times = np.asarray(series.time_s, dtype=np.float64)
        self._values = np.asarray(series.values)

        self.setTitle(title)
        self.setLabel("bottom", "时间", units="s")
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.getViewBox().setMouseEnabled(x=False, y=False)

        for idx in channel_indices:
            if idx >= self._values.shape[1]:
                continue
            label = (
                series.channels[idx]
                if idx < len(series.channels)
                else f"ch_{idx + 1}"
            )
            unit = series.units[idx] if idx < len(series.units) else ""
            legend = f"{label} [{unit}]" if unit else label
            pen = pg.mkPen(
                _PLOT_COLORS[idx % len(_PLOT_COLORS)], width=1.0
            )
            self.plot(
                self._times,
                self._values[:, idx],
                pen=pen,
                name=legend,
            )

        t0 = float(self._times[0]) if self._times.size else 0.0
        self.cursor = pg.InfiniteLine(
            pos=t0, angle=90, movable=False,
            pen=pg.mkPen("#E04040", width=2.0),
        )
        self.cursor.setZValue(100)
        self.addItem(self.cursor)


# ---------------------------------------------------------------------------
# Channel grouping helpers
# ---------------------------------------------------------------------------

def _partition_indices(
    total: int, groups: int
) -> list[tuple[int, ...]]:
    """Split *total* channel indices evenly into *groups* tuples."""
    if total <= 0 or groups <= 0:
        return [()]
    base = total // groups
    remainder = total % groups
    result: list[tuple[int, ...]] = []
    start = 0
    for g in range(groups):
        size = base + (1 if g < remainder else 0)
        result.append(tuple(range(start, start + size)))
        start += size
    return result


def _imu_groups(series: SignalPlayback) -> list[tuple[str, tuple[int, ...]]]:
    """Heuristic: group IMU channels by sensor type prefix, max 3 groups."""
    labels = [str(c).casefold() for c in series.channels]
    col_count = int(series.values.shape[1])
    # Try to split into accel / gyro / other
    accel = tuple(i for i, n in enumerate(labels) if "accel" in n or "acc" in n)
    gyro = tuple(i for i, n in enumerate(labels) if "gyro" in n or "gyr" in n)
    mag = tuple(i for i, n in enumerate(labels) if "mag" in n)
    used = set(accel + gyro + mag)
    other = tuple(i for i in range(col_count) if i not in used)
    groups: list[tuple[str, tuple[int, ...]]] = []
    if accel:
        groups.append(("IMU Accel", accel))
    if gyro:
        groups.append(("IMU Gyro", gyro))
    if mag:
        groups.append(("IMU Mag", mag))
    if other:
        groups.append(("IMU Other", other))
    # Collapse to at most 3 groups
    while len(groups) > 3:
        # merge the two smallest groups
        groups.sort(key=lambda g: len(g[1]))
        name_a, idx_a = groups[0]
        name_b, idx_b = groups[1]
        groups = groups[2:]
        groups.append((f"{name_a} + {name_b}", idx_a + idx_b))
    if not groups:
        for label, idx in _partition_indices(col_count, min(3, col_count)):
            groups.append(("IMU", idx))
    return groups


def _encoder_groups(series: SignalPlayback) -> list[tuple[str, tuple[int, ...]]]:
    """Split encoder channels into 2 windows."""
    col_count = int(series.values.shape[1])
    result: list[tuple[str, tuple[int, ...]]] = []
    for idx_tuple in _partition_indices(col_count, min(2, max(1, col_count))):
        labels = [
            series.channels[i] if i < len(series.channels) else f"ch_{i + 1}"
            for i in idx_tuple
        ]
        result.append(("Encoder " + " / ".join(labels), idx_tuple))
    return result


# ---------------------------------------------------------------------------
# Interactive playback dialog (fixed axes, sweeping cursor)
# ---------------------------------------------------------------------------

class PlaybackDialog(QDialog):
    """Collector-style offline playback with shared sweeping cursor.

    Layout::

        ┌──────────────┬──────────────┐
        │  US Ch 1     │  US Ch 2     │
        │  (waterfall) │  (waterfall) │
        ├──────────────┼──────────────┤
        │  US Ch 3     │  US Ch 4     │
        │  (waterfall) │  (waterfall) │
        ├──────────────┼──────────────┼──────────────┐
        │  IMU group 1 │  IMU group 2 │  IMU group 3 │
        ├──────────────┼──────────────┼──────────────┤
        │  Encoder 1   │  Encoder 2   │              │
        ├──────────────┴──────────────┴──────────────┤
        │  [▶/⏸] ═══════timeline═══════ 1×  t=X/Xs  │
        └─────────────────────────────────────────────┘
    """

    def __init__(self, playback: TrialPlayback, parent: QWidget | None = None) -> None:
        _log.info("=== PlaybackDialog.__init__ 开始 ===")
        _log.info("Trial: %s, t0=%d ns", playback.trial_uuid, playback.formal_t0_host_monotonic_ns)
        super().__init__(parent)
        self.playback = playback
        self.setObjectName("trial_playback_dialog")
        self.setWindowTitle(
            f"离线回放 · {playback.condition_code} · {playback.trial_uuid[:8]}"
        )
        self.resize(1480, 980)
        self._playing = False
        self._last_tick = perf_counter()
        _log.info("计算时间边界…")
        self._time_min, self._time_max = self._playback_bounds(playback)
        _log.info("时间范围: %.3f – %.3f s", self._time_min, self._time_max)
        self._current_time = self._time_min

        # One cursor per plot — synced in set_playback_time
        self._cursors: list[pg.InfiniteLine] = []

        layout = QVBoxLayout(self)

        # --- info banner ---
        _log.info("构建信息横幅…")
        banner = QLabel(
            f"Trial {playback.trial_uuid}  ·  t0 = "
            f"{playback.formal_t0_host_monotonic_ns} ns  ·  降采样回放"
        )
        banner.setObjectName("playback_banner")
        banner.setStyleSheet(
            "QLabel { padding: 4px 8px; background: #eef4fb; color: #16324f;"
            " border: 1px solid #bdd3ea; font-weight: 600; }"
        )
        layout.addWidget(banner)

        # --- plot grid ---
        _log.info("构建绘图网格…")
        grid = QGridLayout()
        row = 0

        # ---- Ultrasound row ----
        us = playback.ultrasound
        if us is not None and us.waterfall.size:
            n_channels = min(int(us.waterfall.shape[0]), 4)
            _log.info("超声瀑布图: %d 通道, waterfall shape=%s, times=%d",
                      n_channels, us.waterfall.shape, us.time_s.size)
            for ch in range(n_channels):
                title = (
                    f"US {us.channels[ch]}"
                    if ch < len(us.channels)
                    else f"US ch_{ch + 1}"
                )
                _log.info("创建 US 通道 %d 瀑布图…", ch)
                wf = _WaterfallPlotWidget(
                    title=title,
                    time_s=us.time_s,
                    data_2d=np.asarray(us.waterfall[ch]).T,
                )
                self._cursors.append(wf.cursor)
                grid.addWidget(wf, row + ch // 2, ch % 2)
            row += 2
        else:
            _log.info("无超声数据")
            no_us = QLabel("无超声数据")
            no_us.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_us.setStyleSheet("color: #888; padding: 40px;")
            grid.addWidget(no_us, row, 0, 2, 2)
            row += 2

        # ---- IMU row ----
        imu = playback.imu
        if imu is not None and imu.time_s.size:
            groups = _imu_groups(imu)
            _log.info("IMU: %d 组, %d 个时间点", len(groups), imu.time_s.size)
            for col, (title, indices) in enumerate(groups[:3]):
                _log.info("创建 IMU 组 '%s' indices=%s…", title, indices)
                sp = _SignalPlotWidget(title, imu, indices)
                self._cursors.append(sp.cursor)
                grid.addWidget(sp, row, col)
            row += 1
        else:
            _log.info("无 IMU 数据")
            no_imu = QLabel("无 IMU 数据")
            no_imu.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_imu.setStyleSheet("color: #888; padding: 20px;")
            grid.addWidget(no_imu, row, 0, 1, 3)
            row += 1

        # ---- Encoder row ----
        enc = playback.encoder
        if enc is not None and enc.time_s.size:
            groups = _encoder_groups(enc)
            _log.info("Encoder: %d 组, %d 个时间点", len(groups), enc.time_s.size)
            for col, (title, indices) in enumerate(groups[:2]):
                _log.info("创建 Encoder 组 '%s' indices=%s…", title, indices)
                sp = _SignalPlotWidget(title, enc, indices)
                self._cursors.append(sp.cursor)
                grid.addWidget(sp, row, col)
            row += 1
        else:
            _log.info("无 Encoder 数据")
            no_enc = QLabel("无编码器数据")
            no_enc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_enc.setStyleSheet("color: #888; padding: 20px;")
            grid.addWidget(no_enc, row, 0, 1, 2)
            row += 1

        _log.info("绘图网格完成，共 %d 个 cursor", len(self._cursors))
        layout.addLayout(grid, 1)

        # --- controls ---
        _log.info("构建播放控件…")
        controls = QHBoxLayout()
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
        self.time_label.setMinimumWidth(200)
        controls.addWidget(self.time_label)

        controls.addWidget(QLabel("速度："))
        self.speed_combo = QComboBox()
        self.speed_combo.setObjectName("playback_speed")
        for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
            self.speed_combo.addItem(f"{speed:g}×", speed)
        self.speed_combo.setCurrentIndex(2)
        controls.addWidget(self.speed_combo)
        layout.addLayout(controls)

        _log.info("启动回放定时器…")
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance_playback)
        self.finished.connect(lambda _result: self._timer.stop())
        self.set_playback_time(self._time_min)
        _log.info("=== PlaybackDialog.__init__ 完成 ===")

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

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
        for cursor in self._cursors:
            cursor.setPos(bounded)

    def _slider_changed(self, slider_value: int) -> None:
        fraction = float(slider_value) / 10_000.0
        self.set_playback_time(
            self._time_min + fraction * (self._time_max - self._time_min)
        )
        self._last_tick = perf_counter()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_playback(self) -> None:
        self._playing = not self._playing
        if self._playing:
            if self._current_time >= self._time_max:
                self.set_playback_time(self._time_min)
            self._last_tick = perf_counter()
            self._timer.start()
            self.play_button.setText("⏸ 暂停")
        else:
            self._timer.stop()
            self.play_button.setText("▶ 播放")

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
            self.play_button.setText("▶ 播放")
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
