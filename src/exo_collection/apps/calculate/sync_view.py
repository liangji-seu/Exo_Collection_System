"""同步标定页：自动同步结果展示 + 人工可视化标定。

自动同步在后台 Worker 完成（见 ``workers.SyncWorker``）；本页只负责展示结果，
并在低置信 / 用户主动时进入人工模式：同屏显示 Gaitway 总/左/右 Fz 与右腿 IMU
冲击包络、原始 XYZ，用户点选对应跺脚峰、拖动 offset 微调，最终「应用为人工」。

显示层按 C3D 100 Hz 或更低抽样（``_downsample``），绝不把 1000 Hz 的 Gaitway
全量塞进曲线。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from exo_collection.apps.calculate.models import SyncMethod, SyncPeakPair, SyncResult
from exo_collection.apps.calculate.workers import SyncDataBundle

_log = logging.getLogger(__name__)

_MAX_DISPLAY_POINTS = 6000


def _downsample(x: np.ndarray, y: np.ndarray, *, max_points: int = _MAX_DISPLAY_POINTS):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size <= max_points:
        return x, y
    stride = int(np.ceil(x.size / max_points))
    return x[::stride], y[::stride]


@dataclass
class _ManualPair:
    imu_time_s: float
    gaitway_time_s: float

    @property
    def offset_s(self) -> float:
        return self.gaitway_time_s - self.imu_time_s


class SyncView(QWidget):
    """同步标定页。"""

    auto_sync_requested = Signal()
    manual_data_requested = Signal()
    manual_applied = Signal(object)   # SyncResult
    sync_cleared = Signal()
    sync_confirmed = Signal()         # 用户确认「MEDIUM/LOW 自动同步」为已确认
    expert_forced = Signal(float, str)  # (offset_s, note)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bundle: SyncDataBundle | None = None
        self._auto_raw: dict[str, Any] | None = None
        self._pairs: list[_ManualPair] = []
        self._selected_imu_time: float | None = None
        self._current_offset_s = 0.0

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_auto_page())
        self._stack.addWidget(self._build_manual_page())
        layout = QVBoxLayout(self)
        layout.addWidget(self._stack)

    # ------------------------------------------------------------------
    # 自动页
    # ------------------------------------------------------------------
    def _build_auto_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        run_row = QHBoxLayout()
        self._auto_button = QPushButton("自动同步")
        self._auto_button.setProperty("buttonRole", "primary")
        self._auto_button.clicked.connect(self.auto_sync_requested.emit)
        self._manual_button = QPushButton("进入人工标定")
        self._manual_button.clicked.connect(self._enter_manual)
        self._confirm_button = QPushButton("确认该同步")
        self._confirm_button.setEnabled(False)
        self._confirm_button.clicked.connect(self.sync_confirmed.emit)
        self._expert_button = QPushButton("专家强制 offset…")
        self._expert_button.clicked.connect(self._prompt_expert_forced)
        run_row.addWidget(self._auto_button)
        run_row.addWidget(self._manual_button)
        run_row.addWidget(self._confirm_button)
        run_row.addWidget(self._expert_button)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self._auto_summary = QLabel("尚未运行自动同步。")
        self._auto_summary.setWordWrap(True)
        layout.addWidget(self._auto_summary)

        self._auto_detail = QLabel("")
        self._auto_detail.setWordWrap(True)
        self._auto_detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._auto_detail)

        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # 人工页
    # ------------------------------------------------------------------
    def _build_manual_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        hint = QLabel(
            "点选跺脚峰：先点击 IMU 峰（上），再点击 Gaitway 峰（下）形成一对；"
            "至少 3 对后计算中位 offset。可拖动微调滑块或直接输入数值。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._axis_combo = QComboBox()
        self._axis_combo.addItem("映射后的 C3D 时间", "c3d")
        self._axis_combo.addItem("各设备原始时间", "device")
        self._axis_combo.currentIndexChanged.connect(self._replot)
        layout.addWidget(self._axis_combo)

        self._imu_plot = pg.PlotWidget()
        self._imu_plot.setLabel("left", "IMU 加速度 (m/s²)")
        self._imu_plot.setLabel("bottom", "时间 (s)")
        self._imu_plot.showGrid(x=True, y=True, alpha=0.3)
        self._imu_plot.scene().sigMouseClicked.connect(self._on_imu_click)
        layout.addWidget(self._imu_plot, 2)

        self._gaitway_plot = pg.PlotWidget()
        self._gaitway_plot.setLabel("left", "Gaitway Fz (N)")
        self._gaitway_plot.setLabel("bottom", "时间 (s)")
        self._gaitway_plot.showGrid(x=True, y=True, alpha=0.3)
        self._gaitway_plot.scene().sigMouseClicked.connect(self._on_gaitway_click)
        layout.addWidget(self._gaitway_plot, 2)

        # offset 微调
        offset_group = QGroupBox("offset 微调（t_gaitway = t_c3d + offset）")
        form = QFormLayout(offset_group)
        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-60.0, 60.0)
        self._offset_spin.setDecimals(4)
        self._offset_spin.setSingleStep(0.001)
        self._offset_spin.valueChanged.connect(self._on_offset_edited)
        form.addRow("offset (s)：", self._offset_spin)

        slider_row = QHBoxLayout()
        self._minus_10ms = QPushButton("-10 ms")
        self._minus_10ms.clicked.connect(lambda: self._nudge(-0.010))
        self._minus_1ms = QPushButton("-1 ms")
        self._minus_1ms.clicked.connect(lambda: self._nudge(-0.001))
        self._plus_1ms = QPushButton("+1 ms")
        self._plus_1ms.clicked.connect(lambda: self._nudge(0.001))
        self._plus_10ms = QPushButton("+10 ms")
        self._plus_10ms.clicked.connect(lambda: self._nudge(0.010))
        for button in (self._minus_10ms, self._minus_1ms, self._plus_1ms, self._plus_10ms):
            slider_row.addWidget(button)
        slider_row.addStretch(1)
        form.addRow("步进：", _wrap_row(slider_row))
        layout.addWidget(offset_group)

        self._pairs_label = QLabel("已配对峰：0 对")
        layout.addWidget(self._pairs_label)

        actions = QHBoxLayout()
        self._restore_button = QPushButton("恢复自动结果")
        self._restore_button.clicked.connect(self._restore_auto)
        self._clear_pairs_button = QPushButton("清除配对")
        self._clear_pairs_button.clicked.connect(self._clear_pairs)
        self._apply_button = QPushButton("应用为人工标定")
        self._apply_button.setProperty("buttonRole", "primary")
        self._apply_button.clicked.connect(self._apply_manual)
        self._cancel_button = QPushButton("取消")
        self._cancel_button.clicked.connect(self._cancel)
        for button in (self._restore_button, self._clear_pairs_button,
                       self._apply_button, self._cancel_button):
            actions.addWidget(button)
        layout.addLayout(actions)

        return page

    # ------------------------------------------------------------------
    # 结果填充
    # ------------------------------------------------------------------
    def set_auto_result(self, raw: dict[str, Any]) -> None:
        """自动同步完成：展示结果摘要与细节。"""
        self._auto_raw = dict(raw)
        confidence = raw.get("confidence", "LOW")
        median = raw.get("median_offset_s")
        mad = raw.get("mad_s")
        drift = raw.get("drift_ppm")
        n_pairs = raw.get("n_pairs", 0)
        rms = raw.get("c3d_h5_match_rms_mm")
        start = raw.get("c3d_start_in_mocap_h5_frame")

        drift_text = "UNASSESSED" if drift is None else f"{drift:+.2f} ppm"
        self._auto_summary.setText(
            f"置信等级：{confidence}   ·   跺脚对数：{n_pairs}   ·   "
            f"offset = {median:.4f} s   ·   MAD = {mad:.4f} s   ·   漂移 {drift_text}"
        )
        detail_lines = [
            f"C3D 第 0 帧 = mocap.h5 第 {start} 帧（RMS {rms:.4g} mm）",
            f"最终 gaitway_offset = {raw.get('gaitway_offset_s'):.4f} s",
            f"IMU 传感器：{raw.get('imu_sensor_label')}（下标 {raw.get('imu_sensor_index')}）",
        ]
        peak_pairs = raw.get("peak_pairs") or []
        for pair in peak_pairs:
            detail_lines.append(
                f"  #{pair['index']}: IMU {pair['imu_time_on_c3d_s']:.4f}s ↔ "
                f"Gaitway {pair['gaitway_time_s']:.4f}s  (offset {pair['offset_s']:.4f}s)"
            )
        self._auto_detail.setText("\n".join(detail_lines))
        self._confirm_button.setEnabled(True)
        self._stack.setCurrentIndex(0)

    def set_auto_failed(self, message: str) -> None:
        self._auto_summary.setText(f"自动同步失败：{message}")
        self._auto_detail.setText("请进入人工标定。")
        self._confirm_button.setEnabled(False)
        self._stack.setCurrentIndex(0)

    def _enter_manual(self) -> None:
        self._stack.setCurrentIndex(1)
        if self._bundle is None:
            self.manual_data_requested.emit()

    def set_manual_data(self, bundle: SyncDataBundle) -> None:
        """加载人工标定原始信号并绘图。"""
        self._bundle = bundle
        # 初始 offset：优先用自动结果，否则 0。
        if self._auto_raw is not None:
            self._current_offset_s = float(self._auto_raw.get("median_offset_s", 0.0))
        else:
            self._current_offset_s = 0.0
        self._offset_spin.blockSignals(True)
        self._offset_spin.setValue(self._current_offset_s)
        self._offset_spin.blockSignals(False)
        self._replot()

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    def _replot(self) -> None:
        if self._bundle is None:
            return
        axis_mode = self._axis_combo.currentData()
        bundle = self._bundle
        self._imu_plot.clear()
        self._gaitway_plot.clear()

        # IMU 时间轴：device 模式直接用原始 c3d 时间（imu_time_s 已映射到 c3d 时间）。
        imu_t = bundle.imu_time_s
        tx, env = _downsample(imu_t, bundle.imu_envelope)
        self._imu_plot.plot(tx, env, pen=pg.mkPen("#0f766e", width=2), name="冲击包络")
        for axis_index, color in ((0, "#c0603a"), (1, "#4a6fa5"), (2, "#7a7a4a")):
            _, raw_axis = _downsample(imu_t, bundle.imu_acc[:, axis_index])
            self._imu_plot.plot(tx, raw_axis, pen=pg.mkPen(color, width=1),
                                name=f"acc[{axis_index}]")

        # Gaitway 时间轴：c3d 模式下平移 -offset，device 模式用自身时间。
        if axis_mode == "c3d":
            gait_t = bundle.gaitway_time_s - self._current_offset_s
        else:
            gait_t = bundle.gaitway_time_s
        gt, total = _downsample(gait_t, bundle.gaitway_total_fz)
        self._gaitway_plot.plot(gt, total, pen=pg.mkPen("#0f766e", width=2), name="total Fz")
        _, fl = _downsample(gait_t, bundle.gaitway_fz_left)
        _, fr = _downsample(gait_t, bundle.gaitway_fz_right)
        self._gaitway_plot.plot(gt, fl, pen=pg.mkPen("#c0603a", width=1), name="FzL")
        self._gaitway_plot.plot(gt, fr, pen=pg.mkPen("#4a6fa5", width=1), name="FzR")

        # marker 第三证据（若可用）
        if bundle.marker_acc_norm is not None and bundle.marker_time_s is not None:
            _, marker_env = _downsample(bundle.marker_time_s, bundle.marker_acc_norm)
            self._imu_plot.plot(bundle.marker_time_s, marker_env,
                                pen=pg.mkPen("#9a5b13", width=1, style=Qt.PenStyle.DashLine),
                                name=f"marker {bundle.marker_name}")

        self._draw_pairs()

    def _draw_pairs(self) -> None:
        if self._bundle is None:
            return
        axis_mode = self._axis_combo.currentData()
        bundle = self._bundle
        imu_scatter = pg.ScatterPlotItem(
            x=[p.imu_time_s for p in self._pairs],
            y=[0.0] * len(self._pairs),
            size=12, brush="#0f766e", pen="w",
        )
        self._imu_plot.addItem(imu_scatter)
        gait_t = [p.gaitway_time_s - self._current_offset_s if axis_mode == "c3d"
                  else p.gaitway_time_s for p in self._pairs]
        gait_scatter = pg.ScatterPlotItem(
            x=gait_t,
            y=[0.0] * len(self._pairs),
            size=12, brush="#c0603a", pen="w",
        )
        self._gaitway_plot.addItem(gait_scatter)
        self._pairs_label.setText(
            f"已配对峰：{len(self._pairs)} 对"
            + (f"  ·  中位 offset = {self._median_offset():.4f} s" if self._pairs else "")
        )

    def _on_imu_click(self, event) -> None:
        if self._bundle is None:
            return
        pos = self._imu_plot.plotItem.vb.mapSceneToView(event.scenePos())
        self._selected_imu_time = float(pos.x())
        self._pairs_label.setText(
            f"已选 IMU 峰 @ {self._selected_imu_time:.3f}s，请点击 Gaitway 对应峰"
        )

    def _on_gaitway_click(self, event) -> None:
        if self._bundle is None or self._selected_imu_time is None:
            return
        axis_mode = self._axis_combo.currentData()
        pos = self._gaitway_plot.plotItem.vb.mapSceneToView(event.scenePos())
        gaitway_x = float(pos.x())
        gaitway_time = gaitway_x + self._current_offset_s if axis_mode == "c3d" else gaitway_x
        self._pairs.append(_ManualPair(self._selected_imu_time, gaitway_time))
        self._selected_imu_time = None
        self._recompute_offset_from_pairs()
        self._replot()

    def _median_offset(self) -> float | None:
        if not self._pairs:
            return None
        offsets = [p.offset_s for p in self._pairs]
        return float(np.median(offsets))

    def _recompute_offset_from_pairs(self) -> None:
        median = self._median_offset()
        if median is not None:
            self._current_offset_s = median
            self._offset_spin.blockSignals(True)
            self._offset_spin.setValue(median)
            self._offset_spin.blockSignals(False)

    def _on_offset_edited(self, value: float) -> None:
        self._current_offset_s = float(value)
        self._replot()

    def _nudge(self, delta: float) -> None:
        self._offset_spin.setValue(self._offset_spin.value() + delta)

    def _clear_pairs(self) -> None:
        self._pairs.clear()
        self._selected_imu_time = None
        self._replot()

    def _restore_auto(self) -> None:
        if self._auto_raw is None:
            return
        self._stack.setCurrentIndex(0)

    def _apply_manual(self) -> None:
        if self._bundle is None:
            return
        # prompt6 §3.2 第 4 条：人工标定必须至少 3 对单调递增的峰对，否则拒绝。
        if len(self._pairs) < 3:
            QMessageBox.warning(
                self,
                "峰对不足",
                f"人工标定需要至少 3 对跺脚峰，当前只有 {len(self._pairs)} 对。\n"
                "请在两个曲线上分别点选对应峰，凑足 3 对后再应用。",
            )
            return
        ordered = sorted(self._pairs, key=lambda p: p.imu_time_s)
        imu_times = [p.imu_time_s for p in ordered]
        gaitway_times = [p.gaitway_time_s for p in ordered]
        imu_monotonic = all(b > a for a, b in zip(imu_times, imu_times[1:]))
        gaitway_monotonic = all(b > a for a, b in zip(gaitway_times, gaitway_times[1:]))
        if not (imu_monotonic and gaitway_monotonic):
            QMessageBox.warning(
                self,
                "峰对不单调",
                "按 IMU 时间排序后，峰对的时间顺序出现交叉或重复，无法得到一致 offset。\n"
                "请清除配对后按时间先后重新点选。",
            )
            return

        offsets = np.asarray([p.offset_s for p in ordered], dtype=np.float64)
        median = float(np.median(offsets))
        mad = float(np.median(np.abs(offsets - median)))
        pairs = tuple(
            SyncPeakPair(
                index=i + 1,
                imu_time_s=p.imu_time_s,
                gaitway_time_s=p.gaitway_time_s,
            )
            for i, p in enumerate(ordered)
        )
        result = SyncResult(
            c3d_start_in_mocap_h5_frame=self._bundle.c3d_h5_start_frame,
            c3d_h5_match_rms_mm=self._bundle.c3d_h5_match_rms_mm,
            c3d_h5_overlap_frames=0,
            gaitway_offset_s=median,
            confidence="MEDIUM",
            peak_pairs=pairs,
            median_offset_s=median,
            mad_s=mad,
            offsets_s=tuple(float(v) for v in offsets),
            manual=True,
            audit_note=f"人工标定：{len(pairs)} 对峰，MAD={mad:.4f}s",
            method=SyncMethod.MANUAL_PAIRED,
        )
        self._current_offset_s = median
        self.manual_applied.emit(result)
        self._stack.setCurrentIndex(0)

    def _cancel(self) -> None:
        self._stack.setCurrentIndex(0)
        self.sync_cleared.emit()

    def _prompt_expert_forced(self) -> None:
        """专家强制 offset：无峰证据的二次确认对话框（prompt6 §3.2 第 5 条）。"""
        dialog = _ExpertForceDialog(self._current_offset_s, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.expert_forced.emit(dialog.offset_s(), dialog.note())

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._auto_raw = None
        self._bundle = None
        self._pairs.clear()
        self._selected_imu_time = None
        self._auto_summary.setText("尚未运行自动同步。")
        self._auto_detail.setText("")
        self._confirm_button.setEnabled(False)
        self._pairs_label.setText("已配对峰：0 对")
        self._stack.setCurrentIndex(0)


def _wrap_row(layout: QHBoxLayout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget


class _ExpertForceDialog(QDialog):
    """专家强制 offset 的二次确认对话框。

    输入一个 offset 值 + 理由；必须勾选「我确认这是无峰证据的强制值」才能点 OK，
    防止误触把无证据的值当作同步结论（prompt6 §3.2 第 5 条）。
    """

    def __init__(self, initial_offset_s: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("专家强制 offset")
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-60.0, 60.0)
        self._offset_spin.setDecimals(4)
        self._offset_spin.setSingleStep(0.001)
        self._offset_spin.setValue(float(initial_offset_s))
        form.addRow("offset (s)：", self._offset_spin)

        self._note_edit = QLineEdit()
        self._note_edit.setPlaceholderText("强制原因（必填）")
        form.addRow("原因：", self._note_edit)
        layout.addLayout(form)

        self._confirm_check = QCheckBox("我确认这是无峰证据的专家强制值（最终 QC 至少判为 WARN）")
        layout.addWidget(self._confirm_check)

        warning = QLabel(
            "该值不经过峰对/MAD 验证，将标记为 EXPERT_FORCED，最终 QC 绝不会判为 PASS。"
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setEnabled(False)
        self._confirm_check.toggled.connect(self._refresh_ok)
        self._note_edit.textChanged.connect(self._refresh_ok)
        layout.addWidget(buttons)
        self._refresh_ok()

    def _refresh_ok(self) -> None:
        self._ok_button.setEnabled(
            self._confirm_check.isChecked() and bool(self._note_edit.text().strip())
        )

    def offset_s(self) -> float:
        return float(self._offset_spin.value())

    def note(self) -> str:
        return self._note_edit.text().strip()


__all__ = ["SyncView"]
