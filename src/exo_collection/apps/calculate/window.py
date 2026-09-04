"""Exo Calculate 主窗口：组合 Session 选择、同步、解算、回放各页。

只负责组合与信号接线；数值计算都在后台 Worker 里。所有原始数据保持只读，
派生结果写到 Session 的 ``derived/``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThreadPool, Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from exo_collection.apps.calculate.controller import CalculateController
from exo_collection.apps.calculate.models import InputCheckReport, SessionRecord, SyncMethod, SyncResult
from exo_collection.apps.calculate.operation import OperationContext
from exo_collection.apps.calculate.processing_view import ProcessingView
from exo_collection.apps.calculate.session_selector import SessionSelector
from exo_collection.apps.calculate.sync_view import SyncView
from exo_collection.apps.calculate.viewer import ViewerWidget
from exo_collection.apps.calculate.workers import (
    LoadSyncDataWorker,
    OpenSimProcessWorker,
    PrepWorker,
    SyncWorker,
)
from exo_collection.apps.collector.theme import COLLECTOR_STYLESHEET
from exo_collection.configuration import SharedAppSettings

_log = logging.getLogger(__name__)


class CalculateWindow(QMainWindow):
    """Exo Calculate 主窗口。"""

    def __init__(self, data_root: Path, settings: SharedAppSettings) -> None:
        super().__init__()
        self._settings = settings
        self._data_root = Path(data_root)
        self._thread_pool = QThreadPool.globalInstance()
        self._controller = CalculateController(data_root)
        self._sync_worker: SyncWorker | None = None
        self._load_worker: LoadSyncDataWorker | None = None
        self._prep_worker: PrepWorker | None = None
        self._opensim_worker: OpenSimProcessWorker | None = None
        self._cancel_file: Path | None = None
        self._busy: bool = False
        self._busy_kind: str | None = None

        self.setWindowTitle("Exo Calculate —— 标定 / 同步 / 解算 / 回放")
        self.setStyleSheet(COLLECTOR_STYLESHEET)
        self.resize(1400, 900)

        central = QWidget()
        root = QVBoxLayout(central)

        self._title = QLabel("Exo Calculate")
        self._title.setObjectName("page_title")
        root.addWidget(self._title)

        # 数据根目录选择（与 Collector / Data Studio 共享同一份持久化设置）。
        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("数据根目录："))
        self._data_root_edit = QLineEdit(str(self._data_root))
        self._data_root_edit.setReadOnly(True)
        root_row.addWidget(self._data_root_edit, 1)
        self._choose_root_button = QPushButton("选择…")
        self._choose_root_button.clicked.connect(self._choose_data_root)
        root_row.addWidget(self._choose_root_button)
        root.addLayout(root_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：Session 选择 + 输入检查报告
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self._selector = SessionSelector(data_root)
        left_layout.addWidget(self._selector)
        self._report_label = QLabel("输入检查：尚未运行")
        self._report_label.setWordWrap(True)
        self._report_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left_layout.addWidget(self._report_label)
        splitter.addWidget(left)

        # 右侧：同步 / 解算 / 回放
        self._tabs = QTabWidget()
        self._sync_view = SyncView()
        self._processing_view = ProcessingView(settings)
        self._viewer = ViewerWidget()
        self._tabs.addTab(self._sync_view, "同步标定")
        self._tabs.addTab(self._processing_view, "解算")
        self._tabs.addTab(self._viewer, "回放")
        splitter.addWidget(self._tabs)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([480, 920])
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

        self._wire()

    # ------------------------------------------------------------------
    def _wire(self) -> None:
        self._selector.dynamic_selected.connect(self._on_dynamic)
        self._selector.static_selected.connect(self._on_static)
        self._selector.check_requested.connect(self._on_check_inputs)

        self._sync_view.auto_sync_requested.connect(self._run_auto_sync)
        self._sync_view.manual_data_requested.connect(self._run_load_sync_data)
        self._sync_view.manual_applied.connect(self._on_manual_applied)
        self._sync_view.sync_confirmed.connect(self._on_sync_confirmed)

        self._processing_view.process_requested.connect(self._on_process_requested)
        self._processing_view.cancel_requested.connect(self._on_cancel_requested)

        self._viewer.export_ground_truth_requested.connect(self._on_export_ground_truth)

        self._controller.state_changed.connect(self._on_state_changed)

    # ------------------------------------------------------------------
    # 异步任务串 Session 防护（operation token，prompt6 §3.4）
    # ------------------------------------------------------------------
    def _begin_task(self, kind: str, *, run_dir: Path | None = None) -> OperationContext | None:
        """启动后台任务前申请上下文；已有任务在跑或输入被锁定时拒绝。

        返回 ``None`` 表示本次启动被拒绝（已有任务未结束），调用方应直接 return。
        """
        if self._busy:
            _log.warning("拒绝启动 %s：已有任务 %s 未结束", kind, self._busy_kind)
            self.statusBar().showMessage(f"已有任务（{self._busy_kind}）运行中，请先取消或等待完成。")
            return None
        self._busy = True
        self._busy_kind = kind
        self._set_inputs_locked(True)
        return self._controller.begin_operation(kind, run_dir=run_dir)

    def _end_task(self, ctx: OperationContext | None) -> None:
        """结束任务：清空 operation 上下文、解锁输入。"""
        self._controller.finish_operation(ctx)
        self._busy = False
        self._busy_kind = None
        self._set_inputs_locked(False)

    def _is_current(self, ctx: OperationContext | None) -> bool:
        if ctx is not None and self._controller.is_current_operation(ctx):
            return True
        _log.warning(
            "丢弃过期回调：op=%s（当前 %s）",
            getattr(ctx, "operation_id", None),
            getattr(self._controller.active_operation, "operation_id", None),
        )
        return False

    def _set_inputs_locked(self, locked: bool) -> None:
        """运行期间锁定数据根/受试者/Session 选择，防止串 Session（§3.4 第 3 条）。"""
        self._selector.setEnabled(not locked)
        self._choose_root_button.setEnabled(not locked)

    # ------------------------------------------------------------------
    # 数据根目录
    # ------------------------------------------------------------------
    def _choose_data_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择外骨骼数据根目录",
            self._data_root_edit.text(),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return
        try:
            normalized = self._settings.set_data_root(selected)
        except (ValueError, RuntimeError) as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"设置数据根目录失败：{exc}")
            return
        self._data_root = normalized
        self._data_root_edit.setText(str(normalized))
        self._report_label.setText("输入检查：尚未运行")
        # 重新扫描 + 按新根自动绑定静态/动态（信号会同步更新 controller）。
        self._selector.set_data_root(normalized)
        self.statusBar().showMessage(f"数据根目录已更新：{normalized}")

    # ------------------------------------------------------------------
    # Session 选择
    # ------------------------------------------------------------------
    def _on_dynamic(self, record: SessionRecord | None) -> None:
        self._controller.set_dynamic(record)
        self._sync_view.reset()
        self._apply_patient_info(record)

    def _apply_patient_info(self, record: SessionRecord | None) -> None:
        """从测力台 TXT 头部读取身高/体重等个人信息并预填解算参数（失败不致命）。"""
        if record is None or record.files.txt_path is None:
            return
        try:
            from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path

            ensure_pipeline_on_path()
            from pipeline.gaitway import read_gaitway_patient_info

            info = read_gaitway_patient_info(record.files.txt_path)
        except Exception as exc:  # noqa: BLE001
            _log.warning("读取测力台个人信息失败：%s", exc)
            return
        if not info:
            return
        self._processing_view.apply_patient_info(info)
        parts = []
        if info.get("weight_kg") is not None:
            parts.append(f"体重 {info['weight_kg']:.1f} kg")
        if info.get("height_m") is not None:
            parts.append(f"身高 {info['height_m']:.2f} m")
        if info.get("name"):
            parts.append(f"姓名 {info['name']}")
        if parts:
            self.statusBar().showMessage("已从测力台读取：" + "，".join(parts))

    def _on_static(self, record: SessionRecord | None) -> None:
        self._controller.set_static(record)

    def _on_check_inputs(self, dynamic: SessionRecord | None, static: SessionRecord | None) -> None:
        if dynamic is None:
            self._report_label.setText("输入检查：请先选择动态 Session。")
            return
        self.statusBar().showMessage("正在只读扫描输入…")
        try:
            from exo_collection.apps.calculate.discovery import check_inputs

            report: InputCheckReport = check_inputs(dynamic, static)
        except Exception as exc:  # noqa: BLE001
            _log.exception("输入检查失败")
            self._report_label.setText(f"输入检查失败：{exc}")
            self.statusBar().showMessage("输入检查失败")
            return
        self._controller.set_input_report(report)
        self._show_report(report)
        self.statusBar().showMessage("输入检查完成")

    def _show_report(self, report: InputCheckReport) -> None:
        lines = ["输入检查结果："]
        lines.append(f"  动态 {report.dynamic_session}：C3D {report.dynamic_c3d_rate_hz or '—'} Hz, "
                     f"{report.dynamic_c3d_duration_s or 0:.1f} s, "
                     f"HH19 {report.dynamic_hh19_markers} 点")
        if report.static_session:
            lines.append(f"  静态 {report.static_session}：C3D {report.static_c3d_rate_hz or '—'} Hz, "
                         f"HH19 {report.static_hh19_markers} 点")
        lines.append(f"  Gaitway {report.gaitway_rate_hz or '—'} Hz, "
                     f"左右力列：{'有' if report.gaitway_has_bilateral_columns else '缺'}")
        for problem in report.problems:
            lines.append(f"  [问题] {problem}")
        for warning in report.warnings:
            lines.append(f"  [警告] {warning}")
        if not report.problems:
            lines.append("  [通过] 输入齐全")
        self._report_label.setText("\n".join(lines))

    # ------------------------------------------------------------------
    # 自动同步
    # ------------------------------------------------------------------
    def _run_auto_sync(self) -> None:
        dynamic = self._controller.dynamic
        if dynamic is None:
            return
        files = dynamic.files
        if not files.has_dynamic_inputs:
            self._sync_view.set_auto_failed("动态 Session 输入不齐全。")
            return
        ctx = self._begin_task("sync")
        if ctx is None:
            return
        self.statusBar().showMessage("自动同步中…")
        self._sync_worker = SyncWorker(
            files.c3d_path,  # type: ignore[arg-type]
            files.mocap_h5_path,  # type: ignore[arg-type]
            files.imu_h5_path,  # type: ignore[arg-type]
            files.txt_path,  # type: ignore[arg-type]
        )
        self._sync_worker.signals.finished.connect(
            lambda result, c=ctx: self._on_auto_sync_done(result, c)
        )
        self._sync_worker.signals.failed.connect(
            lambda message, c=ctx: self._on_auto_sync_failed(message, c)
        )
        self._thread_pool.start(self._sync_worker)

    def _on_auto_sync_done(self, result: dict[str, Any], ctx: OperationContext) -> None:
        if not self._is_current(ctx):
            return
        self.statusBar().showMessage("自动同步完成")
        self._sync_view.set_auto_result(result)
        result_obj = _sync_result_from_raw(result, manual=False)
        self._controller.set_sync(result_obj, raw=result)
        self._save_calibration(result, operator="auto")
        self._end_task(ctx)

    def _on_auto_sync_failed(self, message: str, ctx: OperationContext) -> None:
        if not self._is_current(ctx):
            return
        self.statusBar().showMessage("自动同步失败")
        self._sync_view.set_auto_failed(message)
        self._end_task(ctx)

    def _run_load_sync_data(self) -> None:
        dynamic = self._controller.dynamic
        if dynamic is None or not dynamic.files.has_dynamic_inputs:
            return
        ctx = self._begin_task("load_sync_data")
        if ctx is None:
            return
        files = dynamic.files
        self._load_worker = LoadSyncDataWorker(
            files.c3d_path,  # type: ignore[arg-type]
            files.mocap_h5_path,  # type: ignore[arg-type]
            files.imu_h5_path,  # type: ignore[arg-type]
            files.txt_path,  # type: ignore[arg-type]
        )
        self._load_worker.signals.finished.connect(
            lambda bundle, c=ctx: self._on_load_sync_data_done(bundle, c)
        )
        self._load_worker.signals.failed.connect(
            lambda message, c=ctx: self._on_load_sync_data_failed(message, c)
        )
        self._thread_pool.start(self._load_worker)

    def _on_load_sync_data_done(self, bundle, ctx: OperationContext) -> None:
        if not self._is_current(ctx):
            return
        self._sync_view.set_manual_data(bundle)
        self._end_task(ctx)

    def _on_load_sync_data_failed(self, message: str, ctx: OperationContext) -> None:
        if not self._is_current(ctx):
            return
        self._sync_view.set_auto_failed(message)
        self._end_task(ctx)

    def _on_manual_applied(self, result: SyncResult) -> None:
        self._controller.set_sync(result, method=SyncMethod.MANUAL_PAIRED)
        raw = {
            "median_offset_s": result.gaitway_offset_s,
            "gaitway_offset_s": result.gaitway_offset_s,
            "confidence": result.confidence,
            "n_pairs": len(result.peak_pairs),
            "mad_s": result.mad_s,
            "manual": True,
            "method": SyncMethod.MANUAL_PAIRED.value,
            "c3d_start_in_mocap_h5_frame": result.c3d_start_in_mocap_h5_frame,
            "c3d_h5_match_rms_mm": result.c3d_h5_match_rms_mm,
        }
        self._save_calibration(raw, operator="operator")
        self.statusBar().showMessage("人工标定已应用")

    def _on_sync_confirmed(self) -> None:
        """用户确认 MEDIUM/LOW 自动同步：记录确认元数据并重写标定（§3.2 第 3 条）。"""
        self._controller.confirm_sync()
        result = self._controller.sync_raw or {}
        self._save_calibration(result, operator="operator")
        self.statusBar().showMessage("同步已确认")

    def _save_calibration(self, result: dict[str, Any], *, operator: str) -> None:
        dynamic = self._controller.dynamic
        if dynamic is None:
            return
        files = dynamic.files
        try:
            from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path

            ensure_pipeline_on_path()
            from pipeline.synchronization.sync import save_sync_calibration

            out_dir = dynamic.session_dir / "derived" / "opensim"
            confirm = self._controller.confirm_meta or {}
            save_sync_calibration(
                out_dir,
                result,
                inputs={
                    "c3d": files.c3d_path,
                    "mocap_h5": files.mocap_h5_path,
                    "imu_h5": files.imu_h5_path,
                    "gaitway_txt": files.txt_path,
                },
                operator=operator,
                method=self._controller.sync_method.value,
                dynamic_session_uuid=dynamic.session_uuid,
                trial_uuid=dynamic.trial_uuid,
                auto_candidate=self._controller.auto_candidate_raw,
                operator_type=confirm.get("operator_type"),
                confirmed_at=confirm.get("confirmed_at_utc"),
                adjusted=bool(confirm.get("adjusted", False)),
                note=confirm.get("note"),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("写入 sync_calibration.json 失败：%s", exc)

    # ------------------------------------------------------------------
    # 解算：PrepWorker（EXO 环境预处理）→ OpenSimProcessWorker（opensim 环境）
    # ------------------------------------------------------------------
    def _sync_quality(self) -> dict[str, Any]:
        """把当前同步结论组装成 QC 可消费的同步质量块（prompt6 §3.3）。"""
        sync = self._controller.sync
        raw = self._controller.auto_candidate_raw or self._controller.sync_raw or {}
        return {
            "method": self._controller.sync_method.value,
            "confidence": sync.confidence if sync else "LOW",
            "gaitway_offset_s": sync.gaitway_offset_s if sync else 0.0,
            "n_pairs": len(sync.peak_pairs) if sync else 0,
            "mad_s": sync.mad_s if sync else None,
            "drift_ppm": raw.get("drift_ppm"),
            "c3d_h5_common_markers": len(raw.get("c3d_h5_matched_markers") or []) or None,
            "c3d_h5_rms_mm": raw.get("c3d_h5_match_rms_mm"),
            "c3d_h5_max_error_mm": raw.get("c3d_h5_match_max_error_mm"),
            "c3d_h5_unique": raw.get("c3d_h5_unique"),
            "c3d_h5_exact": raw.get("c3d_h5_exact"),
            "mocap_clock_monotonic": raw.get("mocap_h5_monotonic"),
            "mocap_clock_gaps": raw.get("mocap_h5_clock_gaps"),
            "imu_clock_monotonic": raw.get("imu_clock_monotonic"),
            "imu_clock_gaps": raw.get("imu_clock_gaps"),
        }

    def _on_process_requested(self, config) -> None:
        self._controller.set_config(config)
        dynamic = self._controller.dynamic
        static = self._controller.static
        sync = self._controller.sync
        opensim_python = self._processing_view.opensim_python()

        if dynamic is None:
            self._processing_view.append_log("错误：未选择动态 Session。")
            return
        if static is None or static.files.c3d_path is None:
            self._processing_view.append_log("错误：未选择静态标定 Session（无法缩放模型）。")
            return
        if sync is None:
            self._processing_view.append_log("错误：尚未完成同步标定。")
            return
        if not self._controller.can_solve:
            self._processing_view.append_log("错误：同步未确认，禁止解算（§3.2 门禁）。")
            return
        if opensim_python is None:
            self._processing_view.append_log("错误：未配置 OpenSim 子环境。")
            return
        if not dynamic.files.has_dynamic_inputs:
            self._processing_view.append_log("错误：动态 Session 输入不齐全。")
            return
        generic_model = self._generic_model_path()
        if generic_model is None:
            self._processing_view.append_log("错误：找不到 gait2392 通用模型。")
            return

        # 派生 run 目录（不覆盖旧 run）；原始数据保持只读，只写 derived/。
        run_dir = self._controller.new_run_directory(
            dynamic.session_dir / "derived" / "opensim"
        )
        self._controller.set_run_dir(run_dir)
        self._cancel_file = run_dir / "cancel.flag"
        self._cancel_file.unlink(missing_ok=True)

        ctx = self._begin_task("prep", run_dir=run_dir)
        if ctx is None:
            return

        self._controller.mark_processing()
        self._processing_view.set_running(True)
        self._processing_view.clear_log()
        self._processing_view.append_log(f"run 目录：{run_dir}")
        self._processing_view.append_log(
            f"参数：体重 {config.mass_kg} kg，身高 {config.height_m} m，"
            f"marker 低通 {config.marker_cutoff_hz} Hz，"
            f"同步 offset {sync.gaitway_offset_s:.6f} s（{sync.confidence}）"
        )

        self._prep_worker = PrepWorker(
            static_c3d=static.files.c3d_path,
            dynamic_c3d=dynamic.files.c3d_path,  # type: ignore[arg-type]
            gaitway_txt=dynamic.files.txt_path,  # type: ignore[arg-type]
            generic_model=generic_model,
            out_dir=run_dir,
            subject_id=dynamic.subject_code,
            mass_kg=config.mass_kg,
            height_m=config.height_m,
            gaitway_offset_s=sync.gaitway_offset_s,
            marker_cutoff_hz=config.marker_cutoff_hz,
            grf_cutoff_hz=config.grf_cutoff_hz,
            opensim_x_sign=config.opensim_x_sign,
            opensim_z_sign=config.opensim_z_sign,
            analysis_time_range_s=config.analysis_time_range_s,
            static_time_range_s=config.static_time_range_s,
            sync_confidence=sync.confidence,
            sync_quality=self._sync_quality(),
            marker_adjustment_expert_confirmed=config.marker_adjustment_expert_confirmed,
        )
        self._prep_worker.signals.progress.connect(self._processing_view.append_log)
        self._prep_worker.signals.finished.connect(
            lambda summary, c=ctx: self._on_prep_done(summary, c)
        )
        self._prep_worker.signals.failed.connect(
            lambda message, c=ctx: self._on_process_failed(message, c)
        )
        self._thread_pool.start(self._prep_worker)
        self.statusBar().showMessage("预处理中…")

    def _on_prep_done(self, summary: dict[str, Any], ctx: OperationContext) -> None:
        if not self._is_current(ctx):
            return
        self._processing_view.append_log(
            f"预处理完成：静态 {summary.get('static_n_markers')} 点 / "
            f"动态 {summary.get('dynamic_n_markers')} 点，"
            f"有效双侧帧 {summary.get('n_valid_decomposed_frames')}"
        )
        sw = summary.get("static_window") or {}
        if sw.get("start_s") is not None:
            self._processing_view.append_log(
                f"静态稳定窗口：{sw['start_s']:.2f}~{sw['end_s']:.2f} s"
                f"（{sw.get('method')}，稳定度 {sw.get('mean_velocity_mm_s')} mm/s，"
                f"缺失 {sw.get('missing_markers') or '无'}）"
            )
        opensim_python = self._processing_view.opensim_python()
        if opensim_python is None:
            self._on_process_failed("OpenSim 子环境未配置", ctx)
            return
        script = self._process_session_script()
        # prep 已完成，进入 opensim 阶段：新任务上下文，同一 busy 流程继续锁定输入。
        self._busy_kind = "opensim"
        ctx2 = self._controller.begin_operation("opensim", run_dir=self._controller.run_dir)
        self._opensim_worker = OpenSimProcessWorker(
            opensim_python,
            script,
            ["--manifest", summary["manifest_path"], "--cancel-file", str(self._cancel_file)],
        )
        self._opensim_worker.signals.progress.connect(self._processing_view.append_log)
        self._opensim_worker.signals.finished.connect(
            lambda payload, c=ctx2: self._on_opensim_done(payload, c)
        )
        self._opensim_worker.signals.cancelled.connect(
            lambda c=ctx2: self._on_opensim_cancelled(c)
        )
        self._opensim_worker.signals.failed.connect(
            lambda message, c=ctx2: self._on_process_failed(message, c)
        )
        self._thread_pool.start(self._opensim_worker)
        self.statusBar().showMessage("OpenSim 解算中…")

    def _on_opensim_done(self, payload: dict[str, Any], ctx: OperationContext) -> None:
        if not self._is_current(ctx):
            return
        self._processing_view.append_log("OpenSim 子进程正常结束（退出码 0）。")
        self._processing_view.set_running(False)
        status, detail = self._preliminary_qc(payload, self._controller.config)
        self._controller.mark_completed(status)
        self.statusBar().showMessage(f"解算完成（QC {status}）")

        # 自动载入回放数据并跳到回放页（阶段 E）。
        viewer_dir = payload.get("viewer_dir")
        if viewer_dir is None and self._controller.run_dir is not None:
            viewer_dir = self._controller.run_dir / "viewer"
        self._load_viewer(viewer_dir)
        self._end_task(ctx)

    def _on_opensim_cancelled(self, ctx: OperationContext) -> None:
        if not self._is_current(ctx):
            return
        self._processing_view.set_running(False)
        self._processing_view.append_log("已取消。")
        self._controller.mark_cancelled()
        self.statusBar().showMessage("已取消")
        self._end_task(ctx)

    def _on_process_failed(self, message: str, ctx: OperationContext) -> None:
        if not self._is_current(ctx):
            return
        self._processing_view.set_running(False)
        self._processing_view.append_log(f"失败：{message}")
        self._controller.mark_failed()
        self.statusBar().showMessage("解算失败")
        self._end_task(ctx)

    def _on_cancel_requested(self) -> None:
        self._processing_view.append_log("已请求取消（协作式，等待子进程退出…）")
        if self._cancel_file is not None:
            try:
                self._cancel_file.write_text("cancel", encoding="utf-8")
            except OSError as exc:  # noqa: BLE001
                _log.warning("写取消标记失败：%s", exc)
        if self._opensim_worker is not None:
            # 协作式取消：先置位标记等待子进程自行输出 cancelled 事件；超时后
            # 才 terminate/kill 兜底（prompt6 §3.10 第 3 条）。父进程仍会把结局
            # 标记为 CANCELLED，而非 FAILED。
            self._opensim_worker.cancel()
            QTimer.singleShot(5000, self._opensim_worker.terminate)
            QTimer.singleShot(8000, self._opensim_worker.kill)

    def closeEvent(self, event) -> None:  # noqa: N802 —— Qt 命名约定
        """关闭前安全取消/终止后台子进程，避免回调访问已销毁的 Qt 对象（§3.4 第 6 条）。"""
        if self._cancel_file is not None and self._busy:
            try:
                self._cancel_file.write_text("cancel", encoding="utf-8")
            except OSError:  # noqa: BLE001
                pass
        if self._opensim_worker is not None:
            self._opensim_worker.cancel()
            self._opensim_worker.terminate()
            self._opensim_worker.kill()
        event.accept()

    # ------------------------------------------------------------------
    # 解算辅助
    # ------------------------------------------------------------------
    def _generic_model_path(self) -> Path | None:
        from exo_collection.apps.calculate._pipeline import pipeline_root

        candidate = pipeline_root() / "data" / "models" / "gait2392" / "gait2392_simbody.osim"
        return candidate if candidate.is_file() else None

    def _process_session_script(self) -> Path:
        from exo_collection.apps.calculate._pipeline import pipeline_root

        return pipeline_root() / "scripts" / "process_session.py"

    @staticmethod
    def _preliminary_qc(payload: dict[str, Any], config) -> tuple[str, str]:
        """把 OpenSim 结果映射为 QC 结论。

        权威结论来自子进程已算好的版本化 ``qc``（含同步质量/力覆盖，prompt6 §3.3），
        App 直接采用，保证 ``result.json`` / ``qc_report.json`` / 界面同一结论；
        缺 ``qc`` 块时才本地兜底重算（保守，缺同步质量会判 FAIL）。
        """
        qc = payload.get("qc") if isinstance(payload, dict) else None
        if isinstance(qc, dict) and qc.get("status") in {"PASS", "WARN", "FAIL"}:
            return qc["status"], qc.get("summary", "")

        from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path

        ensure_pipeline_on_path()
        from pipeline.qc.evaluate import evaluate_qc

        mass_kg = config.mass_kg if config is not None else 75.0
        verdict = evaluate_qc(
            marker_qc_overall=payload.get("marker_qc_overall"),
            id_qc=payload.get("id_qc"),
            mass_kg=mass_kg,
        )
        return verdict["status"], verdict["summary"]

    # ------------------------------------------------------------------
    def _load_viewer(self, viewer_dir: Path | None, *, load_imu: bool = True) -> None:
        """载入回放页；失败只记录日志，不影响 QC 结论。

        历史 run 回放同样载入 IMU：起始帧退回会话级 ``sync_calibration.json``
        （其 C3D↔mocap.h5 对齐帧与步态 offset 无关、跨重复同步稳定）。
        """
        try:
            self._viewer.load(viewer_dir)
        except Exception as exc:  # noqa: BLE001
            _log.exception("载入回放数据失败")
            self._processing_view.append_log(f"回放数据载入失败：{exc}")
            return
        if load_imu:
            self._load_imu_trace()
        if self._viewer.has_data:
            self._tabs.setCurrentWidget(self._viewer)
            self._processing_view.append_log("已载入回放页（3D + 力矩曲线）。")

    def _load_imu_trace(self) -> None:
        """把右腿 IMU 三轴加速度映射到 C3D 时间后接入回放页；失败不致命。

        起始帧优先取当前已确认同步；历史 run 回放时控制器可能尚未同步，则退回
        会话级 ``sync_calibration.json``，保证两种回放场景都显示 IMU 曲线。
        """
        import h5py

        from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path

        dynamic = self._controller.dynamic
        if dynamic is None:
            self._viewer.set_imu(None, None)
            return
        files = dynamic.files
        if files.imu_h5_path is None or files.mocap_h5_path is None:
            self._viewer.set_imu(None, None)
            return
        start_frame = self._imu_start_frame()
        if start_frame is None:
            self._viewer.set_imu(None, None)
            return
        try:
            ensure_pipeline_on_path()
            from pipeline.synchronization.clock import (
                find_imu_sensor,
                imu_sensor_on_c3d_time,
                read_host_monotonic_ns,
            )

            with h5py.File(files.mocap_h5_path, "r") as mocap_h5, h5py.File(
                files.imu_h5_path, "r"
            ) as imu_h5:
                mocap_host_ns = read_host_monotonic_ns(mocap_h5)
                if not 0 <= start_frame < mocap_host_ns.size:
                    self._viewer.set_imu(None, None)
                    return
                c3d_t0_host_ns = int(mocap_host_ns[start_frame])
                sensor_index, _ = find_imu_sensor(imu_h5, side="right")
                imu_time_s, accel = imu_sensor_on_c3d_time(
                    imu_h5, c3d_t0_host_ns, sensor_index=sensor_index, axis_slice=slice(0, 3)
                )
            self._viewer.set_imu(imu_time_s, accel)
        except Exception as exc:  # noqa: BLE001
            _log.warning("加载 IMU 回放曲线失败：%s", exc)
            self._viewer.set_imu(None, None)

    def _imu_start_frame(self) -> int | None:
        """C3D→mocap.h5 起始帧：当前同步优先，否则读会话级 sync_calibration。"""
        sync = self._controller.sync
        if sync is not None:
            return int(sync.c3d_start_in_mocap_h5_frame)
        dynamic = self._controller.dynamic
        if dynamic is None:
            return None
        calib = dynamic.session_dir / "derived" / "opensim" / "sync_calibration.json"
        if not calib.is_file():
            return None
        try:
            import json

            document = json.loads(calib.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        for block in ("result", "auto_candidate"):
            frame = (document.get(block) or {}).get("c3d_start_in_mocap_h5_frame")
            if isinstance(frame, (int, float)):
                return int(frame)
        return None

    # ------------------------------------------------------------------
    # 导出真值数据（IMU 12 通道 + 关节力矩，对齐后写 CSV）
    # ------------------------------------------------------------------
    def _on_export_ground_truth(self) -> None:
        """导出训练用真值：右腿 IMU 12 通道 + 关节力矩，对齐到同一 C3D 时间轴。"""
        data = self._viewer.data
        if data is None:
            self._processing_view.append_log("无回放数据，无法导出真值。")
            return
        dynamic = self._controller.dynamic
        if dynamic is None:
            self._processing_view.append_log("未选择动态 Session，无法导出真值。")
            return
        files = dynamic.files
        if files.imu_h5_path is None or files.mocap_h5_path is None:
            self._processing_view.append_log("缺少 imu.h5 / mocap.h5，无法导出真值。")
            return
        start_frame = self._imu_start_frame()
        if start_frame is None:
            self._processing_view.append_log("尚未同步（无起始帧），无法导出真值。")
            return

        # 直接写入当前 Session 文件夹，训练时按 session 直接读取。
        out_path = dynamic.session_dir / "ground_truth.csv"
        try:
            import h5py

            from exo_collection.apps.calculate._pipeline import ensure_pipeline_on_path

            ensure_pipeline_on_path()
            from pipeline.synchronization.clock import (
                find_imu_sensor,
                imu_sensor_on_c3d_time,
                read_host_monotonic_ns,
            )

            with h5py.File(files.mocap_h5_path, "r") as mocap_h5, h5py.File(
                files.imu_h5_path, "r"
            ) as imu_h5:
                mocap_host_ns = read_host_monotonic_ns(mocap_h5)
                if not 0 <= start_frame < mocap_host_ns.size:
                    raise ValueError(f"起始帧 {start_frame} 超出 mocap.h5 范围")
                c3d_t0_host_ns = int(mocap_host_ns[start_frame])
                sensor_index, _ = find_imu_sensor(imu_h5, side="right")
                imu_time_s, imu_signal = imu_sensor_on_c3d_time(
                    imu_h5, c3d_t0_host_ns, sensor_index=sensor_index,
                    axis_slice=slice(0, 12),
                )

            from exo_collection.apps.calculate.ground_truth import (
                align_ground_truth,
                write_ground_truth_csv,
            )

            time_s, imu_aligned, moments = align_ground_truth(
                data.time_s, data.moments, imu_time_s, imu_signal
            )
            write_ground_truth_csv(
                out_path, time_s, imu_aligned, moments, moment_names=data.moment_names
            )
            self.statusBar().showMessage(f"已导出真值数据：{out_path}（{time_s.shape[0]} 帧）")
            self._processing_view.append_log(f"已导出真值数据：{out_path}")
        except Exception as exc:  # noqa: BLE001
            _log.exception("导出真值数据失败")
            QMessageBox.warning(self, "导出失败", f"导出真值数据失败：\n{exc}")

    # ------------------------------------------------------------------
    def _on_state_changed(self, state: str) -> None:
        self.statusBar().showMessage(f"状态：{state}")
        self._processing_view.set_solve_enabled(self._controller.can_solve)


def _sync_result_from_raw(raw: dict[str, Any], *, manual: bool) -> SyncResult:
    """把 ``run_auto_sync`` 的字典转成 models.SyncResult。"""
    from exo_collection.apps.calculate.models import SyncPeakPair

    pairs = tuple(
        SyncPeakPair(
            index=p.get("index", i + 1),
            imu_time_s=p["imu_time_on_c3d_s"],
            gaitway_time_s=p["gaitway_time_s"],
        )
        for i, p in enumerate(raw.get("peak_pairs") or [])
    )
    return SyncResult(
        c3d_start_in_mocap_h5_frame=int(raw.get("c3d_start_in_mocap_h5_frame", 0)),
        c3d_h5_match_rms_mm=float(raw.get("c3d_h5_match_rms_mm", 0.0)),
        c3d_h5_overlap_frames=int(raw.get("c3d_h5_overlap_frames", 0)),
        gaitway_offset_s=float(raw.get("gaitway_offset_s", raw.get("median_offset_s", 0.0))),
        confidence=str(raw.get("confidence", "LOW")),
        peak_pairs=pairs,
        drift_ppm=raw.get("drift_ppm"),
        offsets_s=tuple(raw.get("offsets_s") or ()),
        median_offset_s=raw.get("median_offset_s"),
        mad_s=raw.get("mad_s"),
        manual=manual,
    )


__all__ = ["CalculateWindow"]
