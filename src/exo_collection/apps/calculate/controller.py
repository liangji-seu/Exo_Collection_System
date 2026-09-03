"""Exo Calculate 的 UI 状态机与任务编排（不 import OpenSim）。

职责边界：本模块只负责「选中哪个 Session → 处于哪个状态 → 下一步允许做什么」，
并把结果组装成可持久化的 run。所有数值计算（同步 / OpenSim / QC）都在
``opensim_joint_moment_pipeline`` 或后台 Worker 里完成；本模块不读大文件、
不 import ``opensim``。

状态机刻意区分「程序退出码 0」与「QC PASS」：``ProcessingState`` 里的
``COMPLETED_QC_PASS/WARN/FAIL`` 只反映生物力学 QC 结论，与子进程是否异常
退出（``FAILED``）无关。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from exo_collection.apps.calculate.models import (
    InputCheckReport,
    ProcessingConfig,
    ProcessingState,
    SessionRecord,
    SyncMethod,
    SyncResult,
)
from exo_collection.apps.calculate.operation import OperationContext

_log = logging.getLogger(__name__)

# 允许「开始解算」的状态：同步已确认，或已产生过可复跑的解算结论（同步仍有效）。
_SOLVABLE_STATES = frozenset({
    ProcessingState.SYNC_CONFIRMED,
    ProcessingState.COMPLETED_QC_PASS,
    ProcessingState.COMPLETED_QC_WARN,
    ProcessingState.COMPLETED_QC_FAIL,
})


@dataclass(frozen=True)
class SyncOutcome:
    """一次自动同步的输出（把 pipeline 结果字典转成类型化 DTO）。"""

    raw: dict[str, Any]
    gaitway_offset_s: float
    confidence: str
    n_pairs: int
    mad_s: float | None
    drift_ppm: float | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SyncOutcome":
        return cls(
            raw=dict(raw),
            gaitway_offset_s=float(raw.get("gaitway_offset_s", 0.0)),
            confidence=str(raw.get("confidence", "LOW")),
            n_pairs=int(raw.get("n_pairs", 0)),
            mad_s=raw.get("mad_s"),
            drift_ppm=raw.get("drift_ppm"),
        )


class CalculateController(QObject):
    """Exo Calculate 的核心状态机。

    信号都以「值传递」的方式发出，避免把可变 dict 直接暴露给 UI；UI 只读、
    回写必须走方法，保证状态迁移可审计。
    """

    dynamic_changed = Signal()
    static_changed = Signal()
    report_changed = Signal()
    sync_changed = Signal()
    state_changed = Signal(str)  # ProcessingState.value
    run_changed = Signal(object)  # run directory Path | None

    def __init__(self, data_root: Path) -> None:
        super().__init__()
        self._data_root = Path(data_root)
        self._dynamic: SessionRecord | None = None
        self._static: SessionRecord | None = None
        self._report: InputCheckReport | None = None
        self._sync: SyncResult | None = None
        self._sync_raw: dict[str, Any] | None = None
        self._method: SyncMethod = SyncMethod.AUTO_HIGH
        self._auto_candidate_raw: dict[str, Any] | None = None
        self._confirm_meta: dict[str, Any] | None = None
        self._config: ProcessingConfig | None = None
        self._state = ProcessingState.NOT_SCANNED
        self._run_dir: Path | None = None
        self._operation_counter = 0
        self._active_operation: OperationContext | None = None

    # ------------------------------------------------------------------
    # 数据根
    # ------------------------------------------------------------------
    @property
    def data_root(self) -> Path:
        return self._data_root

    # ------------------------------------------------------------------
    # 当前选中项
    # ------------------------------------------------------------------
    @property
    def dynamic(self) -> SessionRecord | None:
        return self._dynamic

    @property
    def static(self) -> SessionRecord | None:
        return self._static

    @property
    def report(self) -> InputCheckReport | None:
        return self._report

    @property
    def sync(self) -> SyncResult | None:
        return self._sync

    @property
    def sync_raw(self) -> dict[str, Any] | None:
        return self._sync_raw

    @property
    def sync_method(self) -> SyncMethod:
        return self._method

    @property
    def auto_candidate_raw(self) -> dict[str, Any] | None:
        """原始自动同步结果（人工/专家覆盖后仍保留，供审计与 ``sync_calibration``）。"""
        return self._auto_candidate_raw

    @property
    def confirm_meta(self) -> dict[str, Any] | None:
        """同步确认元数据：操作者类型、时间、是否微调、原始自动结果。"""
        return self._confirm_meta

    @property
    def can_solve(self) -> bool:
        """当前是否允许「开始解算」（同步已确认，或已有可复跑结论）。"""
        return self._state in _SOLVABLE_STATES

    @property
    def config(self) -> ProcessingConfig | None:
        return self._config

    @property
    def state(self) -> ProcessingState:
        return self._state

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    # ------------------------------------------------------------------
    # 状态迁移
    # ------------------------------------------------------------------
    def _set_state(self, state: ProcessingState) -> None:
        if state == self._state:
            return
        self._state = state
        _log.info("状态迁移 → %s", state.value)
        self.state_changed.emit(state.value)

    def set_dynamic(self, session: SessionRecord | None) -> None:
        self._dynamic = session
        self._report = None
        self._sync = None
        self._sync_raw = None
        self._method = SyncMethod.AUTO_HIGH
        self._auto_candidate_raw = None
        self._confirm_meta = None
        self._run_dir = None
        self._active_operation = None
        # 不同受试者之间绝不沿用旧静态模型（HH19 标定/缩放属于具体个体）。
        if (
            session is not None
            and self._static is not None
            and self._static.subject_code != session.subject_code
        ):
            self._static = None
            self.static_changed.emit()
        self.dynamic_changed.emit()
        self.report_changed.emit()
        self.sync_changed.emit()
        self.run_changed.emit(None)
        self._set_state(ProcessingState.NOT_SCANNED)

    def set_static(self, session: SessionRecord | None) -> None:
        self._static = session
        self._report = None
        self._sync = None
        self._sync_raw = None
        self._method = SyncMethod.AUTO_HIGH
        self._auto_candidate_raw = None
        self._confirm_meta = None
        self._active_operation = None
        self.static_changed.emit()
        self.report_changed.emit()
        self.sync_changed.emit()

    def set_input_report(self, report: InputCheckReport) -> None:
        self._report = report
        self.report_changed.emit()
        if report.valid and self._dynamic is not None:
            self._set_state(ProcessingState.READY_FOR_SYNC)
        else:
            self._set_state(ProcessingState.INPUT_INVALID)

    def set_sync(
        self,
        result: SyncResult,
        raw: dict[str, Any] | None = None,
        *,
        method: SyncMethod | None = None,
    ) -> None:
        """采纳一个同步结果，并记录其产生方式（method）。

        自动同步（``AUTO_HIGH``）会额外保存原始候选结果供审计；MEDIUM/LOW 的
        自动同步进入 ``SYNC_NEEDS_REVIEW``，HIGH 才自动 ``SYNC_CONFIRMED``。
        """
        if method is None:
            method = SyncMethod.MANUAL_PAIRED if result.manual else SyncMethod.AUTO_HIGH
        self._sync = result
        self._sync_raw = raw
        self._method = method
        self._confirm_meta = None
        if method is SyncMethod.AUTO_HIGH:
            self._auto_candidate_raw = dict(raw) if raw is not None else None
        self.sync_changed.emit()
        if method is SyncMethod.EXPERT_FORCED:
            self._set_state(ProcessingState.SYNC_CONFIRMED)
        elif result.manual:
            self._set_state(ProcessingState.SYNC_CONFIRMED)
        elif result.high_confidence:
            self._set_state(ProcessingState.SYNC_CONFIRMED)
        else:
            self._set_state(ProcessingState.SYNC_NEEDS_REVIEW)

    def clear_sync(self) -> None:
        self._sync = None
        self._sync_raw = None
        self._method = SyncMethod.AUTO_HIGH
        self._auto_candidate_raw = None
        self._confirm_meta = None
        self.sync_changed.emit()
        if self._report is not None and self._report.valid:
            self._set_state(ProcessingState.READY_FOR_SYNC)

    def confirm_sync(
        self,
        *,
        operator_type: str = "operator",
        note: str | None = None,
        adjusted: bool = False,
    ) -> None:
        """记录操作者确认元数据并把状态推进到 ``SYNC_CONFIRMED``（§3.2 第 3 条）。

        只有处于 ``SYNC_NEEDS_REVIEW``（自动同步 MEDIUM/LOW）时才有意义；确认时
        记录操作者类型、时间、原始自动结果与是否微调。
        """
        if self._sync is None:
            return
        self._confirm_meta = {
            "operator_type": operator_type,
            "confirmed_at_utc": datetime.now().astimezone().isoformat(),
            "note": note,
            "adjusted": adjusted,
            "original_auto": self._auto_candidate_raw,
        }
        self._set_state(ProcessingState.SYNC_CONFIRMED)
        self.sync_changed.emit()

    def set_expert_forced(
        self,
        offset_s: float,
        *,
        note: str | None = None,
        operator_type: str = "expert",
    ) -> None:
        """专家直接输入 offset（无峰证据的强制值），二次确认后调用（§3.2 第 5 条）。

        结果标记 ``EXPERT_FORCED``；最终 QC 由 §3.3 依据 method 至少判为 WARN，
        绝不 PASS。C3D↔H5 匹配信息复用已算出的自动结果（若可用），否则占位。
        """
        frame = 0
        rms = 0.0
        overlap = 0
        if self._auto_candidate_raw:
            frame = int(self._auto_candidate_raw.get("c3d_start_in_mocap_h5_frame", 0))
            rms = float(self._auto_candidate_raw.get("c3d_h5_match_rms_mm", 0.0))
            overlap = int(self._auto_candidate_raw.get("c3d_h5_overlap_frames", 0))
        result = SyncResult(
            c3d_start_in_mocap_h5_frame=frame,
            c3d_h5_match_rms_mm=rms,
            c3d_h5_overlap_frames=overlap,
            gaitway_offset_s=float(offset_s),
            confidence="LOW",
            median_offset_s=float(offset_s),
            mad_s=None,
            manual=False,
            audit_note=note or "专家强制 offset（无峰证据）",
            method=SyncMethod.EXPERT_FORCED,
        )
        self._sync = result
        self._sync_raw = None
        self._method = SyncMethod.EXPERT_FORCED
        self._confirm_meta = {
            "operator_type": operator_type,
            "confirmed_at_utc": datetime.now().astimezone().isoformat(),
            "note": note,
            "adjusted": True,
            "original_auto": self._auto_candidate_raw,
        }
        self.sync_changed.emit()
        self._set_state(ProcessingState.SYNC_CONFIRMED)

    def set_config(self, config: ProcessingConfig) -> None:
        self._config = config

    def set_run_dir(self, run_dir: Path | None) -> None:
        self._run_dir = run_dir
        self.run_changed.emit(run_dir)

    def mark_processing(self) -> None:
        self._set_state(ProcessingState.PROCESSING)

    def mark_completed(self, qc_status: str) -> None:
        """``qc_status`` 来自 QC 规则的 PASS / WARN / FAIL（非进程退出码）。"""
        normalized = str(qc_status).upper()
        state = {
            "PASS": ProcessingState.COMPLETED_QC_PASS,
            "WARN": ProcessingState.COMPLETED_QC_WARN,
            "FAIL": ProcessingState.COMPLETED_QC_FAIL,
        }.get(normalized, ProcessingState.COMPLETED_QC_WARN)
        self._set_state(state)

    def mark_failed(self) -> None:
        self._set_state(ProcessingState.FAILED)

    def mark_cancelled(self) -> None:
        self._set_state(ProcessingState.CANCELLED)

    def mark_stale(self) -> None:
        self._set_state(ProcessingState.STALE_INPUTS)

    # ------------------------------------------------------------------
    # 异步任务上下文（operation token，prompt6 §3.4）
    # ------------------------------------------------------------------
    @property
    def active_operation(self) -> OperationContext | None:
        """当前正在进行的后台任务上下文（无则 None）。"""
        return self._active_operation

    def begin_operation(self, kind: str, *, run_dir: Path | None = None) -> OperationContext:
        """开始一个后台任务，返回携带自增 operation_id 的不可变上下文。

        新任务会自动使旧任务的上下文失效（``is_current_operation`` 判 False），
        从而让旧回调被 UI 丢弃。
        """
        self._operation_counter += 1
        dynamic = self._dynamic
        static = self._static
        input_paths: list[str] = []
        if dynamic is not None:
            for path in (
                dynamic.files.c3d_path,
                dynamic.files.txt_path,
                dynamic.files.mocap_h5_path,
                dynamic.files.imu_h5_path,
            ):
                if path is not None:
                    input_paths.append(str(path))
        if static is not None and static.files.c3d_path is not None:
            input_paths.append(str(static.files.c3d_path))
        ctx = OperationContext(
            operation_id=self._operation_counter,
            kind=kind,
            dynamic_session_uuid=dynamic.session_uuid if dynamic else None,
            dynamic_trial_uuid=dynamic.trial_uuid if dynamic else None,
            static_session_uuid=static.session_uuid if static else None,
            input_paths=tuple(input_paths),
            run_dir=run_dir,
        )
        self._active_operation = ctx
        return ctx

    def is_current_operation(self, ctx: OperationContext | None) -> bool:
        """``ctx`` 是否仍是当前任务（旧/过期回调据此丢弃）。"""
        return (
            ctx is not None
            and self._active_operation is not None
            and ctx.operation_id == self._active_operation.operation_id
        )

    def finish_operation(self, ctx: OperationContext | None) -> None:
        """结束一个任务；仅当它仍是当前任务时才清空当前上下文。"""
        if self.is_current_operation(ctx):
            self._active_operation = None

    # ------------------------------------------------------------------
    # 派生 run 目录名（不覆盖旧 run）
    # ------------------------------------------------------------------
    def new_run_directory(self, parent: Path) -> Path:
        """在 ``parent`` 下生成不覆盖旧 run 的 ``run_<timestamp>_<shortid>``。"""
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = datetime.now().strftime("%f")[:4]
        candidate = parent / f"run_{stamp}_{short_id}"
        index = 1
        while candidate.exists():
            candidate = parent / f"run_{stamp}_{short_id}_{index}"
            index += 1
        return candidate


__all__ = ["CalculateController", "SyncOutcome"]
