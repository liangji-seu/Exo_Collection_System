"""Exo Calculate 的不可变 DTO 与处理状态机。

这些类型是 UI 层与计算层之间的契约：计算层（``opensim_joint_moment_pipeline``）
只输出/消费这些结构，不 import Qt。状态机刻意把「进程退出码为 0」与
「QC PASS」区分开 —— 前者是 ``FAILED``/``CANCELLED`` 的判断依据，后者由
版本化 QC 规则单独判定（见 ``pipeline.qc``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ProcessingState(str, Enum):
    """一次「标定 + 解算」工作流的生命周期状态。

    ``COMPLETED_QC_PASS`` / ``COMPLETED_QC_WARN`` / ``COMPLETED_QC_FAIL`` 三者
    表示处理本身成功完成，区别仅在于生物力学 QC 的结论；绝不与 ``FAILED``
    （子进程/程序异常退出）混为一谈。
    """

    NOT_SCANNED = "NOT_SCANNED"
    INPUT_INVALID = "INPUT_INVALID"
    READY_FOR_SYNC = "READY_FOR_SYNC"
    SYNC_NEEDS_REVIEW = "SYNC_NEEDS_REVIEW"
    SYNC_CONFIRMED = "SYNC_CONFIRMED"
    PROCESSING = "PROCESSING"
    COMPLETED_QC_PASS = "COMPLETED_QC_PASS"
    COMPLETED_QC_WARN = "COMPLETED_QC_WARN"
    COMPLETED_QC_FAIL = "COMPLETED_QC_FAIL"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    STALE_INPUTS = "STALE_INPUTS"


@dataclass(frozen=True, slots=True)
class SessionFiles:
    """一个 Session 目录内的输入文件路径（每项可选，缺失为 ``None``）。"""

    c3d_path: Path | None = None
    txt_path: Path | None = None
    mocap_h5_path: Path | None = None
    imu_h5_path: Path | None = None

    @property
    def has_dynamic_inputs(self) -> bool:
        return (
            self.c3d_path is not None
            and self.txt_path is not None
            and self.mocap_h5_path is not None
            and self.imu_h5_path is not None
        )

    def missing(self) -> tuple[str, ...]:
        """返回缺失输入的可读标签（空 tuple 表示齐全）。"""
        labels = (
            ("C3D", self.c3d_path),
            ("Gaitway TXT", self.txt_path),
            ("mocap.h5", self.mocap_h5_path),
            ("imu.h5", self.imu_h5_path),
        )
        return tuple(name for name, path in labels if path is None)


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """从 ``.exo/manifest.json`` 发现的单个 Session（= 一个 Trial 数据包）。"""

    manifest_path: Path
    session_dir: Path
    session_name: str
    subject_code: str
    subject_uuid: str
    project_code: str
    project_name: str
    condition_code: str
    condition_name: str
    condition_level: str | None
    repeat_index: int
    trial_uuid: str
    session_uuid: str
    state: str
    started_at_utc: str
    condition_parameters: dict[str, Any] = field(default_factory=dict)
    files: SessionFiles = field(default_factory=SessionFiles)

    @property
    def is_stand(self) -> bool:
        """静态标定 Session 判定（条件码含 ``STAND``）。"""
        return "STAND" in (self.condition_code or "").upper()

    @property
    def subject_and_condition(self) -> str:
        return f"{self.subject_code} / {self.condition_code} r{self.repeat_index}"


@dataclass(frozen=True, slots=True)
class InputCheckReport:
    """只读扫描结果：采样率、时长、marker 数量、Gaitway 列、缺失项。"""

    subject_code: str
    dynamic_session: str
    static_session: str | None
    dynamic_c3d_rate_hz: float | None = None
    dynamic_c3d_duration_s: float | None = None
    dynamic_hh19_markers: int = 0
    static_c3d_rate_hz: float | None = None
    static_c3d_duration_s: float | None = None
    static_hh19_markers: int = 0
    gaitway_rate_hz: float | None = None
    gaitway_has_bilateral_columns: bool = False
    mocap_h5_rate_hz: float | None = None
    imu_h5_rate_hz: float | None = None
    problems: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.problems


@dataclass(frozen=True, slots=True)
class SyncPeakPair:
    """一对跺脚冲击峰（右腿 IMU ↔ Gaitway 总垂直力）。"""

    index: int
    imu_time_s: float
    gaitway_time_s: float

    @property
    def offset_s(self) -> float:
        return self.gaitway_time_s - self.imu_time_s


class SyncMethod(str, Enum):
    """同步结果的产生方式（写入 ``sync_calibration.json``，prompt6 §3.2 第 7 条）。"""

    AUTO_HIGH = "AUTO_HIGH"        # 自动同步（跺脚 pipeline）产出
    MANUAL_PAIRED = "MANUAL_PAIRED"  # 人工逐对点峰
    EXPERT_FORCED = "EXPERT_FORCED"  # 专家直接输入 offset（无峰证据的强制值）


@dataclass(frozen=True, slots=True)
class SyncResult:
    """自动/人工同步的最终结论。

    约定：``gaitway_time = c3d_time + gaitway_offset_s``（与现有 pipeline 一致）。
    ``drift_ppm`` 为仿射斜率 ``a`` 对应的 ppm 漂移；短记录无法可靠估计时为
    ``None``（表示 UNASSESSED，绝不伪造 0 漂移）。
    """

    c3d_start_in_mocap_h5_frame: int
    c3d_h5_match_rms_mm: float
    c3d_h5_overlap_frames: int
    gaitway_offset_s: float
    confidence: str  # HIGH / MEDIUM / LOW
    peak_pairs: tuple[SyncPeakPair, ...] = ()
    drift_ppm: float | None = None
    offsets_s: tuple[float, ...] = ()
    median_offset_s: float | None = None
    mad_s: float | None = None
    manual: bool = False
    audit_note: str | None = None
    method: SyncMethod = SyncMethod.AUTO_HIGH

    @property
    def high_confidence(self) -> bool:
        return self.confidence == "HIGH"

    @property
    def is_expert_forced(self) -> bool:
        return self.method is SyncMethod.EXPERT_FORCED


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """一次解算的全部参数（写入 ``processing_config.yaml`` 与 run manifest）。"""

    mass_kg: float
    height_m: float
    marker_cutoff_hz: float = 6.0
    grf_cutoff_hz: float = 20.0
    opensim_x_sign: float = -1.0
    opensim_z_sign: float = -1.0
    analysis_time_range_s: tuple[float, float] | None = None
    static_time_range_s: tuple[float, float] | None = None
    marker_adjustment_expert_confirmed: bool = False
    expert_mode: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "mass_kg": self.mass_kg,
            "height_m": self.height_m,
            "marker_cutoff_hz": self.marker_cutoff_hz,
            "grf_cutoff_hz": self.grf_cutoff_hz,
            "opensim_x_sign": self.opensim_x_sign,
            "opensim_z_sign": self.opensim_z_sign,
            "analysis_time_range_s": (
                list(self.analysis_time_range_s) if self.analysis_time_range_s else None
            ),
            "static_time_range_s": (
                list(self.static_time_range_s) if self.static_time_range_s else None
            ),
            "marker_adjustment_expert_confirmed": self.marker_adjustment_expert_confirmed,
            "expert_mode": self.expert_mode,
        }
