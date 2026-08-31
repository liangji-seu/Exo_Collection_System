"""Blocking-state 系统：显式区分 READY / TODO / BLOCKING / WARNING。

每次运行 run_pipeline 都会打印一个 preflight 表，并计算
"当前配置下最大可执行 stage"。下游缺失依赖时，只阻断依赖它的 stage，
上游已经能完成的 stage 继续执行（部分执行）。

stage 依赖顺序（后面的依赖前面的全部 READY）：

    C3D_INSPECTION
        ↓
    MARKER_PREPROCESSING
        ↓
    FORCE_EXTRACTION
        ↓
    FORCE_TO_MOCAP_TRANSFORM      ← 依赖力台标定（当前 BLOCKING）
        ↓
    MOCAP_TO_OPENSIM_TRANSFORM    ← 依赖动捕/OpenSim 轴对应（当前 BLOCKING）
        ↓
    GRF_GENERATION                ← 依赖左右脚分离 + 力方向约定
        ↓
    SCALE                         ← 依赖 gait2392 模型 + 静态 c3d + 质量
        ↓
    IK                            ← 依赖缩放模型 + 动态 trc
        ↓
    ID                            ← 依赖 IK + GRF
        ↓
    EXPORT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    READY = "READY"
    TODO = "TODO"
    BLOCKING = "BLOCKING"
    WARNING = "WARNING"


#: 阶段依赖顺序（索引越大越靠后，前一个阶段 READY 才允许下一个）
STAGES: tuple[str, ...] = (
    "C3D_INSPECTION",
    "MARKER_PREPROCESSING",
    "FORCE_EXTRACTION",
    "FORCE_TO_MOCAP_TRANSFORM",
    "MOCAP_TO_OPENSIM_TRANSFORM",
    "GRF_GENERATION",
    "SCALE",
    "IK",
    "ID",
    "EXPORT",
)

#: 每个 stage 依赖的前置检查项（key = 该 stage 需要 READY 的检查项名）
STAGE_REQUIRES: dict[str, tuple[str, ...]] = {
    "C3D_INSPECTION": ("static_c3d",),
    "MARKER_PREPROCESSING": ("static_c3d", "marker_input_unit"),
    "FORCE_EXTRACTION": ("static_c3d", "analog_channels"),
    "FORCE_TO_MOCAP_TRANSFORM": ("static_c3d", "forceplate_to_mocap_transform"),
    "MOCAP_TO_OPENSIM_TRANSFORM": ("static_c3d", "mocap_to_opensim_transform"),
    "GRF_GENERATION": (
        "forceplate_to_mocap_transform",
        "mocap_to_opensim_transform",
        "grf_left_right",
        "grf_force_convention",
    ),
    "SCALE": ("generic_model", "static_c3d", "subject_mass"),
    "IK": ("scaled_model", "dynamic_c3d"),
    "ID": ("ik_motion", "grf_mot"),
    "EXPORT": ("id_results",),
}


@dataclass
class CheckItem:
    name: str
    status: Status
    reason: str = ""

    def __str__(self) -> str:
        tag = f"[{self.status.value}]".ljust(11)
        text = f"  {tag} {self.name}"
        return text if not self.reason else f"{text}\n           ↳ {self.reason}"


@dataclass
class Preflight:
    items: list[CheckItem] = field(default_factory=list)

    def add(self, name: str, status: Status | str, reason: str = "") -> None:
        if isinstance(status, str):
            status = Status(status)
        self.items.append(CheckItem(name, status, reason))

    def get(self, name: str) -> CheckItem | None:
        for item in self.items:
            if item.name == name:
                return item
        return None

    def is_ready(self, name: str) -> bool:
        item = self.get(name)
        return item is not None and item.status is Status.READY

    def max_executable_stage(self) -> str:
        """返回当前配置下可执行的最后一级 stage 名。"""
        reached = None
        for stage in STAGES:
            reqs = STAGE_REQUIRES.get(stage, ())
            if all(self.is_ready(req) for req in reqs):
                reached = stage
            else:
                break  # 依赖链断裂，后面的都不可能执行
        return reached or "NONE"

    def report(self) -> str:
        lines = ["=" * 55, "OpenSim Pipeline Preflight Check", "=" * 55]
        for item in self.items:
            lines.append(str(item))
        lines.append("-" * 55)
        lines.append(f"Maximum executable stage: {self.max_executable_stage()}")
        return "\n".join(lines)


__all__ = ["Status", "STAGES", "STAGE_REQUIRES", "CheckItem", "Preflight"]
