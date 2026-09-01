"""BLOCKING 状态机：显式管理每个 pipeline stage 的可用性。

原则（prompt2 §2）：
- 未经实测/确认的信息一律显式 ``BLOCKING``，禁止为跑通而补值；
- 上游可运行的 stage 继续跑，下游依赖缺失时终止并报"缺什么/为什么/怎么补"。

stage 状态：
- ``READY``    可执行
- ``BLOCKING`` 缺硬性依赖（标定矩阵 / 模型 / OpenSim 绑定 / 静态数据）
- ``TODO``     未做（非硬阻塞，如滤波 cutoff 待定）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    READY = "READY"
    BLOCKING = "BLOCKING"
    TODO = "TODO"


@dataclass
class Stage:
    key: str
    status: Status
    reason: str = ""
    detail: str = ""

    def line(self) -> str:
        tag = f"[{self.status.value}]"
        if self.reason:
            return f"{tag:<12} {self.key:<32} {self.reason}"
        return f"{tag:<12} {self.key}"


@dataclass
class BlockingReport:
    stages: list[Stage] = field(default_factory=list)

    def add(self, key: str, status: Status | str, reason: str = "", detail: str = "") -> None:
        self.stages.append(Stage(key, Status(status), reason, detail))

    def get(self, key: str) -> Stage | None:
        return next((s for s in self.stages if s.key == key), None)

    def is_blocking(self, key: str) -> bool:
        s = self.get(key)
        return s is not None and s.status == Status.BLOCKING

    def max_executable(self) -> str:
        """返回最后一个连续 READY（不越过 BLOCKING）的 stage key。"""
        last = None
        for s in self.stages:
            if s.status == Status.BLOCKING:
                break
            last = s.key
        return last or "(none)"

    def render(self) -> str:
        lines = [
            "=" * 52,
            "OpenSim Pipeline Preflight Check",
            "=" * 52,
            "",
        ]
        for s in self.stages:
            lines.append(s.line())
        lines += ["", f"Maximum executable stage: {self.max_executable()}"]
        return "\n".join(lines)


__all__ = ["Status", "Stage", "BlockingReport"]
