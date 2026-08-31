"""OpenSim 绑定加载 + 阻断异常。

无绑定时 pipeline 仍然可以跑上游（inspection/预处理），只有 Scale/IK/ID
会抛 PipelineBlockingError，run_pipeline 据此把该 stage 标为 BLOCKING。
"""

from __future__ import annotations

from typing import Any


class PipelineBlockingError(RuntimeError):
    """某 stage 因缺少标定/模型/绑定而无法执行。"""


def get_opensim() -> Any | None:
    """返回 opensim 模块，或 None（未安装）。"""
    try:
        import opensim
        return opensim
    except Exception:  # noqa: BLE001 - 绑定加载失败原因多样，统一视为不可用
        return None


def require_opensim() -> Any:
    osim = get_opensim()
    if osim is None:
        raise PipelineBlockingError(
            "OpenSim 绑定不可用。安装：conda install -c opensim-org opensim。"
            "未安装时只能执行到 C3D inspection / 预处理阶段。"
        )
    return osim


__all__ = ["PipelineBlockingError", "get_opensim", "require_opensim"]
