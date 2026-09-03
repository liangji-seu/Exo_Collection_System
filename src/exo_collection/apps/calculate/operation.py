"""异步任务的不可变上下文（operation token）。

后台 Worker 与 UI 回调都持有同一个 ``OperationContext``：UI 启动一个任务时
从 controller 申请一个自增 ``operation_id``，并把该上下文绑定到 Worker 的回调。
回调触发时先比对 ``controller.is_current_operation(ctx)``，不一致说明用户已
切换到别的 Session / 重新启动了任务，旧回调只能丢弃，绝不能更新当前状态或
写文件（prompt6 §3.4）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OperationContext:
    """一次后台任务的不可变上下文。

    ``operation_id`` 每次 ``begin_operation`` 自增，用于识别「这是不是当前那次」；
    Session UUID 与输入路径用于把回调与它真正处理的数据绑定，避免串 Session。
    """

    operation_id: int
    kind: str                      # "scan" / "sync" / "load_sync_data" / "prep" / "opensim"
    dynamic_session_uuid: str | None = None
    dynamic_trial_uuid: str | None = None
    static_session_uuid: str | None = None
    input_paths: tuple[str, ...] = ()
    run_dir: Path | None = None


__all__ = ["OperationContext"]
