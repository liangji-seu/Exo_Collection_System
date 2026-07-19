"""Spawn-safe process boundary for disk-heavy Data Studio tools."""

from __future__ import annotations

import logging
import multiprocessing as mp
from contextlib import suppress
from multiprocessing.queues import Queue
from queue import Empty
import traceback
from typing import Any, Literal

_log = logging.getLogger(__name__)


ProcessOperation = Literal[
    "catalog_refresh",
    "playback",
    "checksum",
    "management_refresh",
    "management_summary",
    "management_export",
]

_OPERATIONS = {
    "catalog_refresh",
    "playback",
    "checksum",
    "management_refresh",
    "management_summary",
    "management_export",
}


def _local_process_entry(
    operation: ProcessOperation,
    keyword_arguments: dict[str, Any],
    result_queue: Queue[Any],
) -> None:
    """Import tool implementations inside a clean spawned interpreter."""

    _log.info("子进程启动: operation=%s, args_keys=%s", operation, list(keyword_arguments.keys()))
    try:
        if operation == "catalog_refresh":
            from .service import load_catalog_snapshot

            result = load_catalog_snapshot(**keyword_arguments)
        elif operation == "playback":
            _log.info("导入 load_trial_playback…")
            from .local_tools import load_trial_playback

            _log.info("开始执行 load_trial_playback…")
            result = load_trial_playback(**keyword_arguments)
            _log.info("load_trial_playback 返回: %s", type(result).__name__)
        elif operation == "checksum":
            from .local_tools import verify_trial_checksums

            result = verify_trial_checksums(**keyword_arguments)
        elif operation == "management_refresh":
            from .management import load_management_refresh

            result = load_management_refresh(**keyword_arguments)
        elif operation == "management_summary":
            from .management import load_management_summary

            result = load_management_summary(**keyword_arguments)
        elif operation == "management_export":
            from .management import export_manifest_inventory_checked

            result = export_manifest_inventory_checked(**keyword_arguments)
        else:  # pragma: no cover - parent validates this before spawning
            raise ValueError(f"unsupported Data Studio process operation: {operation}")
        _log.info("子进程完成: operation=%s", operation)
        result_queue.put(("completed", result))
    except BaseException:
        _log.exception("子进程失败: operation=%s", operation)
        result_queue.put(("failed", traceback.format_exc(limit=30)))


class DataStudioProcessWorker:
    """Own one spawned, single-result local analysis process."""

    def __init__(
        self,
        operation: ProcessOperation,
        **keyword_arguments: Any,
    ) -> None:
        if operation not in _OPERATIONS:
            raise ValueError(f"unsupported process operation: {operation}")
        self.operation = operation
        self._context = mp.get_context("spawn")
        self._result_queue = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=_local_process_entry,
            args=(operation, keyword_arguments, self._result_queue),
            name=f"data-studio-{operation}",
            daemon=False,
        )

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def start(self) -> None:
        if self._process.pid is not None:
            raise RuntimeError("Data Studio worker can only be started once")
        try:
            self._process.start()
        except BaseException:
            # Windows spawn can fail before a PID is assigned.  No window-level
            # polling will then own this worker, so release IPC handles here.
            if self._process.pid is not None:
                with suppress(BaseException):
                    if self._process.is_alive():
                        self._process.terminate()
                    self._process.join(timeout=3.0)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(timeout=3.0)
            with suppress(BaseException):
                self._result_queue.cancel_join_thread()
                self._result_queue.close()
            with suppress(BaseException):
                self._result_queue.join_thread()
            with suppress(BaseException):
                self._process.close()
            raise

    def poll_result(self) -> tuple[str, object] | None:
        try:
            status, payload = self._result_queue.get_nowait()
        except Empty:
            return None
        return str(status), payload

    def join(self, timeout: float | None = None) -> int | None:
        self._process.join(timeout)
        return self._process.exitcode

    def terminate(self, timeout: float = 3.0) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout)
        if self._process.is_alive():
            raise RuntimeError("Data Studio process worker did not exit after kill")

    def close(self) -> None:
        if self._process.is_alive():
            raise RuntimeError("cannot close a running Data Studio process worker")
        # cancel_join_thread must be called before close/join_thread whenever the
        # process may have been killed (SIGKILL on POSIX / TerminateProcess on
        # Windows).  Without it the Queue feeder thread can wait forever for the
        # dead child to consume buffered data, hanging pytest -q and the UI.
        self._result_queue.cancel_join_thread()
        self._result_queue.close()
        self._result_queue.join_thread()
        self._process.close()


__all__ = ["DataStudioProcessWorker", "ProcessOperation"]
