"""Spawn-safe process boundary for disk-heavy Data Studio tools."""

from __future__ import annotations

import logging
import multiprocessing as mp
from contextlib import suppress
from multiprocessing.queues import Queue
from queue import Empty
import time
import traceback
from typing import Any, Literal

from exo_collection.logging_setup import (
    configure_subprocess_logging,
    current_collector_log_path,
)

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
    log_path: str | None = None,
) -> None:
    """Import tool implementations inside a clean spawned interpreter."""

    # Windows ``spawn`` imports modules in a clean interpreter, so an in-memory
    # parent logging configuration is not inherited. Pass the already protected
    # application log path explicitly; never pass credentials or tool payloads.
    configure_subprocess_logging(log_path=log_path)
    started = time.monotonic()
    _log.info(
        "Data Studio worker started: operation=%s argument_names=%s",
        operation,
        sorted(keyword_arguments),
    )
    try:
        if operation == "catalog_refresh":
            from .service import load_catalog_snapshot

            result = load_catalog_snapshot(**keyword_arguments)
        elif operation == "playback":
            from .local_tools import load_trial_playback

            result = load_trial_playback(**keyword_arguments)
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
        _log.info(
            "Data Studio worker completed: operation=%s result_type=%s elapsed_ms=%.1f",
            operation,
            type(result).__name__,
            (time.monotonic() - started) * 1000.0,
        )
        result_queue.put(("completed", result))
    except BaseException:
        _log.exception(
            "Data Studio worker failed: operation=%s elapsed_ms=%.1f",
            operation,
            (time.monotonic() - started) * 1000.0,
        )
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
        shared_log_path = current_collector_log_path()
        self._process = self._context.Process(
            target=_local_process_entry,
            args=(
                operation,
                keyword_arguments,
                self._result_queue,
                str(shared_log_path) if shared_log_path is not None else None,
            ),
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
            _log.info("Starting Data Studio worker: operation=%s", self.operation)
            self._process.start()
            _log.debug(
                "Data Studio worker process created: operation=%s pid=%s",
                self.operation,
                self._process.pid,
            )
        except BaseException:
            _log.exception("Could not start Data Studio worker: operation=%s", self.operation)
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
        _log.info(
            "Data Studio worker result received: operation=%s status=%s",
            self.operation,
            status,
        )
        return str(status), payload

    def join(self, timeout: float | None = None) -> int | None:
        self._process.join(timeout)
        return self._process.exitcode

    def terminate(self, timeout: float = 3.0) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self._process.is_alive():
            _log.warning(
                "Terminating Data Studio worker: operation=%s pid=%s",
                self.operation,
                self._process.pid,
            )
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
