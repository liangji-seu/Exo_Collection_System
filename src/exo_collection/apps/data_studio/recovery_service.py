"""Spawn-process service for Data Studio recovery discovery and decisions."""

from __future__ import annotations

import multiprocessing as mp
from contextlib import suppress
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
import traceback
from typing import Any, Literal


RecoveryOperation = Literal["scan", "repair", "finalize", "abort"]


def _recovery_process_entry(
    operation: RecoveryOperation,
    keyword_arguments: dict[str, Any],
    result_queue: Queue[Any],
) -> None:
    """Run all filesystem-heavy inspection/mutation outside the Qt process."""

    try:
        from exo_collection.storage.recovery_manager import (
            abort_recording_preserving_data,
            discover_recoverable_trials,
            finalize_prepared_recording,
            repair_recording_directory,
        )

        if operation == "scan":
            result = discover_recoverable_trials(**keyword_arguments)
        elif operation == "repair":
            result = repair_recording_directory(**keyword_arguments)
        elif operation == "finalize":
            result = finalize_prepared_recording(**keyword_arguments)
        elif operation == "abort":
            result = abort_recording_preserving_data(**keyword_arguments)
        else:  # pragma: no cover - validated by the parent
            raise ValueError(f"unsupported recovery operation: {operation}")
        result_queue.put(("completed", operation, result))
    except BaseException:
        result_queue.put(("failed", operation, traceback.format_exc(limit=30)))


class RecoveryBackgroundService:
    """Own at most one spawn worker and expose a non-blocking polling API."""

    def __init__(self) -> None:
        self._context = mp.get_context("spawn")
        self._queue: Queue[Any] | None = None
        self._process: mp.Process | None = None
        self._operation: RecoveryOperation | None = None
        self._terminal_reported = False

    @property
    def busy(self) -> bool:
        return self._process is not None and self._process.exitcode is None

    @property
    def operation(self) -> RecoveryOperation | None:
        return self._operation

    def _start(self, operation: RecoveryOperation, **keyword_arguments: Any) -> None:
        if self._process is not None:
            raise RuntimeError("a recovery operation is already pending cleanup")
        queue = self._context.Queue(maxsize=1)
        process = self._context.Process(
            target=_recovery_process_entry,
            args=(operation, keyword_arguments, queue),
            name=f"data-studio-recovery-{operation}",
            daemon=False,
        )
        self._queue = queue
        self._process = process
        self._operation = operation
        self._terminal_reported = False
        try:
            process.start()
        except BaseException:
            if process.pid is not None:
                with suppress(BaseException):
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=3.0)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=3.0)
            with suppress(BaseException):
                queue.cancel_join_thread()
                queue.close()
            with suppress(BaseException):
                queue.join_thread()
            with suppress(BaseException):
                process.close()
            self._queue = None
            self._process = None
            self._operation = None
            raise

    def start_scan(self, dataset_root: str | Path) -> None:
        self._start("scan", dataset_root=Path(dataset_root).expanduser().resolve())

    def start_repair(self, recording_directory: str | Path) -> None:
        self._start(
            "repair",
            recording_directory=Path(recording_directory).expanduser().resolve(),
        )

    def start_finalize(
        self,
        recording_directory: str | Path,
        *,
        confirmed_by: str | None = None,
    ) -> None:
        self._start(
            "finalize",
            recording_directory=Path(recording_directory).expanduser().resolve(),
            confirmed=True,
            confirmed_by=confirmed_by,
        )

    def start_abort(
        self,
        recording_directory: str | Path,
        *,
        reason: str,
        confirmed_by: str | None = None,
    ) -> None:
        self._start(
            "abort",
            recording_directory=Path(recording_directory).expanduser().resolve(),
            reason=reason,
            confirmed=True,
            confirmed_by=confirmed_by,
        )

    def poll(self) -> tuple[str, RecoveryOperation, object] | None:
        """Return one result without blocking; caller then invokes :meth:`finish`."""

        process = self._process
        queue = self._queue
        operation = self._operation
        if process is None or queue is None or operation is None:
            return None
        try:
            status, queued_operation, payload = queue.get_nowait()
            self._terminal_reported = True
            return str(status), queued_operation, payload
        except Empty:
            if process.exitcode is not None and not self._terminal_reported:
                self._terminal_reported = True
                return (
                    "failed",
                    operation,
                    f"recovery worker exited with code {process.exitcode} without a result",
                )
            return None

    def finish(self, timeout: float = 3.0) -> None:
        process = self._process
        queue = self._queue
        if process is None or queue is None:
            return
        process.join(timeout)
        if process.is_alive():
            raise RuntimeError("cannot finish a recovery worker that is still running")
        queue.cancel_join_thread()
        queue.close()
        queue.join_thread()
        process.close()
        self._queue = None
        self._process = None
        self._operation = None
        self._terminal_reported = False

    def cancel(self, timeout: float = 3.0) -> None:
        """Stop a worker; atomic rename and append-only logs remain crash-safe."""

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout)
        if process is not None and process.is_alive():
            process.kill()
            process.join(timeout)
        if process is not None and process.is_alive():
            raise RuntimeError("recovery worker did not exit after kill")
        if process is not None and process.exitcode is not None:
            self.finish(timeout=0)


__all__ = ["RecoveryBackgroundService", "RecoveryOperation"]
