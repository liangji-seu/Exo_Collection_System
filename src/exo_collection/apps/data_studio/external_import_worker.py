"""Spawn-safe process boundary for immutable external-modality imports."""

from __future__ import annotations

import multiprocessing as mp
from contextlib import suppress
from multiprocessing.queues import Queue
from queue import Empty
import traceback
from typing import Any, Mapping

from exo_collection.external import ExternalImportRequest


def _external_import_entry(
    request_values: Mapping[str, Any],
    result_queue: Queue[Any],
) -> None:
    try:
        from exo_collection.external import import_external_artifact

        result = import_external_artifact(request_values)
        result_queue.put(("completed", result))
    except BaseException:
        result_queue.put(("failed", traceback.format_exc(limit=30)))


class ExternalImportWorker:
    """Own one spawned import process with the same polling API as local tools."""

    def __init__(self, request: ExternalImportRequest) -> None:
        self._context = mp.get_context("spawn")
        self._queue = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=_external_import_entry,
            args=(request.model_dump(mode="python"), self._queue),
            name="data-studio-external-import",
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
            raise RuntimeError("external import worker can only be started once")
        try:
            self._process.start()
        except BaseException:
            if self._process.pid is not None:
                with suppress(BaseException):
                    if self._process.is_alive():
                        self._process.terminate()
                    self._process.join(timeout=3.0)
                    if self._process.is_alive():
                        self._process.kill()
                        self._process.join(timeout=3.0)
            with suppress(BaseException):
                self._queue.cancel_join_thread()
                self._queue.close()
            with suppress(BaseException):
                self._queue.join_thread()
            with suppress(BaseException):
                self._process.close()
            raise

    def poll_result(self) -> tuple[str, object] | None:
        try:
            status, payload = self._queue.get_nowait()
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
            raise RuntimeError("external import worker did not exit after kill")

    def close(self) -> None:
        if self._process.is_alive():
            raise RuntimeError("cannot close a running external import worker")
        self._queue.cancel_join_thread()
        self._queue.close()
        self._queue.join_thread()
        self._process.close()


__all__ = ["ExternalImportWorker"]
