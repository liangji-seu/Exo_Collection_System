"""Spawn-safe process boundary for the ultrasound block writer.

The simulated milestone intentionally sends copied NumPy arrays through a
bounded ``multiprocessing.Queue``.  A real high-throughput hardware backend can
replace that data transport with the project's shared-memory buffer contract
without changing the control/result protocol or the on-disk block format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import wraps
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty, Full
import time
import traceback
from threading import RLock
from types import TracebackType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .binary_block import BlockBinaryWriter, companion_paths


class BlockBinaryWriterProcessError(RuntimeError):
    """A writer child failed to initialize, append, flush, or close."""

    def __init__(
        self,
        message: str,
        *,
        remote_exception_type: str | None = None,
        remote_traceback: str | None = None,
    ) -> None:
        super().__init__(message)
        self.remote_exception_type = remote_exception_type
        self.remote_traceback = remote_traceback


def _synchronized(method: Any) -> Any:
    """Serialize the public lifecycle around one proxy instance."""

    @wraps(method)
    def locked(self: BlockBinaryWriterProcess, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return locked


def _put_result(result_queue: Queue[Any], message: dict[str, Any]) -> None:
    """Best-effort bounded result publication from the writer child."""

    try:
        result_queue.put(message, timeout=1.0)
    except (Full, OSError, ValueError):
        # The parent may already be aborting.  Never keep a failed writer alive
        # solely because its small result pipe has disappeared.
        pass


def _writer_process_entry(
    writer_options: dict[str, Any],
    data_queue: Queue[Any],
    control_queue: Queue[Any],
    result_queue: Queue[Any],
    parent_liveness: Connection,
) -> None:
    """Top-level target required by Windows spawn and PyInstaller."""

    writer: BlockBinaryWriter | None = None
    processed_count = 0
    pending_flush: dict[str, Any] | None = None
    pending_close: dict[str, Any] | None = None
    try:
        writer = BlockBinaryWriter(**writer_options)
        _put_result(
            result_queue,
            {
                "kind": "ready",
                "next_sequence": writer.next_sequence,
                "next_sample_index": writer.next_sample_index,
                "dtype": writer.dtype.str,
                "sample_shape": list(writer.sample_shape),
            },
        )

        while True:
            if parent_liveness.poll():
                try:
                    parent_liveness.recv()
                except EOFError:
                    # The collector-core parent was forcibly terminated.  Do
                    # not become an orphan holding `.partial` file handles.
                    writer.close()
                    return
                raise RuntimeError("unexpected parent-liveness payload")
            while True:
                try:
                    command = control_queue.get_nowait()
                except Empty:
                    break
                kind = command.get("kind")
                if kind == "abort":
                    writer.close()
                    _put_result(
                        result_queue,
                        {"kind": "aborted", "processed_count": processed_count},
                    )
                    return
                if kind == "flush":
                    if pending_flush is not None or pending_close is not None:
                        raise RuntimeError("writer received overlapping lifecycle commands")
                    pending_flush = command
                elif kind == "close":
                    if pending_close is not None:
                        raise RuntimeError("writer received duplicate close commands")
                    pending_close = command
                else:
                    raise ValueError(f"unknown writer control command: {kind!r}")

            if (
                pending_flush is not None
                and processed_count >= int(pending_flush["target_count"])
            ):
                writer.flush(fsync=pending_flush.get("fsync"))
                _put_result(
                    result_queue,
                    {
                        "kind": "flushed",
                        "command_id": pending_flush["command_id"],
                        "processed_count": processed_count,
                    },
                )
                pending_flush = None

            if (
                pending_close is not None
                and processed_count >= int(pending_close["target_count"])
            ):
                writer.close()
                _put_result(
                    result_queue,
                    {
                        "kind": "closed",
                        "command_id": pending_close["command_id"],
                        "processed_count": processed_count,
                        "next_sequence": writer.next_sequence,
                        "next_sample_index": writer.next_sample_index,
                    },
                )
                return

            try:
                item = data_queue.get(timeout=0.05)
            except Empty:
                continue
            if item.get("kind") != "append":
                raise ValueError(f"unknown writer data command: {item.get('kind')!r}")
            writer.append(item["samples"], **item["arguments"])
            processed_count += 1
    except BaseException as exc:
        remote_traceback = traceback.format_exc(limit=30)
        _put_result(
            result_queue,
            {
                "kind": "error",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": remote_traceback,
                "processed_count": processed_count,
            },
        )
        raise SystemExit(1) from None
    finally:
        if writer is not None and not writer.closed:
            try:
                writer.close()
            except BaseException:
                pass
        try:
            parent_liveness.close()
        except OSError:
            pass


class BlockBinaryWriterProcess:
    """Proxy one :class:`BlockBinaryWriter` hosted in a spawned process.

    ``append`` is asynchronous and applies bounded backpressure.  ``flush`` and
    ``close`` are barriers: they wait until every submission made before the
    command has been written, and surface any child exception.  All waits have
    finite defaults so acquisition cleanup cannot hang indefinitely.
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        dtype: np.dtype[Any] | type[Any] | str | None = None,
        sample_shape: Sequence[int] | None = None,
        metadata: Mapping[str, Any] | None = None,
        meta_path: str | Path | None = None,
        index_path: str | Path | None = None,
        mode: str = "x",
        fsync_on_append: bool = False,
        queue_capacity: int = 8,
        startup_timeout_s: float = 15.0,
        enqueue_timeout_s: float = 5.0,
        operation_timeout_s: float = 30.0,
        abort_timeout_s: float = 2.0,
        context: BaseContext | None = None,
    ) -> None:
        if (
            not isinstance(queue_capacity, int)
            or isinstance(queue_capacity, bool)
            or queue_capacity <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer")
        for name, value in (
            ("startup_timeout_s", startup_timeout_s),
            ("enqueue_timeout_s", enqueue_timeout_s),
            ("operation_timeout_s", operation_timeout_s),
            ("abort_timeout_s", abort_timeout_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.data_path = Path(data_path).expanduser().resolve()
        derived_meta, derived_index = companion_paths(self.data_path)
        self.meta_path = (
            Path(meta_path).expanduser().resolve() if meta_path is not None else derived_meta
        )
        self.index_path = (
            Path(index_path).expanduser().resolve()
            if index_path is not None
            else derived_index
        )
        self._startup_timeout_s = float(startup_timeout_s)
        self._enqueue_timeout_s = float(enqueue_timeout_s)
        self._operation_timeout_s = float(operation_timeout_s)
        self._abort_timeout_s = float(abort_timeout_s)
        # Validate every pure option before allocating Windows IPC handles.
        writer_options = {
            "data_path": str(self.data_path),
            "dtype": None if dtype is None else np.dtype(dtype).str,
            "sample_shape": None if sample_shape is None else tuple(sample_shape),
            "metadata": dict(metadata or {}),
            "meta_path": str(self.meta_path),
            "index_path": str(self.index_path),
            "mode": mode,
            "fsync_on_append": bool(fsync_on_append),
        }
        self._lock = RLock()
        self._context = context or mp.get_context("spawn")
        self._data_queue: Queue[Any] = self._context.Queue(maxsize=int(queue_capacity))
        self._control_queue: Queue[Any] = self._context.Queue(maxsize=8)
        self._result_queue: Queue[Any] = self._context.Queue(maxsize=16)
        (
            self._parent_liveness_receiver,
            self._parent_liveness_sender,
        ) = self._context.Pipe(duplex=False)
        self._submitted_count = 0
        self._written_count = 0
        self._next_command_id = 1
        self._closed = False
        self._closing = False
        self._queues_cleaned = False
        self._failure: BlockBinaryWriterProcessError | None = None
        self._exitcode: int | None = None

        self._process = self._context.Process(
            target=_writer_process_entry,
            args=(
                writer_options,
                self._data_queue,
                self._control_queue,
                self._result_queue,
                self._parent_liveness_receiver,
            ),
            name=f"ultrasound-writer-{self.data_path.stem}",
            daemon=False,
        )
        self._pid: int | None = None
        try:
            self._process.start()
            self._pid = self._process.pid
            # The read end must exist only in the child; otherwise killing the
            # Collector would not produce EOF on its copy.
            self._parent_liveness_receiver.close()
            ready = self._wait_for("ready", timeout_s=self._startup_timeout_s)
        except BaseException:
            if self._process.pid is not None:
                self.abort()
            else:
                self._closed = True
                self._cleanup_queues(normal=False)
                self._cleanup_liveness()
                self._close_process_handle()
            raise
        self.dtype = np.dtype(ready["dtype"])
        self.sample_shape = tuple(int(value) for value in ready["sample_shape"])
        self._next_sequence = int(ready["next_sequence"])
        self._next_sample_index = int(ready["next_sample_index"])

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def is_alive(self) -> bool:
        return False if self._closed else self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._exitcode if self._closed else self._process.exitcode

    @property
    def submitted_count(self) -> int:
        return self._submitted_count

    @property
    def written_count(self) -> int:
        return self._written_count

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    @property
    def next_sample_index(self) -> int:
        return self._next_sample_index

    @_synchronized
    def append(
        self,
        samples: ArrayLike,
        *,
        device_timestamp: int | None = None,
        host_monotonic_ns: int | None = None,
        host_utc_ns: int | None = None,
        first_sample_index: int | None = None,
        sequence: int | None = None,
        flags: int = 0,
    ) -> None:
        """Copy and enqueue one block; errors surface here or at the next barrier."""

        self._ensure_open()
        self._drain_results()
        self._ensure_child_alive()
        array = np.asarray(samples, dtype=self.dtype, order="C")
        expected_ndim = len(self.sample_shape) + 1
        if array.ndim != expected_ndim or tuple(array.shape[1:]) != self.sample_shape:
            raise ValueError(
                f"samples must have shape (count, {', '.join(map(str, self.sample_shape))})"
            )
        if array.shape[0] <= 0:
            raise ValueError("samples must contain at least one item")
        # Own the bytes before Queue's feeder thread serializes them; callers
        # may safely reuse or mutate their acquisition buffer after return.
        copied = np.ascontiguousarray(array).copy()

        chosen_sequence = self._next_sequence if sequence is None else int(sequence)
        chosen_first_index = (
            self._next_sample_index
            if first_sample_index is None
            else int(first_sample_index)
        )
        if chosen_sequence < self._next_sequence:
            raise ValueError(
                f"sequence must be at least {self._next_sequence}; got {chosen_sequence}"
            )
        if chosen_first_index < self._next_sample_index:
            raise ValueError(
                "first_sample_index must be at least "
                f"{self._next_sample_index}; got {chosen_first_index}"
            )
        item = {
            "kind": "append",
            "samples": copied,
            "arguments": {
                "device_timestamp": device_timestamp,
                "host_monotonic_ns": host_monotonic_ns,
                "host_utc_ns": host_utc_ns,
                "first_sample_index": chosen_first_index,
                "sequence": chosen_sequence,
                "flags": flags,
            },
        }
        self._put_with_health_checks(
            self._data_queue,
            item,
            timeout_s=self._enqueue_timeout_s,
            operation="enqueue ultrasound block",
        )
        self._submitted_count += 1
        self._next_sequence = chosen_sequence + 1
        self._next_sample_index = chosen_first_index + int(copied.shape[0])
        self._drain_results()

    write = append

    @_synchronized
    def flush(self, *, fsync: bool | None = None) -> None:
        self._ensure_open()
        command_id = self._allocate_command_id()
        self._send_control(
            {
                "kind": "flush",
                "command_id": command_id,
                "target_count": self._submitted_count,
                "fsync": fsync,
            }
        )
        result = self._wait_for(
            "flushed", command_id=command_id, timeout_s=self._operation_timeout_s
        )
        self._written_count = int(result["processed_count"])

    @_synchronized
    def close(self) -> None:
        if self._closed:
            return
        self._ensure_open()
        self._closing = True
        command_id = self._allocate_command_id()
        try:
            self._send_control(
                {
                    "kind": "close",
                    "command_id": command_id,
                    "target_count": self._submitted_count,
                }
            )
            result = self._wait_for(
                "closed", command_id=command_id, timeout_s=self._operation_timeout_s
            )
            self._written_count = int(result["processed_count"])
            if self._written_count != self._submitted_count:
                raise BlockBinaryWriterProcessError(
                    "ultrasound writer close count mismatch: "
                    f"submitted={self._submitted_count}, written={self._written_count}"
                )
            self._next_sequence = int(result["next_sequence"])
            self._next_sample_index = int(result["next_sample_index"])
            exitcode = self.join(timeout=self._operation_timeout_s)
            if exitcode is None:
                raise TimeoutError("ultrasound writer process did not exit after close")
            if exitcode != 0:
                raise BlockBinaryWriterProcessError(
                    f"ultrasound writer exited with code {exitcode} after close"
                )
        except BaseException as original_error:
            try:
                self.abort()
            except BaseException as abort_error:
                original_error.add_note(
                    "additional writer abort failure: "
                    f"{type(abort_error).__name__}: {abort_error}"
                )
            raise
        self._closed = True
        self._exitcode = self._process.exitcode
        self._cleanup_queues(normal=True)
        self._cleanup_liveness()
        self._close_process_handle()

    @_synchronized
    def abort(self) -> None:
        """Bounded best-effort shutdown; queued blocks may intentionally be lost."""

        if self._closed:
            return
        self._closing = True
        if self._process.is_alive():
            try:
                self._put_with_health_checks(
                    self._control_queue,
                    {"kind": "abort"},
                    timeout_s=min(0.5, self._abort_timeout_s),
                    operation="request writer abort",
                    raise_remote_failure=False,
                )
            except BaseException:
                pass
            deadline = time.monotonic() + self._abort_timeout_s
            while self._process.is_alive() and time.monotonic() < deadline:
                try:
                    message = self._result_queue.get(timeout=0.05)
                    if message.get("kind") == "error":
                        self._remember_remote_error(message)
                except (Empty, OSError, EOFError, ValueError):
                    pass
                self._process.join(timeout=0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=self._abort_timeout_s)
        if self._process.is_alive() and hasattr(self._process, "kill"):
            self._process.kill()
            self._process.join(timeout=self._abort_timeout_s)
        if self._process.is_alive():
            # Closing the liveness sender gives a responsive child one final
            # parent-death signal even if its control Queue was damaged.
            self._cleanup_liveness()
            self._process.join(timeout=self._abort_timeout_s)
        if self._process.is_alive():
            self._cleanup_queues(normal=False)
            raise BlockBinaryWriterProcessError(
                "ultrasound writer process remained alive after terminate/kill"
            )
        self._exitcode = self._process.exitcode
        self._closed = True
        self._cleanup_queues(normal=False)
        self._cleanup_liveness()
        self._close_process_handle()

    @_synchronized
    def join(self, timeout: float | None = None) -> int | None:
        """Join for a bounded interval; ``None`` uses the finite operation timeout."""

        if self._closed:
            return self._exitcode
        wait_s = self._operation_timeout_s if timeout is None else max(0.0, timeout)
        self._process.join(timeout=wait_s)
        return self._process.exitcode

    def _allocate_command_id(self) -> int:
        command_id = self._next_command_id
        self._next_command_id += 1
        return command_id

    def _send_control(self, command: dict[str, Any]) -> None:
        self._put_with_health_checks(
            self._control_queue,
            command,
            timeout_s=self._enqueue_timeout_s,
            operation=f"send writer {command['kind']} command",
        )

    def _put_with_health_checks(
        self,
        queue: Queue[Any],
        item: dict[str, Any],
        *,
        timeout_s: float,
        operation: str,
        raise_remote_failure: bool = True,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            if raise_remote_failure:
                self._drain_results()
            if not self._process.is_alive():
                if raise_remote_failure:
                    self._drain_results()
                    if self._failure is not None:
                        raise self._failure
                raise BlockBinaryWriterProcessError(
                    f"cannot {operation}: writer process is not alive"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out while attempting to {operation}")
            try:
                queue.put(item, timeout=min(0.1, remaining))
                return
            except Full:
                continue

    def _wait_for(
        self,
        kind: str,
        *,
        command_id: int | None = None,
        timeout_s: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            if self._failure is not None:
                raise self._failure
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for writer result {kind!r}")
            try:
                message = self._result_queue.get(timeout=min(0.1, remaining))
            except Empty:
                if not self._process.is_alive():
                    # Allow one short turn for the result Queue feeder to make
                    # the terminal message visible after process exit.
                    try:
                        message = self._result_queue.get(timeout=min(0.05, remaining))
                    except Empty:
                        exitcode = self._process.exitcode
                        raise BlockBinaryWriterProcessError(
                            f"writer process exited with code {exitcode} before {kind!r}"
                        )
                    else:
                        pass
                else:
                    continue
            message_kind = message.get("kind")
            if message_kind == "error":
                self._remember_remote_error(message)
                assert self._failure is not None
                raise self._failure
            if message_kind == kind and (
                command_id is None or message.get("command_id") == command_id
            ):
                return message

    def _drain_results(self) -> None:
        while True:
            try:
                message = self._result_queue.get_nowait()
            except Empty:
                break
            if message.get("kind") == "error":
                self._remember_remote_error(message)
        if self._failure is not None:
            raise self._failure

    def _remember_remote_error(self, message: Mapping[str, Any]) -> None:
        self._written_count = int(message.get("processed_count", self._written_count))
        if self._failure is None:
            exception_type = str(message.get("exception_type") or "Exception")
            detail = str(message.get("message") or "unknown writer error")
            self._failure = BlockBinaryWriterProcessError(
                f"ultrasound writer failed with {exception_type}: {detail}",
                remote_exception_type=exception_type,
                remote_traceback=str(message.get("traceback") or ""),
            )

    def _ensure_child_alive(self) -> None:
        if not self._process.is_alive():
            self._drain_results()
            raise BlockBinaryWriterProcessError(
                f"ultrasound writer process exited with code {self._process.exitcode}"
            )

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise ValueError("I/O operation on closed BlockBinaryWriterProcess")

    def _cleanup_queues(self, *, normal: bool) -> None:
        if self._queues_cleaned:
            return
        self._queues_cleaned = True
        for queue in (self._data_queue, self._control_queue, self._result_queue):
            try:
                # CLOSED/FLUSHED barriers prove all normal data was consumed;
                # no lifecycle path needs an unbounded feeder-thread join.
                queue.cancel_join_thread()
                queue.close()
            except (OSError, ValueError, AssertionError):
                pass

    def _cleanup_liveness(self) -> None:
        for connection in (
            self._parent_liveness_sender,
            self._parent_liveness_receiver,
        ):
            try:
                connection.close()
            except OSError:
                pass

    def _close_process_handle(self) -> None:
        if self._process.is_alive():
            return
        try:
            self._process.close()
        except ValueError:
            pass

    def __enter__(self) -> BlockBinaryWriterProcess:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


__all__ = ["BlockBinaryWriterProcess", "BlockBinaryWriterProcessError"]
