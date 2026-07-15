from __future__ import annotations

import ctypes
import os
import multiprocessing as mp
from pathlib import Path
import signal
from threading import Event, Thread
import time

import numpy as np
import pytest

from exo_collection.readers.binary_block import BlockBinaryReader
from exo_collection.writers.block_binary_process import (
    BlockBinaryWriterProcess,
    BlockBinaryWriterProcessError,
)


def _abrupt_writer_parent(path: str, ready_connection: object) -> None:
    """Create a writer grandchild, report its PID, then die without cleanup."""

    writer = BlockBinaryWriterProcess(
        path,
        dtype="uint16",
        sample_shape=(2, 3),
        metadata={"clock_domain": "parent_death_test"},
        operation_timeout_s=5,
    )
    ready_connection.send(writer.pid)  # type: ignore[attr-defined]
    ready_connection.close()  # type: ignore[attr-defined]
    os._exit(0)


def _pid_is_alive(pid: int) -> bool:
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                handle, ctypes.byref(exit_code)
            ):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _process_writer(path: Path, **kwargs: object) -> BlockBinaryWriterProcess:
    return BlockBinaryWriterProcess(
        path,
        dtype="uint16",
        sample_shape=(2, 3),
        metadata={"clock_domain": "ultrasound_test_clock"},
        startup_timeout_s=10,
        operation_timeout_s=10,
        abort_timeout_s=1,
        **kwargs,
    )


def test_process_writer_uses_spawned_process_and_drains_before_close(tmp_path: Path) -> None:
    path = tmp_path / "ultrasound.bin.partial"
    writer = _process_writer(path, queue_capacity=2)
    assert writer.pid is not None
    assert writer.pid != os.getpid()
    assert writer.is_alive

    first = np.ones((2, 2, 3), dtype=np.uint16)
    second = np.full((1, 2, 3), 7, dtype=np.uint16)
    writer.append(
        first,
        device_timestamp=10,
        host_monotonic_ns=100,
        host_utc_ns=200,
        first_sample_index=0,
        sequence=0,
    )
    # append owns a copy before the Queue feeder serializes the array.
    first.fill(99)
    writer.append(
        second,
        device_timestamp=20,
        host_monotonic_ns=300,
        host_utc_ns=400,
        first_sample_index=2,
        sequence=1,
    )
    writer.flush()
    assert writer.submitted_count == 2
    assert writer.written_count == 2
    writer.close()

    assert writer.closed
    assert not writer.is_alive
    assert writer.exitcode == 0
    assert writer.join(timeout=0) == 0
    with BlockBinaryReader(path) as reader:
        assert reader.block_count == 2
        np.testing.assert_array_equal(
            reader.read_block_by_ordinal(0).data,
            np.ones((2, 2, 3), dtype=np.uint16),
        )
        np.testing.assert_array_equal(
            reader.read_block_by_ordinal(1).data,
            np.full((1, 2, 3), 7, dtype=np.uint16),
        )


def test_process_writer_returns_remote_write_error_and_abort_is_bounded(
    tmp_path: Path,
) -> None:
    writer = _process_writer(tmp_path / "invalid.bin.partial", queue_capacity=1)
    started = time.monotonic()
    try:
        # Shape is valid in the parent, while the negative flags value is
        # validated by the real BlockBinaryWriter in the child process.
        try:
            writer.append(np.ones((1, 2, 3), dtype=np.uint16), flags=-1)
        except BlockBinaryWriterProcessError as error:
            captured_error = error
        else:
            with pytest.raises(BlockBinaryWriterProcessError, match="flags") as captured:
                writer.close()
            captured_error = captured.value
        assert "flags" in str(captured_error)
        assert captured_error.remote_exception_type == "ValueError"
        assert captured_error.remote_traceback
    finally:
        writer.abort()
    assert writer.closed
    assert not writer.is_alive
    assert time.monotonic() - started < 6


def test_process_writer_reports_initialization_failure_without_leaking_child(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing.bin.partial"
    path.write_bytes(b"conflict")
    started = time.monotonic()
    with pytest.raises(BlockBinaryWriterProcessError, match="FileExistsError"):
        _process_writer(path)
    assert time.monotonic() - started < 6


def test_process_writer_serializes_concurrent_append_and_close(tmp_path: Path) -> None:
    writer = _process_writer(tmp_path / "concurrent.bin.partial", queue_capacity=1)
    append_entered = Event()
    allow_append = Event()
    failures: list[BaseException] = []
    original_put = writer._put_with_health_checks

    def delayed_put(queue, item, **kwargs):
        if item.get("kind") == "append":
            append_entered.set()
            assert allow_append.wait(timeout=3)
        return original_put(queue, item, **kwargs)

    writer._put_with_health_checks = delayed_put  # type: ignore[method-assign]

    def append() -> None:
        try:
            writer.append(np.ones((1, 2, 3), dtype=np.uint16))
        except BaseException as exc:
            failures.append(exc)

    def close() -> None:
        try:
            writer.close()
        except BaseException as exc:
            failures.append(exc)

    append_thread = Thread(target=append)
    close_thread = Thread(target=close)
    append_thread.start()
    assert append_entered.wait(timeout=3)
    close_thread.start()
    time.sleep(0.05)
    assert close_thread.is_alive()  # waiting on the proxy RLock, not Queue.join_thread
    allow_append.set()
    append_thread.join(timeout=5)
    close_thread.join(timeout=5)

    try:
        assert not append_thread.is_alive()
        assert not close_thread.is_alive()
        assert not failures
        assert writer.closed
        assert writer.submitted_count == writer.written_count == 1
    finally:
        writer.abort()


def test_writer_exits_when_its_collector_parent_dies(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    path = tmp_path / "parent-death.bin.partial"
    parent = context.Process(
        target=_abrupt_writer_parent,
        args=(str(path), send),
        daemon=False,
    )
    parent.start()
    send.close()
    writer_pid = int(receive.recv())
    receive.close()
    parent.join(timeout=10)
    assert parent.exitcode == 0

    deadline = time.monotonic() + 5
    while _pid_is_alive(writer_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert not _pid_is_alive(writer_pid)
        # Windows refuses this rename while the writer still owns the handle.
        renamed = path.with_name("parent-death-closed.bin.partial")
        path.replace(renamed)
        assert renamed.is_file()
    finally:
        if _pid_is_alive(writer_pid):
            os.kill(writer_pid, signal.SIGTERM)
