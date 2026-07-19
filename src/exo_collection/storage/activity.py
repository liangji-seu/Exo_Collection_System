"""Dataset-root acquisition activity lock used by both desktop applications."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4


LOCK_NAME = ".exo/.collector-active.json"
_GUARD_NAME = ".exo/.collector-active.guard"


@dataclass(frozen=True, slots=True)
class AcquisitionActivity:
    pid: int
    hostname: str
    trial_uuid: str | None
    heartbeat_monotonic_ns: int
    heartbeat_utc_ns: int
    owner_token: str = ""


def _decode_activity(path: Path) -> AcquisitionActivity:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("activity document must be an object")
    return AcquisitionActivity(
        pid=int(data["pid"]),
        hostname=str(data["hostname"]),
        trial_uuid=(str(data["trial_uuid"]) if data.get("trial_uuid") is not None else None),
        heartbeat_monotonic_ns=int(data["heartbeat_monotonic_ns"]),
        heartbeat_utc_ns=int(data["heartbeat_utc_ns"]),
        owner_token=str(data.get("owner_token", "")),
    )


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        # ``os.kill(pid, 0)`` is not a harmless existence probe on Windows.
        # CPython routes non-console signals (including 0) through
        # TerminateProcess, which can kill the active Collector we are trying
        # to protect. Query a process handle without termination rights.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL

        handle = open_process(process_query_limited_information, False, pid)
        if not handle:
            # ERROR_INVALID_PARAMETER means no such PID. Access denied and
            # unknown failures are treated conservatively as alive.
            return ctypes.get_last_error() != 87
        try:
            exit_code = wintypes.DWORD()
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            close_handle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # Windows reports ERROR_INVALID_PARAMETER for a nonexistent PID.
        return getattr(exc, "winerror", None) not in {87}
    return True


def _activity_age_ns(activity: AcquisitionActivity) -> int:
    if activity.hostname == socket.gethostname():
        monotonic_age = time.perf_counter_ns() - activity.heartbeat_monotonic_ns
        if monotonic_age >= 0:
            return monotonic_age
    return max(0, time.time_ns() - activity.heartbeat_utc_ns)


@contextmanager
def _interprocess_guard(root: Path) -> Iterator[None]:
    """Serialize lock-file replacement without relying on conditional rename."""

    guard_path = root / _GUARD_NAME
    with guard_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - exercised by CI on non-Windows hosts
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class AcquisitionLock:
    """Exclusive collector lease with automatic heartbeat and stale takeover."""

    def __init__(
        self,
        dataset_root: str | Path,
        trial_uuid: UUID | None = None,
        *,
        stale_after_s: float = 5.0,
        heartbeat_interval_s: float = 1.0,
        release_on_exception: bool = True,
    ) -> None:
        if stale_after_s <= 0 or heartbeat_interval_s <= 0:
            raise ValueError("lock timing values must be positive")
        if heartbeat_interval_s >= stale_after_s:
            raise ValueError("heartbeat_interval_s must be shorter than stale_after_s")
        self.root = Path(dataset_root).expanduser().resolve()
        self.path = self.root / LOCK_NAME
        self.trial_uuid = str(trial_uuid) if trial_uuid else None
        self.stale_after_s = float(stale_after_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.release_on_exception = bool(release_on_exception)
        self.owner_token = uuid4().hex
        self._owned = False
        self._mutex = threading.RLock()
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_error: BaseException | None = None

    def _payload(self) -> dict[str, int | str | None]:
        return {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "trial_uuid": self.trial_uuid,
            "heartbeat_monotonic_ns": time.perf_counter_ns(),
            "heartbeat_utc_ns": time.time_ns(),
            "owner_token": self.owner_token,
        }

    def _is_reclaimable(self) -> bool:
        try:
            activity = _decode_activity(self.path)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
            try:
                age_s = time.time() - self.path.stat().st_mtime
            except OSError:
                return False
            return age_s > self.stale_after_s
        stale = _activity_age_ns(activity) > int(self.stale_after_s * 1_000_000_000)
        if not stale:
            return False
        if activity.hostname == socket.gethostname():
            return not _pid_is_alive(activity.pid)
        # A shared dataset may have been used by another host. Its PID cannot be
        # inspected locally, so a stale UTC heartbeat is the available lease rule.
        return True

    def _write_new(self) -> None:
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self._payload(), stream, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._mutex, _interprocess_guard(self.root):
            if self._owned:
                raise RuntimeError("Acquisition lock is already owned")
            if self.path.exists():
                if not self._is_reclaimable():
                    raise FileExistsError(f"an active or non-stale collector lock exists: {self.path}")
                stale_path = self.path.with_name(
                    f"{self.path.name}.stale.{self.owner_token}"
                )
                os.replace(self.path, stale_path)
                stale_path.unlink(missing_ok=True)
            self._write_new()
            self._owned = True
            self._stop_heartbeat.clear()
            self._heartbeat_error = None
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="collector-activity-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self.heartbeat_interval_s):
            try:
                self.heartbeat()
            except BaseException as exc:  # surfaced on the next explicit heartbeat/release
                self._heartbeat_error = exc
                return

    def heartbeat(self) -> None:
        with self._mutex:
            if not self._owned:
                raise RuntimeError("Acquisition lock is not owned")
            if self._heartbeat_error is not None:
                raise RuntimeError("acquisition heartbeat previously failed") from self._heartbeat_error
            with _interprocess_guard(self.root):
                try:
                    current = _decode_activity(self.path)
                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
                    raise RuntimeError("acquisition lock disappeared or became invalid") from exc
                if current.owner_token != self.owner_token:
                    raise RuntimeError("acquisition lock ownership was lost")
                partial = self.path.with_name(
                    f"{self.path.name}.{self.owner_token}.partial"
                )
                try:
                    with partial.open("w", encoding="utf-8", newline="\n") as stream:
                        json.dump(self._payload(), stream, separators=(",", ":"))
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(partial, self.path)
                finally:
                    partial.unlink(missing_ok=True)

    def release(self) -> None:
        self._stop_heartbeat.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.heartbeat_interval_s * 2))
        with self._mutex:
            if self._owned:
                with _interprocess_guard(self.root):
                    try:
                        current = _decode_activity(self.path)
                    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
                        current = None
                    if current is not None and current.owner_token == self.owner_token:
                        self.path.unlink(missing_ok=True)
                self._owned = False
        self._heartbeat_thread = None

    def __enter__(self) -> "AcquisitionLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, *_exc: object) -> None:
        if exc_type is None or self.release_on_exception:
            self.release()


def read_activity(dataset_root: str | Path, stale_after_s: float = 5.0) -> AcquisitionActivity | None:
    if stale_after_s <= 0:
        raise ValueError("stale_after_s must be positive")
    path = Path(dataset_root).expanduser().resolve() / LOCK_NAME
    try:
        activity = _decode_activity(path)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        # Fail closed while a recently-written lock document is temporarily
        # unreadable (for example during an interrupted heartbeat replace).
        # Treating that situation as idle could let Data Studio start a large
        # checksum, recovery mutation or upload while Collector still owns the
        # dataset. An old malformed file is ignored after the normal lease
        # timeout, so a damaged lock cannot block the system forever.
        try:
            modified_utc_ns = path.stat().st_mtime_ns
        except OSError:
            return None
        age_ns = max(0, time.time_ns() - modified_utc_ns)
        if age_ns > int(stale_after_s * 1_000_000_000):
            return None
        return AcquisitionActivity(
            pid=0,
            hostname="unreadable-lock",
            trial_uuid=None,
            heartbeat_monotonic_ns=time.perf_counter_ns(),
            heartbeat_utc_ns=modified_utc_ns,
            owner_token="unreadable-lock",
        )
    age_ns = _activity_age_ns(activity)
    if age_ns <= int(stale_after_s * 1_000_000_000):
        return activity
    # A heartbeat can be delayed by a temporarily saturated disk or scheduler
    # while the Collector process is still actively writing raw data.  On the
    # local host, process liveness is stronger evidence than lease age; keep
    # Data Studio in lightweight mode until that owner actually exits.
    if activity.hostname == socket.gethostname() and _pid_is_alive(activity.pid):
        return activity
    return None


__all__ = ["AcquisitionActivity", "AcquisitionLock", "LOCK_NAME", "read_activity"]
