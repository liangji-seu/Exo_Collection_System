"""Dataset-root acquisition activity lock used by both desktop applications."""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


LOCK_NAME = ".collector-active.json"


@dataclass(frozen=True, slots=True)
class AcquisitionActivity:
    pid: int
    hostname: str
    trial_uuid: str | None
    heartbeat_monotonic_ns: int
    heartbeat_utc_ns: int


class AcquisitionLock:
    def __init__(self, dataset_root: str | Path, trial_uuid: UUID | None = None) -> None:
        self.root = Path(dataset_root).expanduser().resolve()
        self.path = self.root / LOCK_NAME
        self.trial_uuid = str(trial_uuid) if trial_uuid else None
        self._owned = False

    def _payload(self) -> dict[str, int | str | None]:
        return {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "trial_uuid": self.trial_uuid,
            "heartbeat_monotonic_ns": time.perf_counter_ns(),
            "heartbeat_utc_ns": time.time_ns(),
        }

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self._payload(), stream, separators=(",", ":"))
                stream.flush()
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise
        self._owned = True

    def heartbeat(self) -> None:
        if not self._owned:
            raise RuntimeError("Acquisition lock is not owned")
        partial = self.path.with_suffix(".json.partial")
        partial.write_text(json.dumps(self._payload(), separators=(",", ":")), encoding="utf-8")
        os.replace(partial, self.path)

    def release(self) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False

    def __enter__(self) -> AcquisitionLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def read_activity(dataset_root: str | Path, stale_after_s: float = 5.0) -> AcquisitionActivity | None:
    path = Path(dataset_root).expanduser().resolve() / LOCK_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        activity = AcquisitionActivity(**data)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, OSError):
        return None
    age_ns = time.time_ns() - activity.heartbeat_utc_ns
    return activity if 0 <= age_ns <= int(stale_after_s * 1_000_000_000) else None

