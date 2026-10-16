"""Append-only local audit records; no hardware or network side effects."""
from __future__ import annotations

import dataclasses
import json
import time
import uuid
from pathlib import Path


def _json_default(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class SessionLog:
    def __init__(self, directory: Path, metadata: dict):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
        self._file = self.path.open("x", encoding="utf-8")
        self._last_flush = time.monotonic()
        self.write("session", metadata)

    def write(self, kind: str, payload):
        self._file.write(json.dumps(
            {"kind": kind, "host_monotonic_ns": time.perf_counter_ns(), "payload": payload},
            ensure_ascii=False, default=_json_default, allow_nan=False,
        ) + "\n")
        if time.monotonic() - self._last_flush >= 1:
            self._file.flush()
            self._last_flush = time.monotonic()

    def close(self):
        if not self._file.closed:
            self.write("closed", {})
            self._file.close()
