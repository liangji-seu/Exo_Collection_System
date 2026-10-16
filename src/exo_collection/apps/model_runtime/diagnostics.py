"""Bounded, best-effort application diagnostics, independent of run auditing."""
from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import sys
import time
import traceback
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

from exo_collection.logging_setup import current_collector_log_path


class _QuietRotatingHandler(RotatingFileHandler):
    def handleError(self, record):  # noqa: N802
        # Disk failures must never re-enter safety actions or stderr logging.
        self.last_error = traceback.format_exc()
        if not getattr(self, "_reported_failure", False):
            self._reported_failure = True
            _stderr(f"Online diagnostics write failed: {self.last_error}")


def _stderr(message):
    try:
        if sys.__stderr__ is not None:
            sys.__stderr__.write(message + "\n")
    except Exception:
        pass


def _closed_log(path):
    # Only reap files that explicitly completed. A different process/window
    # may still own any other file, even if its mtime appears old.
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 2048))
            tail = stream.read().splitlines()
        return bool(tail) and json.loads(tail[-1]).get("kind") == "diagnostics_closed"
    except (OSError, ValueError):
        return False


class Diagnostics:
    def __init__(self, directory: Path, *, max_bytes=2 * 1024 * 1024, backups=3):
        self.path = directory / f"online_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:6]}.jsonl"
        self._logger = logging.Logger(f"online_diagnostics.{uuid.uuid4().hex}")
        self._logger.propagate = False
        self._handler = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            # Bound historical launches as well as each launch's rotated files.
            old = sorted((p for p in directory.glob("online_*.jsonl") if _closed_log(p)),
                         key=lambda p: p.stat().st_mtime, reverse=True)
            for path in old[7:]:
                for candidate in (path, *(Path(f"{path}.{i}") for i in range(1, backups + 1))):
                    try:
                        candidate.unlink(missing_ok=True)
                    except OSError:
                        pass
            self._handler = _QuietRotatingHandler(self.path, encoding="utf-8", maxBytes=max_bytes, backupCount=backups)
            self._handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(self._handler)
        except OSError as exc:
            _stderr(f"Online diagnostics unavailable at {self.path}: {exc}")
        versions = {}
        for package in ("PySide6", "numpy", "torch", "pydantic", "pyserial"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "not installed"
        self.record("startup", pid=os.getpid(), executable=sys.executable,
                    python=sys.version, platform=platform.platform(), versions=versions,
                    application_log=str(current_collector_log_path() or "not configured"))

    def record(self, kind, *, exception=None, **fields):
        try:
            if exception is not None:
                fields["traceback"] = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            line = json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "host_monotonic_ns": time.perf_counter_ns(), "kind": kind, **fields},
                ensure_ascii=False, default=str, allow_nan=False)
        except Exception as exc:
            # Retain the event identity even if one diagnostic field is bad.
            line = json.dumps({"kind": str(kind), "serialization_error": str(exc),
                               "host_monotonic_ns": time.perf_counter_ns()})
            if not getattr(self, "_serialization_error_reported", False):
                self._serialization_error_reported = True
                _stderr(f"Online diagnostics serialization failed for {kind}: {exc}")
        try:
            self._logger.info(line)
        except Exception as exc:
            if self._handler is not None and not getattr(self._handler, "_reported_failure", False):
                self._handler._reported_failure = True
                _stderr(f"Online diagnostics write failed: {exc}")

    def close(self):
        if self._handler is not None:
            self.record("diagnostics_closed")
            self._logger.removeHandler(self._handler)
            try:
                self._handler.close()
            except OSError as exc:
                _stderr(f"Online diagnostics close failed: {exc}")
            self._handler = None

    @property
    def available(self):
        return self._handler is not None and not getattr(self._handler, "_reported_failure", False)
