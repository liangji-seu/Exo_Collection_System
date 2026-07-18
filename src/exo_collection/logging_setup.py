"""Centralised logging configuration for Collector and Data Studio applications.

Provides one UTF-8 rotating application log per desktop process launch in the
visible ``Exo_Collection_System/log/`` directory. A launch is identified by
application name, local wall-clock time and PID, so historical sessions are
never appended into one ambiguous shared file.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
DEFAULT_BACKUP_COUNT = 10
_HANDLER_MARKER = "_exo_application_log_handler"
_PROCESS_LAUNCH_TOKEN = datetime.now().astimezone().strftime(
    "%Y%m%d_%H%M%S_%f"
)


def _default_log_dir() -> Path:
    """Return the visible project/install-local ``log`` directory."""

    if getattr(sys, "frozen", False):
        application_root = Path(sys.executable).resolve().parent
    else:
        application_root = Path(__file__).resolve().parents[2]
    return application_root / "log"


def collector_log_path() -> Path:
    """Return this Collector process launch's absolute log path."""
    return _default_log_dir() / (
        f"ExoCollector_{_PROCESS_LAUNCH_TOKEN}_pid{os.getpid()}.log"
    )


def data_studio_log_path() -> Path:
    """Return this Data Studio process launch's absolute log path."""
    return _default_log_dir() / (
        f"ExoDataStudio_{_PROCESS_LAUNCH_TOKEN}_pid{os.getpid()}.log"
    )


def setup_collector_logging(
    *,
    level: int = logging.INFO,
    console: bool = False,
    log_path: Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """Configure the root logger with a rotating file handler.

    When *console* is True (development), a StreamHandler is added so log
    entries also appear on stderr.

    This function is idempotent for the same target. An existing Collector
    handler for another target is replaced, while unrelated handlers remain.
    """
    root = logging.getLogger()
    resolved_path = log_path or collector_log_path()
    resolved_path = resolved_path.expanduser().resolve()
    for handler in list(root.handlers):
        if not bool(getattr(handler, _HANDLER_MARKER, False)):
            continue
        if Path(handler.baseFilename).resolve() == resolved_path:
            return
        root.removeHandler(handler)
        handler.close()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    root.setLevel(level)

    file_handler = RotatingFileHandler(
        filename=str(resolved_path),
        mode="a",
        encoding="utf-8",
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(level)
    setattr(file_handler, _HANDLER_MARKER, True)
    file_handler.addFilter(SensitiveDataFilter())
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.addFilter(SensitiveDataFilter())
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(console_handler)


def setup_data_studio_logging(
    *,
    level: int = logging.INFO,
    console: bool = False,
    log_path: Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """Configure Data Studio logging using the shared protected handler."""

    setup_collector_logging(
        level=level,
        console=console,
        log_path=log_path or data_studio_log_path(),
        max_bytes=max_bytes,
        backup_count=backup_count,
    )


# -- Sensitive key guard ----------------------------------------------------

_SECRET_PATTERNS = frozenset(
    {
        "password",
        "passwd",
        "token",
        "secret",
        "api_key",
        "apikey",
        "access_key",
        "private_key",
        "credential",
    }
)

_SECRET_TEXT_RE = re.compile(
    r"(?i)\b(password|passwd|token|secret|api[-_]?key|access[-_]?key|"
    r"private[-_]?key|credential)\b(\s*[:=]\s*)([^\s,;]+)"
)


def _redact_text(value: object) -> str:
    return _SECRET_TEXT_RE.sub(r"\1\2<REDACTED>", str(value))


class SensitiveDataFilter(logging.Filter):
    """Redact common secret assignments after %-formatting, before disk I/O."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _redact_text(record.getMessage())
            record.args = ()
            if record.exc_info:
                rendered = "".join(traceback.format_exception(*record.exc_info))
                record.exc_text = _redact_text(rendered)
                record.exc_info = None
        except Exception:
            record.msg = "<log message redaction failed>"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
        return True


class SafeLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that redacts values whose keys match common secret patterns.

    Usage::

        logger = SafeLoggerAdapter(logging.getLogger(__name__))

    When the extra dict passed via ``logger.info("msg", extra={"payload": d})``
    contains a key that looks like a secret, the value is replaced with
    ``"<REDACTED>"`` before formatting.
    """

    def process(self, msg: object, kwargs: dict) -> tuple[object, dict]:
        extra = kwargs.get("extra")
        if isinstance(extra, dict):
            sanitized = dict(extra)
            for key in list(sanitized):
                norm = str(key).lower().replace("-", "_")
                if norm in _SECRET_PATTERNS:
                    sanitized[key] = "<REDACTED>"
            kwargs["extra"] = sanitized
        return _redact_text(msg), kwargs


# Module-level convenience: create a safe logger for each module that
# imports ``get_logger``.  The returned logger honours the global
# configuration set by ``setup_collector_logging``.
def get_logger(name: str | None = None) -> SafeLoggerAdapter:
    return SafeLoggerAdapter(logging.getLogger(name))


# Top-level application logger — use ``import logger_module`` then
# ``logger = logger_module.logger`` in simple cases.
logger = get_logger("exo_collection")
