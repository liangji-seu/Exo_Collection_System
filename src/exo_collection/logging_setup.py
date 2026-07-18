"""Centralised logging configuration for Collector and Data Studio applications.

Provides a UTF-8 rotating application log in the main desktop process
into ``%LOCALAPPDATA%/ExoCollectionSystem/logs/collector/collector.log``
on Windows, falling back to ``~/.local/share/ExoCollectionSystem/logs/`` on
other platforms.  The handler is created once at application start and reused
by all modules.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_ORGANIZATION = "ExoCollectionSystem"
LOG_APP_NAME = "collector"
LOG_FILENAME = "collector.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
DEFAULT_BACKUP_COUNT = 10
_HANDLER_MARKER = "_exo_collector_log_handler"


def _default_log_dir() -> Path:
    """Platform-appropriate application log directory."""
    if sys.platform == "win32":
        localappdata = os.environ.get("LOCALAPPDATA")
        if localappdata:
            base = Path(localappdata)
        else:
            # Rare fallback: %USERPROFILE% or CWD
            base = Path.home() / "AppData" / "Local"
        return base / LOG_ORGANIZATION / "logs" / LOG_APP_NAME
    # Non-Windows: XDG_DATA_HOME or ~/.local/share
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        base = Path(xdg)
    else:
        base = Path.home() / ".local" / "share"
    return base / LOG_ORGANIZATION / "logs" / LOG_APP_NAME


def collector_log_path() -> Path:
    """Return the absolute path to the collector log file."""
    return _default_log_dir() / LOG_FILENAME


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
