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

# When a subprocess (preview / trial worker) needs the parent process's log
# path, the parent writes it here.  The subprocess reads it so all processes
# append to the same file.
_SUBPROCESS_LOG_PATH: Path | None = None


def current_collector_log_path() -> Path | None:
    """Return the file-handler path most recently registered by the parent.

    Preview / trial subprocesses call this to share the parent's log file.
    Returns ``None`` when no file handler has been registered.
    """
    resolved = _SUBPROCESS_LOG_PATH
    if resolved is not None:
        return resolved.expanduser().resolve()
    return None


def configure_subprocess_logging(
    *,
    log_path: str | Path | None = None,
    level: int = logging.DEBUG,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """Attach a file handler *and* a stderr stream to the root logger inside a
    spawned process.  stderr is also redirected through the logging system so
    that every ``print``, ``traceback.print_exc``, or C-extension warning that
    writes directly to fd 2 still ends up in the shared log file.

    Must be called after the parent has called ``setup_collector_logging``
    (which stores the shared path).
    """
    resolved_path = (
        Path(log_path).expanduser().resolve()
        if log_path is not None
        else current_collector_log_path()
    )
    if resolved_path is None:
        return
    root = logging.getLogger()
    # Remove the default last-resort handler so only our handlers fire.
    if root.hasHandlers() and len(root.handlers) == 1 and isinstance(
        root.handlers[0], logging.StreamHandler
    ):
        root.removeHandler(root.handlers[0])
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            return
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    root.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

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
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Also stream every log entry to stderr so terminal users see the same
    # messages.  The parent process' own StreamHandler already covers the main
    # thread; this one is for the spawned child.
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.addFilter(SensitiveDataFilter())
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # Redirect bare stderr writes (SDK warnings, traceback.print_exc, etc.)
    # through a logging handler so they land in *both* the file and the
    # terminal.
    _stderr_capture = _StderrToLogHandler()
    sys.stderr = _stderr_capture  # type: ignore[assignment]


class _StderrToLogHandler:
    """Wraps the real stderr and forwards every write to a logger."""

    def __init__(self) -> None:
        self._real_stderr = sys.stderr
        self._logger = logging.getLogger("exo_collection.stderr")

    def write(self, message: str) -> None:
        if message and not message.isspace():
            self._logger.debug(message.rstrip("\n"))
        self._real_stderr.write(message)

    def flush(self) -> None:
        self._real_stderr.flush()

    def fileno(self) -> int:
        return self._real_stderr.fileno()

    def isatty(self) -> bool:
        return self._real_stderr.isatty()


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
    global _SUBPROCESS_LOG_PATH
    _SUBPROCESS_LOG_PATH = resolved_path
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
