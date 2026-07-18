"""Tests for centralized logging configuration and SafeLoggerAdapter."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

from exo_collection.logging_setup import (
    SafeLoggerAdapter,
    collector_log_path,
    data_studio_log_path,
    get_logger,
    setup_collector_logging,
    setup_data_studio_logging,
)


# ── Path tests ──


def test_collector_log_path_returns_absolute(tmp_path: Path) -> None:
    path = collector_log_path()
    repository_root = Path(__file__).resolve().parents[2]
    assert path.parent == repository_root / "log"
    assert path.name.startswith("ExoCollector_")
    assert f"_pid{os.getpid()}.log" in path.name
    assert path.is_absolute()


def test_data_studio_uses_its_own_launch_log_name() -> None:
    path = data_studio_log_path()
    assert path.parent == collector_log_path().parent
    assert path.name.startswith("ExoDataStudio_")
    assert f"_pid{os.getpid()}.log" in path.name


def test_separate_process_launches_use_distinct_log_file_names() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from exo_collection.logging_setup import collector_log_path; "
            "print(collector_log_path())"
        ),
    ]
    first = subprocess.check_output(command, text=True).strip()
    second = subprocess.check_output(command, text=True).strip()
    assert first != second
    assert Path(first).parent.name == "log"
    assert Path(second).parent.name == "log"


def test_setup_creates_log_directory(tmp_path: Path) -> None:
    # Clear any handlers from previous tests so we get a fresh setup.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    log_file = tmp_path / "test_logs" / "collector.log"
    setup_collector_logging(log_path=log_file, max_bytes=1024, backup_count=2)
    assert log_file.parent.exists()
    assert log_file.exists()


def test_data_studio_setup_creates_its_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "studio_logs" / "data-studio.log"
    setup_data_studio_logging(log_path=log_file)

    assert log_file.is_file()


def test_setup_idempotent(tmp_path: Path) -> None:
    """Calling setup_collector_logging twice should not add duplicate handlers."""
    log_file = tmp_path / "dedup" / "collector.log"
    setup_collector_logging(log_path=log_file, max_bytes=10000, backup_count=1)
    root = logging.getLogger()
    handler_count_before = len(root.handlers)
    setup_collector_logging(log_path=log_file, max_bytes=10000, backup_count=1)
    handler_count_after = len(root.handlers)
    assert handler_count_after == handler_count_before


def test_unrelated_rotating_handler_does_not_block_collector_log(
    tmp_path: Path,
) -> None:
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    unrelated = RotatingFileHandler(tmp_path / "third-party.log")
    root.addHandler(unrelated)
    collector_file = tmp_path / "collector" / "collector.log"
    try:
        setup_collector_logging(log_path=collector_file)
        assert collector_file.is_file()
        assert any(
            isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename).resolve() == collector_file.resolve()
            for handler in root.handlers
        )
    finally:
        root.removeHandler(unrelated)
        unrelated.close()


def test_rotation_config_applied(tmp_path: Path) -> None:
    """Verify max_bytes and backup_count are stored on the handler."""
    from logging.handlers import RotatingFileHandler

    log_file = tmp_path / "rotation" / "collector.log"
    # Clear any existing handlers so we get a fresh setup
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    setup_collector_logging(log_path=log_file, max_bytes=5000, backup_count=3)
    handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    assert len(handlers) == 1
    handler = handlers[0]
    assert handler.maxBytes == 5000
    assert handler.backupCount == 3

    # Tear down
    root.removeHandler(handler)
    handler.close()


def test_key_events_written(tmp_path: Path) -> None:
    """Write several log messages and verify they appear in the file."""
    log_file = tmp_path / "events" / "collector.log"
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    setup_collector_logging(log_path=log_file)
    logger = logging.getLogger("exo_collection.test")
    logger.info("应用启动")
    logger.warning("配置文件缺失")
    logger.error("连接失败", exc_info=False)

    for h in root.handlers:
        h.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "应用启动" in content
    assert "配置文件缺失" in content
    assert "连接失败" in content

    # Cleanup
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()


def test_formatted_secrets_are_redacted_before_disk_write(tmp_path: Path) -> None:
    log_file = tmp_path / "redacted" / "collector.log"
    setup_collector_logging(log_path=log_file)
    logger = logging.getLogger("exo_collection.secret_test")
    logger.error(
        "password=%s token=%s device=%s", "secret-pass", "abc123", "imu"
    )
    for handler in logging.getLogger().handlers:
        handler.flush()
    content = log_file.read_text(encoding="utf-8")
    assert "secret-pass" not in content
    assert "abc123" not in content
    assert "password=<REDACTED>" in content
    assert "token=<REDACTED>" in content


# ── SafeLoggerAdapter tests ──


def test_safe_logger_redacts_secrets() -> None:
    """SafeLoggerAdapter should redact values for keys matching secret patterns."""
    logger = logging.getLogger("test_redact")
    safe = SafeLoggerAdapter(logger)

    # Log a message with an extra dict containing a 'password' key
    # The process method should replace the value with '<REDACTED>'
    kwargs = {"extra": {"user": "john", "password": "s3cret!", "token": "abc123"}}
    msg, processed_kwargs = safe.process("auth attempt", kwargs)
    extra = processed_kwargs.get("extra", {})
    assert extra["user"] == "john"
    assert extra["password"] == "<REDACTED>"
    assert extra["token"] == "<REDACTED>"


def test_safe_logger_preserves_non_secret_keys() -> None:
    """Non-secret keys should pass through unmodified."""
    logger = logging.getLogger("test_safe")
    safe = SafeLoggerAdapter(logger)

    kwargs = {
        "extra": {
            "device_id": "elonxi_001",
            "port": 1430,
            "simulated": False,
            "sample_rate_hz": 200.0,
        }
    }
    msg, processed_kwargs = safe.process("device info", kwargs)
    extra = processed_kwargs.get("extra", {})
    assert extra["device_id"] == "elonxi_001"
    assert extra["port"] == 1430
    assert extra["simulated"] is False


def test_safe_logger_handles_no_extra() -> None:
    """When no extra dict is provided, process should no-op."""
    logger = logging.getLogger("test_no_extra")
    safe = SafeLoggerAdapter(logger)
    kwargs: dict = {}
    msg, processed_kwargs = safe.process("simple message", kwargs)
    assert processed_kwargs == {}
    assert msg == "simple message"


def test_get_logger_returns_safe_logger() -> None:
    """get_logger should return a SafeLoggerAdapter."""
    log = get_logger("test_module")
    assert isinstance(log, SafeLoggerAdapter)
