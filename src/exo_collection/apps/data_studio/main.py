"""Command-line entry point for Exo Data Studio."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from PySide6.QtCore import QSettings, QTimer, QtMsgType, qInstallMessageHandler
from PySide6.QtWidgets import QApplication

from exo_collection.apps.data_studio.window import DataStudioWindow
from exo_collection.configuration import SharedAppSettings
from exo_collection.logging_setup import (
    data_studio_log_path,
    setup_data_studio_logging,
)

_log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exo-data-studio")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create the offscreen UI, complete one background refresh, and exit",
    )
    return parser


def _temporary_settings(data_root: Path) -> SharedAppSettings:
    return SharedAppSettings(
        QSettings(
            str(data_root / ".smoke-settings.ini"),
            QSettings.Format.IniFormat,
        )
    )


def _run_ui(
    arguments: list[str],
    data_root: Path,
    settings: SharedAppSettings,
    *,
    smoke_test: bool,
) -> int:
    QApplication.setOrganizationName("Exo Collection System")
    QApplication.setApplicationName("Exo Data Studio")
    app = QApplication.instance()
    if app is None:
        app = QApplication(["exo-data-studio", *arguments])

    window = DataStudioWindow(
        data_root,
        settings=settings,
        autostart_refresh=not smoke_test,
    )
    window.show()

    if not smoke_test:
        return int(app.exec())

    result = {
        "catalog_completed": False,
        "catalog_succeeded": False,
        "management_completed": False,
        "management_succeeded": False,
    }

    def catalog_finished(succeeded: bool) -> None:
        result["catalog_completed"] = True
        result["catalog_succeeded"] = succeeded
        if result["management_completed"]:
            window._thread_pool.waitForDone(10_000)
            QTimer.singleShot(0, app.quit)

    def management_finished(succeeded: bool) -> None:
        result["management_completed"] = True
        result["management_succeeded"] = succeeded
        if result["catalog_completed"]:
            window._thread_pool.waitForDone(10_000)
            QTimer.singleShot(0, app.quit)

    window.refresh_finished.connect(catalog_finished)
    window.management_refresh_finished.connect(management_finished)
    # The timeout is a test failure, while still guaranteeing a non-hanging
    # packaging/startup probe on machines with a broken SQLite or Windows
    # spawn environment. Success requires both the Catalog thread and the
    # management/annex process to import, execute and return a result.
    QTimer.singleShot(30_000, app.quit)
    window.refresh_catalog()
    exit_code = int(app.exec())
    # The QRunnable may still hold SQLAlchemy objects when app.quit() fires.
    # Wait synchronously so Python GC on the main thread does not collide with
    # the worker thread's still-live SQLAlchemy C extensions.
    window._thread_pool.waitForDone(10_000)
    window.close()
    if exit_code != 0:
        return exit_code
    return (
        0
        if result["catalog_completed"]
        and result["catalog_succeeded"]
        and result["management_completed"]
        and result["management_succeeded"]
        else 1
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: SharedAppSettings | None = None,
) -> int:
    multiprocessing.freeze_support()
    setup_data_studio_logging(level=logging.DEBUG, console=True)
    logger = logging.getLogger("exo_collection.data_studio.main")
    logger.info(
        "Exo Data Studio application starting; log_file=%s",
        data_studio_log_path(),
    )

    # ------------------------------------------------------------------
    # Global crash capture: log EVERY unhandled exception before exit
    # ------------------------------------------------------------------
    _original_excepthook = sys.excepthook

    def _crash_logger(exc_type, exc_value, exc_tb):
        logger.critical(
            "UNHANDLED EXCEPTION — Data Studio will terminate\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )
        _original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _crash_logger

    # Qt warnings / errors also routed to the Python log
    def _qt_message_handler(mode, _context, message):
        level = {
            QtMsgType.QtDebugMsg: logging.DEBUG,
            QtMsgType.QtInfoMsg: logging.INFO,
            QtMsgType.QtWarningMsg: logging.WARNING,
            QtMsgType.QtCriticalMsg: logging.CRITICAL,
            QtMsgType.QtFatalMsg: logging.CRITICAL,
        }.get(mode, logging.WARNING)
        logger.log(level, "Qt: %s", message)

    qInstallMessageHandler(_qt_message_handler)

    # ------------------------------------------------------------------
    try:
        logger.debug("Parsing command-line arguments…")
        arguments = list(argv) if argv is not None else sys.argv[1:]
        options = _build_parser().parse_args(arguments)

        if options.smoke_test:
            logger.debug("Smoke-test mode; using offscreen platform + temp dir")
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            with TemporaryDirectory(prefix="exo-data-studio-smoke-") as directory:
                data_root = Path(directory)
                return _run_ui(
                    arguments,
                    data_root,
                    _temporary_settings(data_root),
                    smoke_test=True,
                )

        logger.debug("Loading application settings…")
        settings_store = settings if settings is not None else SharedAppSettings()
        logger.info("Data root: %s", settings_store.data_root)
        logger.debug("Entering UI run loop…")
        exit_code = _run_ui(
            arguments,
            settings_store.data_root,
            settings_store,
            smoke_test=False,
        )
        logger.info("UI run loop ended; exit_code=%d", exit_code)
        return exit_code
    except Exception:
        logger.exception("Exo Data Studio terminated by an unhandled exception")
        raise
    finally:
        logger.info("Exo Data Studio application exiting")


if __name__ == "__main__":
    raise SystemExit(main())
