"""Command-line entry point for the online model-testing module (在线测试)."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

from exo_collection.apps.model_runtime.window import ModelRuntimeWindow
from exo_collection.configuration import SharedAppSettings
from exo_collection.logging_setup import setup_collector_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exo-model-runtime")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create the offscreen UI, connect simulated devices, run the demo "
        "model end to end, and exit without touching hardware",
    )
    return parser


def _temporary_settings(data_root: Path) -> SharedAppSettings:
    settings = SharedAppSettings(
        QSettings(str(data_root / ".smoke-settings.ini"), QSettings.Format.IniFormat)
    )
    # Frozen smoke tests must never probe laboratory hardware.
    settings.set_device_profile_key("simulated")
    return settings


def _set_windows_dpi_awareness() -> None:
    """Must run before the first QApplication construction (see collector)."""
    import ctypes as _ctypes

    if sys.platform != "win32":
        return
    try:
        per_monitor_v2 = _ctypes.c_void_p(-4)
        if not _ctypes.windll.user32.SetProcessDpiAwarenessContext(per_monitor_v2):
            raise OSError("SetProcessDpiAwarenessContext returned false")
    except Exception:
        try:
            _ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor v1
        except Exception:
            try:
                _ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def _run_ui(settings: SharedAppSettings, *, smoke_test: bool) -> int:
    _set_windows_dpi_awareness()
    QApplication.setOrganizationName("Exo Collection System")
    QApplication.setApplicationName("Exo Model Runtime")
    app = QApplication.instance() or QApplication(["exo-model-runtime"])

    window = ModelRuntimeWindow(settings, smoke_test=smoke_test)
    if not smoke_test:
        window.showMaximized()
        QTimer.singleShot(0, window.showMaximized)

    if smoke_test:
        smoke_deadline = time.monotonic() + 30.0
        started = {"value": False}

        def drive_smoke() -> None:
            if not started["value"]:
                started["value"] = True
                window.connect_all()
                window.install_demo_spec()
            if window.torque_history:
                app.exit(0)
                return
            if time.monotonic() >= smoke_deadline:
                window.statusBar().showMessage(
                    "模型在线推理 smoke 超时（无扭矩指令）。"
                )
                app.exit(2)
                return
            QTimer.singleShot(50, drive_smoke)

        QTimer.singleShot(0, drive_smoke)

    exit_code = int(app.exec())
    window.close()
    return exit_code


def main(argv: Sequence[str] | None = None, *, settings: SharedAppSettings | None = None) -> int:
    multiprocessing.freeze_support()
    setup_collector_logging(level=logging.DEBUG, console=True)
    logger = logging.getLogger("exo_collection.model_runtime.main")
    logger.info("Online model-testing module starting")
    try:
        arguments = list(argv) if argv is not None else sys.argv[1:]
        options = _build_parser().parse_args(arguments)
        if options.smoke_test:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            with TemporaryDirectory(prefix="exo-model-runtime-smoke-") as directory:
                return _run_ui(_temporary_settings(Path(directory)), smoke_test=True)
        settings_store = settings if settings is not None else SharedAppSettings()
        return _run_ui(settings_store, smoke_test=False)
    except Exception:
        logger.exception("Online model-testing module terminated by an unhandled exception")
        raise
    finally:
        logger.info("Online model-testing module exiting")


if __name__ == "__main__":
    raise SystemExit(main())
