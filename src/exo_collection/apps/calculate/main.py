"""Exo Calculate 的命令行入口。"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

from exo_collection.apps.calculate.window import CalculateWindow
from exo_collection.configuration import SharedAppSettings
from exo_collection.logging_setup import calculate_log_path, setup_calculate_logging

_log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exo-calculate")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create the offscreen UI, process events, and exit without computing",
    )
    return parser


def _temporary_settings(data_root: Path) -> SharedAppSettings:
    return SharedAppSettings(
        QSettings(str(data_root / ".smoke-settings.ini"), QSettings.Format.IniFormat)
    )


def _run_ui(
    arguments: list[str],
    data_root: Path,
    settings: SharedAppSettings,
    *,
    smoke_test: bool,
) -> int:
    QApplication.setOrganizationName("Exo Collection System")
    QApplication.setApplicationName("Exo Calculate")
    app = QApplication.instance()
    if app is None:
        app = QApplication(["exo-calculate", *arguments])

    window = CalculateWindow(data_root, settings=settings)
    window.show()

    if smoke_test:
        QTimer.singleShot(200, app.quit)
    return int(app.exec())


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: SharedAppSettings | None = None,
) -> int:
    multiprocessing.freeze_support()
    setup_calculate_logging(level=logging.DEBUG, console=True)
    logger = logging.getLogger("exo_collection.calculate.main")
    logger.info("Exo Calculate application starting; log_file=%s", calculate_log_path())

    arguments = list(argv) if argv is not None else sys.argv[1:]
    options = _build_parser().parse_args(arguments)

    if options.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        with TemporaryDirectory(prefix="exo-calculate-smoke-") as directory:
            data_root = Path(directory)
            return _run_ui(
                arguments,
                data_root,
                _temporary_settings(data_root),
                smoke_test=True,
            )

    settings_store = settings if settings is not None else SharedAppSettings()
    return _run_ui(
        arguments,
        settings_store.data_root,
        settings_store,
        smoke_test=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
