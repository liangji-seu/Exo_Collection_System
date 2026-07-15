"""Command-line entry point for Exo Data Studio."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QApplication

from exo_collection.apps.data_studio.window import DataStudioWindow
from exo_collection.configuration import SharedAppSettings


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

    result = {"completed": False, "succeeded": False}

    def finish(succeeded: bool) -> None:
        result["completed"] = True
        result["succeeded"] = succeeded
        QTimer.singleShot(0, app.quit)

    window.refresh_finished.connect(finish)
    # The timeout is a test failure, while still guaranteeing a non-hanging
    # packaging/startup probe on machines with a broken SQLite environment.
    QTimer.singleShot(10_000, app.quit)
    window.refresh_catalog()
    exit_code = int(app.exec())
    window.close()
    if exit_code != 0:
        return exit_code
    return 0 if result["completed"] and result["succeeded"] else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: SharedAppSettings | None = None,
) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    options = _build_parser().parse_args(arguments)
    if options.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        with TemporaryDirectory(prefix="exo-data-studio-smoke-") as directory:
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
