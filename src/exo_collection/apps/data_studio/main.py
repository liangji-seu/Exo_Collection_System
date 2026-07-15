"""Command-line entry point for Exo Data Studio."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QStandardPaths, QTimer
from PySide6.QtWidgets import QApplication

from exo_collection.apps.data_studio.window import DataStudioWindow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exo-data-studio")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Local Exo Collection dataset root (selectable again in the UI)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create the offscreen UI, complete one background refresh, and exit",
    )
    return parser


def _default_data_root() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(base or Path.home() / "ExoCollectionSystem") / "data"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    options = _build_parser().parse_args(arguments)
    if options.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    QApplication.setOrganizationName("Exo Collection System")
    QApplication.setApplicationName("Exo Data Studio")
    app = QApplication.instance()
    if app is None:
        app = QApplication(["exo-data-studio", *arguments])

    data_root = options.data_root or _default_data_root()
    window = DataStudioWindow(
        data_root,
        autostart_refresh=not options.smoke_test,
    )
    window.show()

    if not options.smoke_test:
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


if __name__ == "__main__":
    raise SystemExit(main())
