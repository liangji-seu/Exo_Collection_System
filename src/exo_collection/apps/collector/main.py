"""Command-line entry point for Exo Collector."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QStandardPaths, QTimer
from PySide6.QtWidgets import QApplication

from exo_collection.apps.collector.window import CollectorWindow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exo-collector")
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Local Exo Collection dataset root (selectable again in the UI)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create the offscreen UI, process events, and exit without collecting",
    )
    return parser


def _default_data_root() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(base or Path.home() / "ExoCollectionSystem") / "data"


def main(argv: Sequence[str] | None = None) -> int:
    multiprocessing.freeze_support()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    options = _build_parser().parse_args(arguments)
    if options.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    QApplication.setOrganizationName("Exo Collection System")
    QApplication.setApplicationName("Exo Collector")
    app = QApplication.instance()
    if app is None:
        app = QApplication(["exo-collector", *arguments])

    window = CollectorWindow(options.data_root or _default_data_root())
    window.show()
    if options.smoke_test:
        QTimer.singleShot(50, app.quit)
    exit_code = int(app.exec())
    window.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
