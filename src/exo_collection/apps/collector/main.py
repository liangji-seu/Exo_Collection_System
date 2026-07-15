"""Command-line entry point for Exo Collector."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QStandardPaths, QTimer
from PySide6.QtWidgets import QApplication

from exo_collection.acquisition.messages import WorkerEventType
from exo_collection.acquisition.workers import CollectorWorker
from exo_collection.apps.collector.window import CollectorWindow
from exo_collection.orchestration.models import TrialRunRequest


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
    parser.add_argument(
        "--collect-smoke-test",
        action="store_true",
        help="Run a short simulated Trial through the spawned worker and exit",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.5,
        help="Simulated duration for --collect-smoke-test",
    )
    return parser


def _default_data_root() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(base or Path.home() / "ExoCollectionSystem") / "data"


def _run_collection_smoke(data_root: Path, duration_s: float) -> int:
    worker = CollectorWorker(TrialRunRequest(data_root=data_root, duration_s=duration_s))
    worker.start()
    terminal = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and terminal is None:
        for event in worker.poll_events():
            if event.event_type in {WorkerEventType.COMPLETED, WorkerEventType.FAILED}:
                terminal = event
                break
        if terminal is None:
            time.sleep(0.02)
    if terminal is None and worker.is_alive:
        worker.request_stop()
    shutdown_deadline = time.monotonic() + 20
    while worker.is_alive and time.monotonic() < shutdown_deadline:
        for event in worker.poll_events(limit=1000):
            if event.event_type in {WorkerEventType.COMPLETED, WorkerEventType.FAILED}:
                terminal = event
        worker.join(timeout=0.05)
    if worker.is_alive:
        worker.terminate_for_recovery()
    exitcode = worker.join(timeout=2)
    for event in worker.poll_events(limit=1000):
        if event.event_type in {WorkerEventType.COMPLETED, WorkerEventType.FAILED}:
            terminal = event
    try:
        return int(
            terminal is None
            or terminal.event_type is WorkerEventType.FAILED
            or exitcode != 0
        )
    finally:
        if not worker.is_alive:
            worker.close()


def main(argv: Sequence[str] | None = None) -> int:
    multiprocessing.freeze_support()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    options = _build_parser().parse_args(arguments)
    data_root = options.data_root or _default_data_root()
    if options.collect_smoke_test:
        return _run_collection_smoke(data_root, options.duration)
    if options.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    QApplication.setOrganizationName("Exo Collection System")
    QApplication.setApplicationName("Exo Collector")
    app = QApplication.instance()
    if app is None:
        app = QApplication(["exo-collector", *arguments])

    window = CollectorWindow(data_root)
    window.show()
    if options.smoke_test:
        QTimer.singleShot(50, app.quit)
    exit_code = int(app.exec())
    # QApplication.quit()/Windows session shutdown can end the Qt event loop
    # before closeEvent has completed a live Trial. Keep pumping the existing
    # window until its controlled stop, Writer flush and worker release finish.
    window.close()
    shutdown_deadline = time.monotonic() + 60
    while window.worker is not None:
        window.poll_worker_events()
        app.processEvents()
        if time.monotonic() >= shutdown_deadline and window.worker is not None:
            terminate = getattr(window.worker, "terminate_for_recovery", None)
            if callable(terminate):
                terminate()
            shutdown_deadline = float("inf")
        time.sleep(0.02)
    window.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
