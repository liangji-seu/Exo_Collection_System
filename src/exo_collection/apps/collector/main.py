"""Command-line entry point for Exo Collector."""

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

from exo_collection.acquisition.messages import WorkerEventType
from exo_collection.acquisition.workers import CollectorWorker
from exo_collection.apps.collector.window import CollectorWindow
from exo_collection.configuration import SharedAppSettings
from exo_collection.logging_setup import setup_collector_logging
from exo_collection.orchestration.models import TrialRunRequest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exo-collector")
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


def _temporary_settings(data_root: Path) -> SharedAppSettings:
    settings = SharedAppSettings(
        QSettings(
            str(data_root / ".smoke-settings.ini"),
            QSettings.Format.IniFormat,
        )
    )
    # Frozen smoke tests must never probe laboratory hardware.
    settings.set_device_profile_key("simulated")
    return settings


def _run_ui(
    arguments: list[str],
    data_root: Path,
    settings: SharedAppSettings,
    *,
    smoke_test: bool,
) -> int:
    QApplication.setOrganizationName("Exo Collection System")
    QApplication.setApplicationName("Exo Collector")
    app = QApplication.instance()
    if app is None:
        app = QApplication(["exo-collector", *arguments])

    window = CollectorWindow(data_root, settings=settings)
    window.showMaximized()
    if smoke_test:
        # The existing internal UI smoke also exercises the spawn-based device
        # preflight. This is essential for PyInstaller: a window-only check
        # cannot detect missing frozen child modules or broken freeze_support.
        smoke_deadline = time.monotonic() + 30.0
        smoke_started = {"value": False}

        def poll_preflight_smoke() -> None:
            if not smoke_started["value"]:
                smoke_started["value"] = True
                window.run_preflight()
            if not window.preflight_in_progress:
                app.exit(0 if window.preflight_ready else 1)
                return
            if time.monotonic() >= smoke_deadline:
                window.statusBar().showMessage(
                    "Frozen preflight smoke timed out; terminating child process."
                )
                app.exit(2)
                return
            QTimer.singleShot(50, poll_preflight_smoke)

        QTimer.singleShot(0, poll_preflight_smoke)
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


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: SharedAppSettings | None = None,
) -> int:
    multiprocessing.freeze_support()
    setup_collector_logging()
    logger = logging.getLogger("exo_collection.collector.main")
    logger.info("Exo Collector application starting")
    arguments = list(argv) if argv is not None else sys.argv[1:]
    options = _build_parser().parse_args(arguments)
    if options.collect_smoke_test:
        with TemporaryDirectory(prefix="exo-collector-collect-smoke-") as directory:
            return _run_collection_smoke(Path(directory), options.duration)
    if options.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        with TemporaryDirectory(prefix="exo-collector-ui-smoke-") as directory:
            data_root = Path(directory)
            return _run_ui(
                arguments,
                data_root,
                _temporary_settings(data_root),
                smoke_test=True,
            )

    settings_store = settings if settings is not None else SharedAppSettings()
    try:
        return _run_ui(
            arguments,
            settings_store.data_root,
            settings_store,
            smoke_test=False,
        )
    finally:
        logger.info("Exo Collector application exiting")


if __name__ == "__main__":
    raise SystemExit(main())
