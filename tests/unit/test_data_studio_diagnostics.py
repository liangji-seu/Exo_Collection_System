"""Regression coverage for actionable, non-frame-level Data Studio logs."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys

from exo_collection.apps.data_studio.service import load_catalog_snapshot


def test_catalog_refresh_logs_scan_summary(
    tmp_path: Path,
    caplog,
) -> None:
    caplog.set_level(logging.DEBUG, logger="exo_collection.apps.data_studio.service")

    snapshot = load_catalog_snapshot(tmp_path)

    assert snapshot.data_root == tmp_path.resolve()
    messages = [record.getMessage() for record in caplog.records]
    assert any("Catalog refresh started" in message for message in messages)
    assert any("Manifest scan completed" in message for message in messages)
    assert any("Catalog summaries loaded" in message for message in messages)
    assert any("Catalog refresh finished" in message for message in messages)


def test_spawned_process_can_attach_to_explicit_application_log(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "log" / "studio.log"
    code = (
        "import logging; "
        "from exo_collection.logging_setup import configure_subprocess_logging; "
        f"configure_subprocess_logging(log_path={str(log_path)!r}); "
        "logging.getLogger('exo_collection.data_studio.worker_test').info("
        "'worker diagnostic marker')"
    )

    subprocess.run([sys.executable, "-c", code], check=True)

    content = log_path.read_text(encoding="utf-8")
    assert "worker diagnostic marker" in content
    assert "exo_collection.data_studio.worker_test" in content
