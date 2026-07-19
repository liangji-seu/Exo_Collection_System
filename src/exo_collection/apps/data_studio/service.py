"""Manifest-only catalog refresh service used by Exo Data Studio.

This module deliberately returns small Python summaries.  It never opens an
Artifact, so UI refresh cannot accidentally map or read an active high-volume
``.recording``/``.partial`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

from exo_collection.catalog import Catalog
from exo_collection.catalog.repositories import CatalogRepository, ScanReport
from exo_collection.storage.activity import AcquisitionActivity, read_activity


CATALOG_FILENAME = ".exo/catalog.sqlite3"
_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DataStudioSnapshot:
    """One immutable, UI-safe view of the local catalog."""

    data_root: Path
    tree: list[dict[str, Any]]
    statistics: dict[str, Any]
    scan_report: ScanReport
    acquisition_activity: AcquisitionActivity | None

    @property
    def lightweight_mode(self) -> bool:
        return self.acquisition_activity is not None


def load_catalog_snapshot(data_root: str | Path) -> DataStudioSnapshot:
    """Migrate, scan finalized Manifests, and query a compact catalog snapshot.

    ``CatalogRepository.scan_dataset`` is intentionally the only filesystem
    discovery operation here.  Its storage contract excludes ``.recording``
    Trial directories and reads only published ``manifest.json`` files.
    Artifact payloads are not inspected.
    """

    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _log.info("Catalog refresh started: data_root=%s", root)

    # Read before and after the short metadata refresh so a Collector that
    # starts concurrently is reflected in the mode returned to the UI.
    activity_before = read_activity(root)
    _log.debug(
        "Acquisition activity before scan: active=%s",
        activity_before is not None,
    )
    catalog = Catalog(root / CATALOG_FILENAME)
    try:
        _log.debug("Opening and migrating Catalog: path=%s", root / CATALOG_FILENAME)
        catalog.migrate()
        repository = CatalogRepository(catalog)
        # Lightweight mode browses the existing SQLite summaries only. Even a
        # Manifest-only full-tree walk is deferred so Data Studio does not add
        # disk activity while Collector owns the dataset root.
        activity_for_scan = activity_before or read_activity(root)
        report = (
            ScanReport()
            if activity_for_scan is not None
            else repository.scan_dataset(root)
        )
        _log.info(
            "Manifest scan completed: lightweight=%s indexed=%d failures=%d",
            activity_for_scan is not None,
            report.indexed,
            len(report.failures),
        )
        for failure in report.failures:
            _log.warning("Manifest scan failure: %s", failure)
        tree = repository.tree()
        statistics = repository.statistics()
        _log.debug(
            "Catalog summaries loaded: root_nodes=%d statistics_keys=%s",
            len(tree),
            sorted(statistics),
        )
    except Exception:
        _log.exception("Catalog refresh failed: data_root=%s", root)
        raise
    finally:
        catalog.close()
    activity_after = read_activity(root)

    snapshot = DataStudioSnapshot(
        data_root=root,
        tree=tree,
        statistics=statistics,
        scan_report=report,
        acquisition_activity=activity_after or activity_for_scan,
    )
    _log.info(
        "Catalog refresh finished: root_nodes=%d lightweight=%s elapsed_ms=%.1f",
        len(snapshot.tree),
        snapshot.lightweight_mode,
        (time.monotonic() - started) * 1000.0,
    )
    return snapshot
