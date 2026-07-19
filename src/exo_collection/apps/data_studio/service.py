"""Manifest-only catalog refresh service used by Exo Data Studio.

This module deliberately returns small Python summaries.  It never opens an
Artifact, so UI refresh cannot accidentally map or read an active high-volume
``.recording``/``.partial`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exo_collection.catalog import Catalog
from exo_collection.catalog.repositories import CatalogRepository, ScanReport
from exo_collection.storage.activity import AcquisitionActivity, read_activity


CATALOG_FILENAME = ".exo/catalog.sqlite3"


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

    # Read before and after the short metadata refresh so a Collector that
    # starts concurrently is reflected in the mode returned to the UI.
    activity_before = read_activity(root)
    catalog = Catalog(root / CATALOG_FILENAME)
    try:
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
        tree = repository.tree()
        statistics = repository.statistics()
    finally:
        catalog.close()
    activity_after = read_activity(root)

    return DataStudioSnapshot(
        data_root=root,
        tree=tree,
        statistics=statistics,
        scan_report=report,
        acquisition_activity=activity_after or activity_for_scan,
    )
