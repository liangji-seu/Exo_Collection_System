from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import inspect, text

from exo_collection.catalog import Catalog
from exo_collection.catalog.repositories import CatalogRepository
from exo_collection.domain.models import ArtifactKind, Condition, QualityGrade
from exo_collection.domain.states import TrialState
from exo_collection.storage.manifest import (
    ConfigurationSnapshot,
    ManifestArtifact,
    QualitySummary,
    SoftwareProvenance,
    TrialManifest,
    TrialTiming,
    save_manifest,
)


def make_manifest() -> TrialManifest:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    trial_uuid = uuid4()
    return TrialManifest(
        project_uuid=uuid4(),
        subject_uuid=uuid4(),
        session_uuid=uuid4(),
        trial_uuid=trial_uuid,
        state=TrialState.FINALIZED,
        condition=Condition(
            condition_code="WALK_LEVEL",
            condition_name="Level walking",
            condition_level=2,
            parameters={"speed_mps": 0.8},
            repeat_index=1,
            protocol_version="1.0.0",
            selected_at_utc=now,
        ),
        timing=TrialTiming(
            started_at_utc=now,
            stopped_at_utc=now + timedelta(seconds=4),
            finalized_at_utc=now + timedelta(seconds=5),
            start_host_monotonic_ns=1_000,
            stop_host_monotonic_ns=4_000_001_000,
            finalize_host_monotonic_ns=5_000_001_000,
        ),
        software=SoftwareProvenance(
            application="Exo Collector",
            application_version="0.1.0",
            core_version="0.1.0",
            git_commit="test-commit",
        ),
        configuration=ConfigurationSnapshot(
            config_version="1.0.0",
            protocol_version="1.0.0",
            condition_definition_version="1.0.0",
            content_sha256="b" * 64,
        ),
        artifacts=[
            ManifestArtifact(
                artifact_uuid=uuid4(),
                trial_uuid=trial_uuid,
                modality="imu",
                kind=ArtifactKind.RAW,
                media_type="application/x-hdf5",
                relative_path="raw/imu.h5",
                size_bytes=1234,
                sha256="a" * 64,
                created_at_utc=now,
                finalized_at_utc=now + timedelta(seconds=5),
            )
        ],
        quality=QualitySummary(
            computed_grade=QualityGrade.A,
            required_artifacts_complete=True,
            integrity_checks_passed=True,
            algorithm_version="quality-0.1.0",
            assessed_at_utc=now + timedelta(seconds=5),
        ),
    )


def test_first_migration_and_sqlite_concurrency_pragmas(tmp_path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    catalog.migrate()
    tables = set(inspect(catalog.engine).get_table_names())
    assert {"projects", "subjects", "sessions", "conditions", "trials", "artifacts"} <= tables
    assert "alembic_version" in tables
    with catalog.engine.connect() as connection:
        assert connection.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
        assert connection.execute(text("PRAGMA busy_timeout")).scalar_one() == 5000
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    catalog.close()


def test_manifest_scan_rebuilds_tree_and_statistics(tmp_path) -> None:
    manifest = make_manifest()
    trial_dir = (
        tmp_path
        / str(manifest.project_uuid)
        / str(manifest.subject_uuid)
        / str(manifest.session_uuid)
        / "trials"
        / str(manifest.trial_uuid)
    )
    path = trial_dir / "manifest.json"
    save_manifest(path, manifest)
    recording_manifest = trial_dir.parent / f"{uuid4()}.recording" / "manifest.json"
    recording_manifest.parent.mkdir(parents=True)
    recording_manifest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    with Catalog(tmp_path / "catalog.sqlite3") as catalog:
        repository = CatalogRepository(catalog)
        report = repository.scan_dataset(tmp_path)
        tree = repository.tree()
        statistics = repository.statistics()

    assert report.indexed == 1
    assert not report.failures
    assert len(tree) == 1
    artifact = tree[0]["children"][0]["children"][0]["children"][0]["children"][0]
    assert artifact["label"] == "raw/imu.h5"
    assert statistics["trial_count"] == 1
    assert statistics["finalized_count"] == 1
    assert statistics["total_duration_s"] == 4.0
    assert statistics["by_condition"]["WALK_LEVEL"]["trial_count"] == 1

