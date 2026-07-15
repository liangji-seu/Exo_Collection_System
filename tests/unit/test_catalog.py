from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import event, inspect, text
from sqlalchemy.exc import OperationalError

from exo_collection.catalog import Catalog
from exo_collection.catalog import repositories as catalog_repositories
from exo_collection.catalog.models import (
    ArtifactRow,
    ProjectRow,
    SessionRow,
    SubjectRow,
    TrialRow,
)
from exo_collection.catalog.repositories import CatalogRepository
from exo_collection.domain.models import (
    ArtifactKind,
    Condition,
    Project,
    QualityGrade,
    Session,
    Subject,
)
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


def _migrate_catalog_worker(
    catalog_path: str,
    barrier: Any,
    results: Any,
) -> None:
    try:
        barrier.wait(timeout=20)
        catalog = Catalog(catalog_path)
        try:
            catalog.migrate()
        finally:
            catalog.close()
        results.put(None)
    except BaseException as exc:  # pragma: no cover - failure detail crosses process boundary
        results.put(f"{type(exc).__name__}: {exc}")


def _index_manifest_worker(
    catalog_path: str,
    manifest_json: str,
    manifest_path: str,
    barrier: Any,
    results: Any,
) -> None:
    catalog = Catalog(catalog_path)
    try:
        manifest = TrialManifest.model_validate_json(manifest_json)
        repository = CatalogRepository(catalog)
        barrier.wait(timeout=20)
        repository.index_manifest(manifest, manifest_path)
        results.put(None)
    except BaseException as exc:  # pragma: no cover - failure detail crosses process boundary
        results.put(f"{type(exc).__name__}: {exc}")
    finally:
        catalog.close()


def _make_hierarchy(now: datetime) -> tuple[Project, Subject, Session]:
    project = Project(
        project_name="Exoskeleton Study",
        principal_investigator="PI One",
        protocol_version="1.0.0",
        data_root="D:/exo-data",
        condition_definition_version="1.0.0",
        created_at_utc=now,
        updated_at_utc=now,
    )
    subject = Subject(
        project_uuid=project.project_uuid,
        subject_code="SUB-001",
        group="control",
        attributes={"dominant_side": "right"},
        created_at_utc=now + timedelta(minutes=1),
        updated_at_utc=now + timedelta(minutes=1),
    )
    visit = Session(
        project_uuid=project.project_uuid,
        subject_uuid=subject.subject_uuid,
        operator="Operator One",
        software_version="0.1.0",
        started_at_utc=now + timedelta(minutes=3),
        created_at_utc=now + timedelta(minutes=2),
    )
    return project, subject, visit


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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


def test_register_hierarchy_preserves_creation_and_session_start_times(tmp_path) -> None:
    initial_time = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    project, subject, visit = _make_hierarchy(initial_time)
    refreshed_time = initial_time + timedelta(days=30)
    refreshed_project = Project(
        project_uuid=project.project_uuid,
        project_name="Renamed Exoskeleton Study",
        principal_investigator="PI Two",
        protocol_version="1.1.0",
        data_root="E:/exo-data",
        condition_definition_version="1.1.0",
        created_at_utc=refreshed_time,
        updated_at_utc=refreshed_time,
    )
    refreshed_subject = Subject(
        subject_uuid=subject.subject_uuid,
        project_uuid=project.project_uuid,
        subject_code="SUB-001-UPDATED",
        group="intervention",
        attributes={"dominant_side": "left"},
        created_at_utc=refreshed_time + timedelta(minutes=1),
        updated_at_utc=refreshed_time + timedelta(minutes=1),
    )
    refreshed_visit = Session(
        session_uuid=visit.session_uuid,
        project_uuid=project.project_uuid,
        subject_uuid=subject.subject_uuid,
        operator="Operator Two",
        software_version="0.2.0",
        started_at_utc=refreshed_time + timedelta(minutes=3),
        created_at_utc=refreshed_time + timedelta(minutes=2),
    )

    with Catalog(tmp_path / "catalog.sqlite3") as catalog:
        repository = CatalogRepository(catalog)
        repository.register_hierarchy(project, subject, visit)
        repository.register_hierarchy(refreshed_project, refreshed_subject, refreshed_visit)

        with catalog.session() as db:
            project_row = db.get(ProjectRow, str(project.project_uuid))
            subject_row = db.get(SubjectRow, str(subject.subject_uuid))
            session_row = db.get(SessionRow, str(visit.session_uuid))
            assert project_row is not None
            assert subject_row is not None
            assert session_row is not None
            assert project_row.name == "Renamed Exoskeleton Study"
            assert subject_row.group_label == "intervention"
            assert session_row.operator == "Operator Two"
            assert _aware_utc(project_row.created_utc) == project.created_at_utc
            assert _aware_utc(subject_row.created_utc) == subject.created_at_utc
            assert _aware_utc(session_row.created_utc) == visit.created_at_utc
            assert _aware_utc(session_row.started_utc) == visit.started_at_utc


def test_concurrent_processes_can_migrate_one_catalog(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.sqlite3"
    context = multiprocessing.get_context("spawn")
    process_count = 4
    barrier = context.Barrier(process_count)
    results = context.Queue()
    processes = [
        context.Process(
            target=_migrate_catalog_worker,
            args=(str(catalog_path), barrier, results),
        )
        for _ in range(process_count)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    errors = [results.get(timeout=5) for _ in range(process_count)]
    assert errors == [None] * process_count
    assert all(process.exitcode == 0 for process in processes)
    with Catalog(catalog_path) as catalog:
        assert "alembic_version" in inspect(catalog.engine).get_table_names()


def test_concurrent_processes_index_same_manifest_into_empty_catalog(tmp_path) -> None:
    catalog_path = tmp_path / "catalog.sqlite3"
    with Catalog(catalog_path) as catalog:
        with catalog.session() as db:
            assert db.query(ProjectRow).count() == 0

    manifest = make_manifest()
    manifest_path = tmp_path / "finalized" / "manifest.json"
    context = multiprocessing.get_context("spawn")
    process_count = 2
    barrier = context.Barrier(process_count)
    results = context.Queue()
    processes = [
        context.Process(
            target=_index_manifest_worker,
            args=(
                str(catalog_path),
                manifest.model_dump_json(),
                str(manifest_path),
                barrier,
                results,
            ),
        )
        for _ in range(process_count)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    errors = [results.get(timeout=5) for _ in range(process_count)]
    assert errors == [None] * process_count
    assert all(process.exitcode == 0 for process in processes)
    with Catalog(catalog_path) as catalog:
        with catalog.session() as db:
            assert db.query(ProjectRow).count() == 1
            assert db.query(SubjectRow).count() == 1
            assert db.query(SessionRow).count() == 1
            assert db.query(TrialRow).count() == 1
            assert db.query(ArtifactRow).count() == len(manifest.artifacts)


@pytest.mark.parametrize("operation_name", ["register_hierarchy", "index_manifest"])
def test_catalog_writes_retry_real_sqlite_lock_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    catalog_path = tmp_path / "catalog.sqlite3"
    catalog = Catalog(catalog_path)
    catalog.migrate()
    repository = CatalogRepository(catalog)

    @event.listens_for(catalog.engine, "checkout")
    def use_short_test_timeout(dbapi_connection: object, *_args: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA busy_timeout=1")
        cursor.close()

    catalog.engine.dispose()
    lock_ready = threading.Event()
    release_lock = threading.Event()
    holder_errors: list[BaseException] = []

    def hold_write_lock() -> None:
        connection = sqlite3.connect(catalog_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            lock_ready.set()
            release_lock.wait(timeout=5)
            connection.rollback()
        except BaseException as exc:  # pragma: no cover - reported in the parent assertion
            holder_errors.append(exc)
            lock_ready.set()
        finally:
            connection.close()

    holder = threading.Thread(target=hold_write_lock, daemon=True)
    holder.start()
    assert lock_ready.wait(timeout=5)
    assert not holder_errors

    delays: list[float] = []
    real_sleep = time.sleep

    def track_sleep(delay: float) -> None:
        delays.append(delay)
        real_sleep(delay)

    monkeypatch.setattr(catalog_repositories, "sleep", track_sleep)
    timer = threading.Timer(0.12, release_lock.set)
    timer.start()
    manifest = make_manifest()
    project, subject, visit = _make_hierarchy(
        datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    )
    try:
        if operation_name == "register_hierarchy":
            repository.register_hierarchy(project, subject, visit)
        else:
            repository.index_manifest(manifest, tmp_path / "final" / "manifest.json")
    finally:
        release_lock.set()
        timer.cancel()
        holder.join(timeout=5)

    assert not holder.is_alive()
    assert not holder_errors
    assert delays
    assert delays == list(catalog_repositories._WRITE_RETRY_DELAYS_SECONDS[: len(delays)])
    with catalog.session() as db:
        if operation_name == "register_hierarchy":
            assert db.get(ProjectRow, str(project.project_uuid)) is not None
        else:
            assert db.get(TrialRow, str(manifest.trial_uuid)) is not None
    catalog.close()


def test_catalog_write_lock_retry_is_bounded(tmp_path, monkeypatch) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite3")
    repository = CatalogRepository(catalog)
    attempts = 0
    delays: list[float] = []

    def always_locked(_db) -> None:
        nonlocal attempts
        attempts += 1
        raise OperationalError(
            "INSERT",
            {},
            sqlite3.OperationalError("database is locked"),
        )

    monkeypatch.setattr(catalog_repositories, "sleep", delays.append)
    with pytest.raises(OperationalError, match="database is locked"):
        repository._run_write_transaction(always_locked)

    assert attempts == len(catalog_repositories._WRITE_RETRY_DELAYS_SECONDS) + 1
    assert delays == list(catalog_repositories._WRITE_RETRY_DELAYS_SECONDS)
    catalog.close()
