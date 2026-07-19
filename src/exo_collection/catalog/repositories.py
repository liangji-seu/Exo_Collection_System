"""Short-transaction Catalog operations and Manifest-driven rebuilding."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Callable
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from exo_collection.domain.models import Project, Session as DomainSession, Subject
from exo_collection.storage.layout import (
    iter_finalized_manifest_paths,
    path_has_unpublished_component,
)
from exo_collection.storage.manifest import TrialManifest, load_manifest

from .db import Catalog
from .models import ArtifactRow, ConditionRow, ProjectRow, SessionRow, SubjectRow, TrialRow


_WRITE_RETRY_DELAYS_SECONDS = (0.025, 0.05, 0.1, 0.2)


@dataclass(slots=True)
class ScanReport:
    indexed: int = 0
    unchanged_or_updated: int = 0
    failures: dict[str, str] = field(default_factory=dict)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _utc(value: datetime | None) -> datetime | None:
    return value.astimezone(timezone.utc) if value is not None else None


def _condition_uuid(manifest: TrialManifest) -> str:
    key = (
        f"exo-condition:{manifest.project_uuid}:"
        f"{manifest.condition.protocol_version}:{manifest.condition.condition_code}"
    )
    return str(uuid5(NAMESPACE_URL, key))


def _is_sqlite_lock_error(exc: OperationalError) -> bool:
    """Return whether an OperationalError is SQLite BUSY/LOCKED contention."""

    original = exc.orig
    error_code = getattr(original, "sqlite_errorcode", None)
    if isinstance(error_code, int) and error_code & 0xFF in {
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    }:
        return True
    message = str(original).casefold()
    return "locked" in message and ("database" in message or "table" in message)


def _validate_scanned_manifest_location(
    path: Path,
    manifest: TrialManifest,
) -> None:
    """Reject copied/misplaced Manifests before they can replace Catalog paths.

    UUIDs in the Manifest remain authoritative.  Human F/T and subject-code
    folders are accepted only when they agree with those same Manifest labels;
    the legacy UUID directory layout remains readable.

    Two canonical layouts are supported:

    * **Human-readable** (current):
      ``{root}/{project}/{subject}/{condition}/session{repeat}_{ts}/.exo/manifest.json``
    * **Legacy UUID** (pre-0.2.0):
      ``{root}/{project}/{subject}/trials/{trial_uuid}/manifest.json``
    """

    if path.name != "manifest.json" or len(path.parents) < 6:
        raise ValueError("Manifest is not in a canonical Trial directory")

    _EXO = ".exo"
    if path.parent.name == _EXO:
        # ------------------------------------------------------------------
        # Human-readable layout: .exo/manifest.json is two levels below the
        # trial root.  Only project and subject directory identity is
        # validated; the trial leaf name carries a timestamp and cannot be
        # checked against a single Manifest field.
        # ------------------------------------------------------------------
        trial_directory = path.parent.parent  # session{repeat}_{timestamp}
        condition_directory = trial_directory.parent
        subject_directory = condition_directory.parent
        project_directory = subject_directory.parent
        # The innermost condition directory merely groups trials — its name
        # is not an identity claim, so skip it.
        del condition_directory
    else:
        # ------------------------------------------------------------------
        # Legacy UUID layout: manifest.json sits directly at the trial root
        # and the parent-of-parent is always "trials".
        # ------------------------------------------------------------------
        trial_directory = path.parent
        trials_directory = trial_directory.parent
        session_directory = trials_directory.parent
        subject_directory = session_directory.parent
        project_directory = subject_directory.parent
        if trials_directory.name.casefold() != "trials":
            raise ValueError("Manifest parent hierarchy is missing the trials directory")
        if trial_directory.name.casefold() != str(manifest.trial_uuid).casefold():
            raise ValueError("Trial directory does not match Manifest trial_uuid")
        if session_directory.name.casefold() != str(manifest.session_uuid).casefold():
            raise ValueError("Session directory does not match Manifest session_uuid")

    accepted_subjects = {str(manifest.subject_uuid).casefold()}
    if manifest.subject_code:
        accepted_subjects.add(manifest.subject_code.casefold())
    if subject_directory.name.casefold() not in accepted_subjects:
        raise ValueError("Subject directory does not match Manifest identity")
    accepted_projects = {str(manifest.project_uuid).casefold()}
    if manifest.project_code:
        accepted_projects.add(manifest.project_code.casefold())
    if project_directory.name.casefold() not in accepted_projects:
        raise ValueError("Project directory does not match Manifest identity")


class CatalogRepository:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def _run_write_transaction(self, operation: Callable[[Session], None]) -> None:
        """Run a short write transaction with bounded SQLite lock retries."""

        retry_delays: tuple[float | None, ...] = (*_WRITE_RETRY_DELAYS_SECONDS, None)
        for retry_delay in retry_delays:
            try:
                with self.catalog.session() as db, db.begin():
                    operation(db)
                return
            except OperationalError as exc:
                if not _is_sqlite_lock_error(exc) or retry_delay is None:
                    raise
                sleep(retry_delay)

    def register_hierarchy(self, project: Project, subject: Subject, visit: DomainSession) -> None:
        """Create or refresh hierarchy metadata without rewriting audit anchors."""

        if subject.project_uuid != project.project_uuid:
            raise ValueError("Subject does not belong to Project")
        if visit.project_uuid != project.project_uuid or visit.subject_uuid != subject.subject_uuid:
            raise ValueError("Session hierarchy UUIDs are inconsistent")
        project_code = (
            project.project_code
            or f"{project.project_name}-{str(project.project_uuid)[:8]}"
        )
        project_id = str(project.project_uuid)
        subject_id = str(subject.subject_uuid)
        session_id = str(visit.session_uuid)

        def register(db: Session) -> None:
            project_row = db.get(ProjectRow, project_id)
            if project_row is None:
                project_row = ProjectRow(
                    project_uuid=str(project.project_uuid),
                    project_code=project_code,
                    name=project.project_name,
                    principal_investigator=project.principal_investigator,
                    protocol_version=project.protocol_version,
                    data_root=project.data_root,
                    created_utc=_utc(project.created_at_utc),
                )
                db.add(project_row)
                db.flush()
            else:
                project_row.project_code = project_code
                project_row.name = project.project_name
                project_row.principal_investigator = project.principal_investigator
                project_row.protocol_version = project.protocol_version
                project_row.data_root = project.data_root

            subject_row = db.get(SubjectRow, subject_id)
            if subject_row is None:
                subject_row = SubjectRow(
                    subject_uuid=str(subject.subject_uuid),
                    project_uuid=str(subject.project_uuid),
                    subject_code=subject.subject_code,
                    group_label=subject.group,
                    attributes_json=_json(subject.attributes),
                    created_utc=_utc(subject.created_at_utc),
                )
                db.add(subject_row)
                db.flush()
            else:
                if subject_row.project_uuid != project_id:
                    raise ValueError("Existing Subject belongs to a different Project")
                subject_row.subject_code = subject.subject_code
                subject_row.group_label = subject.group
                subject_row.attributes_json = _json(subject.attributes)

            session_row = db.get(SessionRow, session_id)
            if session_row is None:
                session_row = SessionRow(
                    session_uuid=str(visit.session_uuid),
                    project_uuid=str(visit.project_uuid),
                    subject_uuid=str(visit.subject_uuid),
                    operator=visit.operator,
                    software_version=visit.software_version,
                    started_utc=_utc(visit.started_at_utc),
                    ended_utc=_utc(visit.ended_at_utc),
                    created_utc=_utc(visit.created_at_utc),
                )
                db.add(session_row)
            else:
                if (
                    session_row.project_uuid != project_id
                    or session_row.subject_uuid != subject_id
                ):
                    raise ValueError("Existing Session belongs to a different hierarchy")
                session_row.operator = visit.operator
                session_row.software_version = visit.software_version
                session_row.ended_utc = _utc(visit.ended_at_utc)

        self._run_write_transaction(register)

    def index_manifest(self, manifest: TrialManifest, manifest_path: str | Path) -> None:
        """Upsert one immutable Manifest and its Artifact summaries."""

        path = Path(manifest_path).expanduser().resolve()
        if path_has_unpublished_component(path):
            raise ValueError("Catalog must not index an unpublished Trial package")
        condition_id = _condition_uuid(manifest)
        quality = manifest.quality.reviewed_grade or manifest.quality.computed_grade
        quality_value = quality.value if quality is not None else "INVALID"
        timing = manifest.timing
        if timing.stop_host_monotonic_ns is not None:
            duration_s = max(
                0.0,
                (timing.stop_host_monotonic_ns - timing.start_host_monotonic_ns) / 1_000_000_000,
            )
        elif timing.stopped_at_utc is not None:
            duration_s = max(0.0, (timing.stopped_at_utc - timing.started_at_utc).total_seconds())
        else:
            duration_s = 0.0

        def index(db: Session) -> None:
            now = datetime.now(timezone.utc)
            existing_trial = db.get(TrialRow, str(manifest.trial_uuid))
            if existing_trial is not None:
                expected_identity = {
                    "project_uuid": str(manifest.project_uuid),
                    "subject_uuid": str(manifest.subject_uuid),
                    "session_uuid": str(manifest.session_uuid),
                    "condition_uuid": condition_id,
                    "condition_code": manifest.condition.condition_code,
                    "repeat_index": manifest.condition.repeat_index,
                }
                conflicts = {
                    field_name: (
                        getattr(existing_trial, field_name),
                        expected_value,
                    )
                    for field_name, expected_value in expected_identity.items()
                    if getattr(existing_trial, field_name) != expected_value
                }
                if conflicts:
                    details = ", ".join(
                        f"{field_name}: catalog={actual!r}, manifest={expected!r}"
                        for field_name, (actual, expected) in sorted(conflicts.items())
                    )
                    raise ValueError(
                        f"Catalog trial_uuid conflict for {manifest.trial_uuid}: "
                        f"immutable identity differs ({details})"
                    )
            self._ensure_fallback_hierarchy(db, manifest, path)
            condition_values = {
                "condition_uuid": condition_id,
                "project_uuid": str(manifest.project_uuid),
                "condition_code": manifest.condition.condition_code,
                "condition_name": manifest.condition.condition_name,
                "condition_level": (
                    manifest.condition.condition_level
                    if isinstance(manifest.condition.condition_level, int)
                    else None
                ),
                "protocol_version": manifest.condition.protocol_version,
                "parameters_json": _json(manifest.condition.parameters),
            }
            condition_insert = sqlite_insert(ConditionRow).values(**condition_values)
            db.execute(
                condition_insert.on_conflict_do_update(
                    index_elements=[ConditionRow.condition_uuid],
                    set_={
                        key: getattr(condition_insert.excluded, key)
                        for key in (
                            "condition_name",
                            "condition_level",
                            "parameters_json",
                        )
                    },
                )
            )
            trial_values = {
                "trial_uuid": str(manifest.trial_uuid),
                "project_uuid": str(manifest.project_uuid),
                "subject_uuid": str(manifest.subject_uuid),
                "session_uuid": str(manifest.session_uuid),
                "condition_uuid": condition_id,
                "condition_code": manifest.condition.condition_code,
                "repeat_index": manifest.condition.repeat_index,
                "state": manifest.state.value,
                "quality_grade": quality_value,
                "started_utc": _utc(timing.started_at_utc),
                "stopped_utc": _utc(timing.stopped_at_utc),
                "finalized_utc": _utc(timing.finalized_at_utc),
                "duration_s": duration_s,
                "abnormal_stop": manifest.abnormal_termination.occurred,
                "manifest_path": str(path),
                "manifest_schema_version": manifest.schema_version,
                "updated_utc": now,
            }
            trial_insert = sqlite_insert(TrialRow).values(**trial_values)
            db.execute(
                trial_insert.on_conflict_do_update(
                    index_elements=[TrialRow.trial_uuid],
                    set_={
                        key: getattr(trial_insert.excluded, key)
                        for key in (
                            "state",
                            "quality_grade",
                            "stopped_utc",
                            "finalized_utc",
                            "duration_s",
                            "abnormal_stop",
                            "manifest_path",
                            "manifest_schema_version",
                            "updated_utc",
                        )
                    },
                )
            )
            for artifact in manifest.artifacts:
                artifact_values = {
                    "artifact_uuid": str(artifact.artifact_uuid),
                    "trial_uuid": str(manifest.trial_uuid),
                    "modality": artifact.modality,
                    "artifact_type": artifact.kind.value,
                    "relative_path": artifact.relative_path,
                    "media_type": artifact.media_type,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                    "immutable": artifact.immutable,
                }
                artifact_insert = sqlite_insert(ArtifactRow).values(**artifact_values)
                db.execute(
                    artifact_insert.on_conflict_do_update(
                        index_elements=[ArtifactRow.artifact_uuid],
                        set_={
                            key: getattr(artifact_insert.excluded, key)
                            for key in (
                                "modality",
                                "artifact_type",
                                "media_type",
                                "size_bytes",
                                "sha256",
                                "immutable",
                            )
                        },
                    )
                )

        self._run_write_transaction(index)

    @staticmethod
    def _ensure_fallback_hierarchy(db: Session, manifest: TrialManifest, path: Path) -> None:
        project_id = str(manifest.project_uuid)
        subject_id = str(manifest.subject_uuid)
        session_id = str(manifest.session_uuid)
        project_insert = sqlite_insert(ProjectRow).values(
            project_uuid=project_id,
            project_code=manifest.project_code or f"project-{project_id[:8]}",
            name=manifest.project_name or f"Project {project_id[:8]}",
            principal_investigator=None,
            protocol_version=manifest.condition.protocol_version,
            data_root=str(path.parents[5] if len(path.parents) > 5 else path.parent),
            created_utc=manifest.created_at_utc,
        )
        db.execute(
            project_insert.on_conflict_do_nothing(index_elements=[ProjectRow.project_uuid])
        )
        subject_insert = sqlite_insert(SubjectRow).values(
            subject_uuid=subject_id,
            project_uuid=project_id,
            subject_code=manifest.subject_code or f"subject-{subject_id[:8]}",
            group_label=None,
            attributes_json="{}",
            created_utc=manifest.created_at_utc,
        )
        db.execute(
            subject_insert.on_conflict_do_nothing(index_elements=[SubjectRow.subject_uuid])
        )
        session_insert = sqlite_insert(SessionRow).values(
            session_uuid=session_id,
            project_uuid=project_id,
            subject_uuid=subject_id,
            operator="unknown (Manifest rebuild)",
            software_version=manifest.software.application_version,
            started_utc=manifest.timing.started_at_utc,
            ended_utc=manifest.timing.stopped_at_utc,
            created_utc=manifest.created_at_utc,
        )
        db.execute(
            session_insert.on_conflict_do_nothing(index_elements=[SessionRow.session_uuid])
        )

    def scan_dataset(self, dataset_root: str | Path) -> ScanReport:
        report = ScanReport()
        for path in iter_finalized_manifest_paths(dataset_root):
            try:
                manifest = load_manifest(path)
                _validate_scanned_manifest_location(path, manifest)
                self.index_manifest(manifest, path)
                report.indexed += 1
            except Exception as exc:  # each immutable Trial is independently recoverable
                report.failures[str(path)] = f"{type(exc).__name__}: {exc}"
        return report

    @staticmethod
    def _trial_leaf_label(manifest_path: str | None) -> str:
        """Extract the human-readable trial directory name from a manifest path.

        New layout: ``.../session1_20260718_120000/.exo/manifest.json`` → ``session1_20260718_120000``
        Legacy layout: ``.../trials/<uuid>/manifest.json`` → ``<uuid[:8]>``
        """
        if not manifest_path:
            return "?"
        p = Path(manifest_path)
        if p.parent.name == ".exo":
            return p.parent.parent.name
        return p.parent.name[:8]

    @staticmethod
    def _condition_label(trials: list[TrialRow]) -> str:
        """Return the on-disk condition directory represented by ``trials``."""
        if not trials:
            return "?"
        return trials[0].condition_code or "?"

    def tree(self) -> list[dict[str, object]]:
        """Return a deterministic Project→Subject→Session→Trial→Artifact tree.

        Labels are derived from the filesystem directory names (via
        ``manifest_path``) so the tree mirrors what the operator sees in
        ``data/`` rather than internal UUIDs.
        """

        with self.catalog.session() as db:
            projects = db.scalars(select(ProjectRow).order_by(ProjectRow.name)).all()
            subjects = db.scalars(select(SubjectRow).order_by(SubjectRow.subject_code)).all()
            trials = db.scalars(select(TrialRow).order_by(TrialRow.started_utc)).all()
            artifacts = db.scalars(select(ArtifactRow).order_by(ArtifactRow.relative_path)).all()
        artifacts_by_trial: dict[str, list[ArtifactRow]] = {}
        for item in artifacts:
            artifacts_by_trial.setdefault(item.trial_uuid, []).append(item)
        trials_by_subject_condition: dict[tuple[str, str], list[TrialRow]] = {}
        for item in trials:
            trials_by_subject_condition.setdefault(
                (item.subject_uuid, item.condition_code), []
            ).append(item)
        subjects_by_project: dict[str, list[SubjectRow]] = {}
        for item in subjects:
            subjects_by_project.setdefault(item.project_uuid, []).append(item)
        return [
            {
                "type": "project",
                "uuid": project.project_uuid,
                "label": project.project_code or project.name,
                "children": [
                    {
                        "type": "subject",
                        "uuid": subject.subject_uuid,
                        "label": subject.subject_code or subject.subject_uuid[:8],
                        "children": [
                            {
                                "type": "session",
                                "uuid": condition_trials[0].condition_uuid,
                                "label": self._condition_label(condition_trials),
                                "children": [
                                    {
                                        "type": "trial",
                                        "uuid": trial.trial_uuid,
                                        "label": self._trial_leaf_label(trial.manifest_path),
                                        "state": trial.state,
                                        "quality_grade": trial.quality_grade,
                                        "duration_s": trial.duration_s,
                                        "manifest_path": trial.manifest_path,
                                        "modality_count": len(
                                            {
                                                artifact.modality
                                                for artifact in artifacts_by_trial.get(
                                                    trial.trial_uuid, []
                                                )
                                                if artifact.modality != "trial"
                                            }
                                        ),
                                        "children": [
                                            {
                                                "type": "artifact",
                                                "uuid": artifact.artifact_uuid,
                                                "label": artifact.relative_path,
                                                "modality": artifact.modality,
                                                "size_bytes": artifact.size_bytes,
                                                "sha256": artifact.sha256,
                                                "children": [],
                                            }
                                            for artifact in artifacts_by_trial.get(trial.trial_uuid, [])
                                        ],
                                    }
                                    for trial in condition_trials
                                ],
                            }
                            for (_subject_uuid, _condition), condition_trials in sorted(
                                trials_by_subject_condition.items(),
                                key=lambda item: item[0][1],
                            )
                            if _subject_uuid == subject.subject_uuid and condition_trials
                        ],
                    }
                    for subject in subjects_by_project.get(project.project_uuid, [])
                ],
            }
            for project in projects
        ]

    def statistics(self) -> dict[str, object]:
        with self.catalog.session() as db:
            total_trials = db.scalar(select(func.count()).select_from(TrialRow)) or 0
            total_duration = db.scalar(select(func.coalesce(func.sum(TrialRow.duration_s), 0.0))) or 0.0
            finalized = (
                db.scalar(
                    select(func.count()).select_from(TrialRow).where(TrialRow.state == "FINALIZED")
                )
                or 0
            )
            rows = db.execute(
                select(TrialRow.condition_code, func.count(), func.sum(TrialRow.duration_s))
                .group_by(TrialRow.condition_code)
                .order_by(TrialRow.condition_code)
            ).all()
        return {
            "trial_count": int(total_trials),
            "finalized_count": int(finalized),
            "total_duration_s": float(total_duration),
            "by_condition": {
                code: {"trial_count": int(count), "duration_s": float(duration or 0.0)}
                for code, count, duration in rows
            },
        }
