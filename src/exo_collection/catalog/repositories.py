"""Short-transaction Catalog operations and Manifest-driven rebuilding."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import func, select

from exo_collection.domain.models import Project, Session as DomainSession, Subject
from exo_collection.storage.layout import iter_finalized_manifest_paths
from exo_collection.storage.manifest import TrialManifest, load_manifest

from .db import Catalog
from .models import ArtifactRow, ConditionRow, ProjectRow, SessionRow, SubjectRow, TrialRow


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


class CatalogRepository:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def register_hierarchy(self, project: Project, subject: Subject, visit: DomainSession) -> None:
        """Create or refresh the user-facing hierarchy before a Trial starts."""

        if subject.project_uuid != project.project_uuid:
            raise ValueError("Subject does not belong to Project")
        if visit.project_uuid != project.project_uuid or visit.subject_uuid != subject.subject_uuid:
            raise ValueError("Session hierarchy UUIDs are inconsistent")
        project_code = f"{project.project_name}-{str(project.project_uuid)[:8]}"
        with self.catalog.session() as db, db.begin():
            db.merge(
                ProjectRow(
                    project_uuid=str(project.project_uuid),
                    project_code=project_code,
                    name=project.project_name,
                    principal_investigator=project.principal_investigator,
                    protocol_version=project.protocol_version,
                    data_root=project.data_root,
                    created_utc=_utc(project.created_at_utc),
                )
            )
            db.merge(
                SubjectRow(
                    subject_uuid=str(subject.subject_uuid),
                    project_uuid=str(subject.project_uuid),
                    subject_code=subject.subject_code,
                    group_label=subject.group,
                    attributes_json=_json(subject.attributes),
                    created_utc=_utc(subject.created_at_utc),
                )
            )
            db.merge(
                SessionRow(
                    session_uuid=str(visit.session_uuid),
                    project_uuid=str(visit.project_uuid),
                    subject_uuid=str(visit.subject_uuid),
                    operator=visit.operator,
                    software_version=visit.software_version,
                    started_utc=_utc(visit.started_at_utc),
                    ended_utc=_utc(visit.ended_at_utc),
                    created_utc=_utc(visit.created_at_utc),
                )
            )

    def index_manifest(self, manifest: TrialManifest, manifest_path: str | Path) -> None:
        """Upsert one immutable Manifest and its Artifact summaries."""

        path = Path(manifest_path).expanduser().resolve()
        if any(part.endswith(".recording") for part in path.parts):
            raise ValueError("Catalog must not index an active .recording Trial")
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

        now = datetime.now(timezone.utc)
        with self.catalog.session() as db, db.begin():
            self._ensure_fallback_hierarchy(db, manifest, path)
            db.merge(
                ConditionRow(
                    condition_uuid=condition_id,
                    project_uuid=str(manifest.project_uuid),
                    condition_code=manifest.condition.condition_code,
                    condition_name=manifest.condition.condition_name,
                    condition_level=(
                        manifest.condition.condition_level
                        if isinstance(manifest.condition.condition_level, int)
                        else None
                    ),
                    protocol_version=manifest.condition.protocol_version,
                    parameters_json=_json(manifest.condition.parameters),
                )
            )
            db.merge(
                TrialRow(
                    trial_uuid=str(manifest.trial_uuid),
                    project_uuid=str(manifest.project_uuid),
                    subject_uuid=str(manifest.subject_uuid),
                    session_uuid=str(manifest.session_uuid),
                    condition_uuid=condition_id,
                    condition_code=manifest.condition.condition_code,
                    repeat_index=manifest.condition.repeat_index,
                    state=manifest.state.value,
                    quality_grade=quality_value,
                    started_utc=_utc(timing.started_at_utc),
                    stopped_utc=_utc(timing.stopped_at_utc),
                    finalized_utc=_utc(timing.finalized_at_utc),
                    duration_s=duration_s,
                    abnormal_stop=manifest.abnormal_termination.occurred,
                    manifest_path=str(path),
                    manifest_schema_version=manifest.schema_version,
                    updated_utc=now,
                )
            )
            for artifact in manifest.artifacts:
                db.merge(
                    ArtifactRow(
                        artifact_uuid=str(artifact.artifact_uuid),
                        trial_uuid=str(manifest.trial_uuid),
                        modality=artifact.modality,
                        artifact_type=artifact.kind.value,
                        relative_path=artifact.relative_path,
                        media_type=artifact.media_type,
                        size_bytes=artifact.size_bytes,
                        sha256=artifact.sha256,
                        immutable=artifact.immutable,
                    )
                )

    @staticmethod
    def _ensure_fallback_hierarchy(db: object, manifest: TrialManifest, path: Path) -> None:
        project_id = str(manifest.project_uuid)
        subject_id = str(manifest.subject_uuid)
        session_id = str(manifest.session_uuid)
        if db.get(ProjectRow, project_id) is None:
            db.add(
                ProjectRow(
                    project_uuid=project_id,
                    project_code=f"project-{project_id[:8]}",
                    name=f"Project {project_id[:8]}",
                    principal_investigator=None,
                    protocol_version=manifest.condition.protocol_version,
                    data_root=str(path.parents[5] if len(path.parents) > 5 else path.parent),
                    created_utc=manifest.created_at_utc,
                )
            )
            db.flush()
        if db.get(SubjectRow, subject_id) is None:
            db.add(
                SubjectRow(
                    subject_uuid=subject_id,
                    project_uuid=project_id,
                    subject_code=f"subject-{subject_id[:8]}",
                    group_label=None,
                    attributes_json="{}",
                    created_utc=manifest.created_at_utc,
                )
            )
            db.flush()
        if db.get(SessionRow, session_id) is None:
            db.add(
                SessionRow(
                    session_uuid=session_id,
                    project_uuid=project_id,
                    subject_uuid=subject_id,
                    operator="unknown (Manifest rebuild)",
                    software_version=manifest.software.application_version,
                    started_utc=manifest.timing.started_at_utc,
                    ended_utc=manifest.timing.stopped_at_utc,
                    created_utc=manifest.created_at_utc,
                )
            )
            db.flush()

    def scan_dataset(self, dataset_root: str | Path) -> ScanReport:
        report = ScanReport()
        for path in iter_finalized_manifest_paths(dataset_root):
            try:
                manifest = load_manifest(path)
                self.index_manifest(manifest, path)
                report.indexed += 1
            except Exception as exc:  # each immutable Trial is independently recoverable
                report.failures[str(path)] = f"{type(exc).__name__}: {exc}"
        return report

    def tree(self) -> list[dict[str, object]]:
        """Return a deterministic Project→Subject→Session→Trial→Artifact tree."""

        with self.catalog.session() as db:
            projects = db.scalars(select(ProjectRow).order_by(ProjectRow.name)).all()
            subjects = db.scalars(select(SubjectRow).order_by(SubjectRow.subject_code)).all()
            sessions = db.scalars(select(SessionRow).order_by(SessionRow.started_utc)).all()
            trials = db.scalars(select(TrialRow).order_by(TrialRow.started_utc)).all()
            artifacts = db.scalars(select(ArtifactRow).order_by(ArtifactRow.relative_path)).all()
        artifacts_by_trial: dict[str, list[ArtifactRow]] = {}
        for item in artifacts:
            artifacts_by_trial.setdefault(item.trial_uuid, []).append(item)
        trials_by_session: dict[str, list[TrialRow]] = {}
        for item in trials:
            trials_by_session.setdefault(item.session_uuid, []).append(item)
        sessions_by_subject: dict[str, list[SessionRow]] = {}
        for item in sessions:
            sessions_by_subject.setdefault(item.subject_uuid, []).append(item)
        subjects_by_project: dict[str, list[SubjectRow]] = {}
        for item in subjects:
            subjects_by_project.setdefault(item.project_uuid, []).append(item)
        return [
            {
                "type": "project",
                "uuid": project.project_uuid,
                "label": project.name,
                "children": [
                    {
                        "type": "subject",
                        "uuid": subject.subject_uuid,
                        "label": subject.subject_code,
                        "children": [
                            {
                                "type": "session",
                                "uuid": visit.session_uuid,
                                "label": f"Session {visit.session_uuid[:8]}",
                                "children": [
                                    {
                                        "type": "trial",
                                        "uuid": trial.trial_uuid,
                                        "label": (
                                            f"{trial.condition_code} · repeat {trial.repeat_index} · "
                                            f"{trial.state}"
                                        ),
                                        "state": trial.state,
                                        "quality_grade": trial.quality_grade,
                                        "duration_s": trial.duration_s,
                                        "manifest_path": trial.manifest_path,
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
                                    for trial in trials_by_session.get(visit.session_uuid, [])
                                ],
                            }
                            for visit in sessions_by_subject.get(subject.subject_uuid, [])
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

