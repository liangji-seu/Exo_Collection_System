"""SQLAlchemy mappings for the local, rebuildable metadata catalog."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ProjectRow(Base):
    __tablename__ = "projects"

    project_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_code: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    principal_investigator: Mapped[str | None] = mapped_column(String(200))
    protocol_version: Mapped[str] = mapped_column(String(40), nullable=False)
    data_root: Mapped[str] = mapped_column(Text, nullable=False)
    created_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class SubjectRow(Base):
    __tablename__ = "subjects"
    __table_args__ = (UniqueConstraint("project_uuid", "subject_code", name="uq_subject_project_code"),)

    subject_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_uuid: Mapped[str] = mapped_column(
        ForeignKey("projects.project_uuid", ondelete="CASCADE"), index=True, nullable=False
    )
    subject_code: Mapped[str] = mapped_column(String(80), nullable=False)
    group_label: Mapped[str | None] = mapped_column(String(100))
    attributes_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    project: Mapped[ProjectRow] = relationship()


class SessionRow(Base):
    __tablename__ = "sessions"

    session_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_uuid: Mapped[str] = mapped_column(
        ForeignKey("projects.project_uuid", ondelete="CASCADE"), index=True, nullable=False
    )
    subject_uuid: Mapped[str] = mapped_column(
        ForeignKey("subjects.subject_uuid", ondelete="CASCADE"), index=True, nullable=False
    )
    operator: Mapped[str] = mapped_column(String(120), nullable=False)
    software_version: Mapped[str] = mapped_column(String(40), nullable=False)
    started_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class ConditionRow(Base):
    __tablename__ = "conditions"
    __table_args__ = (
        UniqueConstraint(
            "project_uuid", "condition_code", "protocol_version", name="uq_condition_project_version"
        ),
    )

    condition_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_uuid: Mapped[str] = mapped_column(
        ForeignKey("projects.project_uuid", ondelete="CASCADE"), index=True, nullable=False
    )
    condition_code: Mapped[str] = mapped_column(String(80), nullable=False)
    condition_name: Mapped[str] = mapped_column(String(200), nullable=False)
    condition_level: Mapped[int | None] = mapped_column(Integer)
    protocol_version: Mapped[str] = mapped_column(String(40), nullable=False)
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class TrialRow(Base):
    __tablename__ = "trials"

    trial_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_uuid: Mapped[str] = mapped_column(
        ForeignKey("projects.project_uuid", ondelete="CASCADE"), index=True, nullable=False
    )
    subject_uuid: Mapped[str] = mapped_column(
        ForeignKey("subjects.subject_uuid", ondelete="CASCADE"), index=True, nullable=False
    )
    session_uuid: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_uuid", ondelete="CASCADE"), index=True, nullable=False
    )
    condition_uuid: Mapped[str] = mapped_column(
        ForeignKey("conditions.condition_uuid"), index=True, nullable=False
    )
    condition_code: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    repeat_index: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(24), index=True, nullable=False)
    quality_grade: Mapped[str] = mapped_column(String(24), index=True, nullable=False)
    started_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    stopped_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finalized_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_s: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    abnormal_stop: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    manifest_path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    manifest_schema_version: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class ArtifactRow(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("trial_uuid", "relative_path", name="uq_artifact_trial_path"),)

    artifact_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    trial_uuid: Mapped[str] = mapped_column(
        ForeignKey("trials.trial_uuid", ondelete="CASCADE"), index=True, nullable=False
    )
    modality: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(80), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    immutable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

