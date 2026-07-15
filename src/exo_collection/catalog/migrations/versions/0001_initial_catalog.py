"""Create the first local catalog schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_catalog"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("project_uuid", sa.String(36), primary_key=True),
        sa.Column("project_code", sa.String(80), nullable=False, unique=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("principal_investigator", sa.String(200)),
        sa.Column("protocol_version", sa.String(40), nullable=False),
        sa.Column("data_root", sa.Text(), nullable=False),
        sa.Column("created_utc", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "subjects",
        sa.Column("subject_uuid", sa.String(36), primary_key=True),
        sa.Column("project_uuid", sa.String(36), sa.ForeignKey("projects.project_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("subject_code", sa.String(80), nullable=False),
        sa.Column("group_label", sa.String(100)),
        sa.Column("attributes_json", sa.Text(), nullable=False),
        sa.Column("created_utc", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_uuid", "subject_code", name="uq_subject_project_code"),
    )
    op.create_index("ix_subjects_project_uuid", "subjects", ["project_uuid"])
    op.create_table(
        "sessions",
        sa.Column("session_uuid", sa.String(36), primary_key=True),
        sa.Column("project_uuid", sa.String(36), sa.ForeignKey("projects.project_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("subject_uuid", sa.String(36), sa.ForeignKey("subjects.subject_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("operator", sa.String(120), nullable=False),
        sa.Column("software_version", sa.String(40), nullable=False),
        sa.Column("started_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_utc", sa.DateTime(timezone=True)),
        sa.Column("created_utc", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_sessions_project_uuid", "sessions", ["project_uuid"])
    op.create_index("ix_sessions_subject_uuid", "sessions", ["subject_uuid"])
    op.create_table(
        "conditions",
        sa.Column("condition_uuid", sa.String(36), primary_key=True),
        sa.Column("project_uuid", sa.String(36), sa.ForeignKey("projects.project_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("condition_code", sa.String(80), nullable=False),
        sa.Column("condition_name", sa.String(200), nullable=False),
        sa.Column("condition_level", sa.Integer()),
        sa.Column("protocol_version", sa.String(40), nullable=False),
        sa.Column("parameters_json", sa.Text(), nullable=False),
        sa.UniqueConstraint("project_uuid", "condition_code", "protocol_version", name="uq_condition_project_version"),
    )
    op.create_index("ix_conditions_project_uuid", "conditions", ["project_uuid"])
    op.create_table(
        "trials",
        sa.Column("trial_uuid", sa.String(36), primary_key=True),
        sa.Column("project_uuid", sa.String(36), sa.ForeignKey("projects.project_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("subject_uuid", sa.String(36), sa.ForeignKey("subjects.subject_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("session_uuid", sa.String(36), sa.ForeignKey("sessions.session_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("condition_uuid", sa.String(36), sa.ForeignKey("conditions.condition_uuid"), nullable=False),
        sa.Column("condition_code", sa.String(80), nullable=False),
        sa.Column("repeat_index", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(24), nullable=False),
        sa.Column("quality_grade", sa.String(24), nullable=False),
        sa.Column("started_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stopped_utc", sa.DateTime(timezone=True)),
        sa.Column("finalized_utc", sa.DateTime(timezone=True)),
        sa.Column("duration_s", sa.Float(), nullable=False),
        sa.Column("abnormal_stop", sa.Boolean(), nullable=False),
        sa.Column("manifest_path", sa.Text(), nullable=False, unique=True),
        sa.Column("manifest_schema_version", sa.String(40), nullable=False),
        sa.Column("updated_utc", sa.DateTime(timezone=True), nullable=False),
    )
    for name in ("project_uuid", "subject_uuid", "session_uuid", "condition_uuid", "condition_code", "state", "quality_grade"):
        op.create_index(f"ix_trials_{name}", "trials", [name])
    op.create_table(
        "artifacts",
        sa.Column("artifact_uuid", sa.String(36), primary_key=True),
        sa.Column("trial_uuid", sa.String(36), sa.ForeignKey("trials.trial_uuid", ondelete="CASCADE"), nullable=False),
        sa.Column("modality", sa.String(80), nullable=False),
        sa.Column("artifact_type", sa.String(80), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("media_type", sa.String(120)),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False),
        sa.UniqueConstraint("trial_uuid", "relative_path", name="uq_artifact_trial_path"),
    )
    op.create_index("ix_artifacts_trial_uuid", "artifacts", ["trial_uuid"])
    op.create_index("ix_artifacts_modality", "artifacts", ["modality"])


def downgrade() -> None:
    op.drop_table("artifacts")
    op.drop_table("trials")
    op.drop_table("conditions")
    op.drop_table("sessions")
    op.drop_table("subjects")
    op.drop_table("projects")

