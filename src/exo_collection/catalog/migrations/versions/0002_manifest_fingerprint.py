"""Add a cheap per-manifest file fingerprint for incremental scan skips."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_manifest_fingerprint"
down_revision = "0001_initial_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("trials", sa.Column("manifest_mtime_ns", sa.BigInteger(), nullable=True))
    op.add_column(
        "trials", sa.Column("manifest_size_bytes", sa.BigInteger(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("trials", "manifest_size_bytes")
    op.drop_column("trials", "manifest_mtime_ns")
