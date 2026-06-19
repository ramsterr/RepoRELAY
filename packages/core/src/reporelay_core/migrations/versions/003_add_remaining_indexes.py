"""add partial index on recent star_events and HNSW index on readme_texts embedding

Revision ID: 003
Revises: 002
Create Date: 2026-06-19
"""
from collections.abc import Sequence

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_star_events_recent
        ON star_events (repo_id, starred_at DESC)
        WHERE starred_at > '2024-06-19 00:00:00+00'::timestamptz
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_readme_texts_embedding_hnsw
        ON readme_texts
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_readme_texts_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_star_events_recent")
