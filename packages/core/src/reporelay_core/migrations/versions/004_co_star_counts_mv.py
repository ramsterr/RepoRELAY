"""create co_star_counts materialized view

Revision ID: 004
Revises: 003
Create Date: 2026-06-19
"""
from collections.abc import Sequence

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS co_star_counts AS
        SELECT
            s1.repo_id AS repo_a,
            s2.repo_id AS repo_b,
            COUNT(*) AS co_star_count
        FROM star_events s1
        JOIN star_events s2
            ON s1.user_id = s2.user_id
            AND s1.repo_id != s2.repo_id
        GROUP BY s1.repo_id, s2.repo_id
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_co_star_a
        ON co_star_counts (repo_a, co_star_count DESC)
        """
    )

    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ix_co_star_pair
        ON co_star_counts (repo_a, repo_b)
        """
    )


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS co_star_counts")
    op.execute("DROP INDEX IF EXISTS ix_co_star_a")
    op.execute("DROP INDEX IF EXISTS ix_co_star_pair")
