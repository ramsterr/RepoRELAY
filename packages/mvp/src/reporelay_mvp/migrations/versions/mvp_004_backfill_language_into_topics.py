"""backfill language into topics for existing mvp_repos rows

When we seeded, we only stored the `topics` array that maintainers
explicitly set. For most repos, the language is set but not added to
topics (e.g. a Python repo with topics=['cli','tui'] but not 'python').

This migration adds the language to topics for any row where it's
missing. The application code (data.py:upsert_repo, data.py:
bulk_upsert_from_search) does this on new writes — this is a one-time
backfill for rows written before that fix.

Revision ID: mvp_004
Revises: mvp_003
Create Date: 2026-06-21
"""

from collections.abc import Sequence

from alembic import op

revision: str = "mvp_004"
down_revision: str | None = "mvp_003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE mvp_repos
        SET topics = ARRAY(SELECT DISTINCT unnest(topics || ARRAY[language]))
        WHERE language IS NOT NULL
          AND language <> ''
          AND NOT (language = ANY(topics))
        """
    )


def downgrade() -> None:
    pass
