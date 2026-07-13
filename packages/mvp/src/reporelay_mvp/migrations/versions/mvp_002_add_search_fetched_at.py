"""add search_fetched_at to mvp_repos

When a row was last populated from a GitHub search-result payload
(metadata + topics, no README, no embedding). Used to identify which
rows need a follow-up enrichment pass to embed their README.

Revision ID: mvp_002
Revises: mvp_001
Create Date: 2026-06-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "mvp_002"
down_revision: str | None = "mvp_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "mvp_repos",
        sa.Column("search_fetched_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_mvp_repos_search_fetched_at",
        "mvp_repos",
        ["search_fetched_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_mvp_repos_search_fetched_at", "mvp_repos")
    op.drop_column("mvp_repos", "search_fetched_at")
