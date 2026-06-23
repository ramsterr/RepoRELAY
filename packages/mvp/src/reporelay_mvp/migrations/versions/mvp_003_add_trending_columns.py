"""add trending signal columns to mvp_repos

Stores the per-period star velocity scraped from github.com/trending.
`trending_score` is a normalized 0..1 boost used by the recommender to
surface viral repos even before they have many total stars.

Revision ID: mvp_003
Revises: mvp_002
Create Date: 2026-06-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "mvp_003"
down_revision: str | None = "mvp_002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "mvp_repos",
        sa.Column("stars_today", sa.Integer(), nullable=True),
    )
    op.add_column(
        "mvp_repos",
        sa.Column("stars_this_week", sa.Integer(), nullable=True),
    )
    op.add_column(
        "mvp_repos",
        sa.Column("stars_this_month", sa.Integer(), nullable=True),
    )
    op.add_column(
        "mvp_repos",
        sa.Column("trending_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "mvp_repos",
        sa.Column("trending_fetched_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_mvp_repos_trending_score",
        "mvp_repos",
        ["trending_score"],
    )
    op.create_index(
        "ix_mvp_repos_trending_fetched_at",
        "mvp_repos",
        ["trending_fetched_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_mvp_repos_trending_fetched_at", "mvp_repos")
    op.drop_index("ix_mvp_repos_trending_score", "mvp_repos")
    op.drop_column("mvp_repos", "trending_fetched_at")
    op.drop_column("mvp_repos", "trending_score")
    op.drop_column("mvp_repos", "stars_this_month")
    op.drop_column("mvp_repos", "stars_this_week")
    op.drop_column("mvp_repos", "stars_today")
