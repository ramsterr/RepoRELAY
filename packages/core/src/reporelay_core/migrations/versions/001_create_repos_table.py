"""create repos table

Revision ID: 001
Revises:
Create Date: 2026-06-19
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "repos",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(512), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("homepage", sa.String(1024), nullable=True),
        sa.Column("language", sa.String(64), nullable=True),
        sa.Column("license", sa.String(128), nullable=True),
        sa.Column("stars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("forks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("topics", sa.ARRAY(sa.String(128)), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pushed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("is_template", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("default_branch", sa.String(255), nullable=True),
        sa.Column("raw_metadata", sa.JSON(), nullable=True),
    )

    op.create_index("ix_repos_language", "repos", ["language"])
    op.create_index("ix_repos_topics", "repos", ["topics"], postgresql_using="gin")
    op.create_index("ix_repos_stars", "repos", [sa.text("stars DESC")])
    op.create_index("ix_repos_full_name", "repos", ["full_name"])


def downgrade() -> None:
    op.drop_index("ix_repos_full_name")
    op.drop_index("ix_repos_stars")
    op.drop_index("ix_repos_topics")
    op.drop_index("ix_repos_language")
    op.drop_table("repos")
