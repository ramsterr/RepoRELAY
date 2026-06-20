"""create mvp_repos table

The MVP uses a single, self-contained table. It does not depend on
any of the tables created by the main app's migrations. The
pgvector extension must be available.

Revision ID: mvp_001
Revises:
Create Date: 2026-06-20
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "mvp_001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "mvp_repos",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("owner", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(512), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("language", sa.String(64), nullable=True),
        sa.Column(
            "topics",
            sa.ARRAY(sa.String(128)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("stars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "dependencies",
            sa.ARRAY(sa.String(255)),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("ALTER TABLE mvp_repos ADD COLUMN embedding vector(384)")

    op.create_index("ix_mvp_repos_language", "mvp_repos", ["language"])
    op.create_index("ix_mvp_repos_topics", "mvp_repos", ["topics"], postgresql_using="gin")
    op.create_index("ix_mvp_repos_stars", "mvp_repos", [sa.text("stars DESC")])
    op.create_index("ix_mvp_repos_full_name", "mvp_repos", ["full_name"])

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_mvp_repos_embedding_hnsw
        ON mvp_repos
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_mvp_repos_embedding_hnsw")
    op.drop_index("ix_mvp_repos_full_name", "mvp_repos")
    op.drop_index("ix_mvp_repos_stars", "mvp_repos")
    op.drop_index("ix_mvp_repos_topics", "mvp_repos")
    op.drop_index("ix_mvp_repos_language", "mvp_repos")
    op.drop_table("mvp_repos")
