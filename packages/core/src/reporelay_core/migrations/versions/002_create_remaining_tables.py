"""create users, topics, languages, readme_texts, star_events, fork_events,
contributor_edges, dependency_edges, workflow_cooccurrence, user_blend_states,
two_hop_neighbors

Revision ID: 002
Revises: 001
Create Date: 2026-06-19
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pgvector_loaded = _check_extension("vector")
    if not pgvector_loaded:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("login", sa.String(255), nullable=False, unique=True),
        sa.Column("type", sa.String(32), nullable=False, server_default="User"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_login", "users", ["login"])

    op.create_table(
        "topics",
        sa.Column("name", sa.String(128), primary_key=True),
        sa.Column("display_name", sa.String(255), nullable=True),
    )

    op.create_table(
        "languages",
        sa.Column("name", sa.String(64), primary_key=True),
    )

    op.create_table(
        "readme_texts",
        sa.Column("repo_id", sa.BigInteger(), nullable=False, unique=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        "ALTER TABLE readme_texts ADD COLUMN embedding vector(768)"
    )
    op.create_foreign_key("fk_readme_texts_repo", "readme_texts", "repos", ["repo_id"], ["id"])

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS star_events (
            user_id BIGINT NOT NULL,
            repo_id BIGINT NOT NULL,
            starred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        ) PARTITION BY RANGE (starred_at)
        """
    )
    _create_star_partitions(2025, 1, 2026, 12)
    op.create_index("ix_star_events_user_repo", "star_events", ["user_id", "repo_id"])
    op.create_index("ix_star_events_repo_time", "star_events", ["repo_id", sa.text("starred_at DESC")])

    op.create_table(
        "fork_events",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("repo_id", sa.BigInteger(), nullable=False),
        sa.Column("forked_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_fork_events_repo", "fork_events", ["repo_id"])

    op.create_table(
        "contributor_edges",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("repo_id", sa.BigInteger(), nullable=False),
        sa.Column("commit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("first_commit_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_commit_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_primary_key("pk_contributor_edges", "contributor_edges", ["user_id", "repo_id"])
    op.create_index("ix_contributor_edges_repo", "contributor_edges", ["repo_id", "user_id"])

    op.create_table(
        "dependency_edges",
        sa.Column("repo_id", sa.BigInteger(), nullable=False),
        sa.Column("dependency_name", sa.String(255), nullable=False),
        sa.Column("ecosystem", sa.String(32), nullable=False),
        sa.Column("version_constraint", sa.String(128), nullable=True),
        sa.Column("is_dev", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("file_path", sa.String(512), nullable=True),
        sa.Column("extracted_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_dep_edges_repo", "dependency_edges", ["repo_id", "dependency_name"])
    op.create_index("ix_dep_edges_ecosystem", "dependency_edges", ["ecosystem", "dependency_name"])

    op.create_table(
        "workflow_cooccurrence",
        sa.Column("repo_a", sa.BigInteger(), nullable=False),
        sa.Column("repo_b", sa.BigInteger(), nullable=False),
        sa.Column("co_occurrence_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("workflow_file_path", sa.String(512), nullable=True),
    )
    op.create_primary_key("pk_workflow_coocc", "workflow_cooccurrence", ["repo_a", "repo_b"])

    op.create_table(
        "user_blend_states",
        sa.Column("user_id", sa.String(255), primary_key=True),
        sa.Column("weight_content", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("weight_item_cf", sa.Float(), nullable=False, server_default="0.70"),
        sa.Column("weight_user_cf", sa.Float(), nullable=False, server_default="0.20"),
        sa.Column("weight_exploration", sa.Float(), nullable=False, server_default="0.10"),
        sa.Column("current_data_stage", sa.String(16), nullable=False, server_default="'cold'"),
        sa.Column("total_interactions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_thumbs_up", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_thumbs_down", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("feedback_window", sa.ARRAY(sa.String(8)), nullable=False, server_default="{}"),
    )

    op.create_table(
        "two_hop_neighbors",
        sa.Column("source_repo_id", sa.BigInteger(), nullable=False),
        sa.Column("neighbor_repo_id", sa.BigInteger(), nullable=False),
        sa.Column("path_type", sa.String(32), nullable=False),
        sa.Column("hop_count", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("combined_weight", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_primary_key("pk_two_hop", "two_hop_neighbors", ["source_repo_id", "neighbor_repo_id", "path_type"])
    op.create_index("ix_two_hop_source", "two_hop_neighbors", ["source_repo_id", sa.text("combined_weight DESC")])


def downgrade() -> None:
    op.drop_table("two_hop_neighbors")
    op.drop_table("user_blend_states")
    op.drop_table("workflow_cooccurrence")
    op.drop_table("dependency_edges")
    op.drop_table("contributor_edges")
    op.drop_table("fork_events")
    _drop_star_partitions(2025, 1, 2026, 12)
    op.drop_table("star_events")
    op.drop_table("readme_texts")
    op.drop_table("languages")
    op.drop_table("topics")
    op.drop_table("users")


def _check_extension(name: str) -> bool:
    conn = op.get_bind()
    result = conn.execute(
        sa.text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = :name)"),
        {"name": name},
    )
    return bool(result.scalar())


def _star_partition_name(year: int, month: int) -> str:
    return f"star_events_{year}_{month:02d}"


def _create_star_partitions(from_year: int, from_month: int, to_year: int, to_month: int) -> None:
    year, month = from_year, from_month
    while (year, month) <= (to_year, to_month):
        name = _star_partition_name(year, month)
        next_month = month + 1
        next_year = year
        if next_month > 12:
            next_month = 1
            next_year += 1
        op.execute(
            f"CREATE TABLE IF NOT EXISTS {name} "
            f"PARTITION OF star_events "
            f"FOR VALUES FROM ('{year}-{month:02d}-01') TO ('{next_year}-{next_month:02d}-01')"
        )
        month = next_month
        year = next_year


def _drop_star_partitions(from_year: int, from_month: int, to_year: int, to_month: int) -> None:
    year, month = from_year, from_month
    while (year, month) <= (to_year, to_month):
        name = _star_partition_name(year, month)
        op.execute(f"DROP TABLE IF EXISTS {name}")
        month += 1
        if month > 12:
            month = 1
            year += 1
