"""change embedding dimensions from 384 to 512

Voyage voyage-code-3 (the new embedding model) supports 256, 512, 1024,
and 2048 dims. 512 is the sweet spot for README/description text and
keeps storage ~67 MB total for 11k repos (fits comfortably in Neon's
512 MB free tier).

voyage-code-3 is purpose-built for code retrieval and outperforms the
previous local BAAI/bge-small-en-v1.5 (384 dims) on code tasks.

This migration:
  1. Drops the old HNSW index on embedding (dim change requires rebuild)
  2. Alters the embedding column type to vector(512)
  3. Alters the description_embedding column type to vector(512)
  4. Recreates the HNSW index on the new dim

All existing vectors will be invalid after the dim change. Re-embed
all repos via:
    just mvp embed --limit 11000

revision id: mvp_006_change_embedding_dim_to_512
"""

from alembic import op

revision = "mvp_006"
down_revision = "mvp_005"


def upgrade() -> None:
    # Drop the HNSW index — must be done before column type change
    op.execute("DROP INDEX IF EXISTS ix_mvp_repos_embedding_hnsw;")

    # Null out existing 384-dim vectors (they're incompatible with 512-dim space)
    # The embed pass will repopulate these with new 512-dim vectors.
    op.execute("UPDATE mvp_repos SET embedding = NULL;")
    op.execute("UPDATE mvp_repos SET description_embedding = NULL;")

    # Change column types
    op.execute("ALTER TABLE mvp_repos ALTER COLUMN embedding TYPE vector(512);")
    op.execute(
        "ALTER TABLE mvp_repos ALTER COLUMN description_embedding TYPE vector(512);"
    )

    # Recreate HNSW index with new dim
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_mvp_repos_embedding_hnsw
        ON mvp_repos
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_mvp_repos_embedding_hnsw;")
    op.execute("UPDATE mvp_repos SET embedding = NULL;")
    op.execute("UPDATE mvp_repos SET description_embedding = NULL;")
    op.execute("ALTER TABLE mvp_repos ALTER COLUMN embedding TYPE vector(384);")
    op.execute(
        "ALTER TABLE mvp_repos ALTER COLUMN description_embedding TYPE vector(384);"
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_mvp_repos_embedding_hnsw
        ON mvp_repos
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
        """
    )
