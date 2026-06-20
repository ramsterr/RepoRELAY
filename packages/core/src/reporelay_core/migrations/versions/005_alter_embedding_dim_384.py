"""alter readme_texts embedding column from vector(768) to vector(384)

all-MiniLM-L6-v2 produces 384-dim embeddings, not 768.

Revision ID: 005
Revises: 004
Create Date: 2026-06-19
"""
from collections.abc import Sequence

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_readme_texts_embedding_hnsw")
    op.execute("ALTER TABLE readme_texts ALTER COLUMN embedding TYPE vector(384)")
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
    op.execute("ALTER TABLE readme_texts ALTER COLUMN embedding TYPE vector(768)")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_readme_texts_embedding_hnsw
        ON readme_texts
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200)
        """
    )
