"""add keywords and search_vector columns for full-text search

Adds two columns:
  - keywords TEXT[] — extracted domain keywords for ARRAY overlap queries
  - search_vector tsvector — PostgreSQL full-text search vector

Two GIN indexes:
  - ix_mvp_repos_keywords — GIN on keywords[] for fast ARRAY overlap
  - ix_mvp_repos_search_vector — GIN on search_vector for fast FTS

This enables:
  - Keyword search: WHERE keywords && ARRAY['pixel', 'game', 'physics']
  - Full-text search: WHERE search_vector @@ to_tsquery('pixel & game')
  - Phrase search: WHERE search_vector @@ phraseto_tsquery('game development')
  - Relevancy ranking: ORDER BY ts_rank(search_vector, query) DESC
  - Hybrid search: combine ts_rank + cosine distance for scoring

The keyword extraction happens at embed time (embed_pass.py),
enrich time (github.py), and can be backfilled via CLI:
    just mvp extract-keywords --limit 12000

revision id: mvp_007_add_keyword_search
"""

from alembic import op

revision = "mvp_007"
down_revision = "mvp_006"


def upgrade() -> None:
    # 1. Add keywords array column
    op.execute(
        "ALTER TABLE mvp_repos ADD COLUMN IF NOT EXISTS keywords text[] NOT NULL DEFAULT '{}';"
    )

    # 2. Add search_vector tsvector column
    op.execute(
        "ALTER TABLE mvp_repos ADD COLUMN IF NOT EXISTS search_vector tsvector;"
    )

    # 3. GIN index on keywords for fast ARRAY overlap
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_mvp_repos_keywords
        ON mvp_repos USING gin (keywords);
        """
    )

    # 4. GIN index on search_vector for full-text search
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_mvp_repos_search_vector
        ON mvp_repos USING gin (search_vector);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_mvp_repos_search_vector;")
    op.execute("DROP INDEX IF EXISTS ix_mvp_repos_keywords;")
    op.execute("ALTER TABLE mvp_repos DROP COLUMN IF EXISTS search_vector;")
    op.execute("ALTER TABLE mvp_repos DROP COLUMN IF EXISTS keywords;")
