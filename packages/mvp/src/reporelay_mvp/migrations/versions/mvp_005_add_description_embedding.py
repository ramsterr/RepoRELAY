"""add description_embedding column to mvp_repos

Stores the 384-dim sentence-transformer embedding of the repo's GitHub
description text. A separate column from the README embedding so scoring
can weight description similarity independently from README similarity.

The description embedding is computed by the embed pass (alongside the
README embedding) via the same BAAI/bge-small-en-v1.5 model.

revision id: mvp_005_add_description_embedding
"""

revision = "mvp_005"
down_revision = "mvp_004"

from collections.abc import Sequence

from alembic import op


def upgrade() -> None:
    op.execute(
        "ALTER TABLE mvp_repos ADD COLUMN IF NOT EXISTS "
        "description_embedding vector(384);"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE mvp_repos DROP COLUMN IF EXISTS description_embedding;")
