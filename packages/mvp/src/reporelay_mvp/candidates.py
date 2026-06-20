"""
Stage 3 of the MVP pipeline: candidate generation.

Two-stage filter:
  1. SQL filter for "same language or topic overlap" — uses the GIN +
     btree indexes, fast even on a large DB.
  2. pgvector ANN on the source repo's embedding — narrows to the
     most content-similar repos.

The two sets are merged by id and de-duplicated. The result is a
small pool (~150-250 candidates) that the scorer can evaluate cheaply.
Each candidate also carries its cosine similarity (0.5 for SQL-only
hits that have no embedding-based score).
"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from reporelay_mvp import data
from reporelay_mvp.models import Repo

logger = logging.getLogger(__name__)

NEUTRAL_SIM = 0.5


async def generate_candidates(
    session: AsyncSession,
    source: Repo,
    *,
    pool_size: int = 250,
    vector_k: int = 150,
) -> list[tuple[Repo, float]]:
    sql_pool = await data.fetch_filtered_pool(
        session,
        repo_id=source.id,
        language=source.language,
        topics=source.topics,
        limit=pool_size,
    )
    vector_pool = await data.fetch_vector_neighbors(
        session,
        source_id=source.id,
        exclude_id=source.id,
        limit=vector_k,
    )

    merged: list[tuple[Repo, float]] = []
    seen: set[int] = set()
    for repo, sim in vector_pool.values():
        if repo.id in seen:
            continue
        seen.add(repo.id)
        merged.append((repo, sim))

    for repo in sql_pool:
        if repo.id in seen:
            continue
        seen.add(repo.id)
        merged.append((repo, NEUTRAL_SIM))

    logger.info(
        "candidate pool: sql=%d vector=%d merged=%d (sql-only: %d)",
        len(sql_pool),
        len(vector_pool),
        len(merged),
        sum(1 for r, s in merged if r.id not in {vid for vid, _ in vector_pool.items()}),
    )
    return merged


