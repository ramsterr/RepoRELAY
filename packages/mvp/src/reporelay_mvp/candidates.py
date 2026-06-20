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

When `seed` is not None, the merged pool is shuffled deterministically
with `random.Random(seed)` so different seeds produce different result
orderings while the same seed always produces the same ordering.
"""

from __future__ import annotations

import logging
import random

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
    seed: int | None = None,
    tags: list[str] | None = None,
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

    # tag filtering — keep only repos that have at least one requested tag
    if tags:
        tag_set = {t.lower() for t in tags}
        filtered = [
            (repo, sim) for repo, sim in merged if tag_set & {t.lower() for t in repo.topics}
        ]
        if filtered:
            merged = filtered
            logger.info("tag filter: %d candidates after filtering by %s", len(merged), tags)
        else:
            logger.warning(
                "tag filter eliminated all %d candidates for tags=%s — returning empty pool",
                len(merged),
                tags,
            )
            return []

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(merged)

    logger.info(
        "candidate pool: sql=%d vector=%d merged=%d seed=%s tags=%s",
        len(sql_pool),
        len(vector_pool),
        len(merged),
        seed,
        tags,
    )
    return merged
