"""
Stage 3 of the MVP pipeline: candidate generation.

Three pools, merged and deduplicated:
  1. SQL pool: same language OR topic overlap — uses GIN + btree indexes.
  2. README-vs-desc vector pool: source's README embedding against DB's
     description_embedding column (pgvector ANN on the 11K-strong column).
  3. Description-vs-desc vector pool: source's description embedding
     against DB's description_embedding column.

The result is a pool of ~300-400 candidates ready for scoring.
"""

from __future__ import annotations

import asyncio
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
    vector_k: int = 100,
    seed: int | None = None,
    tags: list[str] | None = None,
) -> list[tuple[Repo, float]]:
    from reporelay_mvp.data import get_session, is_real_vector

    # Use SEPARATE sessions for each query so they truly run in parallel
    # via asyncio.gather. A single SQLAlchemy AsyncSession serializes
    # concurrent queries because it uses one DB connection.
    s1, s2 = await get_session(), await get_session()

    try:
        sql_coro = data.fetch_filtered_pool(
            s1, repo_id=source.id, language=source.language,
            topics=source.topics, limit=pool_size,
        )
        readme_vs_desc_coro = _vector_coro(s2, source, source.embedding, vector_k, "embedding")
        desc_vs_desc_coro = _vector_coro(session, source, source.description_embedding, vector_k, "description_embedding")

        sql_pool, readme_vs_desc_pool, desc_vs_desc_pool = await asyncio.gather(
            sql_coro, readme_vs_desc_coro, desc_vs_desc_coro,
        )
    finally:
        await s1.close()
        await s2.close()

    # Merge and deduplicate
    merged: list[tuple[Repo, float]] = []
    seen: set[int] = set()

    for pool in (readme_vs_desc_pool, desc_vs_desc_pool):
        for repo, sim in pool.values():
            if repo.id in seen:
                continue
            seen.add(repo.id)
            merged.append((repo, sim))

    for repo in sql_pool:
        if repo.id in seen:
            continue
        seen.add(repo.id)
        merged.append((repo, NEUTRAL_SIM))

    if tags:
        tag_set = {t.lower() for t in tags}
        filtered = [(repo, sim) for repo, sim in merged if tag_set & {t.lower() for t in repo.topics}]
        if filtered:
            merged = filtered
        else:
            logger.warning("tag filter eliminated all %d candidates for tags=%s", len(merged), tags)

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(merged)

    logger.info(
        "candidate pool: sql=%d readme_vs_desc=%d desc_vs_desc=%d merged=%d",
        len(sql_pool), len(readme_vs_desc_pool), len(desc_vs_desc_pool), len(merged),
    )
    return merged


async def _vector_coro(
    session: AsyncSession, source: Repo, vec: list[float] | None,
    vector_k: int, tag: str,
) -> dict[int, tuple[Repo, float]]:
    from reporelay_mvp.data import is_real_vector
    if is_real_vector(vec):
        return await data.fetch_desc_vector_neighbors(
            session, source_embedding=vec, exclude_id=source.id, limit=vector_k,
        )
    logger.info("source %s has no real %s — skipping vector pool for it", source.full_name, tag)
    return {}
