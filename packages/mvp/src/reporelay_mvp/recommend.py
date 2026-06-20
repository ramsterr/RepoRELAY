"""
Top-level entry point for the MVP recommender.

`recommend(full_name, limit=10, seed=None)` runs the full 5-stage
pipeline against a single source repo and returns a flat ranked list.

When `seed` is set, the candidate pool is shuffled and the scoring
weights are jittered — giving different results per seed while
remaining deterministic (same seed = same results).

`recommend_random(seed)` picks a random source repo and runs the
pipeline against it — the "surprise me / explore" feature.
"""
from __future__ import annotations

import logging
from typing import Any

from reporelay_mvp import data
from reporelay_mvp.candidates import generate_candidates
from reporelay_mvp.models import Recommendation
from reporelay_mvp.rerank import rerank
from reporelay_mvp.score import score_many

logger = logging.getLogger(__name__)


async def recommend(
    full_name: str,
    *,
    limit: int = 10,
    seed: int | None = None,
) -> Recommendation:
    if limit <= 0:
        raise ValueError("limit must be > 0")

    session = await data.get_session()
    try:
        source = await data.get_repo(session, full_name)
        if source is None:
            raise LookupError(f"repo {full_name!r} not found in mvp_repos")

        candidates = await generate_candidates(session, source, seed=seed)
        logger.info(
            "recommend: source=%s candidates=%d limit=%d seed=%s",
            full_name,
            len(candidates),
            limit,
            seed,
        )

        scored = score_many(source, candidates, seed=seed)
        final = rerank(source, scored, limit=limit)
        return Recommendation(source_repo=full_name, repos=final)
    finally:
        await session.close()


async def recommend_random(
    *,
    seed: int,
    limit: int = 10,
) -> Recommendation:
    if limit <= 0:
        raise ValueError("limit must be > 0")

    session = await data.get_session()
    try:
        source = await data.get_random_repo(session, seed=seed)
        if source is None:
            raise LookupError("no repos in mvp_repos — save some first")

        candidates = await generate_candidates(session, source, seed=seed)
        logger.info(
            "explore: source=%s candidates=%d limit=%d seed=%s",
            source.full_name,
            len(candidates),
            limit,
            seed,
        )

        scored = score_many(source, candidates, seed=seed)
        final = rerank(source, scored, limit=limit)
        return Recommendation(source_repo=source.full_name, repos=final)
    finally:
        await session.close()


async def recommend_dict(
    full_name: str,
    *,
    limit: int = 10,
    seed: int | None = None,
) -> dict[str, Any]:
    rec = await recommend(full_name, limit=limit, seed=seed)
    return {
        "source_repo": rec.source_repo,
        "repos": [repo.model_dump() for repo in rec.repos],
    }
