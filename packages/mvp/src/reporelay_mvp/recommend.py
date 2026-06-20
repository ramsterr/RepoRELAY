"""
Top-level entry point for the MVP recommender.

`recommend(full_name, limit=10, seed=None)` runs the full 5-stage
pipeline against a single source repo and returns a flat ranked list.

When `seed` is set, the candidate pool is shuffled and the scoring
weights are jittered — giving different results per seed while
remaining deterministic (same seed = same results).

If the source repo is not in the DB, it is automatically fetched from
GitHub and saved. If the candidate pool is small (< 20), related repos
are discovered from GitHub and added to the pool.

`recommend_random(seed)` picks a random source repo and runs the
pipeline against it — the "surprise me / explore" feature.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from reporelay_mvp import data
from reporelay_mvp.candidates import generate_candidates
from reporelay_mvp.features import compute_features
from reporelay_mvp.github import save_repo, search_repos
from reporelay_mvp.models import Repo, ScoredRecommendation, ScoredRepo
from reporelay_mvp.rerank import rerank
from reporelay_mvp.score import score_many

logger = logging.getLogger(__name__)

POOL_MIN = 6  # threshold — expand pool if we have fewer candidates than this


def _build_scored_repo(
    source: Any,
    repo: Any,
    score: float,
    cosine_sim: float,
) -> ScoredRepo:
    features = compute_features(source, repo, cosine_sim=cosine_sim)
    source_topic_set = set(source.topics)
    source_lang = source.language

    return ScoredRepo(
        id=repo.id,
        owner=repo.owner,
        name=repo.name,
        full_name=repo.full_name,
        description=repo.description,
        language=repo.language,
        topics=repo.topics,
        stars=repo.stars,
        dependencies=repo.dependencies,
        score=round(score, 4),
        features=features.as_dict(),
        shared_topics=sorted(source_topic_set & set(repo.topics)),
        shared_language=bool(source_lang and repo.language and source_lang == repo.language),
    )


async def recommend(
    full_name: str,
    *,
    limit: int = 10,
    seed: int | None = None,
) -> ScoredRecommendation:
    if limit <= 0:
        raise ValueError("limit must be > 0")

    owner, _, name = full_name.partition("/")
    if not owner or not name:
        raise LookupError(f"repo must be 'owner/name', got {full_name!r}")

    session = await data.get_session()
    try:
        source = await data.get_repo(session, full_name)
        if source is None:
            logger.info("repo %s not in DB — fetching from GitHub", full_name)
            await save_repo(owner, name)
            # Re-fetch after save (with fresh session to see new rows)
            await session.close()
            session = await data.get_session()
            source = await data.get_repo(session, full_name)
            if source is None:
                raise LookupError(f"failed to fetch repo {full_name!r} from GitHub")

        candidates = await _expand_pool(session, source, seed=seed)

        scored = score_many(source, candidates, seed=seed)
        final = rerank(source, scored, limit=limit, seed=seed)

        scored_repos: list[ScoredRepo] = []
        for repo, sc in final:
            cosine_sim = _find_cosine(repo, candidates)
            scored_repos.append(_build_scored_repo(source, repo, sc, cosine_sim))

        return ScoredRecommendation(source_repo=full_name, repos=scored_repos)
    finally:
        await session.close()


async def _expand_pool(
    session: Any,
    source: Repo,
    *,
    seed: int | None = None,
) -> list[tuple[Repo, float]]:
    """Generate candidates, expanding via GitHub if pool is too small."""
    candidates = await generate_candidates(session, source, seed=seed)
    if len(candidates) >= POOL_MIN:
        logger.info("candidate pool: %d (no expansion needed)", len(candidates))
        return candidates

    logger.info("small pool (%d) — discovering related repos from GitHub", len(candidates))
    search_results = await search_repos(
        source.owner, source.name, limit=POOL_MIN
    )
    if not search_results:
        logger.info("no new repos found on GitHub")
        return candidates

    existing_ids: set[int] = {source.id}
    for cand, _ in candidates:
        existing_ids.add(cand.id)

    new_ids: list[int] = []
    for item in search_results:
        rid = int(item["id"])
        if rid in existing_ids:
            continue
        try:
            await save_repo(item["owner"]["login"], item["name"])
            new_ids.append(rid)
            existing_ids.add(rid)
        except Exception:
            logger.debug("failed to save discovered repo %s", item.get("full_name", "?"))

    if new_ids:
        await asyncio.sleep(0.3)

    fresh_session = await data.get_session()
    try:
        new_candidates = await generate_candidates(fresh_session, source, seed=seed)
    finally:
        await fresh_session.close()

    logger.info(
        "expanded pool: %d -> %d (saved %d new)",
        len(candidates),
        len(new_candidates),
        len(new_ids),
    )
    return new_candidates


async def recommend_random(
    *,
    seed: int,
    limit: int = 10,
) -> ScoredRecommendation:
    if limit <= 0:
        raise ValueError("limit must be > 0")

    session = await data.get_session()
    try:
        source = await data.get_random_repo(session, seed=seed)
        if source is None:
            raise LookupError("no repos in mvp_repos — save some first")

        candidates = await _expand_pool(session, source, seed=seed)

        scored = score_many(source, candidates, seed=seed)
        final = rerank(source, scored, limit=limit, seed=seed)

        scored_repos: list[ScoredRepo] = []
        for repo, sc in final:
            cosine_sim = _find_cosine(repo, candidates)
            scored_repos.append(_build_scored_repo(source, repo, sc, cosine_sim))

        return ScoredRecommendation(source_repo=source.full_name, repos=scored_repos)
    finally:
        await session.close()


def _find_cosine(repo: Any, candidates: list[tuple[Any, float]]) -> float:
    for cand, sim in candidates:
        if cand.id == repo.id:
            return sim
    return 0.5


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
