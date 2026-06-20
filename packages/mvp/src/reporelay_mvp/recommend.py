"""
Top-level entry point for the MVP recommender.

`recommend(full_name, limit=10, seed=None)` runs the full 5-stage
pipeline against a single source repo and returns a flat ranked list.

When `seed` is set, the candidate pool is shuffled and the scoring
weights are jittered — giving different results per seed while
remaining deterministic (same seed = same results).

If the source repo is not in the DB, it is automatically fetched from
GitHub and saved. The candidate pool is always built from two sources:
the local DB (fast, has embeddings) and a fresh GitHub search (live,
has variety). Search hits are persisted back to the DB so the corpus
grows over time.

`recommend_random(seed)` picks a random source repo and runs the
pipeline against it — the "surprise me / explore" feature.
"""

from __future__ import annotations

import logging
from typing import Any

from reporelay_mvp import data
from reporelay_mvp.candidates import generate_candidates
from reporelay_mvp.features import compute_features
from reporelay_mvp.github import (
    _auth_client,
    _search_item_to_repo,
    save_repo,
    search_repositories,
)
from reporelay_mvp.models import Repo, ScoredRecommendation, ScoredRepo
from reporelay_mvp.rerank import rerank
from reporelay_mvp.score import score_many
from reporelay_mvp.settings import get_mvp_settings

logger = logging.getLogger(__name__)

SEARCH_LIMIT = 100  # how many fresh candidates to pull from GitHub per call


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
    tags: list[str] | None = None,
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

        candidates = await _expand_pool(session, source, seed=seed, tags=tags)

        scored = score_many(source, candidates, seed=seed, tags=tags)
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
    tags: list[str] | None = None,
) -> list[tuple[Repo, float]]:
    """
    Build the candidate pool from two sources:

    1. The local DB (pgvector ANN + SQL filter) — fast, has
       embeddings for cosine sim, but only knows about rows we've
       already indexed.
    2. A live GitHub search — uses the source's topics OR'd
       together with its language, returns up to SEARCH_LIMIT
       fresh results.

    Search hits are persisted back to the DB so the corpus grows
    over time. They're scored with cosine_sim = 0 (no embedding
    yet); the other four features (language, topics, deps,
    popularity) carry the score for these.
    """
    db_candidates = await generate_candidates(session, source, seed=seed, tags=tags)
    logger.info("db pool: %d candidates", len(db_candidates))

    settings = get_mvp_settings()
    search_items: list[dict[str, Any]] = []
    try:
        async with _auth_client(settings.github_token) as client:
            payload = await search_repositories(
                client,
                topics=source.topics or None,
                language=source.language,
                min_stars=100,
                sort="stars",
                per_page=SEARCH_LIMIT,
                page=1,
            )
            search_items = list(payload.get("items", []))
    except Exception as exc:
        logger.warning("github search failed: %s — falling back to db-only pool", exc)
        return db_candidates

    if not search_items:
        logger.info("github search returned 0 items — db pool only")
        return db_candidates

    written = await data.bulk_upsert_from_search(session, search_items)
    await session.commit()
    logger.info("github search: %d items, %d upserted to db", len(search_items), written)

    candidates: list[tuple[Repo, float]] = list(db_candidates)
    seen: set[int] = {c.id for c, _ in db_candidates}
    seen.add(source.id)

    tag_set = {t.lower() for t in tags} if tags else None
    added = 0
    for item in search_items:
        repo = _search_item_to_repo(item)
        if repo.id in seen:
            continue
        if tag_set and not (tag_set & {t.lower() for t in repo.topics}):
            continue
        seen.add(repo.id)
        candidates.append((repo, 0.0))
        added += 1

    logger.info("merged pool: %d db + %d search = %d", len(db_candidates), added, len(candidates))
    return candidates


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
