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

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from reporelay_mvp import data
from reporelay_mvp.candidates import generate_candidates, NEUTRAL_SIM
from reporelay_mvp.embedding import embed_text
from reporelay_mvp.features import compute_features
from reporelay_mvp.github import (
    _auth_client,
    _search_item_to_repo,
    enrich_repo,
    quick_save,
    save_repo,
    search_repositories,
)
from reporelay_mvp.models import Features, Repo, ScoredRecommendation, ScoredRepo
from reporelay_mvp.rerank import rerank
from reporelay_mvp.score import score_many
from reporelay_mvp.settings import get_mvp_settings

logger = logging.getLogger(__name__)

SEARCH_LIMIT = 100  # how many fresh candidates to pull from GitHub per call

# --- request-level rec cache (in-process, 10 min TTL) ---
import time as _rec_time

_rec_cache: dict[str, tuple[float, Any]] = {}
_REC_CACHE_TTL = 600  # 10 minutes
_REC_CACHE_MAX = 500  # evict oldest if exceeded


def _rec_cache_key(full_name: str, seed: int | None, tags: list[str] | None) -> str:
    tag_str = ",".join(sorted(tags or []))
    return f"{full_name}:{seed}:{tag_str}"


def _rec_cache_get(key: str, now: float) -> Any | None:
    if key not in _rec_cache:
        return None
    ts, value = _rec_cache[key]
    if now - ts >= _REC_CACHE_TTL:
        del _rec_cache[key]
        return None
    return value


def _rec_cache_set(key: str, now: float, value: Any) -> None:
    if len(_rec_cache) >= _REC_CACHE_MAX:
        oldest = min(_rec_cache, key=lambda k: _rec_cache[k][0])
        del _rec_cache[oldest]
    _rec_cache[key] = (now, value)

_cached_search_results: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 300
_DISK_CACHE_TTL = 24 * 3600  # 24h — GitHub search results don't change fast
_DISK_CACHE_PATH = Path(
    os.environ.get("REPORE_LAY_SEARCH_CACHE")
    or Path(tempfile.gettempdir()) / "reporelay_search_cache.json"
)
_MIN_DB_POOL_FOR_SKIP = 200  # if DB pool is already this big, skip the GitHub search

_SEARCH_CACHE: dict[str, dict[str, Any]] = {}


def _load_disk_cache() -> None:
    if _SEARCH_CACHE:
        return
    try:
        if _DISK_CACHE_PATH.exists():
            _SEARCH_CACHE.update(json.loads(_DISK_CACHE_PATH.read_text()))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("search cache load failed: %s", exc)


def _save_disk_cache() -> None:
    try:
        _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DISK_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_SEARCH_CACHE))
        tmp.replace(_DISK_CACHE_PATH)
    except OSError as exc:
        logger.warning("search cache save failed: %s", exc)


async def _cached_search(
    client: Any, *, topics: list[str] | None, language: str | None, **kwargs: Any
) -> dict[str, Any]:
    key = repr((language, tuple(sorted(topics or []))))
    now = time.monotonic()

    if key in _cached_search_results:
        ts, cached = _cached_search_results[key]
        if now - ts < _CACHE_TTL:
            logger.info("search cache hit (memory) for %s", key)
            return cached

    _load_disk_cache()
    disk = _SEARCH_CACHE.get(key)
    if disk and (time.time() - float(disk.get("ts", 0))) < _DISK_CACHE_TTL:
        logger.info("search cache hit (disk) for %s", key)
        result = disk["payload"]
        _cached_search_results[key] = (now, result)
        return result

    result = await search_repositories(client, topics=topics, language=language, **kwargs)
    _cached_search_results[key] = (now, result)
    _SEARCH_CACHE[key] = {"ts": time.time(), "payload": result}
    _save_disk_cache()
    return result


def _build_scored_repo(
    source: Any,
    repo: Any,
    score: float,
    cosine_sim: float,
    features: Features | None = None,
) -> ScoredRepo:
    feats = features if features is not None else compute_features(source, repo, cosine_sim=cosine_sim)
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
        features=feats.as_dict(),
        shared_topics=sorted(source_topic_set & set(repo.topics)),
        shared_language=bool(source_lang and repo.language and source_lang == repo.language),
    )


async def _find_proxy_embedding(session: Any, source: Repo) -> list[float] | None:
    """
    When a repo has no embedding, find the best-matched repo in the DB
    by topic overlap and borrow its embedding for pgvector search.

    Returns the proxy embedding or None if no good match exists.
    """
    if not source.topics:
        return None

    from sqlalchemy import text

    rows = await session.execute(
        text(
            """
            SELECT id, embedding, topics
            FROM mvp_repos
            WHERE embedding IS NOT NULL
              AND topics && :topics
            ORDER BY stars DESC
            LIMIT 20
            """
        ),
        {"topics": source.topics},
    )
    best_id = None
    best_embedding = None
    best_score = -1.0
    source_set = set(source.topics)
    for row in rows:
        cand_topics = list(row.topics or [])
        overlap = len(source_set & set(cand_topics))
        if overlap > best_score:
            best_score = overlap
            best_id = row.id
            emb_raw = row.embedding
            if isinstance(emb_raw, list):
                best_embedding = [float(x) for x in emb_raw]

    if best_embedding:
        logger.info(
            "proxy embedding from repo %d (topic overlap=%d, topics=%s)",
            best_id,
            best_score,
            source.topics,
        )
    return best_embedding


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

    cache_key = _rec_cache_key(full_name, seed, tags)
    cache_now = _rec_time.monotonic()
    cached = _rec_cache_get(cache_key, cache_now)
    if cached is not None:
        logger.info("rec cache hit for %s", cache_key)
        return cached

    session = await data.get_session()
    try:
        source = await data.get_repo(session, full_name)
        is_cold = False
        if source is None:
            logger.info("repo %s not in DB — quick-saving metadata + topics", full_name)
            await quick_save(owner, name)
            await session.close()
            session = await data.get_session()
            source = await data.get_repo(session, full_name)
            if source is None:
                raise LookupError(f"failed to fetch repo {full_name!r} from GitHub")
            is_cold = True
            # Fire background task to fetch README + dependencies
            asyncio.create_task(enrich_repo(owner, name))

        # If the source has no real embedding, borrow one from a
        # topic-similar repo in the DB. This gives instant pgvector-
        # quality results for ANY pasted URL.
        source_has_embedding = source.embedding is not None and any(v != 0.0 for v in source.embedding)
        if not source_has_embedding:
            proxy_emb = await _find_proxy_embedding(session, source)
            if proxy_emb:
                source = source.model_copy(update={"embedding": proxy_emb})
                logger.info("using proxy embedding for %s — full pgvector pipeline", full_name)
            else:
                logger.info("no proxy embedding found for %s — topic/language matching only", full_name)

        candidates = await _expand_pool(session, source, seed=seed, tags=tags)

        filter_emb = None
        if tags:
            filter_text = " ".join(tags)
            logger.info("embedding filter text: %r", filter_text)
            filter_emb = await embed_text(filter_text)

        scored = await score_many(source, candidates, session=session, seed=seed, tags=tags, filter_embedding=filter_emb)
        final = rerank(source, scored, limit=limit, seed=seed)

        cosine_lookup = _build_cosine_lookup(candidates)
        scored_repos: list[ScoredRepo] = []
        for repo, sc, features in final:
            cosine_sim = cosine_lookup.get(repo.id, 0.0)
            scored_repos.append(_build_scored_repo(source, repo, sc, cosine_sim, features=features))

        result = ScoredRecommendation(source_repo=full_name, repos=scored_repos)
        _rec_cache_set(cache_key, cache_now, result)
        return result
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

    if len(db_candidates) >= _MIN_DB_POOL_FOR_SKIP:
        logger.info(
            "db pool has %d candidates (>= %d) — skipping github search for speed",
            len(db_candidates),
            _MIN_DB_POOL_FOR_SKIP,
        )
        return db_candidates

    settings = get_mvp_settings()
    search_items: list[dict[str, Any]] = []
    try:
        async with _auth_client(settings.github_token) as client:
            payload = await _cached_search(
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

        scored = await score_many(source, candidates, session=session, seed=seed)
        final = rerank(source, scored, limit=limit, seed=seed)

        cosine_lookup = _build_cosine_lookup(candidates)
        scored_repos: list[ScoredRepo] = []
        for repo, sc, features in final:
            cosine_sim = cosine_lookup.get(repo.id, 0.0)
            scored_repos.append(_build_scored_repo(source, repo, sc, cosine_sim, features=features))

        return ScoredRecommendation(source_repo=source.full_name, repos=scored_repos)
    finally:
        await session.close()


def _build_cosine_lookup(candidates: list[tuple[Any, float]]) -> dict[int, float]:
    return {cand.id: sim for cand, sim in candidates}


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
