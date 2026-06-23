"""
FastAPI serving layer for the MVP.

Endpoints:
  GET  /health                liveness check
  GET  /recommend?repo=...    ranked recommendations with features
  GET  /explore?seed=...      surprise me — random repo + its recs
  GET  /popular?limit=...     top repos by stars — for the homepage
  GET  /topics?limit=...      top topics by DB frequency — for explore page
"""

from __future__ import annotations

import contextlib
import logging
import os
import time as _time_mod
from collections import defaultdict
from typing import Literal

import httpx as _httpx_mod
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from reporelay_mvp import data as mvp_data
from reporelay_mvp import recommend as recommend_fn
from reporelay_mvp import recommend_random as explore_fn
from reporelay_mvp.trending import USER_AGENT, scrape_trending

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    from reporelay_mvp.embedding import preloadModel
    from reporelay_mvp.features import load_topic_idf

    asyncio.create_task(preloadModel())

    # Load IDF weights from the topic distribution in the DB
    # so rare topics count more than common ones in scoring.
    try:
        session = await mvp_data.get_session()
        try:
            rows = await session.execute(
                text(
                    """
                    SELECT unnest(topics) AS topic, COUNT(*) AS cnt
                    FROM mvp_repos
                    GROUP BY topic
                    """
                )
            )
            topic_counts: dict[str, int] = {r.topic: r.cnt for r in rows if r.topic}
            load_topic_idf(topic_counts)
            logger.info("IDF weights loaded for %d topics", len(topic_counts))
        finally:
            await session.close()
    except Exception as exc:
        logger.warning("failed to load IDF weights (using unweighted fallback): %s", exc)

    logger.info("server ready")
    yield


class ScoredRepoOut(BaseModel):
    id: int
    full_name: str
    description: str | None = None
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    stars: int
    score: float = 0.0
    features: dict[str, float] = Field(default_factory=dict)
    shared_topics: list[str] = Field(default_factory=list)
    shared_language: bool = False


class RecommendResponse(BaseModel):
    source_repo: str
    repos: list[ScoredRepoOut]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str = "0.1.0"


app = FastAPI(
    title="RepoRelay MVP",
    version="0.1.0",
    description="Single-source GitHub repo recommender (5-stage pipeline, no graph/Redis).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.github_webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")

# per-IP rate limiting (in-process, windowed, no Redis dependency)
_rate_limit_buckets: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT_WINDOW_S = 60
_RATE_LIMIT_RULES: dict[str, int] = {
    "/recommend": 10,
    "/explore":   10,
    "/trending":   5,
    "/random":    20,
}


@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    max_req = _RATE_LIMIT_RULES.get(path)
    if max_req is None:
        return await call_next(request)

    ip = request.client.host if request.client else "0.0.0.0"
    now = _time_mod.monotonic()
    window = now - _RATE_LIMIT_WINDOW_S

    bucket = _rate_limit_buckets[ip]
    # Clean expired entries for this IP only
    if bucket and bucket[0] < window:
        bucket[:] = [t for t in bucket if t > window]

    if len(bucket) >= max_req:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded — wait before retrying",
        )

    bucket.append(now)
    return await call_next(request)

from reporelay_mvp_api.webhooks import router as webhooks_router

app.include_router(webhooks_router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


class PopularRepo(BaseModel):
    id: int
    full_name: str
    description: str | None = None
    language: str | None = None
    stars: int
    trending_score: float = 0.0


class PopularResponse(BaseModel):
    repos: list[PopularRepo]


@app.get("/popular", response_model=PopularResponse)
async def popular(
    limit: int = Query(8, ge=1, le=50),
    topic: str | None = Query(None, description="Filter repos by topic"),
) -> PopularResponse:
    """Top repos by stars — used by the homepage examples list and explore page."""
    session = await mvp_data.get_session()
    try:
        if topic:
            rows = await session.execute(
                text(
                    """
                    SELECT id, full_name, description, language, stars,
                           COALESCE(trending_score, 0) AS trending_score
                    FROM mvp_repos
                    WHERE :topic = ANY(topics)
                    ORDER BY stars DESC
                    LIMIT :limit
                    """
                ),
                {"topic": topic, "limit": limit},
            )
        else:
            rows = await session.execute(
                text(
                    """
                    SELECT id, full_name, description, language, stars,
                           COALESCE(trending_score, 0) AS trending_score
                    FROM mvp_repos
                    ORDER BY stars DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
        repos = [
            PopularRepo(
                id=r.id,
                full_name=r.full_name,
                description=r.description,
                language=r.language,
                stars=r.stars,
                trending_score=float(r.trending_score or 0.0),
            )
            for r in rows
        ]
    finally:
        await session.close()
    return PopularResponse(repos=repos)


class TopicInfo(BaseModel):
    topic: str
    count: int


class TopicsResponse(BaseModel):
    topics: list[TopicInfo]


@app.get("/topics", response_model=TopicsResponse)
async def topics(
    limit: int = Query(40, ge=1, le=200),
) -> TopicsResponse:
    """Top topics by DB frequency — used by the explore page."""
    session = await mvp_data.get_session()
    try:
        rows = await session.execute(
            text(
                """
                SELECT unnest(topics) AS topic, COUNT(*) AS cnt
                FROM mvp_repos
                GROUP BY topic
                ORDER BY cnt DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        )
        result = [TopicInfo(topic=r.topic, count=r.cnt) for r in rows if r.topic]
    finally:
        await session.close()
    return TopicsResponse(topics=result)


# trending cache: (since_key) → (fetched_at, repos)
_trending_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_TRENDING_CACHE_TTL = 7200  # 2 hours


class TrendingRepoOut(BaseModel):
    full_name: str
    description: str | None = None
    language: str | None = None
    total_stars: int
    stars_today: int = 0


class TrendingResponse(BaseModel):
    repos: list[TrendingRepoOut]


@app.get("/trending", response_model=TrendingResponse)
async def trending(
    limit: int = Query(8, ge=1, le=25),
    since: str = Query("daily", pattern="^(daily|weekly|monthly)$"),
) -> TrendingResponse:
    """
    Live trending repos from github.com/trending (scraped, not API).
    Cached for 2 hours. Falls back to DB trending_score if scrape fails.
    """
    now = _time_mod.monotonic()
    cache_key = since

    if cache_key in _trending_cache:
        ts, repos = _trending_cache[cache_key]
        if now - ts < _TRENDING_CACHE_TTL:
            return TrendingResponse(repos=repos[:limit])

    repos: list[dict[str, Any]] = []
    scraped = False
    try:
        async with _httpx_mod.AsyncClient(
            timeout=_httpx_mod.Timeout(12.0, connect=5.0)
        ) as client:
            raw = await scrape_trending(client, language="", since=since)
            repos = [
                {
                    "full_name": r.full_name,
                    "description": r.description,
                    "language": r.language,
                    "total_stars": r.total_stars,
                    "stars_today": r.stars_today,
                }
                for r in raw
            ]
            scraped = True
            _trending_cache[cache_key] = (now, repos[:])
            logger.info("trending scrape: %d repos (since=%s)", len(repos), since)
    except Exception as exc:
        logger.warning("trending scrape failed — falling back to DB: %s", exc)

    if not scraped:
        session = await mvp_data.get_session()
        try:
            rows = await session.execute(
                text(
                    """
                    SELECT full_name, description, language, stars,
                           COALESCE(trending_score, 0) AS tscore
                    FROM mvp_repos
                    WHERE trending_score IS NOT NULL AND trending_score > 0
                    ORDER BY trending_score DESC, stars DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            repos = [
                {
                    "full_name": r.full_name,
                    "description": r.description,
                    "language": r.language,
                    "total_stars": r.stars,
                    "stars_today": 0,
                }
                for r in rows
            ]
        finally:
            await session.close()
        if repos:
            _trending_cache[cache_key] = (now, list(repos))

    return TrendingResponse(
        repos=[TrendingRepoOut(**r) for r in repos[:limit]]
    )


@app.get("/random", response_model=PopularResponse)
async def random_repos(
    limit: int = Query(8, ge=1, le=30),
) -> PopularResponse:
    """Random repos from the DB — fresh picks every call."""
    session = await mvp_data.get_session()
    try:
        rows = await session.execute(
            text(
                """
                SELECT id, full_name, description, language, stars,
                       COALESCE(trending_score, 0) AS trending_score
                FROM mvp_repos
                ORDER BY RANDOM()
                LIMIT :limit
                """
            ),
            {"limit": limit},
        )
        repos = [
            PopularRepo(
                id=r.id,
                full_name=r.full_name,
                description=r.description,
                language=r.language,
                stars=r.stars,
                trending_score=float(r.trending_score or 0.0),
            )
            for r in rows
        ]
    finally:
        await session.close()
    return PopularResponse(repos=repos)


@app.get("/recommend", response_model=RecommendResponse)
async def recommend(
    repo: str = Query(..., description="Source repo as owner/name"),
    limit: int = Query(10, ge=1, le=50),
    seed: int | None = Query(None, description="Seed for deterministic shuffle"),
    tags: str | None = Query(
        None, description="Comma-separated tags to filter by (e.g. react,typescript)"
    ),
) -> RecommendResponse:
    if "/" not in repo:
        raise HTTPException(status_code=400, detail="repo must be in 'owner/name' format")

    tag_list: list[str] | None = None
    if tags:
        tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]

    try:
        rec = await recommend_fn(repo, limit=limit, seed=seed, tags=tag_list)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("recommend failed for %s", repo)
        raise HTTPException(status_code=500, detail="internal error") from exc

    return RecommendResponse(
        source_repo=rec.source_repo,
        repos=[ScoredRepoOut(**{k: v for k, v in r.model_dump().items() if k != "dependencies"}) for r in rec.repos],
    )


@app.get("/explore", response_model=RecommendResponse)
async def explore(
    seed: int = Query(..., description="Seed for deterministic random pick"),
    limit: int = Query(10, ge=1, le=50),
) -> RecommendResponse:
    try:
        rec = await explore_fn(seed=seed, limit=limit)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("explore failed")
        raise HTTPException(status_code=500, detail="internal error") from exc

    return RecommendResponse(
        source_repo=rec.source_repo,
        repos=[ScoredRepoOut(**{k: v for k, v in r.model_dump().items() if k != "dependencies"}) for r in rec.repos],
    )
