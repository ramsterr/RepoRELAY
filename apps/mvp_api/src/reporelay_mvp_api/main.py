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
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from reporelay_mvp import data as mvp_data
from reporelay_mvp import recommend as recommend_fn
from reporelay_mvp import recommend_random as explore_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    from reporelay_mvp.embedding import preloadModel

    asyncio.create_task(preloadModel())
    logger.info("model preloading in background — server ready")
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
