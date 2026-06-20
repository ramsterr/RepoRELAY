"""
FastAPI serving layer for the MVP.

Endpoints:
  GET  /health                liveness check
  GET  /recommend?repo=...    ranked recommendations with features
  GET  /explore?seed=...      surprise me — random repo + its recs
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from reporelay_mvp import recommend as recommend_fn
from reporelay_mvp import recommend_random as explore_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/recommend", response_model=RecommendResponse)
async def recommend(
    repo: str = Query(..., description="Source repo as owner/name"),
    limit: int = Query(10, ge=1, le=50),
    seed: int | None = Query(None, description="Seed for deterministic shuffle"),
    tags: str | None = Query(None, description="Comma-separated tags to filter by (e.g. react,typescript)"),
) -> RecommendResponse:
    if "/" not in repo:
        raise HTTPException(
            status_code=400, detail="repo must be in 'owner/name' format"
        )

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
        repos=[ScoredRepoOut(**r.model_dump()) for r in rec.repos],
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
        repos=[ScoredRepoOut(**r.model_dump()) for r in rec.repos],
    )
