from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from reporelay_engine.engine import RecommendationEngine
from reporelay_engine.models import BlendState, Recommendation, SlotName

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Repo(BaseModel):
    id: int
    full_name: str
    description: str | None = None
    stars: int
    language: str | None = None
    topics: list[str] = Field(default_factory=list)


class SlotOut(BaseModel):
    name: SlotName
    repos: list[Repo]


class RecommendResponse(BaseModel):
    source_repo: str
    slots: list[SlotOut]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str = "0.0.0"


class FeedbackRequest(BaseModel):
    user_id: str
    repo: str
    feedback: Literal["up", "down"]


class BlendStateOut(BaseModel):
    user_id: str
    stage: str
    weight_content: float
    weight_item_cf: float
    weight_user_cf: float
    weight_exploration: float
    total_interactions: int
    feedback_quality: float


app = FastAPI(
    title="RepoRelay",
    version="0.0.0",
    description="GitHub repository recommendation engine",
)

engine = RecommendationEngine()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/recommend", response_model=RecommendResponse)
async def recommend(
    repo: str = Query(..., description="Source repo as owner/name"),
    user_id: str | None = Query(None, description="Optional user for personalization"),
    limit: int = Query(10, ge=1, le=50),
) -> RecommendResponse:
    if "/" not in repo:
        raise HTTPException(
            status_code=400,
            detail="repo must be in 'owner/name' format",
        )

    cache_key = f"{repo}:{user_id or 'anon'}:{limit}"

    try:
        from reporelay_core.cache import (
            cache_recommendations,
            get_cached_recommendations,
        )

        cached = await get_cached_recommendations(cache_key)
        if cached is not None:
            logger.info("Cache hit for %s", cache_key)
            return RecommendResponse(**cached)
    except Exception:
        logger.debug("Redis unavailable, skipping cache", exc_info=True)

    rec = await engine.recommend(repo, user_id=user_id, limit=limit)
    response = _to_response(rec)

    try:
        from reporelay_core.cache import cache_recommendations

        await cache_recommendations(
            cache_key,
            response.model_dump(mode="json"),
        )
    except Exception:
        logger.debug("Redis unavailable, skipping cache write", exc_info=True)

    return response


@app.post("/feedback", response_model=BlendStateOut)
async def record_feedback(body: FeedbackRequest) -> BlendStateOut:
    if body.feedback not in ("up", "down"):
        raise HTTPException(status_code=400, detail="feedback must be 'up' or 'down'")
    state = await engine.record_feedback(body.user_id, body.repo, body.feedback)
    result = _to_blend_out(state)

    try:
        from reporelay_core.cache import cache_blend_state

        await cache_blend_state(body.user_id, result.model_dump(mode="json"))
    except Exception:
        logger.debug("Redis unavailable, skipping blend cache", exc_info=True)

    return result


@app.get("/blend/{user_id}", response_model=BlendStateOut)
async def get_blend(user_id: str) -> BlendStateOut:
    try:
        from reporelay_core.cache import get_cached_blend

        cached = await get_cached_blend(user_id)
        if cached is not None:
            return BlendStateOut(**cached)
    except Exception:
        logger.debug("Redis unavailable, skipping blend cache read", exc_info=True)

    profile = engine.get_or_create_profile(user_id)
    return _to_blend_out(profile.blend)


def _to_response(rec: Recommendation) -> RecommendResponse:
    return RecommendResponse(
        source_repo=rec.source_repo,
        slots=[
            SlotOut(
                name=slot.name,
                repos=[
                    Repo(
                        id=r.id,
                        full_name=r.full_name,
                        description=r.description,
                        stars=r.stars,
                        language=r.language,
                        topics=r.topics,
                    )
                    for r in slot.repos
                ],
            )
            for slot in rec.slots
        ],
    )


def _to_blend_out(state: BlendState) -> BlendStateOut:
    window = state.feedback_window
    quality = window.count("up") / len(window) if window else 0.0
    return BlendStateOut(
        user_id=state.user_id,
        stage=state.current_data_stage,
        weight_content=round(state.weight_content, 2),
        weight_item_cf=round(state.weight_item_cf, 2),
        weight_user_cf=round(state.weight_user_cf, 2),
        weight_exploration=round(state.weight_exploration, 2),
        total_interactions=state.total_interactions,
        feedback_quality=round(quality, 2),
    )
