"""
Redis caching layer for RepoRelay.

Tiers:
  - repo_features:     TTL 6h
  - user_blend:        TTL 24h
  - recommendations:   TTL 5min
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from reporelay_core.settings import get_settings

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        settings = get_settings()
        _redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis


async def cache_repo_features(repo_id: int, features: dict[str, Any]) -> None:
    """Store repo feature vector in Redis with TTL=6h."""
    r = await get_redis()
    key = f"repo:features:{repo_id}"
    await r.setex(key, 21600, json.dumps(features))
    logger.debug("Cached features for repo %d", repo_id)


async def get_cached_features(repo_id: int) -> dict[str, Any] | None:
    """Get cached repo features from Redis."""
    r = await get_redis()
    key = f"repo:features:{repo_id}"
    data = await r.get(key)
    if data:
        logger.debug("Cache hit: features for repo %d", repo_id)
        return json.loads(data)
    logger.debug("Cache miss: features for repo %d", repo_id)
    return None


async def cache_blend_state(user_id: str, blend: dict[str, Any]) -> None:
    """Store user blend state in Redis with TTL=24h."""
    r = await get_redis()
    key = f"blend:{user_id}"
    await r.setex(key, 86400, json.dumps(blend))
    logger.debug("Cached blend for user %s", user_id)


async def get_cached_blend(user_id: str) -> dict[str, Any] | None:
    """Get cached user blend state from Redis."""
    r = await get_redis()
    key = f"blend:{user_id}"
    data = await r.get(key)
    if data:
        logger.debug("Cache hit: blend for user %s", user_id)
        return json.loads(data)
    logger.debug("Cache miss: blend for user %s", user_id)
    return None


async def cache_recommendations(key: str, result: dict[str, Any]) -> None:
    """Store recommendation response in Redis with TTL=5min."""
    r = await get_redis()
    cache_key = f"rec:{key}"
    await r.setex(cache_key, 300, json.dumps(result))
    logger.debug("Cached recommendation for key %s", key)


async def get_cached_recommendations(key: str) -> dict[str, Any] | None:
    """Get cached recommendation response from Redis."""
    r = await get_redis()
    cache_key = f"rec:{key}"
    data = await r.get(cache_key)
    if data:
        logger.debug("Cache hit: recommendation for key %s", key)
        return json.loads(data)
    logger.debug("Cache miss: recommendation for key %s", key)
    return None
