from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from reporelay_engine.data import (
    content_based_neighbors,
    co_starred_repos,
    trending_repos,
    user_similarity_repos,
)
from reporelay_engine.models import Repo

logger = logging.getLogger(__name__)


def _dicts_to_repos(rows: list[dict]) -> list[Repo]:
    return [
        Repo(
            id=r["id"],
            full_name=r["full_name"],
            description=r.get("description"),
            stars=r.get("stars", 0),
            language=r.get("language"),
            topics=r.get("topics", []),
        )
        for r in rows
    ]


class BaseStrategy(ABC):
    @abstractmethod
    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        ...


class ContentBasedStrategy(BaseStrategy):
    """Recommend repos with similar README content via pgvector ANN."""

    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        if limit <= 0:
            return []

        cache_key = f"content:{source_repo}:{limit}"
        cached = await _cache_get(cache_key)
        if cached is not None:
            return cached

        from reporelay_engine.data import _get_session

        session = await _get_session()
        async with session:
            rows = await content_based_neighbors(session, source_repo, limit)

        repos = _dicts_to_repos(rows)
        await _cache_set(cache_key, repos, ttl=300)
        return repos


class ItemBasedCFStrategy(BaseStrategy):
    """Recommend repos co-starred with the source repo."""

    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        if limit <= 0:
            return []

        cache_key = f"itemcf:{source_repo}:{limit}"
        cached = await _cache_get(cache_key)
        if cached is not None:
            return cached

        from reporelay_engine.data import _get_session

        session = await _get_session()
        async with session:
            rows = await co_starred_repos(session, source_repo, limit)

        repos = _dicts_to_repos(rows)
        await _cache_set(cache_key, repos, ttl=300)
        return repos


class UserBasedCFStrategy(BaseStrategy):
    """Recommend repos liked by users with similar star patterns."""

    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        if limit <= 0 or user_id is None:
            return []

        cache_key = f"usercf:{user_id}:{source_repo}:{limit}"
        cached = await _cache_get(cache_key)
        if cached is not None:
            return cached

        from reporelay_engine.data import _get_session

        session = await _get_session()
        async with session:
            rows = await user_similarity_repos(session, user_id, source_repo, limit)

        repos = _dicts_to_repos(rows)
        await _cache_set(cache_key, repos, ttl=300)
        return repos


class ExplorationStrategy(BaseStrategy):
    """Recommend trending repos for controlled novelty."""

    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        if limit <= 0:
            return []

        cache_key = f"trending:{limit}"
        cached = await _cache_get(cache_key)
        if cached is not None:
            return cached

        from reporelay_engine.data import _get_session

        session = await _get_session()
        async with session:
            rows = await trending_repos(session, source_repo, limit * 2)

        repos = _dicts_to_repos(rows)
        await _cache_set(cache_key, repos, ttl=300)
        return repos


async def _cache_get(key: str) -> list[Repo] | None:
    try:
        from reporelay_core.cache import get_cached_features

        data = await get_cached_features(key)
        if data and "repos" in data:
            return [Repo(**r) for r in data["repos"]]
    except Exception:
        logger.debug("Cache read skipped for %s", key, exc_info=True)
    return None


async def _cache_set(key: str, repos: list[Repo], ttl: int) -> None:
    try:
        from reporelay_core.cache import get_redis

        import json

        r = await get_redis()
        await r.setex(
            f"repo:features:{key}",
            ttl,
            json.dumps({"repos": [repo.model_dump(mode="json") for repo in repos]}),
        )
    except Exception:
        logger.debug("Cache write skipped for %s", key, exc_info=True)
