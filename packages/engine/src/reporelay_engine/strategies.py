from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from reporelay_engine.models import Repo

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    @abstractmethod
    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        ...


class ContentBasedStrategy(BaseStrategy):
    """Recommend repos with similar content (README embeddings, topics, language)."""

    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        return await _fetch_by_similarity(source_repo, limit, mode="content")


class ItemBasedCFStrategy(BaseStrategy):
    """Recommend repos co-starred / co-used with the source repo."""

    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        return await _fetch_by_similarity(source_repo, limit, mode="co_starring")


class UserBasedCFStrategy(BaseStrategy):
    """Recommend repos liked by users similar to the current user."""

    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        if user_id is None:
            return []
        return await _fetch_by_similarity(source_repo, limit, mode="user_similarity")


class ExplorationStrategy(BaseStrategy):
    """Recommend trending, random, or serendipitous repos for controlled novelty."""

    async def recommend(self, source_repo: str, user_id: str | None, limit: int) -> list[Repo]:
        return await _fetch_by_similarity(source_repo, limit, mode="trending")


async def _fetch_by_similarity(source_repo: str, limit: int, mode: str) -> list[Repo]:
    """
    Placeholder data provider. Replace with real queries against:
      - pgvector for content-based (embedding ANN)
      - Apache AGE for item-based CF (graph traversal on STARRED_BY edges)
      - Two-tower model for user-based CF (user_vector dot item_vector)
    """
    logger.info("fetching %d repos for %s via '%s'", limit, source_repo, mode)
    return []
