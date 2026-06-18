from __future__ import annotations

import logging

from reporelay_engine.blender import ThresholdBlender
from reporelay_engine.models import BlendState, Recommendation, Repo, Slot, SlotName, UserProfile
from reporelay_engine.strategies import (
    BaseStrategy,
    ContentBasedStrategy,
    ExplorationStrategy,
    ItemBasedCFStrategy,
    UserBasedCFStrategy,
)

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Blended recommendation engine for RepoRelay.

    Strategy:
      - Cold user (0 interactions):  content 30%  | item-CF 50%  | exploration 20%
      - Warm user (3+ interactions): content 20%  | item-CF 45%  | user-CF 25%  | exploration 10%
      - Hot user  (10+ interactions):content 10%  | item-CF 35%  | user-CF 45%  | exploration 10%

    Weights adapt per-user based on thumbs feedback.
    Never fully abandons any strategy — blending > switching.
    """

    _SLOTS: tuple[SlotName, ...] = (
        "alternatives",
        "addons",
        "companions",
        "starters",
        "trending",
        "maintainer_wanted",
    )

    def __init__(self) -> None:
        self._content_strategy: BaseStrategy = ContentBasedStrategy()
        self._item_cf_strategy: BaseStrategy = ItemBasedCFStrategy()
        self._user_cf_strategy: BaseStrategy = UserBasedCFStrategy()
        self._exploration_strategy: BaseStrategy = ExplorationStrategy()
        self._profiles: dict[str, UserProfile] = {}

    def get_or_create_profile(self, user_id: str | None) -> UserProfile:
        uid = user_id or "anonymous"
        if uid not in self._profiles:
            self._profiles[uid] = UserProfile(
                user_id=uid,
                blend=BlendState(user_id=uid),
            )
        return self._profiles[uid]

    async def recommend(
        self,
        source_repo: str,
        user_id: str | None = None,
        limit: int = 10,
    ) -> Recommendation:
        profile = self.get_or_create_profile(user_id)
        blender = ThresholdBlender(profile)
        state = blender.adapt_weights([])

        total_quota = limit * len(self._SLOTS)

        quotas = self._compute_quotas(total_quota, state)

        logger.info(
            "recommending for %s quotas=%s",
            source_repo,
            {k: v for k, v in quotas.items()},
        )

        content_repos = await self._content_strategy.recommend(
            source_repo, user_id, quotas["content"]
        )
        item_cf_repos = await self._item_cf_strategy.recommend(
            source_repo, user_id, quotas["item_cf"]
        )
        user_cf_repos = await self._user_cf_strategy.recommend(
            source_repo, user_id, quotas["user_cf"]
        )
        explore_repos = await self._exploration_strategy.recommend(
            source_repo, user_id, quotas["explore"]
        )

        pool = self._merge_and_dedup(
            content_repos, item_cf_repos, user_cf_repos, explore_repos
        )

        pool = pool[:total_quota]

        slots: list[Slot] = []
        start = 0
        for slot_name in self._SLOTS:
            chunk = pool[start : start + limit]
            slots.append(Slot(name=slot_name, repos=chunk))
            start += limit

        return Recommendation(source_repo=source_repo, slots=slots)

    async def record_feedback(
        self, user_id: str, repo_name: str, feedback: str
    ) -> BlendState:
        profile = self.get_or_create_profile(user_id)
        blender = ThresholdBlender(profile)
        state = blender.adapt_weights([feedback])

        if feedback == "up":
            profile.liked_repos.append(repo_name)
        elif feedback == "down":
            profile.disliked_repos.append(repo_name)

        logger.info(
            "feedback recorded user=%s repo=%s feedback=%s stage=%s quality=%.2f",
            user_id,
            repo_name,
            feedback,
            state.current_data_stage,
            blender._feedback_quality(),
        )

        return state

    def _compute_quotas(self, total: int, state: BlendState) -> dict[str, int]:
        quotas = {
            "content": max(0, round(total * state.weight_content)),
            "item_cf": max(0, round(total * state.weight_item_cf)),
            "user_cf": max(0, round(total * state.weight_user_cf)),
            "explore": max(0, round(total * state.weight_exploration)),
        }
        diff = total - sum(quotas.values())
        if diff != 0:
            quotas["item_cf"] += diff
        return quotas

    @staticmethod
    def _merge_and_dedup(*repo_lists: list[Repo]) -> list[Repo]:
        seen: set[int] = set()
        merged: list[Repo] = []
        for repos in repo_lists:
            for repo in repos:
                if repo.id not in seen:
                    seen.add(repo.id)
                    merged.append(repo)
        return merged
