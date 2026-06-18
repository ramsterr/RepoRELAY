from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

SlotName = Literal[
    "alternatives",
    "addons",
    "companions",
    "starters",
    "trending",
    "maintainer_wanted",
]


class Repo(BaseModel):
    id: int
    full_name: str
    description: str | None = None
    stars: int
    language: str | None = None
    topics: list[str] = Field(default_factory=list)


class Slot(BaseModel):
    name: SlotName
    repos: list[Repo]


class Recommendation(BaseModel):
    source_repo: str
    slots: list[Slot]


class StrategyName(StrEnum):
    content_based = "content_based"
    item_based_cf = "item_based_cf"
    user_based_cf = "user_based_cf"


class BlendState(BaseModel):
    user_id: str

    weight_content: float = 0.0
    weight_item_cf: float = 0.70
    weight_user_cf: float = 0.20
    weight_exploration: float = 0.10

    total_interactions: int = 0
    total_thumbs_up: int = 0
    total_thumbs_down: int = 0

    feedback_window: list[str] = Field(
        default_factory=list, description="Last 10 feedback entries: 'up', 'down'"
    )
    current_data_stage: Literal["cold", "warm", "hot"] = "cold"

    content_threshold: float = 0.3
    warm_threshold: int = 3
    hot_threshold: int = 10


class UserProfile(BaseModel):
    user_id: str
    starred_repos: list[str] = Field(default_factory=list)
    liked_repos: list[str] = Field(default_factory=list)
    disliked_repos: list[str] = Field(default_factory=list)
    blend: BlendState
