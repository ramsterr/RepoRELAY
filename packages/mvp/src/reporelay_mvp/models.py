"""
Pydantic models for the MVP.

Kept intentionally simple: a Repo, a Recommendation, and the structured
feature vector used by the scorer. No blend state, no user profile, no
lifecycle stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class Repo(BaseModel):
    id: int
    owner: str
    name: str
    full_name: str
    description: str | None = None
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    stars: int = 0
    dependencies: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    description_embedding: list[float] | None = None
    trending_score: float = 0.0


class Recommendation(BaseModel):
    source_repo: str
    repos: list[Repo]


class ScoredRepo(BaseModel):
    id: int
    owner: str
    name: str
    full_name: str
    description: str | None = None
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    stars: int = 0
    dependencies: list[str] = Field(default_factory=list)
    score: float = 0.0
    features: dict[str, float] = Field(default_factory=dict)
    shared_topics: list[str] = Field(default_factory=list)
    shared_language: bool = False


class ScoredRecommendation(BaseModel):
    source_repo: str
    repos: list[ScoredRepo]


@dataclass
class Features:
    language_match: float
    topic_overlap: float
    cosine_sim: float
    description_sim: float = 0.0
    description_cosine_sim: float = 0.0
    readme_topic_sim: float = 0.0
    dep_overlap: float = 0.0
    popularity_sim: float = 0.0
    trending_boost: float = 0.0
    filter_cosine_sim: float = 0.0
    quality_signal: float = 0.0
    language_diversity: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "language_match": self.language_match,
            "topic_overlap": self.topic_overlap,
            "cosine_sim": self.cosine_sim,
            "description_sim": self.description_sim,
            "description_cosine_sim": self.description_cosine_sim,
            "readme_topic_sim": self.readme_topic_sim,
            "dep_overlap": self.dep_overlap,
            "popularity_sim": self.popularity_sim,
            "trending_boost": self.trending_boost,
            "filter_cosine_sim": self.filter_cosine_sim,
            "quality_signal": self.quality_signal,
            "language_diversity": self.language_diversity,
        }
