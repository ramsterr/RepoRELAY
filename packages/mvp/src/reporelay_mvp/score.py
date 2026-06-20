"""
Stage 4 of the MVP pipeline: scoring.

A fixed weighted sum of the five features. No ML, no blender, no
lifecycle stages. The weights are documented in WEIGHTS; tweak them
in one place.
"""
from __future__ import annotations

from reporelay_mvp.features import compute_features
from reporelay_mvp.models import Features, Repo

WEIGHTS: dict[str, float] = {
    "language_match": 0.30,
    "topic_overlap": 0.30,
    "cosine_sim": 0.20,
    "dep_overlap": 0.15,
    "popularity_sim": 0.05,
}


def score_repo(features: Features) -> float:
    total: float = 0.0
    for name, weight in WEIGHTS.items():
        total += getattr(features, name) * weight
    return total


def score_many(
    source: Repo, candidates: list[tuple[Repo, float]]
) -> list[tuple[Repo, float]]:
    """
    Score all candidates against the source repo. Each candidate is
    paired with a cosine similarity (from the ANN pool) or a neutral
    0.5 (from the SQL-only pool).
    """
    scored: list[tuple[Repo, float]] = []
    for cand, cosine_sim in candidates:
        features = compute_features(source, cand, cosine_sim=cosine_sim)
        scored.append((cand, score_repo(features)))
    return scored
