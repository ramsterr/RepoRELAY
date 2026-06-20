"""
Stage 4 of the MVP pipeline: scoring.

A fixed weighted sum of the five features. No ML, no blender, no
lifecycle stages. The weights are documented in WEIGHTS; tweak them
in one place.

When `tags` are provided, topic_overlap gets a 2x boost and
language_match is reduced, so the results are driven by tag matching.

When `seed` is not None, each weight is jittered by ±10% and the
popularity_sim weight is boosted by 3x (to surface "cooler" repos).
The jitter is deterministic — same seed = same weights.
"""
from __future__ import annotations

import random

from reporelay_mvp.features import compute_features
from reporelay_mvp.models import Features, Repo

WEIGHTS: dict[str, float] = {
    "language_match": 0.30,
    "topic_overlap": 0.30,
    "cosine_sim": 0.20,
    "dep_overlap": 0.15,
    "popularity_sim": 0.05,
}

TAG_WEIGHTS: dict[str, float] = {
    "language_match": 0.15,
    "topic_overlap": 0.50,
    "cosine_sim": 0.15,
    "dep_overlap": 0.10,
    "popularity_sim": 0.10,
}


def _get_weights(seed: int | None, *, use_tags: bool = False) -> dict[str, float]:
    base = dict(TAG_WEIGHTS if use_tags else WEIGHTS)
    if seed is None:
        return base
    rng = random.Random(seed)
    w = {}
    for name, v in base.items():
        jitter = 1.0 + rng.uniform(-0.10, 0.10)
        w[name] = v * jitter
    # boost popularity for "cool repos" effect when exploring
    w["popularity_sim"] *= 3.0
    return w


def score_repo(features: Features, *, seed: int | None = None, use_tags: bool = False) -> float:
    weights = _get_weights(seed, use_tags=use_tags)
    total: float = 0.0
    for name, weight in weights.items():
        total += getattr(features, name) * weight
    return total


def score_many(
    source: Repo,
    candidates: list[tuple[Repo, float]],
    *,
    seed: int | None = None,
    tags: list[str] | None = None,
) -> list[tuple[Repo, float]]:
    """
    Score all candidates against the source repo. Each candidate is
    paired with a cosine similarity (from the ANN pool) or a neutral
    0.5 (from the SQL-only pool).

    When `tags` are provided, topic_overlap gets a 2x weight boost so
    results are driven by tag alignment. When `seed` is set, noise is
    added to each score. Both are deterministic — same inputs = same scores.
    """
    use_tags = bool(tags)
    rng = random.Random(seed) if seed is not None else None
    scored: list[tuple[Repo, float]] = []
    for cand, cosine_sim in candidates:
        features = compute_features(source, cand, cosine_sim=cosine_sim)
        s = score_repo(features, seed=seed, use_tags=use_tags)
        if rng is not None:
            s += rng.uniform(-0.08, 0.08)
        scored.append((cand, s))
    return scored
