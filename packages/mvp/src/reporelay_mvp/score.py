"""
Stage 4 of the MVP pipeline: scoring.

A fixed weighted sum of the five features. No ML, no blender, no
lifecycle stages. The weights are documented in WEIGHTS; tweak them
in one place.

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


def _get_weights(seed: int | None) -> dict[str, float]:
    if seed is None:
        return dict(WEIGHTS)
    rng = random.Random(seed)
    w = {}
    for name, base in WEIGHTS.items():
        jitter = 1.0 + rng.uniform(-0.10, 0.10)
        w[name] = base * jitter
    # boost popularity for "cool repos" effect when exploring
    w["popularity_sim"] *= 3.0
    return w


def score_repo(features: Features, *, seed: int | None = None) -> float:
    weights = _get_weights(seed)
    total: float = 0.0
    for name, weight in weights.items():
        total += getattr(features, name) * weight
    return total


def score_many(
    source: Repo,
    candidates: list[tuple[Repo, float]],
    *,
    seed: int | None = None,
) -> list[tuple[Repo, float]]:
    """
    Score all candidates against the source repo. Each candidate is
    paired with a cosine similarity (from the ANN pool) or a neutral
    0.5 (from the SQL-only pool).

    When `seed` is set, noise is added to each score so the ranking
    varies per seed. Noise is deterministic — same seed = same noise.
    """
    rng = random.Random(seed) if seed is not None else None
    scored: list[tuple[Repo, float]] = []
    for cand, cosine_sim in candidates:
        features = compute_features(source, cand, cosine_sim=cosine_sim)
        s = score_repo(features, seed=seed)
        if rng is not None:
            s += rng.uniform(-0.08, 0.08)
        scored.append((cand, s))
    return scored
