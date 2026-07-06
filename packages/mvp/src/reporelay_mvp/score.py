"""
Stage 4 of the MVP pipeline: scoring.

A fixed weighted sum of the features. No ML, no blender, no
lifecycle stages. The weights are documented in WEIGHTS; tweak them
in one place.

When `tags` are provided, a new filter_cosine_sim feature is added:
the filter text ("machine learning free courses") is embedded at query
time and compared against every candidate's README embedding. This
gives semantic matching — repos about ML education score high even
if they don't have that exact tag. The feature captures the 35%
weight, with topic_overlap at 25% as the secondary signal.

When `seed` is not None, each weight is jittered by +/- 10% and the
popularity_sim weight is boosted by 3x (to surface "cooler" repos).
The jitter is deterministic — same seed = same weights.
"""

from __future__ import annotations

import logging
import random

from sqlalchemy.ext.asyncio import AsyncSession

from reporelay_mvp import data
from reporelay_mvp.embedding import cosine_batch_one_vs_many
from reporelay_mvp.features import compute_features
from reporelay_mvp.features import tag_match as _tag_match
from reporelay_mvp.models import Features, Repo

logger = logging.getLogger(__name__)

WEIGHTS: dict[str, float] = {
    "language_match":          0.03,
    "topic_overlap":           0.10,
    "cosine_sim":              0.16,
    "description_sim":         0.03,
    "description_cosine_sim":  0.13,
    "readme_topic_sim":        0.15,
    "readme_vs_desc_cosine_sim": 0.13,
    "dep_overlap":             0.06,
    "popularity_sim":          0.03,
    "star_ratio":              0.04,
    "trending_boost":          0.02,
    "quality_signal":          0.03,
    "language_diversity":      0.09,
}

TAG_WEIGHTS: dict[str, float] = {
    "language_match":          0.02,
    "topic_overlap":           0.06,
    "cosine_sim":              0.11,
    "description_sim":         0.02,
    "description_cosine_sim":  0.09,
    "readme_topic_sim":        0.11,
    "readme_vs_desc_cosine_sim": 0.09,
    "filter_cosine_sim":       0.25,
    "dep_overlap":             0.05,
    "popularity_sim":          0.02,
    "star_ratio":              0.03,
    "trending_boost":          0.02,
    "quality_signal":          0.02,
    "language_diversity":      0.11,
}


def _embedding_weights(
    weights: dict[str, float],
    has_readme_emb: bool,
    has_desc_emb: bool,
) -> dict[str, float]:
    w = dict(weights)

    if not has_readme_emb:
        # Source has no README embedding — can't compute cosine_sim or
        # readme_vs_desc_cosine_sim. Redistribute their weights to the
        # features that still work.
        for key in ("cosine_sim", "readme_vs_desc_cosine_sim"):
            moved = w.pop(key, 0.0)
            if moved > 0:
                half = moved / 2
                w["topic_overlap"] = w.get("topic_overlap", 0.0) + half
                w["readme_topic_sim"] = w.get("readme_topic_sim", 0.0) + half

    if not has_desc_emb:
        moved = w.pop("description_cosine_sim", 0.0)
        if moved > 0:
            half = moved / 2
            w["description_sim"] = w.get("description_sim", 0.0) + half
            w["readme_topic_sim"] = w.get("readme_topic_sim", 0.0) + half

    return w


def _readme_weights(weights: dict[str, float], has_readme: bool) -> dict[str, float]:
    if not has_readme:
        return weights
    w = dict(weights)
    # When the source README is available, boost readme_topic_sim by
    # borrowing from topic_overlap. README tokens are a strong cross-
    # language signal — they match keywords like "chess" against
    # candidate topic tags even when source and candidate have no
    # shared topic.
    boost = 0.05
    w["readme_topic_sim"] = w.get("readme_topic_sim", 0.0) + boost
    w["topic_overlap"] = w.get("topic_overlap", 0.13) - boost
    return w


def _topicless_weights(weights: dict[str, float], has_topics: bool, has_readme: bool) -> dict[str, float]:
    if has_topics:
        return weights
    w = dict(weights)
    moved = w.pop("topic_overlap", 0.0)
    if moved <= 0:
        return w
    half = moved / 2
    w["description_sim"] = w.get("description_sim", 0.0) + half
    if has_readme and "readme_topic_sim" in w:
        w["readme_topic_sim"] = w.get("readme_topic_sim", 0.0) + half
    else:
        w["description_sim"] = w.get("description_sim", half) + half
    return w


def _get_weights(
    seed: int | None, *,
    use_tags: bool = False,
    has_readme_emb: bool = False,
    has_desc_emb: bool = False,
    has_readme_keywords: bool = False,
    has_topics: bool = False,
) -> dict[str, float]:
    base = dict(TAG_WEIGHTS if use_tags else WEIGHTS)
    base = _embedding_weights(base, has_readme_emb, has_desc_emb)
    base = _readme_weights(base, has_readme_keywords)
    base = _topicless_weights(base, has_topics, has_readme_keywords)
    if seed is None:
        return base
    rng = random.Random(seed)
    w = {}
    for name, v in base.items():
        jitter = 1.0 + rng.uniform(-0.10, 0.10)
        w[name] = v * jitter
    # The old behavior tripled popularity_sim when a seed was set,
    # which actively hurt domain relevance. Now we just apply a mild
    # boost (1.3x) so explore mode surfaces a wider range of repos
    # without drowning out semantic signals.
    w["popularity_sim"] = w.get("popularity_sim", 0.0) * 1.3
    return w


def score_repo(
    features: Features, *, seed: int | None = None, use_tags: bool = False,
    has_readme_emb: bool = False, has_desc_emb: bool = False,
    has_readme_keywords: bool = False, has_topics: bool = False,
) -> float:
    weights = _get_weights(seed, use_tags=use_tags,
                           has_readme_emb=has_readme_emb,
                           has_desc_emb=has_desc_emb,
                           has_readme_keywords=has_readme_keywords,
                           has_topics=has_topics)
    total: float = 0.0
    for name, weight in weights.items():
        total += getattr(features, name) * weight
    return total


async def score_many(
    source: Repo,
    candidates: list[tuple[Repo, float]],
    *,
    session: AsyncSession | None = None,
    seed: int | None = None,
    tags: list[str] | None = None,
    filter_embedding: list[float] | None = None,
    source_readme_tokens: set[str] | None = None,
) -> list[tuple[Repo, float, Features]]:
    """
    Score all candidates against the source repo.

    All candidate embeddings (description_embedding, embedding) are read
    from the candidate Repo objects already in memory — no DB roundtrips.
    This is critical for remote databases (Neon) where each query costs
    0.5-2s of network latency.
    """
    from reporelay_mvp.data import is_real_vector

    use_tags = bool(tags)
    rng = random.Random(seed) if seed is not None else None

    source_has_readme_emb = is_real_vector(source.embedding)
    source_has_desc_emb = is_real_vector(source.description_embedding)

    embeddings: dict[int, list[float]] = {}
    fc_by_id: dict[int, float] = {}

    # Exact tag matching — works without the embedding model
    if tags:
        for cand, _ in candidates:
            fc_by_id[cand.id] = _tag_match(tags, cand.topics)

    # Description embeddings — compute cosine for source's description vs candidates.
    # The candidates already have description_embedding in memory (fetched as
    # part of EXPECTED_COLUMNS in the candidate query). No need for a second
    # DB roundtrip.
    desc_cosine_by_id: dict[int, float] = {}
    if source_has_desc_emb:
        ids_ordered = []
        vecs_ordered = []
        for cand, _ in candidates:
            if is_real_vector(cand.description_embedding):
                ids_ordered.append(cand.id)
                vecs_ordered.append(cand.description_embedding)
        if vecs_ordered:
            scores = cosine_batch_one_vs_many(source.description_embedding, vecs_ordered)
            desc_cosine_by_id = dict(zip(ids_ordered, scores, strict=True))

    # Cross-modal: source README vs candidate descriptions.
    # Reuses the same in-memory vectors — no extra DB call.
    readme_vs_desc_by_id: dict[int, float] = {}
    if source_has_readme_emb and vecs_ordered:
        scores = cosine_batch_one_vs_many(source.embedding, vecs_ordered)
        readme_vs_desc_by_id = dict(zip(ids_ordered, scores, strict=True))

    if filter_embedding:
        # For the tag filter, we need README embeddings (the `embedding` column).
        # These are also already in the candidate objects.
        filter_ids = []
        filter_vecs = []
        for cand, _ in candidates:
            if is_real_vector(cand.embedding):
                filter_ids.append(cand.id)
                filter_vecs.append(cand.embedding)
        if filter_vecs:
            scores = cosine_batch_one_vs_many(filter_embedding, filter_vecs)
            for cid, score in zip(filter_ids, scores, strict=True):
                fc_by_id[cid] = max(fc_by_id.get(cid, 0.0), score)
        else:
            logger.info("no candidate embeddings for semantic tag filter — falling back to topic overlap")

    scored: list[tuple[Repo, float, Features]] = []
    has_readme = source_readme_tokens is not None and len(source_readme_tokens) > 0
    has_topics = source.topics is not None and len(source.topics) > 0
    if has_readme:
        from reporelay_mvp.features import readme_topic_sim as _rts
        rts_by_id: dict[int, float] = {}
        for cand, _ in candidates:
            rts_by_id[cand.id] = _rts(source_readme_tokens, cand.topics)
    else:
        rts_by_id = {}

    for cand, cosine_sim in candidates:
        fc = fc_by_id.get(cand.id, 0.0)
        desc_cos = desc_cosine_by_id.get(cand.id, 0.0)
        rts = rts_by_id.get(cand.id, 0.0) if has_readme else 0.0
        rvd = readme_vs_desc_by_id.get(cand.id, 0.0)
        features = compute_features(
            source, cand, cosine_sim=cosine_sim, filter_cosine_sim=fc,
            description_cosine_sim=desc_cos, readme_topic_sim=rts,
            readme_vs_desc_cosine_sim=rvd,
        )
        s = score_repo(features, seed=seed, use_tags=use_tags,
                        has_readme_emb=source_has_readme_emb,
                        has_desc_emb=source_has_desc_emb,
                        has_readme_keywords=has_readme,
                        has_topics=has_topics)
        if rng is not None:
            # Tight noise band — ±0.04 gives meaningful variation for
            # "explore" mode without reordering the top 10. The old
            # ±0.08 was too aggressive and could swap position-1 with
            # position-10 candidates.
            s += rng.uniform(-0.04, 0.04)
        scored.append((cand, s, features))
    return scored
