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
from reporelay_mvp.features import compute_features, tag_match as _tag_match
from reporelay_mvp.models import Features, Repo

logger = logging.getLogger(__name__)

WEIGHTS: dict[str, float] = {
    "language_match":          0.08,
    "topic_overlap":           0.18,
    "cosine_sim":              0.15,
    "description_sim":         0.05,
    "description_cosine_sim":  0.15,
    "readme_topic_sim":        0.00,
    "dep_overlap":             0.12,
    "popularity_sim":          0.10,
    "trending_boost":          0.07,
    "quality_signal":          0.05,
    "language_diversity":      0.05,
}

TAG_WEIGHTS: dict[str, float] = {
    "language_match":          0.05,
    "topic_overlap":           0.15,
    "cosine_sim":              0.10,
    "description_sim":         0.05,
    "description_cosine_sim":  0.10,
    "readme_topic_sim":        0.00,
    "filter_cosine_sim":       0.25,
    "dep_overlap":             0.08,
    "popularity_sim":          0.07,
    "trending_boost":          0.05,
    "quality_signal":          0.05,
    "language_diversity":      0.05,
}


def _embedding_weights(
    weights: dict[str, float],
    has_readme_emb: bool,
    has_desc_emb: bool,
) -> dict[str, float]:
    w = dict(weights)

    if not has_readme_emb:
        moved = w.pop("cosine_sim", 0.0)
        if moved > 0:
            half = moved / 2
            w["topic_overlap"] = w.get("topic_overlap", 0.18) + half
            w["description_sim"] = w.get("description_sim", 0.05) + half

    if not has_desc_emb:
        moved = w.pop("description_cosine_sim", 0.0)
        if moved > 0:
            w["description_sim"] = w.get("description_sim", 0.0) + moved

    return w


def _readme_weights(weights: dict[str, float], has_readme: bool) -> dict[str, float]:
    if not has_readme:
        return weights
    w = dict(weights)
    w["readme_topic_sim"] = 0.15
    w["topic_overlap"] = w.get("topic_overlap", 0.18) - 0.15
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
    w["popularity_sim"] *= 3.0
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
    session: AsyncSession,
    seed: int | None = None,
    tags: list[str] | None = None,
    filter_embedding: list[float] | None = None,
    source_readme_tokens: set[str] | None = None,
) -> list[tuple[Repo, float, Features]]:
    """
    Score all candidates against the source repo.

    When `filter_embedding` is provided (embedding of the tag filter
    text), the batch fetch of candidate embeddings is done and
    filter_cosine_sim is computed per candidate. This gives semantic
    tag filtering — the tag text becomes a vector query.

    When `source_readme_tokens` is provided, a readme_topic_sim
    feature is computed per candidate by Jaccard-matching the source's
    README tokens against each candidate's description tokens.

    Returns (repo, score, features) tuples for downstream use.
    """
    use_tags = bool(tags)
    rng = random.Random(seed) if seed is not None else None

    source_has_readme_emb = (
        source.embedding is not None
        and any(v != 0.0 for v in source.embedding)
    )
    source_has_desc_emb = (
        source.description_embedding is not None
        and any(v != 0.0 for v in source.description_embedding)
    )

    embeddings: dict[int, list[float]] = {}
    fc_by_id: dict[int, float] = {}

    # Exact tag matching — works without the embedding model
    if tags:
        for cand, _ in candidates:
            fc_by_id[cand.id] = _tag_match(tags, cand.topics)

    # Description embeddings — compute cosine for source's description vs candidates
    desc_cosine_by_id: dict[int, float] = {}
    if source_has_desc_emb:
        candidate_ids = [c.id for c, _ in candidates]
        desc_embs = await data.get_description_embeddings_batch(session, candidate_ids)
        if desc_embs:
            ids_ordered = [c.id for c, _ in candidates if c.id in desc_embs]
            vecs_ordered = [desc_embs[cid] for cid in ids_ordered]
            scores = cosine_batch_one_vs_many(source.description_embedding, vecs_ordered)
            desc_cosine_by_id = dict(zip(ids_ordered, scores, strict=True))

    if filter_embedding:
        candidate_ids = [c.id for c, _ in candidates]
        if candidate_ids:
            embeddings = await data.get_embeddings_batch(session, candidate_ids)
        if embeddings:
            ids_ordered = [c.id for c, _ in candidates if c.id in embeddings]
            vecs_ordered = [embeddings[cid] for cid in ids_ordered]
            scores = cosine_batch_one_vs_many(filter_embedding, vecs_ordered)
            for cid, score in zip(ids_ordered, scores, strict=True):
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
        features = compute_features(
            source, cand, cosine_sim=cosine_sim, filter_cosine_sim=fc,
            description_cosine_sim=desc_cos, readme_topic_sim=rts,
        )
        s = score_repo(features, seed=seed, use_tags=use_tags,
                        has_readme_emb=source_has_readme_emb,
                        has_desc_emb=source_has_desc_emb,
                        has_readme_keywords=has_readme,
                        has_topics=has_topics)
        if rng is not None:
            s += rng.uniform(-0.08, 0.08)
        scored.append((cand, s, features))
    return scored
