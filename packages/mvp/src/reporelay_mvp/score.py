"""
Stage 4 of the MVP pipeline: scoring.

Fixed weighted sum of features.  All features are in [0, 1].

Primary signals (description-embedding based):
  readme_vs_desc_cosine_sim (0.30) — source README vs DB descriptions
  description_cosine_sim      (0.20) — source desc    vs DB descriptions
  readme_topic_sim            (0.15) — source README tokens vs candidate topics

Secondary signals:
  topic_overlap               (0.10) — IDF-weighted Jaccard of topic sets
  star_ratio                  (0.08) — how close the popularity levels are
  language_diversity          (0.07) — bonus for cross-language discoveries
  dep_overlap                 (0.05) — shared dependency names
  language_match              (0.03) — same primary language
  quality_signal              (0.02) — maintenance proxy

When tags are provided, filter_cosine_sim (0.25) captures semantic tag filtering
against candidate description embeddings.
"""

from __future__ import annotations

import logging
import random

from reporelay_mvp.data import is_real_vector
from reporelay_mvp.embedding import cosine_batch_one_vs_many
from reporelay_mvp.features import compute_features
from reporelay_mvp.features import tag_match as _tag_match
from reporelay_mvp.models import Features, Repo

logger = logging.getLogger(__name__)

WEIGHTS: dict[str, float] = {
    "language_match":               0.03,
    "topic_overlap":                0.07,
    "description_cosine_sim":       0.20,
    "readme_topic_sim":             0.12,
    "readme_vs_desc_cosine_sim":    0.25,
    "keyword_match":                0.08,
    "keyword_topic_match":          0.05,
    "dep_overlap":                  0.05,
    "star_ratio":                   0.06,
    "language_diversity":           0.07,
    "quality_signal":               0.02,
}

TAG_WEIGHTS: dict[str, float] = {
    "language_match":               0.02,
    "topic_overlap":                0.05,
    "description_cosine_sim":       0.14,
    "readme_topic_sim":             0.08,
    "readme_vs_desc_cosine_sim":    0.16,
    "filter_cosine_sim":            0.28,
    "keyword_match":                0.06,
    "keyword_topic_match":          0.04,
    "dep_overlap":                  0.04,
    "star_ratio":                   0.05,
    "language_diversity":           0.06,
    "quality_signal":               0.02,
}


def _get_weights(
    seed: int | None, *,
    use_tags: bool = False,
    has_readme_emb: bool = False,
    has_desc_emb: bool = False,
    has_readme_keywords: bool = False,
    has_topics: bool = False,
) -> dict[str, float]:
    base = dict(TAG_WEIGHTS if use_tags else WEIGHTS)

    # If source has no README embedding, can't compute readme_vs_desc_cosine_sim.
    # Redistribute to description_cosine_sim, keyword_match, and readme_topic_sim.
    if not has_readme_emb:
        moved = base.pop("readme_vs_desc_cosine_sim", 0.0)
        if moved > 0:
            third = moved / 3
            base["description_cosine_sim"] = base.get("description_cosine_sim", 0.0) + third
            base["readme_topic_sim"] = base.get("readme_topic_sim", 0.0) + third
            base["keyword_topic_match"] = base.get("keyword_topic_match", 0.0) + third

    # If source has no description embedding, same treatment
    if not has_desc_emb:
        moved = base.pop("description_cosine_sim", 0.0)
        if moved > 0:
            third = moved / 3
            base["readme_vs_desc_cosine_sim"] = base.get("readme_vs_desc_cosine_sim", 0.0) + third
            base["keyword_match"] = base.get("keyword_match", 0.0) + third
            base["star_ratio"] = base.get("star_ratio", 0.0) + third

    # Boost readme_topic_sim when README tokens are available
    if has_readme_keywords:
        boost = 0.04
        base["readme_topic_sim"] = base.get("readme_topic_sim", 0.0) + boost
        base["topic_overlap"] = base.get("topic_overlap", 0.0) - boost

    # If no topics, redistribute topic_overlap weight
    if not has_topics:
        moved = base.pop("topic_overlap", 0.0)
        if moved > 0:
            base["language_diversity"] = base.get("language_diversity", 0.0) + moved / 2
            base["star_ratio"] = base.get("star_ratio", 0.0) + moved / 2

    if seed is None:
        return base

    rng = random.Random(seed)
    w = {}
    for name, v in base.items():
        jitter = 1.0 + rng.uniform(-0.10, 0.10)
        w[name] = v * jitter
    w["star_ratio"] = w.get("star_ratio", 0.0) * 1.3
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
    seed: int | None = None,
    tags: list[str] | None = None,
    filter_embedding: list[float] | None = None,
    source_readme_tokens: set[str] | None = None,
) -> list[tuple[Repo, float, Features]]:
    """Score all candidates against the source.

    All embeddings are read from the candidate Repo objects in memory
    (fetched as part of EXPECTED_COLUMNS). Zero DB roundtrips.
    """
    use_tags = bool(tags)
    rng = random.Random(seed) if seed is not None else None

    source_has_readme_emb = is_real_vector(source.embedding)
    source_has_desc_emb = is_real_vector(source.description_embedding)

    fc_by_id: dict[int, float] = {}

    if tags:
        for cand, _ in candidates:
            fc_by_id[cand.id] = _tag_match(tags, cand.topics)

    # ── Description cosine: source desc vs candidate desc ────────────
    desc_cosine_by_id: dict[int, float] = {}
    if source_has_desc_emb:
        ids_ordered, vecs_ordered = _extract_desc_vecs(candidates)
        if vecs_ordered:
            scores = cosine_batch_one_vs_many(source.description_embedding, vecs_ordered)
            desc_cosine_by_id = dict(zip(ids_ordered, scores, strict=True))

    # ── Cross-modal: source README vs candidate desc ─────────────────
    readme_vs_desc_by_id: dict[int, float] = {}
    if source_has_readme_emb and vecs_ordered:
        scores = cosine_batch_one_vs_many(source.embedding, vecs_ordered)
        readme_vs_desc_by_id = dict(zip(ids_ordered, scores, strict=True))

    # ── Tag filter: semantic matching against candidate desc ─────────
    if filter_embedding and vecs_ordered:
        scores = cosine_batch_one_vs_many(filter_embedding, vecs_ordered)
        for cid, score in zip(ids_ordered, scores, strict=True):
            fc_by_id[cid] = max(fc_by_id.get(cid, 0.0), score)

    # ── README token matching ────────────────────────────────────────
    has_readme = source_readme_tokens is not None and len(source_readme_tokens) > 0
    has_topics = source.topics is not None and len(source.topics) > 0
    if has_readme:
        from reporelay_mvp.features import readme_topic_sim as _rts
        rts_by_id: dict[int, float] = {}
        for cand, _ in candidates:
            rts_by_id[cand.id] = _rts(source_readme_tokens, cand.topics)
    else:
        rts_by_id = {}

    # ── Score each candidate ─────────────────────────────────────────
    scored: list[tuple[Repo, float, Features]] = []
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
        # Tight noise band for explore mode
        if rng is not None:
            s += rng.uniform(-0.04, 0.04)
        scored.append((cand, s, features))
    return scored


def _extract_desc_vecs(
    candidates: list[tuple[Repo, float]],
) -> tuple[list[int], list[list[float]]]:
    """Extract real description_embedding vectors from candidate objects."""
    ids = []
    vecs = []
    for cand, _ in candidates:
        if is_real_vector(cand.description_embedding):
            ids.append(cand.id)
            vecs.append(cand.description_embedding)
    return ids, vecs
