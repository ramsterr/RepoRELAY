"""
Relevance feedback: "Show me more repos like these selected ones."

When a user picks repos from recommendations and clicks "Show more
like these", we:
  1. Merge them into a virtual source → find repos aligned with ALL picks
  2. Run each picked repo individually → show each repo's specific recs

The response has three sections:
  - merged: repos matching all picked repos (centroid in vector space)
  - picked: per-repo recommendations (one per picked repo, sub-categorized)

This lets the user see both the intersection and the per-repo specifics
in one response.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter

from reporelay_mvp.models import Repo
from reporelay_mvp.recommend import recommend as recommend_fn
from reporelay_mvp.embedding import embed_text as _embed_text

logger = logging.getLogger(__name__)


def merge_sources(sources: list[Repo]) -> Repo:
    """Merge multiple repos into a virtual source Repo by averaging."""
    if not sources:
        raise ValueError("At least one source repo is required")

    n = len(sources)

    langs = [s.language for s in sources if s.language]
    language = Counter(langs).most_common(1)[0][0] if langs else None

    all_topics: list[str] = []
    for s in sources:
        all_topics.extend(s.topics)
    topics = list(dict.fromkeys(all_topics))

    all_deps: list[str] = []
    for s in sources:
        all_deps.extend(s.dependencies)
    deps = list(dict.fromkeys(all_deps))

    avg_stars = sum(s.stars for s in sources) // n

    embeddings = [s.embedding for s in sources if s.embedding and any(v != 0.0 for v in s.embedding)]
    merged_embedding = None
    if embeddings:
        merged_embedding = [sum(vec[i] for vec in embeddings) / len(embeddings) for i in range(len(embeddings[0]))]

    desc_embs = [s.description_embedding for s in sources if s.description_embedding and any(v != 0.0 for v in s.description_embedding)]
    merged_desc_emb = None
    if desc_embs:
        merged_desc_emb = [sum(vec[i] for vec in desc_embs) / len(desc_embs) for i in range(len(desc_embs[0]))]

    description = next((s.description for s in sources if s.description), None)
    avg_trending = sum(s.trending_score for s in sources) / n

    virtual = Repo(
        id=0, owner="virtual", name="merged", full_name="virtual/merged",
        description=description, language=language, topics=topics,
        stars=avg_stars, dependencies=deps,
        embedding=merged_embedding, description_embedding=merged_desc_emb,
        trending_score=avg_trending,
    )
    return virtual


async def more_like_these(
    picked: list[str],
    *,
    limit: int = 10,
    seed: int | None = None,
):
    """Get merged + per-repo recommendations for user-selected repos.

    Returns a dict with:
      - source_repo: "A + B"
      - merged: recommendations for the virtual merged source
      - picked: list of {repo: ..., groups: ...} per picked repo
    """
    if not picked:
        raise ValueError("At least one repo must be selected")

    from reporelay_mvp import data

    # Fetch all picked repos from DB
    sources: list[Repo] = []
    session = await data.get_session()
    try:
        for full_name in picked:
            source = await data.get_repo(session, full_name)
            if source:
                sources.append(source)
            else:
                logger.warning("picked repo %s not in DB — skipping", full_name)
    finally:
        await session.close()

    if not sources:
        raise ValueError("None of the picked repos were found in the DB")

    # 1. Merged: virtual source from all picked repos
    virtual = merge_sources(sources)
    merged_result = await recommend_fn(
        "virtual/merged", limit=limit, seed=seed,
        _source_override=virtual,
    )

    # 2. Per-repo: each picked repo individually
    picked_results = []
    for src in sources:
        try:
            rec = await recommend_fn(
                src.full_name, limit=limit // 2, seed=seed,
                _source_override=src,
            )
            groups_out = [{"label": g.label, "signal": g.signal, "repos": [r.model_dump() for r in g.repos]} for g in rec.groups]
            picked_results.append({"repo": src.full_name, "groups": groups_out})
        except Exception as exc:
            logger.warning("per-repo rec failed for %s: %s", src.full_name, exc)

    merged_groups = [{"label": g.label, "signal": g.signal, "repos": [r.model_dump() for r in g.repos]} for g in merged_result.groups]

    return {
        "source_repo": " + ".join(picked[:5]) + (f" +{len(picked)-5} more" if len(picked) > 5 else ""),
        "merged": merged_groups,
        "picked": picked_results,
        "from_cache": False,
    }
