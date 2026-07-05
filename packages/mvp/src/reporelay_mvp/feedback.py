"""
Relevance feedback: "Show me more repos like these selected ones."

When a user picks repos from recommendations and clicks "Show more
like these", we merge their properties into a virtual source repo,
then run the standard recommendation pipeline against it.

The merge is an intelligent average:
  - Embeddings: element-wise average (centroid in vector space)
  - Topics: union (captures the domain intersection)
  - Language: most common
  - Dependencies: union
  - Stars: average

This produces a "virtual repo" that captures the user's intent —
e.g., picking "flask" + "django" + "fastapi" creates a virtual
source that represents "Python web frameworks" better than any
single repo could.
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
    """Merge multiple repos into a virtual source Repo.

    Each property is averaged/unioned to create a single virtual repo
    that captures the common signals across all picked repos.

    Args:
        sources: List of Repo objects to merge. Must be non-empty.

    Returns:
        A single Repo with merged properties.
    """
    if not sources:
        raise ValueError("At least one source repo is required")

    n = len(sources)

    # ── Language: most common ────────────────────────────────────
    langs = [s.language for s in sources if s.language]
    language = Counter(langs).most_common(1)[0][0] if langs else None

    # ── Topics: union ────────────────────────────────────────────
    all_topics: list[str] = []
    for s in sources:
        all_topics.extend(s.topics)
    topics = list(dict.fromkeys(all_topics))  # dedupe, preserve order

    # ── Dependencies: union ──────────────────────────────────────
    all_deps: list[str] = []
    for s in sources:
        all_deps.extend(s.dependencies)
    deps = list(dict.fromkeys(all_deps))

    # ── Stars: average ───────────────────────────────────────────
    avg_stars = sum(s.stars for s in sources) // n

    # ── Embedding (README): element-wise average ─────────────────
    embeddings = [s.embedding for s in sources if s.embedding and any(v != 0.0 for v in s.embedding)]
    if embeddings:
        merged_embedding = [
            sum(vec[i] for vec in embeddings) / len(embeddings)
            for i in range(len(embeddings[0]))
        ]
    else:
        merged_embedding = None

    # ── Description embedding: element-wise average ──────────────
    desc_embs = [
        s.description_embedding
        for s in sources
        if s.description_embedding and any(v != 0.0 for v in s.description_embedding)
    ]
    if desc_embs:
        merged_desc_emb = [
            sum(vec[i] for vec in desc_embs) / len(desc_embs)
            for i in range(len(desc_embs[0]))
        ]
    else:
        merged_desc_emb = None

    # ── Description: first non-empty ─────────────────────────────
    description = next((s.description for s in sources if s.description), None)

    # ── Trending: average ────────────────────────────────────────
    avg_trending = sum(s.trending_score for s in sources) / n

    # ── Build the virtual repo ───────────────────────────────────
    virtual = Repo(
        id=0,  # placeholder — virtual, doesn't exist in DB
        owner="virtual",
        name="merged",
        full_name="virtual/merged",
        description=description,
        language=language,
        topics=topics,
        stars=avg_stars,
        dependencies=deps,
        embedding=merged_embedding,
        description_embedding=merged_desc_emb,
        trending_score=avg_trending,
    )

    logger.info(
        "merged %d repos → virtual source (%s, %d topics, %d deps, embeddings=%s/%s)",
        n,
        language or "?",
        len(topics),
        len(deps),
        "yes" if merged_embedding else "no",
        "yes" if merged_desc_emb else "no",
    )
    return virtual


async def more_like_these(
    picked: list[str],
    *,
    limit: int = 10,
    seed: int | None = None,
    tags: list[str] | None = None,
):
    """Get repos similar to a set of user-selected repos.

    Fetches each picked repo from the DB (live-embeds if missing),
    merges them into a virtual source, and runs the standard
    recommendation pipeline.

    Args:
        picked: List of repo full_names to merge (e.g. ["flask/flask", "django/django"])
        limit: Number of results
        seed: Deterministic variation seed
        tags: Optional tag filter

    Returns:
        ScoredRecommendation with the virtual source and ranked results.
    """
    if not picked:
        raise ValueError("At least one repo must be selected")

    from reporelay_mvp import data

    sources: list[Repo] = []
    session = await data.get_session()
    try:
        for full_name in picked:
            source = await data.get_repo(session, full_name)
            if source is None:
                logger.warning("picked repo %s not in DB — skipping", full_name)
                continue
            sources.append(source)
    finally:
        await session.close()

    if not sources:
        raise ValueError("None of the picked repos were found in the DB")

    # Merge into virtual source
    virtual_source = merge_sources(sources)

    # Run standard pipeline with "virtual/merged" as the source name
    result = await recommend_fn(
        "virtual/merged",
        limit=limit,
        seed=seed,
        tags=tags,
        _source_override=virtual_source,
    )

    # Override source_repo to show what the user picked
    result.source_repo = " + ".join(picked[:5])
    if len(picked) > 5:
        result.source_repo += f" +{len(picked) - 5} more"
    return result
