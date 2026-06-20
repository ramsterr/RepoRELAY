"""
Stage 5 of the MVP pipeline: reranking.

Three small rules, applied in order:

  1. Drop the source repo itself (defensive — we exclude it in SQL
     already, but a stale read shouldn't poison the list).
  2. Drop repos from the same owner as the source. Recommending
     sibling repos of the source is rarely useful.
  3. Enforce owner diversity — at most one repo per owner in the
     final list, to avoid "ten forks of the same project."

We keep the rest of the list ordered by score.
"""
from __future__ import annotations

from reporelay_mvp.models import Repo


def rerank(
    source: Repo,
    scored: list[tuple[Repo, float]],
    *,
    limit: int = 10,
) -> list[tuple[Repo, float]]:
    source_owner = source.owner.lower()
    seen_owners: set[str] = set()
    out: list[tuple[Repo, float]] = []

    sorted_scored = sorted(scored, key=lambda pair: pair[1], reverse=True)

    for repo, score in sorted_scored:
        if repo.id == source.id:
            continue

        owner = repo.owner.lower()
        if owner == source_owner:
            continue

        if owner in seen_owners:
            continue

        seen_owners.add(owner)
        out.append((repo, score))

        if len(out) >= limit:
            break

    return out
