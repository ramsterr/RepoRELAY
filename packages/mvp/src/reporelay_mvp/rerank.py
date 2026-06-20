"""
Stage 5 of the MVP pipeline: reranking.

Three small rules, applied in order:

  1. Drop the source repo itself (defensive — we exclude it in SQL
     already, but a stale read shouldn't poison the list).
  2. Drop repos from the same owner as the source. Recommending
     sibling repos of the source is rarely useful.
  3. Enforce owner diversity — at most one repo per owner in the
     final list, to avoid "ten forks of the same project."

When `seed` is None, the list is sorted by score (highest first) so
we apply the rules against the top-scoring repos. When `seed` is set,
the candidate pool has already been shuffled — we preserve that order
so the seed actually changes which repos survive the diversity filter.
"""

from __future__ import annotations

from typing import Any

from reporelay_mvp.models import Repo


def rerank(
    source: Repo,
    scored: list[tuple[Repo, float, Any]],
    *,
    limit: int = 10,
    seed: int | None = None,
) -> list[tuple[Repo, float, Any]]:
    source_owner = source.owner.lower()
    seen_owners: set[str] = set()
    out: list[tuple[Repo, float, Any]] = []

    if seed is not None:
        working = list(scored)
    else:
        working = sorted(scored, key=lambda pair: pair[1], reverse=True)

    for repo, score, meta in working:
        if repo.id == source.id:
            continue

        owner = repo.owner.lower()
        if owner == source_owner:
            continue

        if owner in seen_owners:
            continue

        seen_owners.add(owner)
        out.append((repo, score, meta))

        if len(out) >= limit:
            break

    if seed is not None:
        out.sort(key=lambda pair: pair[1], reverse=True)

    return out
