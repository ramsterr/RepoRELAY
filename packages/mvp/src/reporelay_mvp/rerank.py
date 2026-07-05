"""
Stage 5 of the MVP pipeline: reranking.

Five rules, applied in order:

  1. Drop the source repo itself (defensive — we exclude it in SQL
     already, but a stale read shouldn't poison the list).
  2. Drop repos from the same owner as the source. Recommending
     sibling repos of the source is rarely useful.
  3. Enforce owner diversity — at most one repo per owner in the
     final list, to avoid "ten forks of the same project."
  4. Cross-language diversity — ensure at least 25% of results
     are in a different language from the source (if available).
     This prevents "all Python repos" for a Python source when
     there are equivalent repos in Rust, Go, etc.
  5. Sort by score descending.

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
    source_lang = source.language.lower() if source.language else None
    seen_owners: set[str] = set()
    out: list[tuple[Repo, float, Any]] = []
    cross_lang_pool: list[tuple[Repo, float, Any]] = []

    if seed is not None:
        working = list(scored)
    else:
        working = sorted(scored, key=lambda pair: pair[1], reverse=True)

    # Pass 1: build the main list with one-per-owner rule
    for repo, score, meta in working:
        if repo.id == source.id:
            continue

        owner = repo.owner.lower()
        if owner == source_owner:
            continue

        cand_lang = repo.language.lower() if repo.language else ""

        # Collect cross-language candidates separately
        if source_lang and cand_lang and cand_lang != source_lang:
            if owner not in seen_owners:
                cross_lang_pool.append((repo, score, meta))
                # Only add to cross-lang pool, let the main loop handle selection

        if owner in seen_owners:
            continue

        seen_owners.add(owner)
        out.append((repo, score, meta))

        if len(out) >= limit:
            break

    # Pass 2: cross-language diversity — ensure at least min_diverse repos
    # are from a different language than the source
    if source_lang and cross_lang_pool and len(out) >= 4:
        min_diverse = max(1, limit // 4)  # 25% cross-language minimum
        same_lang = [r for r in out if (r[0].language or "").lower() != source_lang]
        diff_lang = [r for r in out if (r[0].language or "").lower() != source_lang]
        currently_diverse = len(diff_lang)

        if currently_diverse < min_diverse:
            # Remove the lowest-scoring same-language repos and replace
            # with the highest-scoring cross-language candidates
            same_lang_sorted = sorted(same_lang, key=lambda r: r[1])
            cross_sorted = sorted(
                cross_lang_pool,
                key=lambda r: r[1],
                reverse=True,
            )

            # Filter out cross-lang candidates whose owners are already in the list
            out_owners = {r[0].owner.lower() for r in out}
            cross_filtered = [
                r for r in cross_sorted
                if r[0].owner.lower() not in out_owners
            ]

            swaps_needed = min_diverse - currently_diverse
            swaps_done = 0
            for i, same_item in enumerate(same_lang_sorted):
                if swaps_done >= swaps_needed:
                    break
                if i >= len(cross_filtered):
                    break
                # Swap: remove same-item from out, add cross-item
                try:
                    out_idx = out.index(same_item)
                    out[out_idx] = cross_filtered[swaps_done]
                    swaps_done += 1
                except ValueError:
                    continue

    if seed is not None:
        out.sort(key=lambda pair: pair[1], reverse=True)

    return out
