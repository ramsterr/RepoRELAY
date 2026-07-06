"""
Stage 5 of the MVP pipeline: reranking.

Rules, applied in order:
  1. Drop the source repo itself.
  2. Drop repos from the same owner as the source.
  3. One repo per owner maximum (no "ten forks of X").
  4. Cross-language diversity: ensure at least 25% of results
     are in a different language from the source.
  5. Sort by score descending.
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

    if seed is not None:
        working = list(scored)
    else:
        working = sorted(scored, key=lambda p: p[1], reverse=True)

    seen_owners: set[str] = set()
    out: list[tuple[Repo, float, Any]] = []
    cross_lang_pool: list[tuple[Repo, float, Any]] = []

    for repo, score, meta in working:
        if repo.id == source.id:
            continue

        owner = repo.owner.lower()
        if owner == source_owner:
            continue

        cand_lang = (repo.language or "").lower()

        # Collect cross-language candidates for diversity pass
        if source_lang and cand_lang and cand_lang != source_lang:
            if owner not in seen_owners:
                cross_lang_pool.append((repo, score, meta))

        if owner in seen_owners:
            continue

        seen_owners.add(owner)
        out.append((repo, score, meta))

        if len(out) >= limit:
            break

    # Diversity: swap lowest-scoring same-language repos with
    # cross-language candidates to hit 25% diversity
    if source_lang and cross_lang_pool and len(out) >= 4:
        min_diverse = max(1, limit // 4)

        same_lang_in_out = [
            (i, r) for i, r in enumerate(out)
            if (r[0].language or "").lower() == source_lang
        ]
        diff_lang_count = len(out) - len(same_lang_in_out)

        if diff_lang_count < min_diverse and same_lang_in_out:
            swaps_needed = min_diverse - diff_lang_count
            cross_sorted = sorted(cross_lang_pool, key=lambda r: r[1], reverse=True)
            out_owners = {r[0].owner.lower() for r in out}
            cross_filtered = [
                r for r in cross_sorted
                if r[0].owner.lower() not in out_owners
                and (r[0].language or "").lower() != source_lang
            ]

            same_lang_sorted = sorted(same_lang_in_out, key=lambda x: x[1][1])

            for i, (out_idx, _) in enumerate(same_lang_sorted):
                if i >= swaps_needed or i >= len(cross_filtered):
                    break
                out[out_idx] = cross_filtered[i]

    if seed is not None:
        out.sort(key=lambda p: p[1], reverse=True)

    return out
