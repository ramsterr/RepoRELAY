"""
Bulk index the corpus from GitHub search.

Walks the GitHub search API across a set of languages, paginates
through results, and bulk-upserts them into mvp_repos. The metadata
+ topics + stars from the search response are enough for the
recommender to function — the README/embedding enrichment is a
separate pass (see `reporelay_mvp.embed_pass`).

Rate-limit budget:
  - Search API: 30 requests / minute (authenticated)
  - At 100 results per page and 1 request / 2s, we index
    ~3,000 repos / minute from search alone.
  - The bulk-upsert is local SQL — no extra API calls.

Usage:
  just mvp seed                       # default 300 per language × 10 langs = 3,000
  just mvp seed --per-language 500    # 500 per language
  just mvp seed --languages python,go # specific languages
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from reporelay_mvp import data
from reporelay_mvp.github import (
    _auth_client,
    search_repositories,
)
from reporelay_mvp.settings import get_mvp_settings

logger = logging.getLogger(__name__)


DEFAULT_LANGUAGES: tuple[str, ...] = (
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "csharp",
    "cpp",
    "ruby",
    "php",
)


async def _fetch_pages(
    client: httpx.AsyncClient,
    *,
    language: str,
    per_page: int,
    pages: int,
    min_stars: int,
    page_delay_s: float,
) -> list[dict[str, Any]]:
    """Hit the search API `pages` times for one language, ~2s apart."""
    items: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        try:
            payload = await search_repositories(
                client,
                language=language,
                min_stars=min_stars,
                sort="stars",
                per_page=per_page,
                page=page,
            )
        except Exception as exc:
            logger.warning("search failed for %s page %d: %s", language, page, exc)
            break
        page_items = payload.get("items", [])
        if not page_items:
            break
        items.extend(page_items)
        logger.info(
            "  %s page %d: +%d items (running total: %d)",
            language,
            page,
            len(page_items),
            len(items),
        )
        if page < pages:
            await asyncio.sleep(page_delay_s)
    return items


async def seed_corpus(
    *,
    languages: list[str] | None = None,
    per_language: int = 300,
    per_page: int = 100,
    min_stars: int = 100,
    page_delay_s: float = 2.0,
) -> dict[str, int]:
    """
    Index `per_language` repos for each language. Returns counts by
    language and a grand total. Idempotent: re-running upserts and
    updates `search_fetched_at`.
    """
    if languages is None:
        languages = list(DEFAULT_LANGUAGES)
    pages_per_lang = max(1, (per_language + per_page - 1) // per_page)

    settings = get_mvp_settings()
    session = await data.get_session()
    totals: dict[str, int] = {}

    try:
        async with _auth_client(settings.github_token) as client:
            for language in languages:
                logger.info(
                    "indexing language=%s target=%d (%d page(s))",
                    language,
                    per_language,
                    pages_per_lang,
                )
                items = await _fetch_pages(
                    client,
                    language=language,
                    per_page=per_page,
                    pages=pages_per_lang,
                    min_stars=min_stars,
                    page_delay_s=page_delay_s,
                )
                if not items:
                    totals[language] = 0
                    continue
                items = items[:per_language]
                written = await data.bulk_upsert_from_search(session, items)
                await session.commit()
                totals[language] = written
                logger.info(
                    "  -> %s: %d repos written to mvp_repos",
                    language,
                    written,
                )
    finally:
        await session.close()

    grand_total = sum(totals.values())
    logger.info("seed complete: %d total repos across %d languages", grand_total, len(languages))
    return {"totals": totals, "grand_total": grand_total}
