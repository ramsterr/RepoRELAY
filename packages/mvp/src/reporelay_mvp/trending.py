"""
Scrape github.com/trending to surface viral repos that the search API misses.

The trending page is public HTML, not part of the REST/Search API rate-limit
budget. Each language has its own page; the same repos appear in daily/weekly
/monthly buckets. We pull daily by default to catch the freshest signal.

Notes:
  - Be polite: User-Agent, 1 req/10s, respect robots.txt (the page is allowed).
  - Fragile by nature: GitHub changes HTML without notice. Failures are logged
    and the cron keeps going.
  - Catches repos with viral growth even when total stars are still < 100,
    so they enter the recommender pool long before search API would surface
    them via `stars:>100`.

Usage:
  reporelay-mvp trending --since daily
  reporelay-mvp trending --languages python,rust --since weekly
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)

TRENDING_URL = "https://github.com/trending/{language}?since={since}"

DEFAULT_LANGUAGES: tuple[str, ...] = (
    "",
    "python",
    "javascript",
    "typescript",
    "go",
    "rust",
    "java",
    "ruby",
    "csharp",
    "cpp",
)

VALID_SINCE: tuple[str, ...] = ("daily", "weekly", "monthly")

USER_AGENT = "reporelay-bot (+https://github.com/ramsterr/RepoRELAY)"


@dataclass
class TrendingRepo:
    full_name: str
    description: str | None
    language: str | None
    stars_today: int
    total_stars: int

    @property
    def repo_id(self) -> int | None:
        return None


def _parse_int(text: str | None) -> int:
    if not text:
        return 0
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 0


def _parse_trending_html(html: str, *, fallback_language: str | None) -> list[TrendingRepo]:
    tree = HTMLParser(html)
    repos: list[TrendingRepo] = []

    for article in tree.css("article.Box-row"):
        try:
            anchor = article.css_first("h2 a")
            if anchor is None:
                continue
            href = (anchor.attributes.get("href") or "").strip().lstrip("/")
            if not href or "/" not in href:
                continue
            full_name = href

            desc_el = article.css_first("p.col-9")
            description = desc_el.text(strip=True) if desc_el is not None else None

            lang_el = article.css_first("span[itemprop='programmingLanguage']")
            language = lang_el.text(strip=True) if lang_el is not None else fallback_language

            stars_today = 0
            for span in article.css("span.d-inline-block.float-sm-right"):
                txt = span.text(strip=True)
                if "stars today" in txt or "stars this week" in txt or "stars this month" in txt:
                    stars_today = _parse_int(txt)
                    break

            total_stars = 0
            star_link = article.css_first("a[href$='/stargazers']")
            if star_link is not None:
                total_stars = _parse_int(star_link.text(strip=True))

            repos.append(
                TrendingRepo(
                    full_name=full_name,
                    description=description,
                    language=language,
                    stars_today=stars_today,
                    total_stars=total_stars,
                )
            )
        except Exception as exc:
            logger.warning("failed to parse trending entry: %s", exc)
            continue

    return repos


async def scrape_trending(
    client: httpx.AsyncClient,
    *,
    language: str = "",
    since: str = "daily",
) -> list[TrendingRepo]:
    if since not in VALID_SINCE:
        raise ValueError(f"since must be one of {VALID_SINCE}, got {since!r}")

    url = TRENDING_URL.format(language=language, since=since)
    response = await client.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        follow_redirects=True,
    )
    response.raise_for_status()
    return _parse_trending_html(response.text, fallback_language=language or None)


async def scrape_all(
    *,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
    since: str = "daily",
    delay_s: float = 10.0,
) -> list[TrendingRepo]:
    seen: dict[str, TrendingRepo] = {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for lang in languages:
            try:
                repos = await scrape_trending(client, language=lang, since=since)
            except Exception as exc:
                logger.warning("trending scrape failed for lang=%r: %s", lang, exc)
                continue

            for r in repos:
                existing = seen.get(r.full_name)
                if existing is None or r.stars_today > existing.stars_today:
                    seen[r.full_name] = r

            logger.info("trending %r (%s): %d repos", lang or "all", since, len(repos))

            if delay_s > 0:
                await asyncio.sleep(delay_s)

    return list(seen.values())


def to_upsert_rows(repos: list[TrendingRepo], *, since: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in repos:
        rows.append(
            {
                "full_name": r.full_name,
                "description": r.description,
                "language": r.language,
                "stars_today": r.stars_today,
                "total_stars": r.total_stars,
                "trending_since": since,
            }
        )
    return rows
