"""
CLI for the MVP.

Commands:
  reporelay-mvp save owner/name       fetch + persist + embed a repo
  reporelay-mvp count                 show how many repos are stored
  reporelay-mvp recommend owner/name  print ranked recommendations
  reporelay-mvp explore               surprise me — random repo, recs
  reporelay-mvp seed                  bulk-index the corpus from GitHub search
  reporelay-mvp embed                 embed READMEs of un-embedded repos
  reporelay-mvp trending              scrape github.com/trending for viral repos
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler

from reporelay_mvp import data
from reporelay_mvp import recommend as recommend_func
from reporelay_mvp import recommend_random as explore_func
from reporelay_mvp.embed_pass import embed_top
from reporelay_mvp.github import save_repo
from reporelay_mvp.seed import DEFAULT_LANGUAGES, seed_corpus
from reporelay_mvp.seed_topics import DEFAULT_TOPICS, seed_topics_weighted
from reporelay_mvp.seed_topics import seed_topics as seed_topics_fn
from reporelay_mvp.settings import get_mvp_settings
from reporelay_mvp.topic_inference import infer_topics_for_repo
from reporelay_mvp.trending import DEFAULT_LANGUAGES as TRENDING_LANGUAGES
from reporelay_mvp.trending import scrape_all

app = typer.Typer(help="RepoRelay MVP CLI", no_args_is_help=True)
console = Console()


def _configure_logging() -> None:
    settings = get_mvp_settings()
    logging.basicConfig(
        level=settings.log_level if hasattr(settings, "log_level") else "INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def save(
    repo: str = typer.Argument(..., help="owner/name"),
) -> None:
    """Fetch a repo from GitHub, persist, and embed its README."""
    _configure_logging()
    if "/" not in repo:
        console.print("[red]expected owner/name[/red]")
        raise typer.Exit(code=1)
    owner, name = repo.split("/", 1)
    repo_id = asyncio.run(save_repo(owner, name))
    console.print(f"[bold green]saved {repo} (id={repo_id})[/bold green]")


@app.command()
def count() -> None:
    """Print how many repos are currently in mvp_repos."""
    from reporelay_mvp import data

    async def run() -> int:
        session = await data.get_session()
        try:
            return await data.count_repos(session)
        finally:
            await session.close()

    n = asyncio.run(run())
    console.print(f"[bold]{n}[/bold] repos in mvp_repos")


@app.command("status")
def status_cmd() -> None:
    """Show current embedding mode and DB stats."""
    from reporelay_mvp.embedding import DIMENSION, embedding_mode

    console.print(f"[bold]embedding mode:[/bold] {embedding_mode()}")
    console.print(f"[bold]embedding dim:[/bold] {DIMENSION}")
    has_key = bool(get_mvp_settings().openai_api_key) or bool(
        os.environ.get("OPENAI_API_KEY", "")
    )
    console.print(f"[bold]openai key set:[/bold] {'yes' if has_key else 'no'}")


@app.command()
def recommend(
    repo: str = typer.Argument(..., help="owner/name"),
    limit: int = typer.Option(10, help="number of recommendations to return"),
    seed: int | None = typer.Option(None, help="seed for different results (deterministic)"),
    json_output: bool = typer.Option(False, "--json", help="emit JSON instead of a table"),
) -> None:
    """Run the recommendation pipeline against a stored repo."""
    _configure_logging()
    try:
        rec = asyncio.run(recommend_func(repo, limit=limit, seed=seed))
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if json_output:
        payload: dict[str, Any] = {
            "source_repo": rec.source_repo,
            "repos": [r.model_dump() for r in rec.repos],
        }
        if seed is not None:
            payload["seed"] = seed
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return

    _print_results(rec)


@app.command()
def explore(
    seed: int = typer.Option(..., help="seed for deterministic random pick"),
    limit: int = typer.Option(10, help="number of recommendations"),
    json_output: bool = typer.Option(False, "--json", help="emit JSON"),
) -> None:
    """Pick a random repo and show its recommendations (surprise me)."""
    _configure_logging()
    try:
        rec = asyncio.run(explore_func(seed=seed, limit=limit))
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if json_output:
        payload: dict[str, Any] = {
            "source_repo": rec.source_repo,
            "repos": [r.model_dump() for r in rec.repos],
            "seed": seed,
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return

    _print_results(rec)


@app.command()
def seed(
    per_language: int = typer.Option(300, help="repos to index per language"),
    languages: str = typer.Option(
        "", help="comma-separated languages (default: top 10 by repo count)"
    ),
    min_stars: int = typer.Option(100, help="GitHub stars floor"),
    page_delay: float = typer.Option(
        2.0, help="seconds between search API calls (2.0 = 30 req/min)"
    ),
) -> None:
    """
    Bulk-index the corpus from GitHub search.

    Default is 300 repos × 10 languages = 3,000 repos, no extra REST
    calls beyond the 30 search requests. Idempotent — re-running
    upserts and refreshes search_fetched_at.
    """
    _configure_logging()
    lang_list: list[str] | None = None
    if languages:
        lang_list = [s.strip() for s in languages.split(",") if s.strip()]
    else:
        lang_list = list(DEFAULT_LANGUAGES)

    console.print(
        f"[bold]seeding corpus: {per_language} repos × {len(lang_list)} languages "
        f"= {per_language * len(lang_list)} target rows[/bold]"
    )
    console.print(f"[dim]languages: {', '.join(lang_list)}[/dim]")
    console.print(f"[dim]min stars: {min_stars}, page delay: {page_delay}s[/dim]")

    result = asyncio.run(
        seed_corpus(
            languages=lang_list,
            per_language=per_language,
            min_stars=min_stars,
            page_delay_s=page_delay,
        )
    )
    console.print(f"[bold green]done — {result['grand_total']} repos indexed[/bold green]")
    for lang, count in result["totals"].items():
        console.print(f"  {lang}: {count}")


@app.command()
def seed_topics(
    topics: str = typer.Option(
        "",
        help="Comma-separated topics (uses built-in default list if empty)",
    ),
    per_topic: int = typer.Option(200, help="repos per topic (max ~200 per search page × 2 pages)"),
    min_stars: int = typer.Option(20, help="minimum star count for search"),
    weighted: bool = typer.Option(
        False,
        "--weighted",
        help="Use weighted distribution from topics_config.py (500+ topics, 50k repos total)",
    ),
    category: str = typer.Option(
        "",
        "--category",
        help="With --weighted: only seed topics from this category",
    ),
    total_limit: int = typer.Option(
        0,
        "--total-limit",
        help="Cap total repos seeded (0 = unlimited). Safety valve.",
    ),
) -> None:
    """Seed repos by topic — broad software categories, not language-only.

    Without --weighted: uniform per_topic across default topics (~163 topics).
    With --weighted: per-topic counts from topics_config.py (~500+ topics, ~50k repos).
    """
    _configure_logging()

    if weighted:
        from reporelay_mvp.topics_config import (
            WEIGHTED_TOPICS,
            get_category_names,
            get_topics_for_category,
        )

        if category:
            weighted_list = get_topics_for_category(category.strip())
            if not weighted_list:
                names = ", ".join(get_category_names())
                console.print(f"[red]unknown category: {category!r}[/red]")
                console.print(f"Available: {names}")
                raise typer.Exit(code=1)
            console.print(
                f"[bold]seed-topics-weighted[/bold] — category={category}, "
                f"{len(weighted_list)} topics, stars ≥ {min_stars}"
            )
        else:
            weighted_list = list(WEIGHTED_TOPICS)
            total = sum(c for c, _ in weighted_list)
            console.print(
                f"[bold]seed-topics-weighted[/bold] — {len(weighted_list)} topics, "
                f"~{total} repos total, stars ≥ {min_stars}"
            )

        if total_limit:
            console.print(f"  total_limit={total_limit}")
        console.print()

        total = asyncio.run(
            seed_topics_weighted(
                topics=weighted_list,
                min_stars=min_stars,
                total_limit=total_limit,
            )
        )
        console.print(f"[bold green]done — {total} repos indexed[/bold green]")
        return

    # Uniform mode (existing behavior)
    topic_list = None
    if topics.strip():
        topic_list = [t.strip().lower() for t in topics.split(",") if t.strip()]
    else:
        topic_list = DEFAULT_TOPICS

    console.print(f"[bold]seed-topics[/bold] — {len(topic_list)} topics, {per_topic}/topic, stars ≥ {min_stars}")
    console.print("  " + ", ".join(topic_list[:10]) + (" …" if len(topic_list) > 10 else ""))
    console.print()

    total = asyncio.run(
        seed_topics_fn(
            topics=topic_list,
            per_topic=per_topic,
            min_stars=min_stars,
        )
    )
    console.print(f"[bold green]done — {total} repos indexed across {len(topic_list)} topics[/bold green]")


@app.command()
def embed(
    limit: int = typer.Option(7000, help="how many top-by-stars repos to embed"),
    concurrency: int = typer.Option(8, help="parallel readme fetches (unused in desc-only mode)"),
    batch_size: int = typer.Option(96, help="texts per batched Gemini API call (max=96)"),
    description_only: bool = typer.Option(
        True,
        "--description-only/--with-readme",
        help="Embed descriptions only (default, no GitHub API calls). "
        "--with-readme fetches READMEs from GitHub too.",
    ),
) -> None:
    """
    Compute and store embeddings for repos indexed from search but not
    yet embedded. Unlocks pgvector ANN semantic search.

    Defaults to description-only mode — embeds the stored description
    without calling GitHub. Uses Gemini paid tier (1500 RPM) with
    batched API calls (96 texts/request).

    With --with-readme, fetches READMEs from GitHub and embeds both
    README + description (uses GitHub API rate limit).
    """
    _configure_logging()
    mode_label = "description-only" if description_only else "readme+description"
    console.print(
        f"[bold]embedding top {limit} repos ({mode_label}, batch_size={batch_size})[/bold]"
    )

    result = asyncio.run(
        embed_top(
            limit=limit,
            concurrency=concurrency,
            batch_size=batch_size,
            description_only=description_only,
        )
    )
    if result["attempted"] == 0:
        console.print("[yellow]no repos need embedding[/yellow]")
        return
    console.print(
        f"[bold green]done — {result['succeeded']}/{result['attempted']} embedded, "
        f"{result['failed']} failed[/bold green]"
    )


@app.command()
def register_webhooks(
    min_stars: int = typer.Option(
        1000, help="only register webhooks for repos with at least this many stars"
    ),
    callback_url: str = typer.Option(
        ...,
        help="public URL of the deployed API (e.g. https://reporelay-mvp-api-0w1k.onrender.com)",
    ),
    secret: str = typer.Option(
        ...,
        help="GITHUB_WEBHOOK_SECRET value (must match what the API was started with)",
    ),
) -> None:
    """
    Register GitHub webhooks on top-starred repos so push events trigger re-embed.

    Run once after deploy. Safe to re-run — duplicate registrations are 409'd.
    """
    _configure_logging()

    import httpx

    headers = {
        "Authorization": f"Bearer {get_mvp_settings().github_token}",
        "Accept": "application/vnd.github+json",
    }

    async def register_one(client: httpx.AsyncClient, repo: dict[str, Any]) -> bool:
        owner, name = repo["owner"]["login"], repo["name"]
        url = f"https://api.github.com/repos/{owner}/{name}/hooks"
        body = {
            "name": "web",
            "active": True,
            "events": ["push"],
            "config": {
                "url": f"{callback_url.rstrip('/')}/webhooks/github",
                "content_type": "json",
                "secret": secret,
                "insecure_ssl": "0",
            },
        }
        r = await client.post(url, headers=headers, json=body)
        if r.status_code == 201:
            console.print(f"  [green]✓[/green] {owner}/{name}")
            return True
        if r.status_code == 422:
            console.print(f"  [yellow]~[/yellow] {owner}/{name} (already has webhook)")
            return False
        console.print(f"  [red]✗[/red] {owner}/{name}: {r.status_code} {r.text[:120]}")
        return False

    async def run() -> int:
        registered = 0
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            for page in range(1, 11):
                params = {"q": f"stars:>{min_stars}", "per_page": 100, "page": page}
                r = await client.get(
                    "https://api.github.com/search/repositories",
                    headers=headers,
                    params=params,
                )
                r.raise_for_status()
                items = r.json().get("items", [])
                if not items:
                    break
                for item in items:
                    if await register_one(client, item):
                        registered += 1
        return registered

    n = asyncio.run(run())
    console.print(f"[bold green]done — {n} webhooks registered[/bold green]")


@app.command()
def trending(
    languages: str = typer.Option(
        "",
        help="comma-separated languages (default: top 10 + 'all languages')",
    ),
    since: str = typer.Option(
        "daily",
        help="time window: daily, weekly, or monthly",
    ),
    delay_s: float = typer.Option(
        10.0,
        help="seconds between language scrapes (be polite to github.com)",
    ),
) -> None:
    """
    Scrape github.com/trending for viral repos. Free (no API rate limit),
    catches repos the search API misses because total stars are still low.

    Updates the `trending_score` column on existing mvp_repos rows.
    """
    _configure_logging()
    if since not in ("daily", "weekly", "monthly"):
        console.print(f"[red]since must be daily/weekly/monthly, got {since!r}[/red]")
        raise typer.Exit(code=1)

    lang_list: list[str]
    if languages:
        lang_list = [s.strip() for s in languages.split(",") if s.strip()]
    else:
        lang_list = list(TRENDING_LANGUAGES)

    console.print(
        f"[bold]scraping trending[/bold] since={since} langs={len(lang_list)}"
    )

    repos = asyncio.run(
        scrape_all(languages=lang_list, since=since, delay_s=delay_s)
    )
    if not repos:
        console.print("[yellow]no trending repos found — github.com may have changed HTML[/yellow]")
        return

    star_field = {
        "daily": "stars_today",
        "weekly": "stars_this_week",
        "monthly": "stars_this_month",
    }[since]

    rows = [
        {
            "full_name": r.full_name,
            "description": r.description,
            "language": r.language,
            "stars_period": r.stars_today,
            "total_stars": r.total_stars,
        }
        for r in repos
    ]
    rows = [{**r, "stars_period": getattr(r_obj, star_field, r["stars_period"])} for r, r_obj in zip(rows, repos, strict=True)]

    async def apply() -> int:
        session = await data.get_session()
        try:
            return await data.bulk_apply_trending_signal(session, rows, since=since)
        finally:
            await session.close()

    updated = asyncio.run(apply())
    console.print(
        f"[bold green]done — {updated}/{len(repos)} trending repos updated in mvp_repos[/bold green]"
    )


@app.command("infer-topics")
def infer_topics_cmd(
    limit: int = typer.Option(1000, help="max repos to process"),
    min_topics: int = typer.Option(3, help="repos with fewer than this many topics get inference"),
    refetch: bool = typer.Option(
        False,
        "--refetch",
        help="re-fetch README from GitHub for better inference (uses API rate limit)",
    ),
) -> None:
    """
    Backfill inferred topics on repos with sparse topic lists.

    Uses README + description text to match against a keyword vocabulary
    and add relevant GitHub-style topic tags. Without --refetch, only
    the stored description is used (fast, no API calls). With --refetch,
    the README is re-fetched from GitHub for more accurate inference.
    """
    _configure_logging()

    async def run() -> tuple[int, int]:
        session = await data.get_session()
        try:
            repos = await data.list_repos_needing_topic_inference(
                session, limit=limit, min_topic_count=min_topics,
            )
        finally:
            await session.close()

        if not repos:
            console.print("[yellow]no repos need topic inference[/yellow]")
            return 0, 0

        console.print(f"[bold]inferring topics for {len(repos)} repos (min_topics < {min_topics})[/bold]")

        updated = 0
        from reporelay_mvp.github import _auth_client
        from reporelay_mvp.github import fetch_readme as gh_fetch_readme

        settings = get_mvp_settings()

        for i, repo in enumerate(repos):
            readme_text: str | None = None

            if refetch and settings.github_token:
                try:
                    async with _auth_client(settings.github_token) as client:
                        readme_text = await gh_fetch_readme(client, repo.owner, repo.name)
                except Exception:
                    pass  # fall back to description-only

            inferred = infer_topics_for_repo(
                repo.description, readme_text, repo.topics,
            )

            if not inferred:
                continue

            session = await data.get_session()
            try:
                await data.update_topics(session, repo_id=repo.id, topics=inferred)
                await session.commit()
            finally:
                await session.close()

            updated += 1
            new_topics_str = ", ".join(inferred)
            console.print(
                f"  [{i+1}/{len(repos)}] {repo.full_name}: +[green]{new_topics_str}[/green]"
            )

        return updated, len(repos)

    updated, total = asyncio.run(run())
    console.print(
        f"[bold green]done — {updated}/{total} repos got new topics[/bold green]"
    )


@app.command("extract-keywords")
def extract_keywords_cmd(
    limit: int = typer.Option(5000, help="max repos to process"),
    concurrency: int = typer.Option(8, help="parallel processing"),
    no_readme: bool = typer.Option(
        False, "--no-readme",
        help="skip GitHub README fetch — use descriptions only (10x faster)",
    ),
) -> None:
    """Extract search keywords from repo descriptions + READMEs.

    Extracts domain-specific technical keywords and stores them in the
    keywords[] column for fast search via GIN index. Also populates
    the search_vector tsvector column for full-text search.

    By default, only repos with short/null descriptions get their README
    fetched from GitHub (slow). Repos with solid descriptions extract
    keywords locally (fast). Use --no-readme to skip ALL GitHub fetches.
    """
    _configure_logging()

    from reporelay_mvp import data
    from reporelay_mvp.keyword_extractor import extract_keywords, extract_keywords_from_repo
    from reporelay_mvp.github import _auth_client, fetch_readme
    from reporelay_mvp.settings import get_mvp_settings

    async def run() -> tuple[int, int]:
        session = await data.get_session()
        try:
            repos = await data.list_repos_needing_keywords(session, limit=limit)
        finally:
            await session.close()

        if not repos:
            console.print("[yellow]no repos need keyword extraction[/yellow]")
            return 0, 0

        # Count how many have solid descriptions (no GitHub needed)
        desc_ok = sum(1 for r in repos if r.description and len(r.description.strip()) >= 50)
        needs_readme = len(repos) - desc_ok
        console.print(f"[bold]extracting keywords for {len(repos)} repos[/bold]")
        console.print(f"  {desc_ok} with solid descriptions (fast, no GitHub)")
        console.print(f"  {needs_readme} need README fetch (slow, GitHub API)")

        settings = get_mvp_settings()
        sem = asyncio.Semaphore(concurrency)
        updated = 0

        async def process_one(repo: Any) -> bool:
            async with sem:
                readme: str | None = None

                # Only fetch README from GitHub if description is too short/null
                # AND --no-readme is not set
                desc_short = not repo.description or len(repo.description.strip()) < 50
                if desc_short and not no_readme:
                    try:
                        async with _auth_client(settings.github_token) as client:
                            fetched = await fetch_readme(client, repo.owner, repo.name)
                            if fetched and fetched.strip():
                                readme = fetched
                    except Exception:
                        pass

                if readme:
                    keywords = extract_keywords_from_repo(repo.description, readme)
                elif repo.description:
                    keywords = extract_keywords(repo.description)
                else:
                    keywords = []

                if not keywords:
                    return False

                session = await data.get_session()
                try:
                    await data.set_keywords(session, repo_id=repo.id, keywords=keywords)
                    await session.commit()
                finally:
                    await session.close()
                return True

        tasks = [process_one(r) for r in repos]
        for i, ok in enumerate(await asyncio.gather(*tasks)):
            if ok:
                updated += 1
            if (i + 1) % 200 == 0 or (i + 1) == len(repos):
                console.print(f"  progress: {i+1}/{len(repos)}")
                await asyncio.sleep(0.05)

        return updated, len(repos)

    updated, total = asyncio.run(run())
    console.print(
        f"[bold green]done — {updated}/{total} repos got keywords extracted[/bold green]"
    )


def _print_results(rec: Any) -> None:
    console.print(f"[bold]recommendations for {rec.source_repo}[/bold]\n")
    for i, r in enumerate(rec.repos, start=1):
        lang = r.language or "—"
        topics = ", ".join(r.topics[:3]) if r.topics else "—"
        console.print(
            f"  {i:2d}. [cyan]{r.full_name}[/cyan]  "
            f"[dim]({lang}, {r.stars} stars, topics: {topics})[/dim]"
        )


if __name__ == "__main__":
    app()
