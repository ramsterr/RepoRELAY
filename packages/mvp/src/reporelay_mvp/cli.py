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
import sys
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler

from reporelay_mvp import recommend as recommend_func
from reporelay_mvp import recommend_random as explore_func
from reporelay_mvp import data
from reporelay_mvp.seed_topics import seed_topics as seed_topics_fn, DEFAULT_TOPICS
from reporelay_mvp.embed_pass import embed_top
from reporelay_mvp.github import save_repo
from reporelay_mvp.seed import DEFAULT_LANGUAGES, seed_corpus
from reporelay_mvp.settings import get_mvp_settings
from reporelay_mvp.trending import DEFAULT_LANGUAGES as TRENDING_LANGUAGES, scrape_all

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
) -> None:
    """Seed repos by topic — broad software categories, not language-only."""
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
    limit: int = typer.Option(1000, help="how many top-by-stars repos to embed"),
    concurrency: int = typer.Option(4, help="parallel readme fetches"),
) -> None:
    """
    Compute and store README embeddings for repos indexed from
    search but not yet embedded. Unlocks pgvector ANN.

    Each repo = 1 readme fetch + 1 embed call. Paced to stay
    under the 5,000 req/hr REST limit. The model downloads on
    first run (~11s) and stays in memory.
    """
    _configure_logging()
    console.print(
        f"[bold]embedding top {limit} repos (concurrency={concurrency})[/bold]"
    )
    console.print(
        "[dim]first run downloads the embedding model (~80MB, ~11s); subsequent runs are fast[/dim]"
    )

    result = asyncio.run(embed_top(limit=limit, concurrency=concurrency))
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
    rows = [{**r, "stars_period": getattr(r_obj, star_field, r["stars_period"])} for r, r_obj in zip(rows, repos)]

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
