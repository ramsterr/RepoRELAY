"""
CLI for the MVP.

Commands:
  reporelay-mvp save owner/name       fetch + persist + embed a repo
  reporelay-mvp count                 show how many repos are stored
  reporelay-mvp recommend owner/name  print ranked recommendations
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
from reporelay_mvp.github import save_repo
from reporelay_mvp.settings import get_mvp_settings

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
    json_output: bool = typer.Option(False, "--json", help="emit JSON instead of a table"),
) -> None:
    """Run the recommendation pipeline against a stored repo."""
    _configure_logging()
    try:
        rec = asyncio.run(recommend_func(repo, limit=limit))
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if json_output:
        payload: dict[str, Any] = {
            "source_repo": rec.source_repo,
            "repos": [r.model_dump() for r in rec.repos],
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return

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
