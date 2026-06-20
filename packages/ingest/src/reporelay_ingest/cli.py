from __future__ import annotations

import asyncio
import logging

import httpx
import typer
from rich.console import Console
from rich.logging import RichHandler

from reporelay_core.settings import get_settings
from reporelay_ingest.github import GitHubClient
from reporelay_ingest.storage import (
    insert_contributors,
    insert_topics,
    upsert_readme,
    upsert_repo,
)

app = typer.Typer(help="RepoRelay data ingestion CLI")
console = Console()


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def fetch_repo(
    owner: str = typer.Argument(..., help="Repo owner"),
    name: str = typer.Argument(..., help="Repo name"),
) -> None:
    """Fetch a single repo's metadata and print it."""
    _configure_logging()

    async def run() -> None:
        async with GitHubClient() as client:
            try:
                repo = await client.get_repo(owner, name)
            except httpx.HTTPError as exc:
                console.print(f"[red]failed:[/red] {exc}")
                raise typer.Exit(code=1) from exc
            console.print(
                f"[green]{repo['full_name']}[/green] - "
                f"{repo['stargazers_count']} stars - "
                f"{repo.get('language', '?')} - "
                f"{repo.get('description') or '(no description)'}"
            )

    asyncio.run(run())


@app.command()
def save_repo(
    owner: str = typer.Argument(..., help="Repo owner"),
    name: str = typer.Argument(..., help="Repo name"),
) -> None:
    """Fetch a repo and all associated data, then persist to the database."""
    _configure_logging()

    async def run() -> None:
        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        try:
            async with (
                GitHubClient() as client,
                session,
            ):
                console.print(f"[bold]Fetching {owner}/{name}...[/bold]")

                repo = await client.get_repo(owner, name)
                console.print(f"  repo: [green]{repo['full_name']}[/green] ({repo['id']})")

                repo_id = await upsert_repo(session, repo)
                console.print(f"  [dim]upserted repos row[/dim]")

                try:
                    readme_text = await client.get_readme(owner, name)
                    await upsert_readme(session, repo_id, readme_text)
                    console.print(f"  [dim]upserted readme ({len(readme_text)} chars)[/dim]")
                except httpx.HTTPStatusError:
                    console.print("  [yellow]no README found, skipping[/yellow]")

                topics = await client.get_topics(owner, name)
                if topics:
                    await insert_topics(session, repo_id, topics)
                    console.print(f"  [dim]inserted {len(topics)} topics[/dim]")

                languages = await client.get_languages(owner, name)
                console.print(f"  languages: {languages}")

                contributors = await client.get_contributors(owner, name)
                if contributors:
                    await insert_contributors(session, repo_id, contributors)
                    console.print(f"  [dim]inserted {len(contributors)} contributors[/dim]")

                await session.commit()
                console.print(f"\n[bold green]Saved {owner}/{name} to database.[/bold green]")

        except httpx.HTTPError as exc:
            console.print(f"[red]GitHub API error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    asyncio.run(run())


@app.command()
def whoami() -> None:
    """Check the GitHub auth state and remaining rate limit."""
    _configure_logging()

    async def run() -> None:
        async with GitHubClient() as client:
            response = await client._client.get("/rate_limit")
            data = response.json()
            core = data.get("resources", {}).get("core", {})
            console.print(
                f"[bold]limit:[/bold] {core.get('limit')} - "
                f"[bold]used:[/bold] {core.get('used')} - "
                f"[bold]remaining:[/bold] {core.get('remaining')}"
            )

    asyncio.run(run())


if __name__ == "__main__":
    app()
