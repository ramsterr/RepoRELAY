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
    insert_dependencies,
    insert_star_events,
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


@app.command()
def seed_topics(
    topics: list[str] = typer.Argument(
        ..., help="Topics to seed (e.g. react vue machine-learning)"
    ),
    per_topic: int = typer.Option(20, help="Repos per topic"),
) -> None:
    """Fetch top repos per topic via GitHub search and persist them."""
    _configure_logging()

    async def run() -> None:
        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        async with GitHubClient() as client, session:
            for topic in topics:
                console.print(f"[bold]Seeding topic: {topic}[/bold]")
                try:
                    results = await client.search_repos(
                        f"topic:{topic}", per_page=per_topic
                    )
                except httpx.HTTPError as exc:
                    console.print(f"  [red]search failed: {exc}[/red]")
                    continue

                for item in results:
                    full_name = item["full_name"]
                    try:
                        repo = await client.get_repo(
                            item["owner"]["login"], item["name"]
                        )
                        repo["topics"] = item.get("topics", [])
                        repo_id = await upsert_repo(session, repo)
                        console.print(f"  [green]{full_name}[/green] [dim]({repo_id})[/dim]")

                        try:
                            readme_text = await client.get_readme(
                                item["owner"]["login"], item["name"]
                            )
                            await upsert_readme(session, repo_id, readme_text)
                        except httpx.HTTPStatusError:
                            pass

                        repo_topics = await client.get_topics(
                            item["owner"]["login"], item["name"]
                        )
                        if repo_topics:
                            await insert_topics(session, repo_id, repo_topics)

                        contributors = await client.get_contributors(
                            item["owner"]["login"], item["name"], max_results=30
                        )
                        if contributors:
                            await insert_contributors(session, repo_id, contributors)

                        await session.commit()
                    except httpx.HTTPError as exc:
                        console.print(f"  [yellow]skipped {full_name}: {exc}[/yellow]")

            row = await session.execute(
                __import__("sqlalchemy").text("SELECT COUNT(*) FROM repos")
            )
            total = row.scalar()
            console.print(f"\n[bold green]Total repos in database: {total}[/bold green]")

    asyncio.run(run())


@app.command()
def seed_languages(
    languages: list[str] = typer.Argument(
        ..., help="Languages to seed (e.g. Python Rust TypeScript)"
    ),
    per_language: int = typer.Option(20, help="Repos per language"),
) -> None:
    """Fetch top repos per language via GitHub search and persist them."""
    _configure_logging()

    async def run() -> None:
        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        async with GitHubClient() as client, session:
            for lang in languages:
                console.print(f"[bold]Seeding language: {lang}[/bold]")
                try:
                    results = await client.search_repos(
                        f"language:{lang}", per_page=per_language
                    )
                except httpx.HTTPError as exc:
                    console.print(f"  [red]search failed: {exc}[/red]")
                    continue

                for item in results:
                    full_name = item["full_name"]
                    try:
                        repo = await client.get_repo(
                            item["owner"]["login"], item["name"]
                        )
                        repo["topics"] = item.get("topics", [])
                        repo_id = await upsert_repo(session, repo)
                        console.print(f"  [green]{full_name}[/green] [dim]({repo_id})[/dim]")

                        try:
                            readme_text = await client.get_readme(
                                item["owner"]["login"], item["name"]
                            )
                            await upsert_readme(session, repo_id, readme_text)
                        except httpx.HTTPStatusError:
                            pass

                        repo_topics = await client.get_topics(
                            item["owner"]["login"], item["name"]
                        )
                        if repo_topics:
                            await insert_topics(session, repo_id, repo_topics)

                        await session.commit()
                    except httpx.HTTPError as exc:
                        console.print(f"  [yellow]skipped {full_name}: {exc}[/yellow]")

            row = await session.execute(
                __import__("sqlalchemy").text("SELECT COUNT(*) FROM repos")
            )
            total = row.scalar()
            console.print(f"\n[bold green]Total repos in database: {total}[/bold green]")

    asyncio.run(run())


@app.command()
def deps(
    owner: str = typer.Argument(..., help="Repo owner"),
    name: str = typer.Argument(..., help="Repo name"),
) -> None:
    """Fetch dependency manifests from a GitHub repo and persist them."""
    _configure_logging()

    async def run() -> None:
        from reporelay_ingest.manifest import MANIFEST_FILES, parse_manifest

        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        async with GitHubClient() as client, session:
            console.print(f"[bold]Scanning deps for {owner}/{name}...[/bold]")

            row = await session.execute(
                __import__("sqlalchemy").text(
                    "SELECT id FROM repos WHERE owner = :owner AND name = :name"
                ),
                {"owner": owner, "name": name},
            )
            result = row.fetchone()
            if not result:
                console.print(f"[red]Repo {owner}/{name} not in database. Run save-repo first.[/red]")
                raise typer.Exit(code=1)
            repo_id = result[0]

            for manifest_name in MANIFEST_FILES:
                content = await client.get_file_content(owner, name, manifest_name)
                if content is None:
                    continue
                console.print(f"  found: {manifest_name} ({len(content)} bytes)")
                deps = parse_manifest(manifest_name, content)
                await insert_dependencies(session, repo_id, deps)
                console.print(f"  [dim]inserted {len(deps)} dependencies[/dim]")

            await session.commit()
            console.print("[bold green]Dependencies saved.[/bold green]")

    asyncio.run(run())


@app.command()
def deps_all() -> None:
    """Ingest dependencies for all repos in the database."""
    _configure_logging()

    async def run() -> None:
        from reporelay_ingest.manifest import MANIFEST_FILES, parse_manifest

        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        async with GitHubClient() as client, session:
            row = await session.execute(
                __import__("sqlalchemy").text("SELECT id, owner, name FROM repos")
            )
            repos = row.fetchall()

            for repo_id, owner, name in repos:
                console.print(f"[dim]Scanning {owner}/{name}...[/dim]")
                for manifest_name in MANIFEST_FILES:
                    content = await client.get_file_content(owner, name, manifest_name)
                    if content is None:
                        continue
                    deps = parse_manifest(manifest_name, content)
                    if deps:
                        await insert_dependencies(session, repo_id, deps)
                        console.print(f"  {owner}/{name}: {len(deps)} deps from {manifest_name}")

            await session.commit()

            row = await session.execute(
                __import__("sqlalchemy").text("SELECT COUNT(*) FROM dependency_edges")
            )
            total = row.scalar()
            console.print(f"\n[bold green]Total dependency edges: {total}[/bold green]")

    asyncio.run(run())


@app.command()
def load_stars(
    file: str = typer.Argument(..., help="Path to stars JSONL file"),
) -> None:
    """Bulk load star events from a JSONL file into star_events."""
    _configure_logging()

    async def run() -> None:
        from reporelay_ingest.stars import load_stars_jsonl

        import reporelay_ingest.storage as storage_mod

        events = load_stars_jsonl(file)
        session = await storage_mod._get_session()
        async with session:
            count = await insert_star_events(session, events)
            await session.commit()
            console.print(f"[bold green]Loaded {count} star events.[/bold green]")

    asyncio.run(run())


@app.command()
def gen_stars(
    file: str = typer.Argument(..., help="Output JSONL file path"),
    events_per_repo: int = typer.Option(50, help="Star events per repo"),
) -> None:
    """Generate synthetic star events for all repos in the database."""
    _configure_logging()

    async def run() -> None:
        from reporelay_ingest.stars import generate_sample_stars, save_stars_jsonl

        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        async with session:
            row = await session.execute(
                __import__("sqlalchemy").text("SELECT id FROM repos")
            )
            repo_ids = [r[0] for r in row.fetchall()]

            row = await session.execute(
                __import__("sqlalchemy").text("SELECT id FROM users")
            )
            user_ids = [r[0] for r in row.fetchall()]

        if not repo_ids or not user_ids:
            console.print("[red]No repos or users in database. Seed first.[/red]")
            raise typer.Exit(code=1)

        events = generate_sample_stars(repo_ids, user_ids, events_per_repo)
        save_stars_jsonl(events, file)
        console.print(f"[bold green]Generated {len(events)} star events → {file}[/bold green]")

    asyncio.run(run())


@app.command()
def refresh_co_stars() -> None:
    """Refresh the co_star_counts materialized view."""
    _configure_logging()

    async def run() -> None:
        import reporelay_ingest.storage as storage_mod
        from sqlalchemy import text

        session = await storage_mod._get_session()
        async with session:
            await session.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY co_star_counts"))
            await session.commit()
            console.print("[bold green]Materialized view refreshed.[/bold green]")

    asyncio.run(run())


@app.command()
def compute_2hop(
    repo_id: int = typer.Argument(..., help="Source repo ID"),
) -> None:
    """Compute 2-hop neighbors for a repo and store in two_hop_neighbors table."""
    _configure_logging()

    async def run() -> None:
        from reporelay_ingest.graph import sync_two_hop_table

        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        async with session:
            count = await sync_two_hop_table(session, repo_id)
            await session.commit()
            console.print(f"[bold green]Computed {count} two-hop neighbors for repo {repo_id}.[/bold green]")

    asyncio.run(run())


@app.command()
def compute_2hop_all() -> None:
    """Compute 2-hop neighbors for all repos in the database."""
    _configure_logging()

    async def run() -> None:
        from reporelay_ingest.graph import sync_two_hop_table
        from sqlalchemy import text

        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        async with session:
            row = await session.execute(text("SELECT id, full_name FROM repos"))
            repos = row.fetchall()

            total = 0
            for repo_id, full_name in repos:
                count = await sync_two_hop_table(session, repo_id)
                total += count
                console.print(f"  {full_name}: {count} two-hop neighbors")

            await session.commit()
            console.print(f"\n[bold green]Total two-hop neighbors computed: {total}[/bold green]")

    asyncio.run(run())


@app.command(name="refresh-2hop")
def refresh_2hop() -> None:
    """Refresh all two-hop neighbor computations."""
    compute_2hop_all()


@app.command()
def embed_readme(
    repo_id: int = typer.Argument(..., help="Repo ID to embed"),
) -> None:
    """Compute and store a README embedding for a repo."""
    _configure_logging()

    async def run() -> None:
        from reporelay_ingest.embedding import embed_readme as do_embed

        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        async with session:
            ok = await do_embed(session, repo_id)
            if ok:
                await session.commit()
                console.print(f"[bold green]Embedded repo {repo_id}.[/bold green]")
            else:
                console.print(f"[yellow]No README text for repo {repo_id}.[/yellow]")

    asyncio.run(run())


@app.command()
def embed_all() -> None:
    """Embed all repos with README text but no embedding."""
    _configure_logging()

    async def run() -> None:
        from reporelay_ingest.embedding import embed_all as do_embed_all

        import reporelay_ingest.storage as storage_mod

        session = await storage_mod._get_session()
        async with session:
            count = await do_embed_all(session)
            await session.commit()
            console.print(f"[bold green]Embedded {count} repos.[/bold green]")

    asyncio.run(run())


if __name__ == "__main__":
    app()
