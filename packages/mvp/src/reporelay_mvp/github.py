"""
GitHub fetch + persist for the MVP.

A thin wrapper over httpx that pulls the four pieces of data the MVP
actually uses: metadata, README, topics, and dependency names.

We deliberately do not parse manifests here — the MVP gets dependency
names from the GitHub API dependency graph if available, otherwise we
leave the dependency list empty. The dependency feature still works
as long as some repos have deps populated.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx

from reporelay_mvp import data
from reporelay_mvp.embedding import embed_text
from reporelay_mvp.models import Repo
from reporelay_mvp.settings import get_mvp_settings

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubError(Exception):
    pass


class _RateLimited(Exception):
    pass


def _auth_headers(token: str) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "RepoRelay-MVP/0.1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _auth_client(token: str) -> httpx.AsyncClient:
    """Build a pre-configured httpx async client for the GitHub API."""
    return httpx.AsyncClient(
        base_url=GITHUB_API,
        headers=_auth_headers(token),
        timeout=httpx.Timeout(5.0, connect=3.0),
    )


async def _get(client: httpx.AsyncClient, path: str, **params: Any) -> dict[str, Any]:
    response = await client.get(path, params=params, follow_redirects=True)
    if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
        reset = response.headers.get("X-RateLimit-Reset", "?")
        logger.warning("rate limited, reset at %s", reset)
        raise _RateLimited(reset)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def _decode_base64_text(content: str) -> str:
    if not content:
        return ""
    padding = "=" * (-len(content) % 4)
    return base64.b64decode(content + padding).decode("utf-8", errors="replace")


async def fetch_repo_metadata(client: httpx.AsyncClient, owner: str, name: str) -> dict[str, Any]:
    return await _get(client, f"/repos/{owner}/{name}")


async def fetch_readme(client: httpx.AsyncClient, owner: str, name: str) -> str:
    try:
        data_dict = await _get(client, f"/repos/{owner}/{name}/readme")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return ""
        raise
    return _decode_base64_text(data_dict.get("content", ""))


async def fetch_topics(client: httpx.AsyncClient, owner: str, name: str) -> list[str]:
    try:
        response = await client.get(
            f"/repos/{owner}/{name}/topics",
            headers={"Accept": "application/vnd.github.mercy-preview+json"},
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return []
    if response.status_code != 200:
        return []
    payload = response.json()
    return list(payload.get("names", []))


async def fetch_dependencies(client: httpx.AsyncClient, owner: str, name: str) -> list[str]:
    """
    Use the GitHub dependency graph if exposed. Returns the package
    names (no version constraints). Empty list if the API is not
    available for this repo.
    """
    try:
        response = await client.get(f"/repos/{owner}/{name}/dependencies", follow_redirects=True)
    except httpx.HTTPError:
        return []
    if response.status_code != 200:
        return []
    payload = response.json()
    packages: list[str] = []
    for group in payload.get("packages", []):
        ecosystem = group.get("ecosystem", "").lower()
        if ecosystem not in {"npm", "pip", "cargo", "rubygems"}:
            continue
        for pkg in group.get("package_name", []) or []:
            packages.append(pkg)
    return packages


async def search_repos(
    owner: str, name: str, *, limit: int = 15, seed: int | None = None
) -> list[Repo]:
    """
    Discover related repos from GitHub and return ephemeral Repo objects.

    Results are NOT persisted — they're used as temporary candidates
    only. The seed varies the search (which topic, sort order, page)
    so different seeds return meaningfully different candidates.
    """
    import random as _random

    settings = get_mvp_settings()
    headers = _auth_headers(settings.github_token)
    timeout = httpx.Timeout(10.0, connect=8.0)

    async with httpx.AsyncClient(base_url=GITHUB_API, headers=headers, timeout=timeout) as client:
        topics = await fetch_topics(client, owner, name)
        metadata = await fetch_repo_metadata(client, owner, name)
        language = metadata.get("language")

    if not topics and not language:
        return []

    rng = _random.Random(seed) if seed is not None else _random.Random()

    # seed-aware: pick a different topic each time
    topic = rng.choice(topics) if topics else ""
    sort_choice = rng.choice(["stars", "updated", "forks"])
    page = rng.randint(1, max(1, limit // 5)) if seed is not None else 1
    per_page = min(limit * 2, 100)

    try:
        async with httpx.AsyncClient(
            base_url=GITHUB_API, headers=headers, timeout=timeout
        ) as client:
            raw = await search_repositories(
                client,
                topics=[topic] if topic else None,
                language=language,
                min_stars=100,
                sort=sort_choice,
                per_page=per_page,
                page=page,
            )
    except Exception:
        return []

    return [_search_item_to_repo(item) for item in raw.get("items", [])]


def _search_item_to_repo(item: dict[str, Any]) -> Repo:
    return Repo(
        id=int(item["id"]),
        owner=item["owner"]["login"],
        name=item["name"],
        full_name=item["full_name"],
        description=item.get("description"),
        language=item.get("language"),
        topics=list(item.get("topics") or []),
        stars=int(item.get("stargazers_count") or 0),
        dependencies=[],
        embedding=None,
    )


async def search_repositories(
    client: httpx.AsyncClient,
    *,
    topics: list[str] | None = None,
    language: str | None = None,
    min_stars: int = 100,
    sort: str = "stars",
    order: str = "desc",
    per_page: int = 100,
    page: int = 1,
) -> dict[str, Any]:
    """
    Single GitHub search/repositories call. Returns the raw response
    payload (with `items`, `total_count`, etc.) so callers can either
    use the rows directly or bulk-upsert them.

    Query construction:
      - one topic at a time (the GitHub search API does NOT allow
        `topic:X OR topic:Y` — it returns 422; OR of qualifiers
        is unsupported)
      - language is used as a FALLBACK when no topics are available
      - `archived:false` is always added
      - stars floor is configurable
    """
    query_parts: list[str] = []
    if topics:
        # Filter out empty topics and pick the first one — see the
        # `iter_search_by_topics` helper below for proper OR-of-topics
        # semantics (one search per topic, results merged by the caller).
        first = next((t for t in topics if t), None)
        if first:
            query_parts.append(f"topic:{first}")
    elif language:
        query_parts.append(f"language:{language}")
    query_parts.append("archived:false")
    query_parts.append(f"stars:>{min_stars}")
    query = " ".join(query_parts)

    return await _get(
        client,
        "/search/repositories",
        q=query,
        sort=sort,
        order=order,
        per_page=per_page,
        page=page,
    )


async def quick_save(owner: str, name: str) -> int:
    """
    Lightweight fetch: metadata + topics only. Returns in ~2s — enough for
    topic/language-based recommendations while README + deps are fetched
    in the background. Returns the repo id.

    If the GitHub API is rate-limited, we create a skeleton DB row so the
    recommendation pipeline can still run with whatever signals are available.
    """
    settings = get_mvp_settings()
    headers = _auth_headers(settings.github_token)
    timeout = httpx.Timeout(10.0, connect=5.0)

    try:
        async with httpx.AsyncClient(base_url=GITHUB_API, headers=headers, timeout=timeout) as client:
            metadata, topics = await asyncio.gather(
                fetch_repo_metadata(client, owner, name),
                fetch_topics(client, owner, name),
            )
            repo_id = int(metadata["id"])
            language = metadata.get("language")
            stars = int(metadata.get("stargazers_count") or 0)
            description = metadata.get("description")
    except Exception as exc:
        logger.warning("quick_save API call failed — creating skeleton row: %s", exc)
        repo_id = abs(hash(f"{owner}/{name}")) % (10**9)
        language = None
        stars = 1
        description = None
        topics = []

    full_name = f"{owner}/{name}"

    session = await data.get_session()
    try:
        await data.upsert_repo(
            session,
            repo_id=repo_id,
            owner=owner,
            name=name,
            full_name=full_name,
            description=description,
            language=language,
            topics=topics,
            stars=stars,
            dependencies=[],
        )
        await data.set_embedding(session, repo_id=repo_id, embedding=[0.0] * 384)
        await session.commit()
    finally:
        await session.close()

    logger.info("quick-saved %s/%s (id=%d)", owner, name, repo_id)
    return repo_id


async def enrich_repo(owner: str, name: str) -> None:
    """
    Background task: fetch README + dependencies + embed. Called after
    quick_save to backfill the full data for future requests.
    """
    settings = get_mvp_settings()
    headers = _auth_headers(settings.github_token)
    timeout = httpx.Timeout(30.0, connect=10.0)

    try:
        async with httpx.AsyncClient(base_url=GITHUB_API, headers=headers, timeout=timeout) as client:
            readme, deps = await asyncio.gather(
                fetch_readme(client, owner, name),
                fetch_dependencies(client, owner, name),
            )

        session = await data.get_session()
        try:
            full_name = f"{owner}/{name}"
            existing = await data.get_repo(session, full_name)
            if existing is None:
                return
            if readme.strip():
                embedding = await embed_text(readme[:8000])
                await data.set_embedding(session, repo_id=existing.id, embedding=embedding)
            if deps:
                await data.upsert_repo(
                    session,
                    repo_id=existing.id,
                    owner=owner,
                    name=name,
                    full_name=full_name,
                    description=existing.description,
                    language=existing.language,
                    topics=existing.topics,
                    stars=existing.stars,
                    dependencies=deps,
                )
            await session.commit()
        finally:
            await session.close()

        logger.info("enriched %s/%s (deps=%d, embedded=%s)", owner, name, len(deps), bool(readme.strip()))
    except Exception as exc:
        logger.warning("background enrich failed for %s/%s: %s", owner, name, exc)


async def save_repo(owner: str, name: str) -> int:
    """Full fetch: metadata + README + topics + deps + embed. Used by CLI."""
    settings = get_mvp_settings()
    headers = _auth_headers(settings.github_token)
    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(base_url=GITHUB_API, headers=headers, timeout=timeout) as client:
        metadata = await fetch_repo_metadata(client, owner, name)
        repo_id = int(metadata["id"])

        readme, topics, deps = await asyncio.gather(
            fetch_readme(client, owner, name),
            fetch_topics(client, owner, name),
            fetch_dependencies(client, owner, name),
        )

        language = metadata.get("language")
        stars = int(metadata.get("stargazers_count") or 0)

        session = await data.get_session()
        try:
            await data.upsert_repo(
                session,
                repo_id=repo_id,
                owner=owner,
                name=name,
                full_name=f"{owner}/{name}",
                description=metadata.get("description"),
                language=language,
                topics=topics,
                stars=stars,
                dependencies=deps,
            )
            if readme.strip():
                embedding = await embed_text(readme[:8000])
                await data.set_embedding(session, repo_id=repo_id, embedding=embedding)
            await session.commit()
        finally:
            await session.close()

    logger.info("saved %s/%s (id=%d)", owner, name, repo_id)
    return repo_id
