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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from reporelay_mvp import data
from reporelay_mvp.embedding import embed_text
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


@retry(
    retry=retry_if_exception_type(_RateLimited),
    wait=wait_exponential(multiplier=2, min=30, max=600),
    stop=stop_after_attempt(3),
    reraise=True,
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


async def fetch_repo_metadata(
    client: httpx.AsyncClient, owner: str, name: str
) -> dict[str, Any]:
    return await _get(client, f"/repos/{owner}/{name}")


async def fetch_readme(
    client: httpx.AsyncClient, owner: str, name: str
) -> str:
    try:
        data_dict = await _get(client, f"/repos/{owner}/{name}/readme")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return ""
        raise
    return _decode_base64_text(data_dict.get("content", ""))


async def fetch_topics(
    client: httpx.AsyncClient, owner: str, name: str
) -> list[str]:
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


async def fetch_dependencies(
    client: httpx.AsyncClient, owner: str, name: str
) -> list[str]:
    """
    Use the GitHub dependency graph if exposed. Returns the package
    names (no version constraints). Empty list if the API is not
    available for this repo.
    """
    try:
        response = await client.get(
            f"/repos/{owner}/{name}/dependencies", follow_redirects=True
        )
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
    owner: str, name: str, *, limit: int = 5
) -> list[dict[str, Any]]:
    """
    Discover related repos from GitHub via a single topic+language search.

    Returns popular repos similar to the source, limited to avoid rate
    limits. Used to expand small candidate pools.
    """
    settings = get_mvp_settings()
    headers = _auth_headers(settings.github_token)
    timeout = httpx.Timeout(15.0, connect=10.0)

    async with httpx.AsyncClient(
        base_url=GITHUB_API, headers=headers, timeout=timeout
    ) as client:
        topics = await fetch_topics(client, owner, name)
        metadata = await fetch_repo_metadata(client, owner, name)
        language = metadata.get("language")

    primary_topic = topics[0] if topics else ""
    if not primary_topic and not language:
        return []

    query_parts: list[str] = []
    if primary_topic:
        query_parts.append(f"topic:{primary_topic}")
    if language:
        query_parts.append(f"language:{language}")
    query_parts.append("stars:>500")
    query = " ".join(query_parts)

    try:
        async with httpx.AsyncClient(
            base_url=GITHUB_API, headers=headers, timeout=timeout
        ) as client:
            raw = await _get(
                client,
                "/search/repositories",
                q=query,
                sort="stars",
                order="desc",
                per_page=limit,
            )
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for item in raw.get("items", []):
        results.append(item)
    return results


async def save_repo(owner: str, name: str) -> int:
    """
    Fetch a single repo from GitHub, persist its row, and embed its
    README. Returns the repo id.
    """
    settings = get_mvp_settings()
    headers = _auth_headers(settings.github_token)
    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(
        base_url=GITHUB_API, headers=headers, timeout=timeout
    ) as client:
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
                owner=metadata["owner"]["login"],
                name=metadata["name"],
                full_name=metadata["full_name"],
                description=metadata.get("description"),
                language=language,
                topics=topics,
                stars=stars,
                dependencies=deps,
            )
            if readme.strip():
                embedding = embed_text(readme[:8000])
                await data.set_embedding(session, repo_id=repo_id, embedding=embedding)
            await session.commit()
        finally:
            await session.close()

    logger.info("saved %s/%s (id=%d)", owner, name, repo_id)
    return repo_id
