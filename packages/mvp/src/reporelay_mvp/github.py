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
from reporelay_mvp.purpose import get_effective_description
from reporelay_mvp.settings import get_mvp_settings
from reporelay_mvp.topic_inference import infer_topics_for_repo

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
        timeout=httpx.Timeout(15.0, connect=5.0),
    )


# ── Token rotation: use GITHUB_TOKEN_2 as fallback when primary is rate-limited ──
_active_github_token_index = 0


def _get_active_token() -> str:
    """Return the currently active GitHub token, rotating on rate limit."""
    from reporelay_mvp.settings import get_mvp_settings

    settings = get_mvp_settings()
    tokens = [t for t in (settings.github_token, settings.github_token_2) if t]
    if not tokens:
        return ""
    return tokens[_active_github_token_index % len(tokens)]


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
    """Fetch a repo's README. Handles GitHub rate limits with backoff.

    GitHub returns rate-limit info in headers:
      X-RateLimit-Remaining — requests left in current hour
      X-RateLimit-Reset     — unix timestamp when quota resets
      Retry-After          — seconds to wait (sent on 429/403 secondary rate limits)
    """
    # Conservative per-request pacing: GitHub authenticated limit is
    # 5000 req/hr = 1.4 req/sec. With 8 concurrent fetches we hit
    # that easily, so we add a small sleep between requests.
    # This is checked BEFORE the request via a global counter.
    await _pace_github_request()

    try:
        data_dict = await _get(client, f"/repos/{owner}/{name}/readme")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return ""
        if exc.response.status_code in (403, 429):
            # Rate limited. Check headers for wait time.
            await _handle_github_rate_limit(exc.response)
            # Retry once
            try:
                data_dict = await _get(client, f"/repos/{owner}/{name}/readme")
            except httpx.HTTPStatusError as exc2:
                if exc2.response.status_code == 404:
                    return ""
                raise
            return _decode_base64_text(data_dict.get("content", ""))
        raise
    return _decode_base64_text(data_dict.get("content", ""))


# ── Global rate limiter for GitHub API ─────────────────────────────────
import time
import asyncio
from collections import deque

_github_request_times: deque[float] = deque()
_github_lock = asyncio.Lock()
_GITHUB_RATE_LIMIT = 4900  # use full 5000/hr quota, GitHub's own 403 handles the rest
_GITHUB_WINDOW_SECONDS = 3600.0  # 1 hour


async def _pace_github_request() -> None:
    """Block until a GitHub request slot is available.

    Tracks request times in a sliding window. Limits to
    _GITHUB_RATE_LIMIT requests per hour (conservative — actual
    limit is 5000 for authenticated users).
    """
    async with _github_lock:
        now = time.monotonic()
        # Drop timestamps older than 1 hour
        while _github_request_times and (now - _github_request_times[0]) > _GITHUB_WINDOW_SECONDS:
            _github_request_times.popleft()
        if len(_github_request_times) >= _GITHUB_RATE_LIMIT:
            sleep_for = _GITHUB_WINDOW_SECONDS - (now - _github_request_times[0]) + 1.0
            logger.warning(
                "GitHub rate limit approaching: %d requests in last hour, "
                "sleeping %.0fs",
                len(_github_request_times), sleep_for,
            )
            await asyncio.sleep(sleep_for)
            # Reset after sleep
            _github_request_times.clear()
        _github_request_times.append(time.monotonic())


async def _handle_github_rate_limit(response: Any) -> None:
    """Sleep until GitHub rate limit resets, based on response headers."""
    global _active_github_token_index

    retry_after = response.headers.get("Retry-After")
    reset_at = response.headers.get("X-RateLimit-Reset")
    remaining = int(response.headers.get("X-RateLimit-Remaining", "1"))

    if remaining > 0:
        # Not rate-limited on this token — just respect Retry-After
        if retry_after:
            wait = float(retry_after)
            logger.warning("GitHub Retry-After: sleeping %.0fs", wait)
            await asyncio.sleep(wait)
        return

    # Rate limited. Switch to the next token if available.
    _active_github_token_index += 1
    next_token = _get_active_token()[:15] + "..."
    logger.warning("GitHub token rate limited — switched to token %s (index %d)", next_token, _active_github_token_index)

    if retry_after:
        wait = float(retry_after)
        await asyncio.sleep(wait)
    elif reset_at:
        now = time.time()
        wait = max(0, float(reset_at) - now)
        logger.warning("GitHub rate limit resets in %.0fs", wait)
        await asyncio.sleep(wait)
    else:
        await asyncio.sleep(60)
        logger.warning("GitHub rate limited, sleeping 60s")
        await asyncio.sleep(60)


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


async def fetch_all(owner: str, name: str) -> dict[str, Any]:
    """Fetch ALL GitHub data for a repo in ONE roundtrip.

    Uses a single httpx client and fires 4 parallel requests:
      - metadata (owner/name, language, stars, description)
      - topics
      - README (decoded from base64)
      - dependencies

    Returns a dict with keys: metadata, topics, readme, deps, error.
    If the GitHub API fails, error is set and the other keys have
    safe defaults.

    This replaces the old pattern of 3 separate roundtrips:
      - quick_save() → metadata + topics
      - _embed_description_fast() → readme + deps
      - _expand_pool() → search
    """
    settings = get_mvp_settings()
    timeout = httpx.Timeout(12.0, connect=5.0)

    try:
        async with _auth_client(_get_active_token()) as client:
            # Fire all 4 requests at once — one roundtrip
            metadata_coro = fetch_repo_metadata(client, owner, name)
            topics_coro = fetch_topics(client, owner, name)
            readme_coro = fetch_readme(client, owner, name)
            deps_coro = fetch_dependencies(client, owner, name)

            metadata, topics, readme_text, deps = await asyncio.gather(
                metadata_coro, topics_coro, readme_coro, deps_coro,
                return_exceptions=True,
            )

        # Handle partial failures gracefully
        error_parts: list[str] = []
        if isinstance(metadata, Exception):
            error_parts.append(f"metadata: {metadata}")
            metadata = {"id": abs(hash(f"{owner}/{name}")) % (10**9)}
        if isinstance(topics, Exception):
            error_parts.append(f"topics: {topics}")
            topics = []
        if isinstance(readme_text, Exception):
            error_parts.append(f"readme: {readme_text}")
            readme_text = ""
        if isinstance(deps, Exception):
            error_parts.append(f"deps: {deps}")
            deps = []

        return {
            "metadata": metadata if not isinstance(metadata, Exception) else {},
            "topics": topics if not isinstance(topics, Exception) else [],
            "readme": readme_text if not isinstance(readme_text, Exception) else "",
            "deps": deps if not isinstance(deps, Exception) else [],
            "error": "; ".join(error_parts) if error_parts else None,
        }
    except Exception as exc:
        logger.warning("fetch_all failed for %s/%s: %s", owner, name, exc)
        return {
            "metadata": {"id": abs(hash(f"{owner}/{name}")) % (10**9)},
            "topics": [],
            "readme": "",
            "deps": [],
            "error": str(exc),
        }


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
        description_embedding=None,
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
        # Leave embedding columns NULL so:
        # 1. `embedding IS NULL` cleanly identifies rows that need to be
        #    embedded (see list_repos_needing_embedding, recommend flow)
        # 2. fetch_vector_neighbors won't match a zero-vector source
        #    (which breaks cosine distance)
        # 3. It's obvious from a quick DB inspection whether a row has
        #    been embedded yet.
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

            # Fill in a good description from README if the existing one
            # is missing, too short, or generic filler. This ensures the
            # description_cosine_sim feature has real signal.
            effective_desc = get_effective_description(
                existing.description, readme if readme.strip() else None,
            )
            desc_changed = effective_desc and effective_desc != existing.description
            if desc_changed:
                await data.upsert_repo(
                    session,
                    repo_id=existing.id,
                    owner=owner,
                    name=name,
                    full_name=full_name,
                    description=effective_desc,
                    language=existing.language,
                    topics=existing.topics,
                    stars=existing.stars,
                    dependencies=existing.dependencies,
                )
                # Re-embed the description so the new text gets a vector
                desc_emb = await embed_text(effective_desc)
                await data.set_description_embedding(
                    session, repo_id=existing.id, description_embedding=desc_emb,
                )

            # Infer topics from README + description if repo has few
            inferred_topics: list[str] = []
            if len(existing.topics) < 3:
                inferred_topics = infer_topics_for_repo(
                    effective_desc or existing.description, readme, existing.topics,
                )
                if inferred_topics:
                    await data.update_topics(
                        session, repo_id=existing.id, topics=inferred_topics,
                    )

            if deps:
                await data.upsert_repo(
                    session,
                    repo_id=existing.id,
                    owner=owner,
                    name=name,
                    full_name=full_name,
                    description=effective_desc or existing.description,
                    language=existing.language,
                    topics=existing.topics,
                    stars=existing.stars,
                    dependencies=deps,
                )
            await session.commit()
        finally:
            await session.close()

        logger.info(
            "enriched %s/%s (deps=%d, embedded=%s, inferred_topics=%d, desc_changed=%s)",
            owner, name, len(deps), bool(readme.strip()), len(inferred_topics), desc_changed,
        )
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
        raw_description = metadata.get("description")

        # Use the best available description — the repo's own if it's
        # substantive, otherwise extract purpose from the README.
        description = get_effective_description(raw_description, readme)

        # Infer topics from README + description if GitHub gave us few
        if len(topics) < 3:
            inferred = infer_topics_for_repo(description, readme, topics)
            if inferred:
                topics = list(dict.fromkeys(topics + inferred))  # dedupe, preserve order

        session = await data.get_session()
        try:
            await data.upsert_repo(
                session,
                repo_id=repo_id,
                owner=owner,
                name=name,
                full_name=f"{owner}/{name}",
                description=description,
                language=language,
                topics=topics,
                stars=stars,
                dependencies=deps,
            )
            if readme.strip():
                embedding = await embed_text(readme[:8000])
                await data.set_embedding(session, repo_id=repo_id, embedding=embedding)
            if description:
                desc_emb = await embed_text(description)
                await data.set_description_embedding(
                    session, repo_id=repo_id, description_embedding=desc_emb,
                )
            await session.commit()
        finally:
            await session.close()

    logger.info("saved %s/%s (id=%d, topics=%d)", owner, name, repo_id, len(topics))
    return repo_id
