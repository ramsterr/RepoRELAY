"""
Bulk embed the top N repos by stars.

The seed phase stores metadata + topics from the search API but
doesn't compute embeddings (the search payload doesn't carry the
README). This pass fills the gap so pgvector ANN has actual
vectors to work with.

For each repo:
  1. fetch /repos/{owner}/{name}/readme (1 REST call)
  2. embed_text() → 384-dim vector
  3. UPDATE mvp_repos SET embedding = ..., embedded_at = NOW()

We only fetch the README — metadata, topics, language, stars are
already in the row from the seed. README truncation to 8000 chars
happens inside embed_text() (already wired).

Pacing: defaults to concurrency=4 and 0.1s sleep between
launches, which keeps us under the 5,000 REST calls/hr limit
(~1.4 req/s budget) while still completing 1,000 repos in
~3 min on a warm box.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from reporelay_mvp import data
from reporelay_mvp.embedding import embed_text
from reporelay_mvp.github import _auth_client, fetch_readme
from reporelay_mvp.settings import get_mvp_settings

logger = logging.getLogger(__name__)


async def _embed_one(
    client: httpx.AsyncClient,
    session: Any,
    repo_id: int,
    owner: str,
    name: str,
    description: str | None = None,
) -> bool:
    """Fetch README + embed README + embed description + persist. Returns True on success."""
    try:
        readme = await fetch_readme(client, owner, name)
    except Exception as exc:
        logger.warning("readme fetch failed for %s/%s: %s", owner, name, exc)
        return False
    if not readme.strip():
        logger.info("  %s/%s: empty readme, skipping", owner, name)
        return False
    try:
        embedding = await embed_text(readme[:8000])
    except Exception as exc:
        logger.warning("embedding failed for %s/%s: %s", owner, name, exc)
        return False

    # Write README embedding with Neon retry
    if not await _safe_set_embedding(session, repo_id, embedding):
        return False

    if description and description.strip():
        try:
            desc_emb = await embed_text(description)
            await _safe_set_description(session, repo_id, desc_emb)
        except Exception as exc:
            logger.warning("description embedding failed for %s/%s: %s", owner, name, exc)
    return True


async def _safe_set_embedding(session: Any, repo_id: int, embedding: list[float]) -> bool:
    try:
        await data.set_embedding(session, repo_id=repo_id, embedding=embedding)
        return True
    except Exception as exc:
        logger.warning("set_embedding failed for %d (Neon): %s", repo_id, exc)
    try:
        await session.rollback()
        new_session = await data.get_session()
        try:
            await data.set_embedding(new_session, repo_id=repo_id, embedding=embedding)
            await new_session.commit()
            return True
        finally:
            await new_session.close()
    except Exception as exc2:
        logger.warning("retry also failed for %d: %s", repo_id, exc2)
        return False


async def _safe_set_description(session: Any, repo_id: int, desc_emb: list[float]) -> bool:
    try:
        await data.set_description_embedding(session, repo_id=repo_id, description_embedding=desc_emb)
        return True
    except Exception:
        pass
    try:
        await session.rollback()
        new_session = await data.get_session()
        try:
            await data.set_description_embedding(new_session, repo_id=repo_id, description_embedding=desc_emb)
            await new_session.commit()
            return True
        finally:
            await new_session.close()
    except Exception as exc:
        logger.warning("desc_embedding retry failed for %d: %s", repo_id, exc)
        return False


async def embed_top(
    *,
    limit: int = 1000,
    concurrency: int = 4,
) -> dict[str, int]:
    """
    Embed the top `limit` repos (by stars) that don't yet have an
    embedding. Returns a stats dict with attempted / succeeded /
    failed counts.
    """
    settings = get_mvp_settings()
    session = await data.get_session()

    try:
        repos = await data.list_repos_needing_embedding(session, limit=limit)
        if not repos:
            logger.info("no repos need embedding")
            return {"attempted": 0, "succeeded": 0, "failed": 0}

        logger.info(
            "embedding %d repos (concurrency=%d, model will load on first use)",
            len(repos), concurrency,
        )

        succeeded = 0
        failed = 0
        sem = asyncio.Semaphore(concurrency)

        async with _auth_client(settings.github_token) as client:

            async def run_one(repo: Any) -> bool:
                async with sem:
                    ok = await _embed_one(
                        client, session, repo.id, repo.owner, repo.name,
                        description=repo.description,
                    )
                    await asyncio.sleep(0.1)
                    return ok

            tasks = [run_one(r) for r in repos]
            for i, result in enumerate(await asyncio.gather(*tasks), start=1):
                if result:
                    succeeded += 1
                else:
                    failed += 1
                if i % 50 == 0 or i == len(repos):
                    logger.info("  progress: %d/%d", i, len(repos))

        await session.commit()
    finally:
        await session.close()

    logger.info(
        "embed pass complete: %d attempted, %d succeeded, %d failed",
        len(repos), succeeded, failed,
    )
    return {"attempted": len(repos), "succeeded": succeeded, "failed": failed}
