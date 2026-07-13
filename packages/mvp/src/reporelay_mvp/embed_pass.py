"""
Bulk embed the top N repos by stars.

Two modes:
  - Full: fetch READMEs from GitHub + embed both README and description
  - Description-only: embed stored descriptions only, no GitHub calls

The seed phase stores metadata + topics from the search API but
doesn't compute embeddings. This pass fills the gap so pgvector ANN
has actual vectors to work with.

For Gemini paid tier (1500 RPM), batching 96 texts per call means
50k repos finish in ~520 batched calls (~20 min).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from reporelay_mvp import data
from reporelay_mvp.embedding import (
    _configure_gemini,
    embed_text,
    embedding_mode,
)
from reporelay_mvp.github import _auth_client, fetch_readme
from reporelay_mvp.keyword_extractor import extract_keywords_from_repo
from reporelay_mvp.purpose import get_effective_description, clean_description
from reporelay_mvp.settings import get_mvp_settings

logger = logging.getLogger(__name__)


# Retry on rate-limit and transient API errors. Don't retry on 404.
_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
try:
    from google.api_core.exceptions import (
        InternalServerError,
        ServiceUnavailable,
        TooManyRequests,
    )
    _RETRYABLE_EXCEPTIONS = (
        TooManyRequests,
        ServiceUnavailable,
        InternalServerError,
        *_RETRYABLE_EXCEPTIONS,
    )
except ImportError:
    pass


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(8),
    reraise=True,
)
async def _embed_with_retry(text: str) -> list[float]:
    """Embed single text with exponential backoff on rate limits."""
    return await embed_text(text)


# Simple fixed-pacing rate limiter for Gemini API
# Free tier: 3s. Paid tier: 0s — 1500 RPM handles anything.
_GEMINI_CALL_DELAY = 0.0  # paid tier — zero delay, full speed


def _pace_gemini_call() -> None:
    """Minimal delay between Gemini API calls. Paid tier: 0.2s to avoid burst limits."""
    time.sleep(_GEMINI_CALL_DELAY)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _embed_batch_gemini(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch of texts via Gemini's batch API.

    Gemini supports batching up to 100 texts per call. This is the
    key optimization that turns 22k individual calls into ~220 batched
    calls (100x reduction).

    For gemini-embedding-001 with output_dimensionality=512.
    Retries on 429 rate limits with short exponential backoff.
    Paces requests to stay under 80 RPM.
    """
    _configure_gemini()
    import google.generativeai as genai  # type: ignore[import-not-found]

    def _call() -> list[list[float]]:
        _pace_gemini_call()  # blocks until a slot is available
        result = genai.embed_content(  # type: ignore[attr-defined]
            model="models/gemini-embedding-001",
            content=texts,
            task_type=task_type,
            output_dimensionality=512,
        )
        return [[float(x) for x in emb] for emb in result["embedding"]]

    return await asyncio.to_thread(_call)


async def _process_repo(
    client: httpx.AsyncClient,
    repo: Any,
) -> tuple[int, str | None, str | None] | None:
    """Fetch README + prepare texts for a single repo.

    Returns (repo_id, readme, description) tuple, or None only if the
    repo has nothing embeddable at all (no readme AND no description).

    If the README fetch fails (rate limit, 404, network error), we
    still return a tuple with just the description — the description
    alone is enough to produce a useful embedding.
    """
    readme: str | None = None
    try:
        readme = await fetch_readme(client, repo.owner, repo.name)
    except Exception as exc:
        # Don't skip — we may still have a description
        logger.debug("readme fetch failed for %s/%s: %s", repo.owner, repo.name, exc)
        readme = None

    if readme and not readme.strip():
        readme = None

    if not readme and not (repo.description and repo.description.strip()):
        # Nothing to embed at all
        return None

    # Build the description: prefer repo's own, fall back to README extraction
    if repo.description and len(repo.description.strip()) >= 25:
        effective_desc = repo.description
    elif readme:
        effective_desc = get_effective_description(repo.description, readme)
    else:
        effective_desc = None

    # Extract keywords from description + README
    keywords: list[str] = []
    try:
        keywords = extract_keywords_from_repo(repo.description, readme)
    except Exception as exc:
        logger.debug("keyword extraction failed for %s/%s: %s", repo.owner, repo.name, exc)

    return (repo.id, readme[:8000] if readme else None, effective_desc, keywords)


async def embed_top(
    *,
    limit: int = 7000,
    concurrency: int = 8,
    batch_size: int = 96,
    description_only: bool = True,
) -> dict[str, int]:
    """
    Embed the top `limit` repos (by stars) that don't yet have an embedding.

    Batches multiple texts per API call for efficiency. For Gemini
    paid tier (1500 RPM), batching 96 texts per call means 50k repos
    finish in ~520 batched calls (~20 min total).

    If `description_only=True` (default), only embed descriptions
    (no GitHub API calls). Uses stored description text from the DB.
    This is the fast path — no rate limit concerns.

    If `description_only=False`, also fetches READMEs from GitHub
    and embeds both README + description (uses GitHub API rate limit).

    Returns a stats dict with attempted / succeeded / failed counts.
    """
    settings = get_mvp_settings()
    session = await data.get_session()

    try:
        if description_only:
            repos = await data.list_repos_needing_description_embedding(
                session, limit=limit
            )
        else:
            repos = await data.list_repos_needing_embedding(session, limit=limit)

        if not repos:
            logger.info("no repos need embedding")
            return {"attempted": 0, "succeeded": 0, "failed": 0}

        mode = embedding_mode()
        logger.info(
            "embedding %d repos (concurrency=%d, batch_size=%d, mode=%s, description_only=%s)",
            len(repos), concurrency, batch_size, mode, description_only,
        )

        succeeded = 0
        failed = 0
        sem = asyncio.Semaphore(concurrency)

        # ── Phase 1: prepare texts ──────────────────────────────────
        batch_inputs: list[tuple[int, str | None, str | None, list[str]]] = []

        if description_only:
            for repo in repos:
                desc = repo.description if repo.description else None
                if not desc or not desc.strip():
                    continue
                from reporelay_mvp.purpose import is_good_description

                if not is_good_description(desc):
                    continue
                batch_inputs.append((repo.id, None, desc, []))
        else:
            async with _auth_client(settings.github_token) as client:

                async def fetch_one(repo: Any) -> Any:
                    async with sem:
                        return await _process_repo(client, repo)

                fetched_results = await asyncio.gather(
                    *[fetch_one(r) for r in repos],
                    return_exceptions=True,
                )

            for result in fetched_results:
                if isinstance(result, Exception):
                    failed += 1
                    continue
                if result is None:
                    continue
                batch_inputs.append(result)

        if not batch_inputs:
            logger.info("no repos with usable texts")
            return {"attempted": len(repos), "succeeded": 0, "failed": failed}

        # ── Phase 2: batch embed via Gemini API ──────────────────────
        all_texts: list[str] = []
        text_meta: list[tuple[int, str]] = []  # (repo_id, "readme"|"desc")
        for repo_id, readme, desc, _keywords in batch_inputs:
            if readme and readme.strip():
                all_texts.append(readme)
                text_meta.append((repo_id, "readme"))
            if desc and desc.strip():
                all_texts.append(clean_description(desc))
                text_meta.append((repo_id, "desc"))

        chunk_size = batch_size
        total_chunks = (len(all_texts) + chunk_size - 1) // chunk_size

        for i in range(0, len(all_texts), chunk_size):
            chunk_num = i // chunk_size + 1
            chunk_texts = all_texts[i : i + chunk_size]
            chunk_meta = text_meta[i : i + chunk_size]
            try:
                vectors = await _embed_batch_gemini(
                    chunk_texts, task_type="retrieval_document"
                )

                for (repo_id, kind), vector in zip(chunk_meta, vectors, strict=True):
                    try:
                        if kind == "readme":
                            await _safe_set_embedding(session, repo_id, vector)
                        else:
                            await _safe_set_description(session, repo_id, vector)
                    except Exception as exc:
                        logger.warning("DB write failed for repo %d %s: %s", repo_id, kind, exc)
                        continue

                # Persist keywords for repos in this chunk
                for repo_id, _readme, _desc, keywords in batch_inputs:
                    if keywords:
                        try:
                            await data.set_keywords(session, repo_id=repo_id, keywords=keywords)
                        except Exception:
                            pass

                await session.commit()
                succeeded += len({repo_id for repo_id, _ in chunk_meta})

                if chunk_num % 10 == 0 or chunk_num == total_chunks:
                    logger.info(
                        "  embed progress: chunk %d/%d (%d repos done so far)",
                        chunk_num, total_chunks, succeeded,
                    )
            except Exception as exc:
                logger.warning("batch embed failed at chunk %d: %s", i, exc)
                failed += len({repo_id for repo_id, _ in chunk_meta})
                continue

        logger.info(
            "embed pass complete: %d attempted, %d succeeded, %d failed",
            len(repos), succeeded, failed,
        )
        return {"attempted": len(repos), "succeeded": succeeded, "failed": failed}
    finally:
        await session.close()


async def _safe_set_embedding(session: Any, repo_id: int, embedding: list[float]) -> bool:
    try:
        await data.set_embedding(session, repo_id=repo_id, embedding=embedding)
        return True
    except Exception as exc:
        logger.warning("set_embedding failed for %d: %s", repo_id, exc)
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
        await data.set_description_embedding(
            session, repo_id=repo_id, description_embedding=desc_emb
        )
        return True
    except Exception:
        pass
    try:
        await session.rollback()
        new_session = await data.get_session()
        try:
            await data.set_description_embedding(
                new_session, repo_id=repo_id, description_embedding=desc_emb
            )
            await new_session.commit()
            return True
        finally:
            await new_session.close()
    except Exception as exc:
        logger.warning("desc_embedding retry failed for %d: %s", repo_id, exc)
        return False
