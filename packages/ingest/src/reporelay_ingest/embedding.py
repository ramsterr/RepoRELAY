"""
Embedding pipeline for RepoRelay readme_texts.

Uses sentence-transformers all-MiniLM-L6-v2 (384-dim) for MVP.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL: Any = None


def _get_model() -> Any:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model...")
        start = time.monotonic()
        _EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        elapsed = time.monotonic() - start
        logger.info("Model loaded in %.1fs (dim=%d)", elapsed, _get_embedding_dim())
    return _EMBEDDING_MODEL


def _get_embedding_dim() -> int:
    model = _get_model()
    return model.get_embedding_dimension()


def compute_embedding(text: str) -> list[float]:
    """Compute a 384-dim embedding vector for the given text."""
    if not text or not text.strip():
        dim = 384
        return [0.0] * dim
    model = _get_model()
    embedding = model.encode(text, show_progress_bar=False, normalize_embeddings=False)
    return embedding.tolist()


async def embed_readme(session: AsyncSession, repo_id: int) -> bool:
    """Read raw_text for a repo, compute embedding, and store it."""
    row = await session.execute(
        text("SELECT raw_text FROM readme_texts WHERE repo_id = :repo_id"),
        {"repo_id": repo_id},
    )
    result = row.fetchone()
    if not result or not result[0]:
        logger.warning("No README text for repo %d", repo_id)
        return False

    raw_text = result[0]
    embedding = compute_embedding(raw_text)

    await session.execute(
        text(
            """
            UPDATE readme_texts
            SET embedding = :embedding,
                embedded_at = NOW()
            WHERE repo_id = :repo_id
            """
        ),
        {"repo_id": repo_id, "embedding": embedding},
    )
    await session.flush()
    logger.info("Embedded repo %d (%d dims)", repo_id, len(embedding))
    return True


async def embed_all(session: AsyncSession) -> int:
    """Embed all repos that have README text but no embedding."""
    row = await session.execute(
        text(
            """
            SELECT repo_id, raw_text FROM readme_texts
            WHERE raw_text IS NOT NULL
              AND (embedding IS NULL OR embedded_at IS NULL)
            """
        )
    )
    rows = row.fetchall()
    count = 0
    for repo_id, _raw_text in rows:
        success = await embed_readme(session, repo_id)
        if success:
            count += 1
    return count
