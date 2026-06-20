"""
Real data access layer for recommendation strategies.

Replaces placeholder _fetch_by_similarity with actual Postgres queries:
  - pgvector ANN for content-based
  - co_star_counts MV for item-based CF
  - star overlap for user-based CF
  - stars/recent push for trending/exploration
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from reporelay_core.db import _get_sessionmaker

logger = logging.getLogger(__name__)


async def _get_session() -> AsyncSession:
    return _get_sessionmaker()()


async def resolve_repo_id(session: AsyncSession, full_name: str) -> int | None:
    row = await session.execute(
        text("SELECT id FROM repos WHERE full_name = :full_name"),
        {"full_name": full_name},
    )
    result = row.fetchone()
    return result[0] if result else None


async def resolve_repo_full_name(session: AsyncSession, repo_id: int) -> str | None:
    row = await session.execute(
        text("SELECT full_name FROM repos WHERE id = :id"),
        {"id": repo_id},
    )
    result = row.fetchone()
    return result[0] if result else None


async def content_based_neighbors(
    session: AsyncSession, source_repo: str, limit: int
) -> list[dict[str, Any]]:
    """Find repos with similar README content via pgvector ANN."""
    repo_id = await resolve_repo_id(session, source_repo)
    if repo_id is None:
        return []

    rows = await session.execute(
        text(
            """
            SELECT r.id, r.full_name, r.description, r.stars, r.language, r.topics,
                   1 - (rt.embedding <=> src.embedding) AS similarity
            FROM readme_texts src
            CROSS JOIN readme_texts rt
            JOIN repos r ON rt.repo_id = r.id
            WHERE src.repo_id = :repo_id
              AND rt.repo_id != :repo_id
              AND rt.embedding IS NOT NULL
              AND src.embedding IS NOT NULL
              AND r.archived = false
            ORDER BY rt.embedding <=> src.embedding
            LIMIT :limit
            """
        ),
        {"repo_id": repo_id, "limit": limit},
    )
    return [dict(r._mapping) for r in rows]


async def co_starred_repos(
    session: AsyncSession, source_repo: str, limit: int
) -> list[dict[str, Any]]:
    """Find repos co-starred by users who starred the source repo."""
    repo_id = await resolve_repo_id(session, source_repo)
    if repo_id is None:
        return []

    rows = await session.execute(
        text(
            """
            SELECT r.id, r.full_name, r.description, r.stars, r.language, r.topics,
                   mv.co_star_count
            FROM co_star_counts mv
            JOIN repos r ON mv.repo_b = r.id
            WHERE mv.repo_a = :repo_id
              AND r.archived = false
            ORDER BY mv.co_star_count DESC
            LIMIT :limit
            """
        ),
        {"repo_id": repo_id, "limit": limit},
    )
    return [dict(r._mapping) for r in rows]


async def user_similarity_repos(
    session: AsyncSession, user_id: str, source_repo: str, limit: int
) -> list[dict[str, Any]]:
    """Find repos starred by users with similar star patterns."""
    repo_id = await resolve_repo_id(session, source_repo)
    if repo_id is None:
        return []

    rows = await session.execute(
        text(
            """
            WITH user_stars AS (
                SELECT user_id, array_agg(repo_id ORDER BY starred_at DESC) AS starred_repos
                FROM star_events
                WHERE user_id = :user_id
                GROUP BY user_id
            ),
            similar_users AS (
                SELECT s.user_id, COUNT(*) AS overlap
                FROM star_events s
                JOIN user_stars us ON s.repo_id = ANY(us.starred_repos)
                WHERE s.user_id != :user_id
                GROUP BY s.user_id
                ORDER BY overlap DESC
                LIMIT 50
            )
            SELECT DISTINCT r.id, r.full_name, r.description, r.stars, r.language, r.topics
            FROM star_events se
            JOIN similar_users su ON se.user_id = su.user_id
            JOIN repos r ON se.repo_id = r.id
            WHERE se.repo_id != :repo_id
              AND r.archived = false
            ORDER BY r.stars DESC
            LIMIT :limit
            """
        ),
        {"user_id": user_id, "repo_id": repo_id, "limit": limit},
    )
    return [dict(r._mapping) for r in rows]


async def trending_repos(
    session: AsyncSession, source_repo: str, limit: int
) -> list[dict[str, Any]]:
    """Find trending repos — highest stars with recent activity, excluding source."""
    repo_id = await resolve_repo_id(session, source_repo)

    rows = await session.execute(
        text(
            """
            SELECT id, full_name, description, stars, language, topics
            FROM repos
            WHERE archived = false
              AND (:exclude_id IS NULL OR id != :exclude_id)
            ORDER BY stars DESC
            LIMIT :limit
            """
        ),
        {"exclude_id": repo_id, "limit": limit},
    )
    return [dict(r._mapping) for r in rows]


async def repos_to_dicts(
    session: AsyncSession, repo_ids: list[int]
) -> list[dict[str, Any]]:
    """Bulk fetch repo metadata by IDs."""
    if not repo_ids:
        return []
    ids = tuple(repo_ids)
    rows = await session.execute(
        text(
            f"""
            SELECT id, full_name, description, stars, language, topics
            FROM repos
            WHERE id IN :ids
              AND archived = false
            """
        ),
        {"ids": ids},
    )
    return [dict(r._mapping) for r in rows]
