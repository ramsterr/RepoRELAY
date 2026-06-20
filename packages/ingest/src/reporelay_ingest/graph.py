"""
Graph layer for RepoRelay using SQL-based traversal on existing Postgres tables.

Apache AGE is not available in the current Postgres image (pgvector/pgvector:pg16).
Instead, we implement graph operations as optimized SQL JOINs on the relational
tables, which avoids the operational complexity of a separate graph extension while
providing equivalent query power for 1-2 hop traversals.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def get_one_hop_neighbors(
    session: AsyncSession, repo_id: int
) -> dict[str, list[dict[str, Any]]]:
    """Get all 1-hop neighbors of a repo across all edge types."""
    result: dict[str, list[dict[str, Any]]] = {}

    rows = await session.execute(
        text(
            """
            SELECT 'contributor' AS edge_type, users.id AS neighbor_id, users.login AS neighbor_name,
                   contributor_edges.commit_count AS weight
            FROM contributor_edges
            JOIN users ON contributor_edges.user_id = users.id
            WHERE contributor_edges.repo_id = :repo_id
            """
        ),
        {"repo_id": repo_id},
    )
    result["contributors"] = [dict(r._mapping) for r in rows]

    rows = await session.execute(
        text(
            """
            SELECT 'dependency' AS edge_type, dependency_name AS neighbor_name,
                   1.0 AS weight, ecosystem
            FROM dependency_edges
            WHERE repo_id = :repo_id
            """
        ),
        {"repo_id": repo_id},
    )
    result["dependencies"] = [dict(r._mapping) for r in rows]

    rows = await session.execute(
        text(
            """
            SELECT 'star_user' AS edge_type, users.id AS neighbor_id, users.login AS neighbor_name,
                   COUNT(*) AS weight
            FROM star_events
            JOIN users ON star_events.user_id = users.id
            WHERE star_events.repo_id = :repo_id
            GROUP BY users.id, users.login
            """
        ),
        {"repo_id": repo_id},
    )
    result["star_users"] = [dict(r._mapping) for r in rows]

    return result


async def get_two_hop_neighbors(
    session: AsyncSession, repo_id: int
) -> list[dict[str, Any]]:
    """Get 2-hop repos: repos starred by users who starred this repo."""
    rows = await session.execute(
        text(
            """
            SELECT
                s2.repo_id AS neighbor_repo_id,
                r.full_name AS neighbor_full_name,
                'star' AS path_type,
                2 AS hop_count,
                COUNT(*)::float AS combined_weight
            FROM star_events s1
            JOIN star_events s2
                ON s1.user_id = s2.user_id
                AND s1.repo_id != s2.repo_id
            JOIN repos r ON s2.repo_id = r.id
            WHERE s1.repo_id = :repo_id
            GROUP BY s2.repo_id, r.full_name
            ORDER BY combined_weight DESC
            LIMIT 200
            """
        ),
        {"repo_id": repo_id},
    )
    return [dict(r._mapping) for r in rows]


async def sync_two_hop_table(
    session: AsyncSession, repo_id: int
) -> int:
    """Compute 2-hop neighbors for a repo and upsert into two_hop_neighbors."""
    neighbors = await get_two_hop_neighbors(session, repo_id)

    for n in neighbors:
        await session.execute(
            text(
                """
                INSERT INTO two_hop_neighbors
                    (source_repo_id, neighbor_repo_id, path_type, hop_count, combined_weight)
                VALUES
                    (:source, :neighbor, :path_type, :hop_count, :weight)
                ON CONFLICT (source_repo_id, neighbor_repo_id, path_type) DO UPDATE SET
                    combined_weight = EXCLUDED.combined_weight,
                    hop_count = EXCLUDED.hop_count,
                    computed_at = NOW()
                """
            ),
            {
                "source": repo_id,
                "neighbor": n["neighbor_repo_id"],
                "path_type": n["path_type"],
                "hop_count": n["hop_count"],
                "weight": n["combined_weight"],
            },
        )

    await session.flush()
    return len(neighbors)
