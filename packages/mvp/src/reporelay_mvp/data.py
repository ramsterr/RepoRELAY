"""
Direct database access for the MVP.

Five small queries, each one obvious from its name. No graph traversal,
no co-star counts, no materialized views.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from reporelay_mvp.db import _get_sessionmaker
from reporelay_mvp.models import Repo

EXPECTED_COLUMNS = (
    "id, owner, name, full_name, description, language, topics, stars, dependencies"
)


def _row_to_repo(row: Any) -> Repo:
    data = dict(row._mapping)
    return Repo(
        id=data["id"],
        owner=data["owner"],
        name=data["name"],
        full_name=data["full_name"],
        description=data.get("description"),
        language=data.get("language"),
        topics=list(data.get("topics") or []),
        stars=int(data.get("stars") or 0),
        dependencies=list(data.get("dependencies") or []),
    )


async def get_session() -> AsyncSession:
    return _get_sessionmaker()()


async def upsert_repo(
    session: AsyncSession,
    *,
    repo_id: int,
    owner: str,
    name: str,
    full_name: str,
    description: str | None,
    language: str | None,
    topics: list[str],
    stars: int,
    dependencies: list[str],
) -> None:
    await session.execute(
        text(
            """
            INSERT INTO mvp_repos (
                id, owner, name, full_name, description, language,
                topics, stars, dependencies, updated_at
            ) VALUES (
                :id, :owner, :name, :full_name, :description, :language,
                :topics, :stars, :dependencies, NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                owner = EXCLUDED.owner,
                name = EXCLUDED.name,
                full_name = EXCLUDED.full_name,
                description = EXCLUDED.description,
                language = EXCLUDED.language,
                topics = EXCLUDED.topics,
                stars = EXCLUDED.stars,
                dependencies = EXCLUDED.dependencies,
                updated_at = NOW()
            """
        ),
        {
            "id": repo_id,
            "owner": owner,
            "name": name,
            "full_name": full_name,
            "description": description,
            "language": language,
            "topics": topics,
            "stars": stars,
            "dependencies": dependencies,
        },
    )


async def set_embedding(
    session: AsyncSession,
    *,
    repo_id: int,
    embedding: list[float],
) -> None:
    await session.execute(
        text(
            """
            UPDATE mvp_repos
            SET embedding = :embedding, embedded_at = NOW()
            WHERE id = :id
            """
        ),
        {"id": repo_id, "embedding": embedding},
    )


async def get_repo(
    session: AsyncSession, full_name: str
) -> Repo | None:
    rows = await session.execute(
        text(f"SELECT {EXPECTED_COLUMNS} FROM mvp_repos WHERE full_name = :full_name"),
        {"full_name": full_name},
    )
    row = rows.fetchone()
    return _row_to_repo(row) if row else None


async def get_repo_by_id(session: AsyncSession, repo_id: int) -> Repo | None:
    rows = await session.execute(
        text(f"SELECT {EXPECTED_COLUMNS} FROM mvp_repos WHERE id = :id"),
        {"id": repo_id},
    )
    row = rows.fetchone()
    return _row_to_repo(row) if row else None


async def get_embedding(
    session: AsyncSession, repo_id: int
) -> list[float] | None:
    rows = await session.execute(
        text("SELECT embedding FROM mvp_repos WHERE id = :id"),
        {"id": repo_id},
    )
    row = rows.fetchone()
    if not row or row[0] is None:
        return None
    return list(row[0])


async def count_repos(session: AsyncSession) -> int:
    rows = await session.execute(text("SELECT COUNT(*) FROM mvp_repos"))
    return int(rows.scalar() or 0)


async def get_random_repo(session: AsyncSession, *, seed: int) -> Repo | None:
    """Pick a random repo using a seed for deterministic random selection."""
    total = await count_repos(session)
    if total == 0:
        return None
    import random
    rng = random.Random(seed)
    offset = rng.randint(0, total - 1)
    rows = await session.execute(
        text(
            f"SELECT {EXPECTED_COLUMNS} FROM mvp_repos ORDER BY id LIMIT 1 OFFSET :offset"
        ),
        {"offset": offset},
    )
    row = rows.fetchone()
    return _row_to_repo(row) if row else None


async def fetch_filtered_pool(
    session: AsyncSession,
    *,
    repo_id: int,
    language: str | None,
    topics: list[str],
    limit: int,
) -> list[Repo]:
    """
    SQL-side filter: same language OR topic overlap, excluding the source.

    Uses the GIN index on `topics` and the btree on `language`. Returns
    up to `limit` candidates that are at least plausibly in the same
    ecosystem.
    """
    where = ["id != :repo_id"]
    params: dict[str, Any] = {"repo_id": repo_id, "limit": limit}

    if language:
        where.append("language = :language")
        params["language"] = language
    if topics:
        where.append("topics && :topics")
        params["topics"] = topics

    sql = text(
        f"""
        SELECT {EXPECTED_COLUMNS}
        FROM mvp_repos
        WHERE {' AND '.join(where)}
        ORDER BY stars DESC
        LIMIT :limit
        """
    )
    rows = await session.execute(sql, params)
    return [_row_to_repo(r) for r in rows]


async def fetch_vector_neighbors(
    session: AsyncSession,
    *,
    source_id: int,
    exclude_id: int,
    limit: int,
) -> dict[int, tuple[Repo, float]]:
    """
    pgvector ANN: nearest neighbors of the source repo's embedding,
    excluding the source itself. The embedding is read from the source
    row in the same query — no parameter binding needed for the vector.

    Returns a dict mapping repo_id -> (Repo, cosine_similarity).
    """
    rows = await session.execute(
        text(
            """
            SELECT
                mvp_repos.id, mvp_repos.owner, mvp_repos.name,
                mvp_repos.full_name, mvp_repos.description,
                mvp_repos.language, mvp_repos.topics, mvp_repos.stars,
                mvp_repos.dependencies,
                1 - (mvp_repos.embedding <=> src.embedding) AS cosine_sim
            FROM mvp_repos
            CROSS JOIN mvp_repos AS src
            WHERE src.id = :source_id
              AND mvp_repos.id != :exclude_id
              AND mvp_repos.embedding IS NOT NULL
              AND src.embedding IS NOT NULL
            ORDER BY mvp_repos.embedding <=> src.embedding
            LIMIT :limit
            """
        ),
        {"source_id": source_id, "exclude_id": exclude_id, "limit": limit},
    )
    result: dict[int, tuple[Repo, float]] = {}
    for row in rows:
        repo = _row_to_repo(row)
        sim = float(row._mapping["cosine_sim"])
        result[repo.id] = (repo, sim)
    return result
