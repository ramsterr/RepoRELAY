"""
Direct database access for the MVP.

Five small queries, each one obvious from its name. No graph traversal,
no co-star counts, no materialized views.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from reporelay_mvp.db import _get_sessionmaker
from reporelay_mvp.models import Repo

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = (
    "id, owner, name, full_name, description, language, topics, stars, "
    "dependencies, trending_score, embedding, description_embedding"
)


def is_real_vector(vec: list[float] | None) -> bool:
    """Return True if vec is a non-zero, NaN-free, Inf-free embedding.

    This is the canonical "do we have a usable embedding?" check.
    Used before storing, before querying, and as a defensive guard in
    score_many so we never compute meaningless cosine distances.
    """
    if vec is None or len(vec) == 0:
        return False
    for x in vec:
        f = float(x)
        if f != f:  # NaN
            return False
        if f == float("inf") or f == float("-inf"):
            return False
    return any(float(x) != 0.0 for x in vec)


def _row_to_repo(row: Any) -> Repo:
    data = dict(row._mapping)
    embedding_raw = data.get("embedding")
    desc_embedding_raw = data.get("description_embedding")
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
        trending_score=float(data.get("trending_score") or 0.0),
        embedding=_parse_embedding(embedding_raw) if embedding_raw is not None else None,
        description_embedding=_parse_embedding(desc_embedding_raw) if desc_embedding_raw is not None else None,
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
    if language and language.lower() not in (t.lower() for t in topics):
        topics = [*topics, language]
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


async def bulk_upsert_from_search(
    session: AsyncSession,
    items: list[dict[str, Any]],
) -> int:
    """
    Upsert a batch of GitHub search-result items into mvp_repos.

    Search results carry everything we need for the recommender
    (metadata, language, topics, stars, description) — no per-repo
    REST call is required. We mark `search_fetched_at` so a follow-up
    pass can identify rows that still need a README + embedding.

    Returns the number of rows written.
    """
    if not items:
        return 0
    params: list[dict[str, Any]] = []
    for item in items:
        lang = item.get("language")
        topics = list(item.get("topics") or [])
        if lang and lang.lower() not in (t.lower() for t in topics):
            topics.append(lang)
        params.append(
            {
                "id": int(item["id"]),
                "owner": item["owner"]["login"],
                "name": item["name"],
                "full_name": item["full_name"],
                "description": item.get("description"),
                "language": lang,
                "topics": topics,
                "stars": int(item.get("stargazers_count") or 0),
            }
        )
    await session.execute(
        text(
            """
            INSERT INTO mvp_repos (
                id, owner, name, full_name, description, language,
                topics, stars, dependencies, updated_at, search_fetched_at
            ) VALUES (
                :id, :owner, :name, :full_name, :description, :language,
                :topics, :stars, '{}', NOW(), NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
                owner = EXCLUDED.owner,
                name = EXCLUDED.name,
                full_name = EXCLUDED.full_name,
                description = EXCLUDED.description,
                language = EXCLUDED.language,
                topics = EXCLUDED.topics,
                stars = EXCLUDED.stars,
                updated_at = NOW(),
                search_fetched_at = NOW()
            """
        ),
        params,
    )
    return len(params)


async def bulk_apply_trending_signal(
    session: AsyncSession,
    rows: list[dict[str, Any]],
    *,
    since: str,
) -> int:
    """
    Apply per-period star velocity (scraped from github.com/trending) to
    rows that already exist in mvp_repos. New repos not in the DB are
    skipped — the search/seed pipeline is the source of truth for
    repo existence; trending is purely a velocity signal.

    `trending_score` is computed as min(1.0, stars_period / 100) so it
    sits in [0, 1] and can be weighted directly in the recommender.

    Returns the number of rows updated.
    """
    if not rows:
        return 0

    star_col = {
        "daily": "stars_today",
        "weekly": "stars_this_week",
        "monthly": "stars_this_month",
    }[since]

    updated = 0
    for row in rows:
        full_name = row["full_name"]
        stars = int(row.get("stars_period") or 0)
        trending_score = min(1.0, stars / 100.0) if stars > 0 else 0.0

        result = await session.execute(
            text(
                f"""
                UPDATE mvp_repos
                SET {star_col} = :stars,
                    trending_score = :trending_score,
                    trending_fetched_at = NOW(),
                    description = COALESCE(:description, description),
                    language = COALESCE(:language, language),
                    stars = GREATEST(stars, :total_stars)
                WHERE full_name = :full_name
                """
            ),
            {
                "full_name": full_name,
                "stars": stars,
                "trending_score": trending_score,
                "description": row.get("description"),
                "language": row.get("language"),
                "total_stars": int(row.get("total_stars") or 0),
            },
        )
        updated += result.rowcount or 0

    await session.commit()
    return updated


async def list_repos_needing_embedding(session: AsyncSession, *, limit: int) -> list[Repo]:
    """
    Return repos that have no embedding yet — candidates for the
    enrichment pass that fetches the README and computes the vector.

    Covers both search-indexed rows and manually-saved rows
    (save_repo sets embedding inline but may be out of date).
    """
    rows = await session.execute(
        text(
            f"""
            SELECT {EXPECTED_COLUMNS}
            FROM mvp_repos
            WHERE embedding IS NULL
            ORDER BY stars DESC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    )
    return [_row_to_repo(r) for r in rows]


async def set_embedding(
    session: AsyncSession,
    *,
    repo_id: int,
    embedding: list[float],
) -> None:
    if not is_real_vector(embedding):
        raise ValueError(
            f"refusing to store a zero/NaN embedding for repo_id={repo_id}"
        )
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


async def clear_embedding_for_reembed(
    session: AsyncSession, *, full_name: str
) -> int:
    """
    Mark a repo for re-embedding by nulling its embedding column.
    Called from the GitHub webhook receiver when a watched repo
    receives a push to its default branch.

    Returns the rowcount (0 if the repo isn't in mvp_repos, 1 if cleared).
    """
    result = await session.execute(
        text(
            """
            UPDATE mvp_repos
            SET embedding = NULL, embedded_at = NULL
            WHERE full_name = :full_name
            """
        ),
        {"full_name": full_name},
    )
    await session.commit()
    return result.rowcount or 0


async def get_repo(session: AsyncSession, full_name: str) -> Repo | None:
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


def _parse_embedding(raw: Any) -> list[float] | None:
    """Parse a pgvector column value returned as a JSON-array string by psycopg."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        try:
            return [float(x) for x in json.loads(raw)]
        except (json.JSONDecodeError, ValueError):
            return None
    return None


async def get_embedding(session: AsyncSession, repo_id: int) -> list[float] | None:
    rows = await session.execute(
        text("SELECT embedding FROM mvp_repos WHERE id = :id"),
        {"id": repo_id},
    )
    row = rows.fetchone()
    if not row or row[0] is None:
        return None
    return _parse_embedding(row[0])


async def get_embeddings_batch(session: AsyncSession, repo_ids: list[int]) -> dict[int, list[float]]:
    rows = await session.execute(
        text(
            """
            SELECT id, embedding
            FROM mvp_repos
            WHERE id = ANY(:ids) AND embedding IS NOT NULL
            """
        ),
        {"ids": repo_ids},
    )
    result: dict[int, list[float]] = {}
    for row in rows:
        parsed = _parse_embedding(row[1])
        if parsed is not None:
            result[int(row[0])] = parsed
    return result


async def set_description_embedding(
    session: AsyncSession,
    *,
    repo_id: int,
    description_embedding: list[float],
) -> None:
    if not is_real_vector(description_embedding):
        raise ValueError(
            f"refusing to store a zero/NaN description_embedding for repo_id={repo_id}"
        )
    await session.execute(
        text(
            """
            UPDATE mvp_repos
            SET description_embedding = :emb
            WHERE id = :id
            """
        ),
        {"id": repo_id, "emb": description_embedding},
    )


async def get_description_embeddings_batch(
    session: AsyncSession, repo_ids: list[int]
) -> dict[int, list[float]]:
    rows = await session.execute(
        text(
            """
            SELECT id, description_embedding
            FROM mvp_repos
            WHERE id = ANY(:ids) AND description_embedding IS NOT NULL
            """
        ),
        {"ids": repo_ids},
    )
    result: dict[int, list[float]] = {}
    for row in rows:
        parsed = _parse_embedding(row[1])
        if parsed is not None:
            result[int(row[0])] = parsed
    return result


async def count_repos(session: AsyncSession) -> int:
    rows = await session.execute(text("SELECT COUNT(*) FROM mvp_repos"))
    return int(rows.scalar() or 0)


async def update_topics(
    session: AsyncSession,
    *,
    repo_id: int,
    topics: list[str],
) -> None:
    """Merge new topics into an existing repo's topic list.

    Does a UNION of existing + new topics so nothing is lost.
    """
    await session.execute(
        text(
            """
            UPDATE mvp_repos
            SET topics = (
                SELECT ARRAY(
                    SELECT DISTINCT unnest(topics || :new_topics)
                )
            ),
            updated_at = NOW()
            WHERE id = :id
            """
        ),
        {"id": repo_id, "new_topics": topics},
    )


async def list_repos_needing_topic_inference(
    session: AsyncSession,
    *,
    limit: int = 1000,
    min_topic_count: int = 3,
) -> list[Repo]:
    """Return repos with fewer than min_topic_count topics.

    These are candidates for topic inference from their README text.
    Ordered by stars descending (highest-impact repos first).
    """
    rows = await session.execute(
        text(
            f"""
            SELECT {EXPECTED_COLUMNS}
            FROM mvp_repos
            WHERE array_length(topics, 1) IS NULL
               OR array_length(topics, 1) < :min_topic_count
            ORDER BY stars DESC
            LIMIT :limit
            """
        ),
        {"min_topic_count": min_topic_count, "limit": limit},
    )
    return [_row_to_repo(r) for r in rows]


async def get_random_repo(session: AsyncSession, *, seed: int) -> Repo | None:
    """Pick a random repo using a seed for deterministic random selection."""
    total = await count_repos(session)
    if total == 0:
        return None
    import random

    rng = random.Random(seed)
    offset = rng.randint(0, total - 1)
    rows = await session.execute(
        text(f"SELECT {EXPECTED_COLUMNS} FROM mvp_repos ORDER BY id LIMIT 1 OFFSET :offset"),
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
    where: list[str] = []
    params: dict[str, Any] = {"repo_id": repo_id, "limit": limit}

    if language is not None and topics:
        where.append("(language = :language OR topics && :topics) AND id != :repo_id")
        params["language"] = language
        params["topics"] = topics
    elif language is not None:
        where.append("language = :language AND id != :repo_id")
        params["language"] = language
    else:
        where.append("id != :repo_id")

    sql = text(
        f"""
        SELECT {EXPECTED_COLUMNS}
        FROM mvp_repos
        WHERE {" AND ".join(where) if where else "TRUE"}
        ORDER BY stars DESC
        LIMIT :limit
        """
    )
    rows = await session.execute(sql, params)
    return [_row_to_repo(r) for r in rows]


async def fetch_vector_neighbors(
    session: AsyncSession,
    *,
    source_embedding: list[float],
    exclude_id: int,
    limit: int,
) -> dict[int, tuple[Repo, float]]:
    """
    pgvector ANN: nearest neighbors of an arbitrary source embedding,
    excluding one repo by id.

    The source embedding is passed in directly (not joined from the
    mvp_repos table) so the caller can use a freshly-computed vector
    without first having to persist it. This is critical for the
    "paste a new GitHub URL" flow — we just computed the README vector
    in-process, and we want to use it immediately to find similar repos
    in the DB rather than waiting for the HNSW index to reflect a write.

    Returns a dict mapping repo_id -> (Repo, cosine_similarity).
    Returns an empty dict if source_embedding is empty/zero.
    """
    if not source_embedding or all(v == 0.0 for v in source_embedding):
        return {}
    rows = await session.execute(
        text(
            """
            SELECT
                mvp_repos.id, mvp_repos.owner, mvp_repos.name,
                mvp_repos.full_name, mvp_repos.description,
                mvp_repos.language, mvp_repos.topics, mvp_repos.stars,
                mvp_repos.dependencies,
                1 - (mvp_repos.embedding <=> CAST(:source_embedding AS vector)) AS cosine_sim
            FROM mvp_repos
            WHERE mvp_repos.id != :exclude_id
              AND mvp_repos.embedding IS NOT NULL
            ORDER BY mvp_repos.embedding <=> CAST(:source_embedding AS vector)
            LIMIT :limit
            """
        ),
        {
            "source_embedding": _to_pgvector(source_embedding),
            "exclude_id": exclude_id,
            "limit": limit,
        },
    )
    result: dict[int, tuple[Repo, float]] = {}
    skipped_zero = 0
    for row in rows:
        repo = _row_to_repo(row)
        # Guard: skip candidates whose embedding is all zeros.
        # Old quick_save wrote [0.0]*512 for repos that haven't been
        # embedded yet — these pass `embedding IS NOT NULL` in SQL
        # but produce meaningless cosine distances (pgvector returns
        # ≈1.0 for zero-vectors, making them look like perfect matches).
        if not is_real_vector(repo.embedding):
            skipped_zero += 1
            continue
        sim = float(row._mapping["cosine_sim"])
        # Final safety: clamp cosine similarity to [0, 1].  pgvector
        # can return values slightly outside this range for degenerate
        # vectors (e.g. very small norms).  We want features to stay
        # in [0, 1] so the weighted scorer behaves predictably.
        sim = max(0.0, min(1.0, sim))
        result[repo.id] = (repo, sim)
    if skipped_zero:
        logger.info(
            "fetch_vector_neighbors: skipped %d candidates with zero embeddings",
            skipped_zero,
        )
    return result


def _to_pgvector(vec: list[float]) -> str:
    """Serialize a Python list[float] into the textual form pgvector accepts
    in a query parameter (e.g. '[0.1,0.2,...]').

    Using a parameter keeps the query plan stable and lets us use the
    HNSW index. We cast in SQL via CAST(... AS vector).

    Raises ValueError for empty, NaN, or all-zero vectors — these would
    produce meaningless cosine distances.
    """
    if not vec:
        raise ValueError("embedding vector is empty")
    cleaned: list[str] = []
    for x in vec:
        f = float(x)
        if f != f:  # NaN check
            raise ValueError("embedding vector contains NaN")
        if f == float("inf") or f == float("-inf"):
            raise ValueError("embedding vector contains Inf")
        cleaned.append(repr(f))
    if all(float(x) == 0.0 for x in vec):
        raise ValueError("embedding vector is all zeros — cannot compute cosine distance")
    return "[" + ",".join(cleaned) + "]"
