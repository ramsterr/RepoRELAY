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
    "dependencies, trending_score, embedding, description_embedding, keywords"
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
        keywords=list(data.get("keywords") or []),
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


async def list_repos_needing_description_embedding(
    session: AsyncSession, *, limit: int
) -> list[Repo]:
    """Return repos that have no description_embedding yet.

    Sorted by stars DESC so the most important repos get embedded first.
    Used by description-only embedding pass (fast — no GitHub API calls).
    """
    rows = await session.execute(
        text(
            f"""
            SELECT {EXPECTED_COLUMNS}
            FROM mvp_repos
            WHERE description_embedding IS NULL
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
    """pgvector ANN on the `embedding` (README) column — legacy path.
    
    Only 355/11918 repos have README embeddings, so this pool is small.
    Prefer fetch_desc_vector_neighbors which uses the description_embedding
    column (11217/11918 repos).
    """
    return await _fetch_vector_neighbors(
        session, source_embedding=source_embedding,
        exclude_id=exclude_id, limit=limit,
        column="embedding",
    )


async def fetch_desc_vector_neighbors(
    session: AsyncSession,
    *,
    source_embedding: list[float],
    exclude_id: int,
    limit: int,
) -> dict[int, tuple[Repo, float]]:
    """pgvector ANN on the `description_embedding` column.

    This is the PRIMARY vector search path. 11,217 out of 11,918 repos
    have description embeddings — 30x more than the README embedding
    column. The source's README and description embeddings are compared
    against this column.
    """
    return await _fetch_vector_neighbors(
        session, source_embedding=source_embedding,
        exclude_id=exclude_id, limit=limit,
        column="description_embedding",
    )


async def _fetch_vector_neighbors(
    session: AsyncSession,
    *,
    source_embedding: list[float],
    exclude_id: int,
    limit: int,
    column: str,
) -> dict[int, tuple[Repo, float]]:
    """pgvector ANN: nearest neighbors of a source embedding against a column.

    The source embedding is passed in directly (in-memory, freshly computed).
    No DB join — we use CAST(:param AS vector) for the computation.

    Sets `hnsw.ef_search = max(limit * 2, 100)` on the session to
    maintain high ANN recall at the requested limit. Without this,
    pgvector defaults ef_search to max(limit, 40), which causes
    noticeable recall degradation above ~100 neighbors on 50k+ vectors.

    Returns a dict mapping repo_id -> (Repo, cosine_similarity).
    Returns an empty dict if source_embedding is empty/zero/NaN.
    """
    if not source_embedding or all(v == 0.0 for v in source_embedding):
        return {}

    _ef_search = max(limit * 2, 100)
    await session.execute(
        text("SET LOCAL hnsw.ef_search = :ef"),
        {"ef": _ef_search},
    )

    rows = await session.execute(
        text(
            f"""
            SELECT {EXPECTED_COLUMNS},
                   1 - ({column} <=> CAST(:source_embedding AS vector)) AS cosine_sim
            FROM mvp_repos
            WHERE mvp_repos.id != :exclude_id
              AND {column} IS NOT NULL
            ORDER BY {column} <=> CAST(:source_embedding AS vector)
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
        vec = repo.description_embedding if column == "description_embedding" else repo.embedding
        if not is_real_vector(vec):
            skipped_zero += 1
            continue
        sim = float(row._mapping["cosine_sim"])
        sim = max(0.0, min(1.0, sim))
        result[repo.id] = (repo, sim)
    if skipped_zero:
        logger.info(
            "_fetch_vector_neighbors(%s): skipped %d candidates with zero embeddings",
            column, skipped_zero,
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


# ── Full-Text & Keyword Search ──────────────────────────────────────


async def search_keywords(
    session: AsyncSession,
    *,
    query_terms: list[str],
    limit: int = 50,
    min_stars: int = 0,
) -> list[Repo]:
    """Fast keyword search using GIN-indexed ARRAY overlap.

    Uses PostgreSQL's GIN index on the keywords[] column for
    sub-millisecond lookup. Finds repos whose extracted keywords
    overlap with the query terms.

    This is the fastest search path — pure index scan, no ranking.
    For ranked results, use search_fulltext() instead.
    """
    if not query_terms:
        return []

    rows = await session.execute(
        text(
            f"""
            SELECT {EXPECTED_COLUMNS}
            FROM mvp_repos
            WHERE keywords && :terms
              AND stars >= :min_stars
            ORDER BY stars DESC
            LIMIT :limit
            """
        ),
        {"terms": query_terms, "min_stars": min_stars, "limit": limit},
    )
    return [_row_to_repo(r) for r in rows]


async def search_fulltext(
    session: AsyncSession,
    *,
    query_text: str,
    limit: int = 50,
    min_stars: int = 0,
) -> list[tuple[Repo, float]]:
    """Ranked full-text search using PostgreSQL tsvector + ts_rank.

    Converts the query to a tsquery, matches against the search_vector
    column, and ranks by ts_rank (relevance). Handles multi-word queries
    with AND semantics.

    Returns (repo, relevance_score) tuples sorted by relevance.
    """
    if not query_text or not query_text.strip():
        return []

    # Convert query to tsquery format: "game development pixel" → "game & development & pixel"
    terms = [t.strip() for t in query_text.split() if t.strip() and len(t.strip()) > 1]
    if not terms:
        return []
    tsquery = " & ".join(terms)

    rows = await session.execute(
        text(
            f"""
            SELECT {EXPECTED_COLUMNS},
                   ts_rank(search_vector, to_tsquery('english', :query)) AS relevance
            FROM mvp_repos
            WHERE search_vector @@ to_tsquery('english', :query)
              AND stars >= :min_stars
            ORDER BY relevance DESC
            LIMIT :limit
            """
        ),
        {"query": tsquery, "min_stars": min_stars, "limit": limit},
    )
    return [(_row_to_repo(r), float(r._mapping["relevance"])) for r in rows]


async def search_hybrid(
    session: AsyncSession,
    *,
    query_text: str,
    query_embedding: list[float] | None = None,
    limit: int = 50,
    min_stars: int = 0,
    fts_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> list[tuple[Repo, float]]:
    """Hybrid search: full-text relevance + semantic similarity.

    Combines PostgreSQL ts_rank (full-text relevance) with pgvector
    cosine distance (semantic similarity) into a single weighted score.

    If query_embedding is None (no semantic component), falls back to
    pure full-text search.

    Returns (repo, combined_score) tuples sorted by combined score.
    """
    if not query_text or not query_text.strip():
        return []

    terms = [t.strip() for t in query_text.split() if t.strip() and len(t.strip()) > 1]
    if not terms:
        return []
    tsquery = " & ".join(terms)

    if query_embedding and len(query_embedding) > 0 and any(v != 0.0 for v in query_embedding):
        # Hybrid: FTS + semantic
        pg_vector = _to_pgvector(query_embedding)
        rows = await session.execute(
            text(
                f"""
                SELECT {EXPECTED_COLUMNS},
                       (:fts_w * ts_rank(search_vector, to_tsquery('english', :query))) +
                       (:sem_w * (1.0 - (description_embedding <=> :embedding::vector))) AS combined_score
                FROM mvp_repos
                WHERE search_vector @@ to_tsquery('english', :query)
                  AND description_embedding IS NOT NULL
                  AND stars >= :min_stars
                ORDER BY combined_score DESC
                LIMIT :limit
                """
            ),
            {
                "query": tsquery,
                "embedding": pg_vector.strip(),
                "fts_w": fts_weight,
                "sem_w": semantic_weight,
                "min_stars": min_stars,
                "limit": limit,
            },
        )
    else:
        # Pure FTS fallback
        rows = await session.execute(
            text(
                f"""
                SELECT {EXPECTED_COLUMNS},
                       ts_rank(search_vector, to_tsquery('english', :query)) AS combined_score
                FROM mvp_repos
                WHERE search_vector @@ to_tsquery('english', :query)
                  AND stars >= :min_stars
                ORDER BY combined_score DESC
                LIMIT :limit
                """
            ),
            {"query": tsquery, "min_stars": min_stars, "limit": limit},
        )

    return [(_row_to_repo(r), float(r._mapping["combined_score"])) for r in rows]


async def set_keywords(
    session: AsyncSession,
    *,
    repo_id: int,
    keywords: list[str],
) -> None:
    """Store extracted keywords and update the search_vector."""
    # Limit keywords to prevent tsvector from timing out on Neon
    keywords = keywords[:30]
    search_text = " ".join(keywords)[:500]
    await session.execute(
        text(
            """
            UPDATE mvp_repos
            SET keywords = :keywords,
                search_vector = to_tsvector('english', :search_text)
            WHERE id = :id
            """
        ),
        {"id": repo_id, "keywords": keywords, "search_text": search_text},
    )


async def list_repos_needing_keywords(
    session: AsyncSession,
    *,
    limit: int = 1000,
) -> list[Repo]:
    """Return repos that have descriptions but no keywords yet."""
    rows = await session.execute(
        text(
            f"""
            SELECT {EXPECTED_COLUMNS}
            FROM mvp_repos
            WHERE description IS NOT NULL
              AND description != ''
              AND (keywords IS NULL OR array_length(keywords, 1) IS NULL OR array_length(keywords, 1) = 0)
            ORDER BY stars DESC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    )
    return [_row_to_repo(r) for r in rows]
