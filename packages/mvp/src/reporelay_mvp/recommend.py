"""
Top-level entry point for the MVP recommender.

`recommend(full_name, limit=10, seed=None)` runs the full 5-stage
pipeline against a single source repo and returns a flat ranked list.

When `seed` is set, the candidate pool is shuffled and the scoring
weights are jittered — giving different results per seed while
remaining deterministic (same seed = same results).

If the source repo is not in the DB, it is automatically fetched from
GitHub and saved. The candidate pool is always built from two sources:
the local DB (fast, has embeddings) and a fresh GitHub search (live,
has variety). Search hits are persisted back to the DB so the corpus
grows over time.

`recommend_random(seed)` picks a random source repo and runs the
pipeline against it — the "surprise me / explore" feature.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from reporelay_mvp import data
from reporelay_mvp.candidates import generate_candidates, NEUTRAL_SIM
from reporelay_mvp.embedding import embed_text
from reporelay_mvp.features import compute_features
from reporelay_mvp.github import (
    _auth_client_active,
    fetch_all,
    fetch_dependencies,
    fetch_readme,
    fetch_repo_metadata,
    search_repositories,
)
from reporelay_mvp.models import (
    CategorizedRecommendation,
    Features,
    RecommendationGroup,
    Repo,
    ScoredRecommendation,
    ScoredRepo,
)
from reporelay_mvp.rerank import rerank
from reporelay_mvp.score import score_many
from reporelay_mvp.settings import get_mvp_settings
from reporelay_mvp.purpose import clean_description

logger = logging.getLogger(__name__)


SEARCH_LIMIT = 100  # how many fresh candidates to pull from GitHub per call

# --- request-level rec cache (in-process, 10 min TTL) ---
import time as _rec_time

_rec_cache: dict[str, tuple[float, Any]] = {}
_REC_CACHE_TTL = 600  # 10 minutes
_REC_CACHE_MAX = 500  # evict oldest if exceeded


def _rec_cache_key(full_name: str, seed: int | None, tags: list[str] | None) -> str:
    tag_str = ",".join(sorted(tags or []))
    return f"{full_name}:{seed}:{tag_str}"


def _rec_cache_get(key: str, now: float) -> Any | None:
    if key not in _rec_cache:
        return None
    ts, value = _rec_cache[key]
    if now - ts >= _REC_CACHE_TTL:
        del _rec_cache[key]
        return None
    return value


def _rec_cache_set(key: str, now: float, value: Any) -> None:
    if len(_rec_cache) >= _REC_CACHE_MAX:
        oldest = min(_rec_cache, key=lambda k: _rec_cache[k][0])
        del _rec_cache[oldest]
    _rec_cache[key] = (now, value)

_cached_search_results: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 300
_DISK_CACHE_TTL = 24 * 3600  # 24h — GitHub search results don't change fast
_DISK_CACHE_PATH = Path(
    os.environ.get("REPORE_LAY_SEARCH_CACHE")
    or Path(tempfile.gettempdir()) / "reporelay_search_cache.json"
)
_MIN_DB_POOL_FOR_SKIP = 300  # if DB pool is already this big, skip the GitHub search

_SEARCH_CACHE: dict[str, dict[str, Any]] = {}


def _load_disk_cache() -> None:
    if _SEARCH_CACHE:
        return
    try:
        if _DISK_CACHE_PATH.exists():
            _SEARCH_CACHE.update(json.loads(_DISK_CACHE_PATH.read_text()))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("search cache load failed: %s", exc)


def _save_disk_cache() -> None:
    try:
        _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DISK_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_SEARCH_CACHE))
        tmp.replace(_DISK_CACHE_PATH)
    except OSError as exc:
        logger.warning("search cache save failed: %s", exc)


async def _cached_search(
    client: Any, *, topics: list[str] | None, language: str | None, **kwargs: Any
) -> dict[str, Any]:
    key = repr((language, tuple(sorted(topics or []))))
    now = time.monotonic()

    if key in _cached_search_results:
        ts, cached = _cached_search_results[key]
        if now - ts < _CACHE_TTL:
            logger.info("search cache hit (memory) for %s", key)
            return cached

    _load_disk_cache()
    disk = _SEARCH_CACHE.get(key)
    if disk and (time.time() - float(disk.get("ts", 0))) < _DISK_CACHE_TTL:
        logger.info("search cache hit (disk) for %s", key)
        result = disk["payload"]
        _cached_search_results[key] = (now, result)
        return result

    result = await search_repositories(client, topics=topics, language=language, **kwargs)
    _cached_search_results[key] = (now, result)
    _SEARCH_CACHE[key] = {"ts": time.time(), "payload": result}
    _save_disk_cache()
    return result


def _build_scored_repo(
    source: Any,
    repo: Any,
    score: float,
    cosine_sim: float,
    features: Features | None = None,
) -> ScoredRepo:
    feats = features if features is not None else compute_features(
        source, repo, cosine_sim=cosine_sim, description_cosine_sim=0.0, readme_topic_sim=0.0,
    )
    source_topic_set = set(source.topics)
    source_lang = source.language

    return ScoredRepo(
        id=repo.id,
        owner=repo.owner,
        name=repo.name,
        full_name=repo.full_name,
        description=repo.description,
        language=repo.language,
        topics=repo.topics,
        stars=repo.stars,
        dependencies=repo.dependencies,
        score=round(score, 4),
        features=feats.as_dict(),
        shared_topics=sorted(source_topic_set & set(repo.topics)),
        shared_language=bool(source_lang and repo.language and source_lang == repo.language),
    )


async def _find_proxy_embedding(session: Any, source: Repo) -> list[float] | None:
    """
    When a repo has no embedding, find the best-matched repo in the DB
    by topic overlap + language + comparable popularity and borrow its
    embedding for pgvector search.

    If source has no topics (rate-limited skeleton), falls back to
    finding ANY non-zero embedding in the DB — better than nothing.

    Returns the proxy embedding or None if no good match exists.
    """
    if not source.topics:
        # Fallback: any embedding is better than none. Pick the most
        # popular repo with a real embedding.
        from sqlalchemy import text

        rows = await session.execute(
            text(
                """
                SELECT id, embedding
                FROM mvp_repos
                WHERE embedding IS NOT NULL
                ORDER BY stars DESC
                LIMIT 5
                """
            ),
        )
        for row in rows:
            emb_raw = row.embedding
            if isinstance(emb_raw, list) and any(v != 0.0 for v in emb_raw[:3]):
                logger.info("fallback proxy: no topics, using popular repo %d", row.id)
                return [float(x) for x in emb_raw]
        return None

    import math as _math

    from sqlalchemy import text

    rows = await session.execute(
        text(
            """
            SELECT id, embedding, topics, language, stars
            FROM mvp_repos
            WHERE embedding IS NOT NULL
              AND topics && :topics
            ORDER BY stars DESC
            LIMIT 20
            """
        ),
        {"topics": source.topics},
    )
    best_id = None
    best_embedding = None
    best_score = -1.0
    source_set = set(source.topics)
    src_lang = source.language
    src_stars = max(1, source.stars)
    for row in rows:
        cand_topics = list(row.topics or [])
        overlap = len(source_set & set(cand_topics))
        lang_score = 1.0 if (src_lang and getattr(row, "language", None) == src_lang) else 0.0
        star_score = min(1.0, _math.log1p(max(1, getattr(row, "stars", 0) or 0)) / _math.log1p(src_stars))
        composite = overlap * 0.5 + lang_score * 0.3 + star_score * 0.2
        if composite > best_score:
            best_score = composite
            best_id = row.id
            emb_raw = row.embedding
            if isinstance(emb_raw, list):
                best_embedding = [float(x) for x in emb_raw]

    if best_embedding:
        logger.info(
            "proxy embedding from repo %d (topic overlap=%d, topics=%s)",
            best_id,
            best_score,
            source.topics,
        )
    return best_embedding


async def _find_desc_proxy_embedding(session: Any, source: Repo) -> list[float] | None:
    """Same as _find_proxy_embedding but for description_embedding column."""
    if not source.topics:
        from sqlalchemy import text

        rows = await session.execute(
            text(
                """
                SELECT id, description_embedding
                FROM mvp_repos
                WHERE description_embedding IS NOT NULL
                ORDER BY stars DESC
                LIMIT 5
                """
            ),
        )
        for row in rows:
            desc_raw = row.description_embedding
            if isinstance(desc_raw, list) and any(v != 0.0 for v in desc_raw[:3]):
                logger.info("fallback desc proxy: no topics, using popular repo %d", row.id)
                return [float(x) for x in desc_raw]
        return None

    import math as _math
    from sqlalchemy import text

    rows = await session.execute(
        text(
            """
            SELECT id, description_embedding, topics, language, stars
            FROM mvp_repos
            WHERE description_embedding IS NOT NULL
              AND topics && :topics
            ORDER BY stars DESC
            LIMIT 20
            """
        ),
        {"topics": source.topics},
    )
    best_id = None
    best_embedding = None
    best_score = -1.0
    source_set = set(source.topics)
    src_lang = source.language
    src_stars = max(1, source.stars)
    for row in rows:
        cand_topics = list(row.topics or [])
        overlap = len(source_set & set(cand_topics))
        lang_score = 1.0 if (src_lang and getattr(row, "language", None) == src_lang) else 0.0
        star_score = min(1.0, _math.log1p(max(1, getattr(row, "stars", 0) or 0)) / _math.log1p(src_stars))
        composite = overlap * 0.5 + lang_score * 0.3 + star_score * 0.2
        if composite > best_score:
            best_score = composite
            best_id = row.id
            desc_raw = row.description_embedding
            if isinstance(desc_raw, list):
                best_embedding = [float(x) for x in desc_raw]

    if best_embedding:
        logger.info(
            "proxy desc embedding from repo %d (topic overlap=%d, topics=%s)",
            best_id, best_score, source.topics,
        )
    return best_embedding


async def recommend(
    full_name: str,
    *,
    limit: int = 10,
    seed: int | None = None,
    tags: list[str] | None = None,
    _source_override: Repo | None = None,
) -> ScoredRecommendation:
    """Run the full recommendation pipeline.

    If _source_override is provided, it is used as the source repo
    directly (skipping DB lookup and live embed). This is used by
    relevance feedback (feedback.py) to run the pipeline against a
    virtual merged repo.
    """
    if limit <= 0:
        raise ValueError("limit must be > 0")

    # ── Source override (relevance feedback): skip DB, go straight to pipeline ──
    if _source_override is not None:
        source = _source_override
        filter_emb = None
        if tags:
            filter_emb = await embed_text(" ".join(tags))
        session = await data.get_session()
        try:
            candidates = await generate_candidates(session, source, seed=seed, tags=tags)
            scored = await score_many(
                source, candidates, seed=seed, tags=tags,
                filter_embedding=filter_emb, source_readme_tokens=None,
            )
            result = categorize_results(source, scored, limit, seed)
            return result
        finally:
            await session.close()

    # ── Normal flow ─────────────────────────────────────────────────────────
    owner, _, name = full_name.partition("/")
    if not owner or not name:
        raise LookupError(f"repo must be 'owner/name', got {full_name!r}")

    # Check cache first — but skip if the previous result had failed
    # embeddings (the cache entry was stored before our fix).
    cache_key = _rec_cache_key(full_name, seed, tags)
    cache_now = _rec_time.monotonic()
    cached = _rec_cache_get(cache_key, cache_now)
    if cached is not None:
        emb_status = getattr(cached, "embed_status", {})
        if emb_status.get("desc_emb") == "missing" and emb_status.get("readme_emb") == "missing":
            logger.info("rec cache hit for %s but embeddings were missing — recomputing", cache_key)
            del _rec_cache[cache_key]
        else:
            logger.info("rec cache hit for %s", cache_key)
            cached.from_cache = True
            return cached

    session = await data.get_session()
    try:
        # ── Step 1: Load or fetch source repo ────────────────────────────
        import time as _time
        _t0 = _time.monotonic()
        source = await data.get_repo(session, full_name)

        if source is None:
            logger.info("repo %s not in DB — fetching from GitHub", full_name)
            _t_gh = _time.monotonic()
            gh = await fetch_all(owner, name)
            logger.info("fetch_all took %.1fs for %s", _time.monotonic() - _t_gh, full_name)

            if gh["error"] and not gh["metadata"]:
                raise LookupError(f"failed to fetch repo {full_name!r}: {gh['error']}")

            metadata = gh["metadata"]
            repo_id = int(metadata.get("id", 0))
            language = metadata.get("language")
            stars = int(metadata.get("stargazers_count") or 0)
            description = metadata.get("description")
            topics = gh["topics"]
            deps = gh["deps"]

            await data.upsert_repo(
                session, repo_id=repo_id, owner=owner, name=name,
                full_name=full_name, description=description,
                language=language, topics=topics, stars=stars,
                dependencies=deps,
            )
            await session.commit()
            source = await data.get_repo(session, full_name)
            if source is None:
                raise LookupError(f"failed to persist repo {full_name!r}")

        # ── Step 2: Ensure source has vectors ────────────────────────────
        from reporelay_mvp.data import is_real_vector

        source_has_desc_emb = is_real_vector(source.description_embedding)
        source_has_readme_emb = is_real_vector(source.embedding)
        source_needs_embed = not source_has_desc_emb or not source_has_readme_emb

        embed_status: dict[str, str] = {
            "desc_emb": "cached" if source_has_desc_emb else "missing",
            "readme_emb": "cached" if source_has_readme_emb else "missing",
        }
        source_readme_tokens: set[str] | None = None

        if source_needs_embed:
            # ── Keyword-based semantic search ──────────────────────────
            # Instead of borrowing a proxy from a random topic-similar repo
            # (which gives "python" repos for a "data science" repo just
            # because they share the "python" topic), we extract keywords
            # from the source's own description + README, embed those
            # keywords as a query vector, and search against ALL 11k
            # description embeddings already stored in the DB.
            #
            # This is fundamentally different from proxy:
            #   - Proxy: "this repo is Python → find any Python repo"
            #   - Keyword: "keywords are data science, education" →
            #     find repos with description embeddings matching those concepts

            # 1. Fetch README to extract keywords
            _generic_repo_names = frozenset({
                "ossu", "awesome", "list", "collection", "repo", "course", "courses",
                "curriculum", "guide", "project", "tool", "library", "framework", "app", "api",
                "data", "code", "src", "main", "test", "docs", "master", "dev", "prod",
                "v1", "v2", "v3", "go", "py", "js", "ts", "rs", "rb", "cpp", "cc",
                "example", "demo", "sample", "tutorial", "learn", "learning",
                "starter", "template", "boilerplate", "scaffold", "cookiecutter",
                "dotfiles", "config", "setup", "install", "build", "deploy",
            })
            try:
                settings = get_mvp_settings()
                async with _auth_client_active() as client:
                    readme_text = await fetch_readme(client, owner, name)
                    meta = await fetch_repo_metadata(client, owner, name)
                desc_text = meta.get("description", "") or source.description or ""
                if readme_text and readme_text.strip():
                    from reporelay_mvp.keyword_extractor import extract_keywords_from_repo
                    keywords = extract_keywords_from_repo(desc_text, readme_text)
                elif desc_text:
                    from reporelay_mvp.keyword_extractor import extract_keywords
                    keywords = extract_keywords(desc_text)
                else:
                    keywords = []

                # Build search query from THREE sources (priority order):
                #   1. Description (most purpose-dense — one sentence)
                #   2. Repo name tokens (differentiator: "data-science" vs "computer-science")
                #   3. Top distinctive keywords (from keyword extraction)
                # This prevents two repos from the same org (ossu/data-science
                # and ossu/computer-science) from producing identical search vectors.
                query_parts: list[str] = []

                # Priority 1: Description (most purpose-dense text)
                if desc_text and desc_text.strip():
                    query_parts.append(clean_description(desc_text))

                # Priority 2: Repo name tokens (differentiating signal)
                # Include composite name: "ai-job-search" → "ai job search"
                name_compounded = name.lower().replace("-", " ").replace("_", " ").replace(".", " ")
                name_tokens = name_compounded.split()
                name_tokens = [t for t in name_tokens if t not in _generic_repo_names and len(t) >= 2]
                if name_tokens:
                    query_parts.append(" ".join(name_tokens[:8]))

                # Priority 3: Top distinctive keywords
                if keywords:
                    # Filter generic education keywords that appear in many repos
                    filtered_kw = [kw for kw in keywords
                                  if kw.lower() not in {"self-taught", "self-taught education", "free self-taught",
                                                        "path", "education", "learning", "course", "courses",
                                                        "curriculum", "tutorial", "guide", "introduction",
                                                        "getting started", "best practices", "awesome list"}]
                    query_parts.append(" ".join(filtered_kw[:15]))

                query_text = " ".join(query_parts)
                if query_text:
                    logger.info("  search query for %s: %s", full_name, query_text[:150])

                    # 2. Embed the query as a single vector — one Gemini call
                    query_emb = await embed_text(query_text)
                    if is_real_vector(query_emb):
                        source = source.model_copy(update={
                            "embedding": query_emb,
                            "description_embedding": query_emb,
                            "keywords": keywords,
                        })
                        embed_status = {"desc_emb": "keyword", "readme_emb": "keyword"}
                        logger.info("  keyword semantic search active for %s (%d keywords)",
                                    full_name, len(keywords))
            except Exception as exc:
                logger.info("  using description fallback for %s (GitHub may be rate-limited)", full_name)

                # Embed the full description — it has the richest semantic
                # signal. Google's text-embedding-004 handles sentence-level
                # text well. Keywords are used for structured matching
                # (keyword_match / keyword_topic_match scoring features),
                # not for the embedding vector.
                desc_text = source.description or ""
                if desc_text and desc_text.strip():
                    try:
                        query_emb = await embed_text(clean_description(desc_text))
                        if is_real_vector(query_emb):
                            source = source.model_copy(update={
                                "embedding": query_emb,
                                "description_embedding": query_emb,
                            })
                            embed_status = {"desc_emb": "desc-only", "readme_emb": "desc-only"}
                            logger.info("  description-only embedding for %s (GitHub rate-limited)", full_name)
                    except Exception as exc2:
                        logger.warning("  description embed also failed for %s: %s", full_name, exc2)

            # ── Fallback to proxy if keyword extraction failed ──────────
            if not is_real_vector(source.description_embedding) and not is_real_vector(source.embedding):
                if not source_has_readme_emb:
                    proxy_emb = await _find_proxy_embedding(session, source)
                    if proxy_emb:
                        source = source.model_copy(update={"embedding": proxy_emb})
                        embed_status["readme_emb"] = "proxy"

                if not source_has_desc_emb:
                    proxy_desc = await _find_desc_proxy_embedding(session, source)
                    if proxy_desc:
                        source = source.model_copy(update={"description_embedding": proxy_desc})
                        embed_status["desc_emb"] = "proxy"

            # ── Background: launch real Gemini embed for next visit ───────
            _bg_owner, _bg_name = owner, name
            _bg_source_id = source.id
            _bg_full_name = full_name
            async def _background_embed() -> None:
                """Embed description + README in background."""
                try:
                    bg_session = await data.get_session()
                    try:
                        settings = get_mvp_settings()
                        async with _auth_client_active() as client:
                            readme_text = await fetch_readme(client, _bg_owner, _bg_name)
                            meta = await fetch_repo_metadata(client, _bg_owner, _bg_name)
                        desc_text = meta.get("description", "") or ""

                        if not readme_text or not readme_text.strip():
                            return

                        from reporelay_mvp.purpose import get_effective_description
                        desc = get_effective_description(desc_text, readme_text) or desc_text or readme_text[:200].replace("\n", " ")

                        desc_emb, readme_emb = await asyncio.wait_for(
                            asyncio.gather(embed_text(clean_description(desc)), embed_text(readme_text[:8000])),
                            timeout=25.0,
                        )
                        if is_real_vector(readme_emb):
                            await data.set_embedding(bg_session, repo_id=_bg_source_id, embedding=readme_emb)
                        if is_real_vector(desc_emb):
                            await data.set_description_embedding(bg_session, repo_id=_bg_source_id, description_embedding=desc_emb)
                        await bg_session.commit()
                        logger.info("  background embed complete for %s", _bg_full_name)
                    finally:
                        await bg_session.close()
                except Exception as exc:
                    logger.warning("  background embed failed for %s: %s", _bg_full_name, exc)

            asyncio.create_task(_background_embed())

        # ── Step 3: Generate candidates ──────────────────────────────────
        _t_cand = _time.monotonic()
        candidates = await generate_candidates(session, source, seed=seed, tags=tags)
        logger.info("generate_candidates took %.1fs (%d candidates) for %s",
                    _time.monotonic() - _t_cand, len(candidates), full_name)

        # ── Step 4: Score ────────────────────────────────────────────────
        filter_emb = None
        if tags:
            filter_emb = await embed_text(" ".join(tags))

        _t_score = _time.monotonic()
        scored = await score_many(
            source, candidates, seed=seed, tags=tags,
            filter_embedding=filter_emb, source_readme_tokens=source_readme_tokens,
        )
        logger.info("score_many took %.1fs for %s", _time.monotonic() - _t_score, full_name)

        # ── Step 5: Categorize and return ────────────────────────────────
        result = categorize_results(source, scored, limit, seed)
        result.embed_status = embed_status
        result.from_cache = False
        # Don't cache results where both embeddings are missing —
        # next request should retry the full embed flow.
        if embed_status.get("desc_emb") != "missing" or embed_status.get("readme_emb") != "missing":
            _rec_cache_set(cache_key, cache_now, result)
        logger.info("TOTAL recommend() took %.1fs for %s (%d groups, %d repos)",
                    _time.monotonic() - _t0, full_name,
                    len(result.groups), len(result.flat_repos))
        return result
    finally:
        await session.close()


def _pool_fraction(pool_size: int, fraction: float, min_val: int, max_val: int) -> int:
    """Compute a soft cap as a fraction of the deduped pool, clamped.

    Ensures category sizes scale with pool breadth: a deep pool of 300
    unique owners gets more slots per category than a shallow pool of 80.
    """
    return max(min_val, min(max_val, int(pool_size * fraction)))


def categorize_results(
    source: Repo,
    scored: list[tuple[Repo, float, Features]],
    limit: int,
    seed: int | None,
) -> CategorizedRecommendation:
    """Split scored candidates into labeled groups by primary signal.

    Categories are ordered by relevance intent:
      1. Top Picks — best overall composite scores
      2. Topic-specific — one category per source topic (dynamic)
      3. Same Stack — repos sharing dependencies
      4. Hidden Gems — high semantic relevance, lower popularity
      5. Cross-Language — different language, related concepts
      6. Trending — repos gaining traction
      7. Also Good — remaining quality candidates

    Caps are soft and scale with the deduped pool size. No hard
    total cap — the pool's natural size (~200–250 unique owners
    for 51K corpus) bounds total output to ~45–70 repos.

    See pool_size.md for corpus-to-pool scaling guidance.
    """
    source_lang = source.language.lower() if source.language else None
    source_owner = source.owner.lower()

    # ── Filter and deduplicate by owner ───────────────────────────
    filtered = [
        (repo, sc, feats) for repo, sc, feats in scored
        if repo.id != source.id
        and repo.owner.lower() != source_owner
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)

    seen_owners: set[str] = set()
    deduped: list[tuple[Repo, float, Features]] = []
    for repo, sc, feats in filtered:
        owner = repo.owner.lower()
        if owner in seen_owners:
            continue
        seen_owners.add(owner)
        deduped.append((repo, sc, feats))

    pool_n = len(deduped)
    used_ids: set[int] = set()
    groups: list[RecommendationGroup] = []

    # ── Group 1: Top Picks — best overall composite scores ─────────
    top_n = _pool_fraction(pool_n, 0.06, 4, 10)
    top_candidates = sorted(deduped, key=lambda x: x[1], reverse=True)
    g = _make_group(top_candidates, "Top Picks", "best overall match", used_ids, top_n,
                     source_topics=source.topics)
    if g:
        groups.append(g)

    # ── Group 2: Topic categories (one per source topic) ──────────
    src_lang_lower = (source.language or "").lower()
    meaningful_topics = [
        t for t in source.topics
        if t.lower() not in (src_lang_lower, "") and len(t) > 1
    ]
    meaningful_topics.sort(key=lambda t: -len(t))

    topic_n = _pool_fraction(pool_n, 0.025, 2, 5)
    _max_topic_groups = 8
    for topic in meaningful_topics:
        if len(groups) >= _max_topic_groups + 1:
            break
        topic_lower = topic.lower()
        topic_matches = [
            r for r in deduped
            if r[0].id not in used_ids
            and any(t.lower() == topic_lower for t in r[0].topics)
        ]
        if topic_matches:
            g = _make_group(topic_matches, topic.title(), f"repos tagged {topic}", used_ids, topic_n,
                            source_topics=source.topics)
            if g:
                groups.append(g)

    # ── Group 3: Same Stack — shared dependencies ─────────────────
    deps_n = _pool_fraction(pool_n, 0.025, 2, 5)
    same_stack = [
        r for r in deduped
        if r[0].id not in used_ids
        and r[2].dep_overlap > 0.0
    ]
    if same_stack:
        g = _make_group(same_stack, "Same Stack", "shared dependencies & tooling", used_ids, deps_n,
                        source_topics=source.topics)
        if g:
            groups.append(g)

    # ── Group 4: Hidden Gems — high relevance + low popularity ────
    gems_n = _pool_fraction(pool_n, 0.025, 2, 5)
    hidden_gems = [
        r for r in deduped
        if r[0].id not in used_ids
        and r[2].readme_vs_desc_cosine_sim > 0.6
        and r[2].star_ratio < 0.3
    ]
    if hidden_gems:
        g = _make_group(hidden_gems, "Hidden Gems", "high relevance, less discovered", used_ids, gems_n,
                        source_topics=source.topics)
        if g:
            groups.append(g)

    # ── Group 5: Cross-Language — different language, similar ideas ──
    cross_n = _pool_fraction(pool_n, 0.035, 3, 8)
    if source_lang:
        cross = [
            r for r in deduped
            if r[0].id not in used_ids
            and r[0].language and r[0].language.lower() != source_lang
            and (r[2].topic_overlap > 0.05 or r[2].readme_vs_desc_cosine_sim > 0.15)
        ]
        if cross:
            g = _make_group(cross, "Cross-Language", "different language, similar ideas", used_ids, cross_n,
                            source_topics=source.topics)
            if g:
                groups.append(g)

    # ── Group 6: Trending — gaining traction ──────────────────────
    trending_n = _pool_fraction(pool_n, 0.02, 1, 4)
    trending = [
        r for r in deduped
        if r[0].id not in used_ids
        and r[2].trending_boost > 0.0
        and r[1] > 0.1
    ]
    if trending:
        g = _make_group(trending, "Trending", "rising in popularity", used_ids, trending_n,
                        source_topics=source.topics)
        if g:
            groups.append(g)

    # ── Group 7: Also Good — remaining above quality threshold ────
    remaining_n = _pool_fraction(pool_n, 0.10, 5, 15)
    remaining = [r for r in deduped if r[0].id not in used_ids and r[1] > 0.08]
    if remaining:
        g = _make_group(remaining, "Also Good", "more repos you might like", used_ids, remaining_n,
                        source_topics=source.topics)
        if g:
            groups.append(g)

    all_repos: list[ScoredRepo] = []
    for g in groups:
        all_repos.extend(g.repos)

    return CategorizedRecommendation(
        source_repo=source.full_name,
        flat_repos=all_repos,
        groups=groups,
    )


def _make_group(
    candidates: list[tuple[Repo, float, Features]],
    label: str,
    signal: str,
    used_ids: set[int],
    max_repos: int,
    *,
    source_topics: list[str] | None = None,
) -> RecommendationGroup | None:
    src_topics = set(source_topics or [])
    repos: list[ScoredRepo] = []
    for repo, sc, feats in candidates:
        if repo.id in used_ids:
            continue
        if len(repos) >= max_repos:
            break
        used_ids.add(repo.id)
        shared = [t for t in repo.topics if t.lower() in (st.lower() for st in src_topics)]
        repos.append(ScoredRepo(
            id=repo.id, owner=repo.owner, name=repo.name,
            full_name=repo.full_name, description=repo.description,
            language=repo.language, topics=repo.topics, stars=repo.stars,
            dependencies=repo.dependencies, score=sc,
            features=feats.as_dict(),
            shared_topics=shared,
            shared_language=feats.language_match >= 1.0,
        ))
    if not repos:
        return None
    return RecommendationGroup(label=label, signal=signal, repos=repos)


async def _embed_description_fast(
    source: Repo,
    owner: str,
    name: str,
    session: Any,
) -> tuple[Repo, set[str] | None, list[str], str, dict[str, str]]:
    """Fast path: fetch README + embed description only (1 Gemini call).

    Returns in ~5-8s (GitHub fetch + 1 Gemini call) instead of 15-25s
    (GitHub fetch + 2 Gemini calls). The README text is returned so the
    caller can schedule a background embed.

    Returns (updated_source, readme_tokens, dependencies, readme_text, status).
    """
    from reporelay_mvp.purpose import get_effective_description

    settings = get_mvp_settings()
    deps: list[str] = []
    readme_text: str = ""

    # 1. Fetch README + dependencies from GitHub
    try:
        async with _auth_client_active() as client:
            readme_text, deps = await asyncio.wait_for(
                asyncio.gather(
                    fetch_readme(client, owner, name),
                    fetch_dependencies(client, owner, name),
                ),
                timeout=15.0,
            )
    except asyncio.TimeoutError:
        logger.error("GitHub fetch timed out for %s/%s after 15s", owner, name)
        raise EmbedError(
            f"GitHub API timed out fetching README for {owner}/{name}"
        ) from None
    except Exception as exc:
        logger.exception("GitHub fetch failed for %s/%s", owner, name)
        raise EmbedError(
            f"could not fetch README from GitHub for {owner}/{name}: {exc}"
        ) from exc

    if not readme_text or not readme_text.strip():
        raise EmbedError(f"repo {owner}/{name} has no README — cannot embed")

    # 2. Effective description
    effective_description = get_effective_description(source.description, readme_text)
    if not effective_description:
        effective_description = readme_text.strip()[:200].replace("\n", " ")

    if effective_description != (source.description or ""):
        try:
            await data.upsert_repo(
                session,
                repo_id=source.id, owner=source.owner, name=source.name,
                full_name=source.full_name, description=effective_description,
                language=source.language, topics=source.topics,
                stars=source.stars, dependencies=deps,
            )
            source = source.model_copy(update={"description": effective_description})
        except Exception:
            logger.exception("failed to persist description for %s/%s", owner, name)

    # 3. Embed the description (1 Gemini call — the fast part)
    try:
        desc_emb = await embed_text(clean_description(effective_description))
    except Exception as exc:
        logger.exception("description embedding failed for %s/%s", owner, name)
        raise EmbedError(f"description embedding failed for {owner}/{name}: {exc}") from exc

    from reporelay_mvp.data import is_real_vector
    if not is_real_vector(desc_emb):
        raise EmbedError(f"description embedding returned zero/NaN for {owner}/{name}")

    try:
        await data.set_description_embedding(session, repo_id=source.id, description_embedding=desc_emb)
        source = source.model_copy(update={"description_embedding": desc_emb})
    except Exception:
        logger.exception("failed to persist description_embedding for %s/%s", owner, name)
        raise

    # 4. Extract README tokens (for readme_topic_sim — works without embedding)
    from reporelay_mvp.features import _tokenize_readme
    readme_tokens = _tokenize_readme(source.full_name, readme_text) if readme_text.strip() else None

    try:
        await session.commit()
    except Exception:
        logger.exception("DB commit failed for %s/%s", owner, name)
        raise

    status = {
        "desc_emb": "ok",
        "readme_emb": "pending",
        "effective_description": effective_description[:200],
    }
    logger.info("fast-embedded description for %s/%s (%d chars)", owner, name, len(effective_description))
    return source, readme_tokens, deps, readme_text, status


# ── Background README embedding ─────────────────────────────────────

# Fire-and-forget task list. We use asyncio.create_task from the
# endpoint handler. If the server restarts, pending tasks are lost —
# that's fine: the next request will re-schedule them.
_readme_bg_tasks: set[asyncio.Task] = set()


def _schedule_readme_background(repo_id: int, owner: str, name: str, readme_text: str) -> None:
    """Schedule a background task to embed the README via Gemini.

    This runs AFTER the HTTP response is sent, so the user doesn't
    wait for it. The next request for this repo will find the README
    embedding already in the DB.
    """
    async def _bg_embed() -> None:
        from reporelay_mvp.data import is_real_vector, get_session, set_embedding
        try:
            readme_emb = await asyncio.wait_for(
                embed_text(readme_text[:8000]),
                timeout=20.0,
            )
            if is_real_vector(readme_emb):
                bg_session = await get_session()
                try:
                    await set_embedding(bg_session, repo_id=repo_id, embedding=readme_emb)
                    await bg_session.commit()
                    logger.info("background: embedded README for %s/%s", owner, name)
                finally:
                    await bg_session.close()
            else:
                logger.warning("background: README embedding returned zero for %s/%s", owner, name)
        except Exception:
            logger.exception("background: README embedding failed for %s/%s", owner, name)

    task = asyncio.create_task(_bg_embed())
    _readme_bg_tasks.add(task)
    task.add_done_callback(_readme_bg_tasks.discard)


async def _embed_source_live(
    source: Repo,
    owner: str,
    name: str,
    session: Any,
) -> tuple[Repo, set[str] | None, list[str], dict[str, str]]:
    """Fetch README + embed description + embed README via live Gemini API.

    Stores vectors in DB so future visits are instant (no API call needed).

    Strategy:
      1. Fetch README + dependencies from GitHub. If this fails, raise
         EmbedError — the caller can't recommend a repo it can't read.
      2. Build an "effective description": prefer the GitHub description,
         but if it's missing/short/filler, fall back to a purpose statement
         extracted from the README. This is what we embed and store as
         `description` for the recommendation signal.
      3. Embed the effective description via Gemini → description_embedding.
      4. Embed the first 8000 chars of the README via Gemini → embedding.
      5. Persist both vectors + the new description in one transaction.

    If either embedding call fails, we raise EmbedError. The caller
    (recommend()) will surface this to the user instead of returning
    results that look like noise.

    Returns (updated_source, readme_tokens, dependencies, status).
    Status keys: "desc_emb", "readme_emb", "effective_description".
    """
    from reporelay_mvp.purpose import get_effective_description

    settings = get_mvp_settings()
    deps: list[str] = []
    readme_text: str = ""

    # 1. Fetch README + dependencies from GitHub
    try:
        async with _auth_client_active() as client:
            readme_text, deps = await asyncio.wait_for(
                asyncio.gather(
                    fetch_readme(client, owner, name),
                    fetch_dependencies(client, owner, name),
                ),
                timeout=20.0,
            )
    except asyncio.TimeoutError:
        logger.error("GitHub fetch timed out for %s/%s after 20s", owner, name)
        raise EmbedError(
            f"GitHub API timed out fetching README for {owner}/{name}"
        ) from None
    except Exception as exc:
        logger.exception("GitHub fetch failed for %s/%s", owner, name)
        raise EmbedError(
            f"could not fetch README from GitHub for {owner}/{name}: {exc}"
        ) from exc

    if not readme_text or not readme_text.strip():
        raise EmbedError(
            f"repo {owner}/{name} has no README — cannot embed"
        )

    # 2. Effective description: prefer the GitHub one, else extract from README
    effective_description = get_effective_description(
        source.description, readme_text,
    )
    if not effective_description:
        # Last-ditch fallback: use the first ~200 chars of the cleaned README
        effective_description = readme_text.strip()[:200].replace("\n", " ")

    # If the effective description differs from what's in the DB, persist it
    # so future visits and the description_cosine_sim signal both see it.
    if effective_description != (source.description or ""):
        try:
            await data.upsert_repo(
                session,
                repo_id=source.id,
                owner=source.owner,
                name=source.name,
                full_name=source.full_name,
                description=effective_description,
                language=source.language,
                topics=source.topics,
                stars=source.stars,
                dependencies=deps,
            )
            source = source.model_copy(update={"description": effective_description})
        except Exception:
            logger.exception("failed to persist effective description for %s/%s", owner, name)

    # 3. Embed the effective description
    try:
        desc_emb = await embed_text(clean_description(effective_description))
    except Exception as exc:
        logger.exception("description embedding failed for %s/%s", owner, name)
        raise EmbedError(
            f"description embedding via Gemini failed for {owner}/{name}: {exc}"
        ) from exc

    from reporelay_mvp.data import is_real_vector
    if not is_real_vector(desc_emb):
        raise EmbedError(
            f"description embedding returned zero/NaN vector for {owner}/{name} — "
            "check EMBEDDING_API mode and API key"
        )

    try:
        await data.set_description_embedding(
            session, repo_id=source.id, description_embedding=desc_emb,
        )
        source = source.model_copy(update={"description_embedding": desc_emb})
    except Exception:
        logger.exception("failed to persist description_embedding for %s/%s", owner, name)
        raise

    # 4. Embed the README
    readme_for_embed = readme_text[:8000]
    try:
        readme_emb = await embed_text(readme_for_embed)
    except Exception as exc:
        logger.exception("README embedding failed for %s/%s", owner, name)
        raise EmbedError(
            f"README embedding via Gemini failed for {owner}/{name}: {exc}"
        ) from exc

    if not is_real_vector(readme_emb):
        raise EmbedError(
            f"README embedding returned zero/NaN vector for {owner}/{name} — "
            "check EMBEDDING_API mode and API key"
        )

    try:
        await data.set_embedding(session, repo_id=source.id, embedding=readme_emb)
        source = source.model_copy(update={"embedding": readme_emb})
    except Exception:
        logger.exception("failed to persist embedding for %s/%s", owner, name)
        raise

    # 5. Extract README tokens for the readme_topic_sim feature
    from reporelay_mvp.features import _tokenize_readme

    readme_tokens: set[str] | None = None
    if readme_text and readme_text.strip():
        readme_tokens = _tokenize_readme(source.full_name, readme_text)

    # 6. Persist everything in one commit so the next request sees it all
    try:
        await session.commit()
    except Exception:
        logger.exception("DB commit failed after live embed for %s/%s", owner, name)
        raise

    logger.info(
        "live embedding complete for %s/%s (desc=%d chars → vec, readme=%d chars → vec, tokens=%d)",
        owner, name, len(effective_description), len(readme_for_embed),
        len(readme_tokens) if readme_tokens else 0,
    )
    return source, readme_tokens, deps, {
        "desc_emb": "ok",
        "readme_emb": "ok",
        "effective_description": effective_description[:200],
    }


class EmbedError(Exception):
    """Raised when we can't produce a real embedding for a new source repo."""
    pass


async def _expand_pool(
    session: Any,
    source: Repo,
    *,
    seed: int | None = None,
    tags: list[str] | None = None,
) -> list[tuple[Repo, float]]:
    """
    Build the candidate pool from two sources:

    1. The local DB (pgvector ANN + SQL filter) — fast, has
       embeddings for cosine sim, but only knows about rows we've
       already indexed.
    2. A live GitHub search — uses the source's topics OR'd
       together with its language, returns up to SEARCH_LIMIT
       fresh results.

    Search hits are persisted back to the DB so the corpus grows
    over time. They're scored with cosine_sim = 0 (no embedding
    yet); the other four features (language, topics, deps,
    popularity) carry the score for these.
    """
    db_candidates = await generate_candidates(session, source, seed=seed, tags=tags)
    logger.info("db pool: %d candidates", len(db_candidates))

    if len(db_candidates) >= _MIN_DB_POOL_FOR_SKIP:
        logger.info(
            "db pool has %d candidates (>= %d) — skipping github search for speed",
            len(db_candidates),
            _MIN_DB_POOL_FOR_SKIP,
        )
        return db_candidates

    settings = get_mvp_settings()
    search_items: list[dict[str, Any]] = []
    try:
        # Seed-aware GitHub search: pick different topics, sort orders,
        # and pages so different seeds produce genuinely different candidates.
        # Without this, every seed returns the same popular repos sorted
        # by stars — which makes the top recommendations redundant.
        topics_for_search: list[str] | None = None
        sort_for_search = "stars"
        page_for_search = 1
        search_language: str | None = source.language

        if seed is not None:
            import random as _rnd
            rng = _rnd.Random(seed)

            # Pick a random topic (not always the first/language topic)
            if source.topics:
                # Filter out the language topic if there are better topics available
                good_topics = [t for t in source.topics if t.lower() != (source.language or "").lower()]
                if good_topics and rng.random() < 0.6:
                    topics_for_search = [rng.choice(good_topics)]
                elif source.topics:
                    topics_for_search = [rng.choice(source.topics)]

            # Vary sort order (stars, updated, forks)
            sort_options = ["stars", "updated", "forks"]
            sort_for_search = rng.choice(sort_options)

            # Vary page
            page_for_search = rng.randint(1, 3)

            # 25% chance: search without language to get cross-language results
            if rng.random() < 0.25:
                search_language = None

        async with _auth_client_active() as client:
            payload = await _cached_search(
                client,
                topics=topics_for_search,
                language=search_language,
                min_stars=100,
                sort=sort_for_search,
                per_page=SEARCH_LIMIT,
                page=page_for_search,
            )
            search_items = list(payload.get("items", []))
    except Exception as exc:
        logger.warning("github search failed: %s — falling back to db-only pool", exc)
        return db_candidates

    if not search_items:
        logger.info("github search returned 0 items — db pool only")
        return db_candidates

    written = await data.bulk_upsert_from_search(session, search_items)
    await session.commit()
    logger.info("github search: %d items, %d upserted to db", len(search_items), written)

    candidates: list[tuple[Repo, float]] = list(db_candidates)
    seen: set[int] = {c.id for c, _ in db_candidates}
    seen.add(source.id)

    tag_set = {t.lower() for t in tags} if tags else None
    added = 0
    for item in search_items:
        repo = _search_item_to_repo(item)
        if repo.id in seen:
            continue
        if tag_set and not (tag_set & {t.lower() for t in repo.topics}):
            continue
        seen.add(repo.id)
        candidates.append((repo, 0.0))
        added += 1

    logger.info("merged pool: %d db + %d search = %d", len(db_candidates), added, len(candidates))
    return candidates


async def recommend_random(
    *,
    seed: int,
    limit: int = 10,
) -> ScoredRecommendation:
    if limit <= 0:
        raise ValueError("limit must be > 0")

    session = await data.get_session()
    try:
        source = await data.get_random_repo(session, seed=seed)
        if source is None:
            raise LookupError("no repos in mvp_repos — save some first")

        candidates = await _expand_pool(session, source, seed=seed)

        scored = await score_many(source, candidates, seed=seed)
        result = categorize_results(source, scored, limit, seed)
        return result
    finally:
        await session.close()


def _build_cosine_lookup(candidates: list[tuple[Any, float]]) -> dict[int, float]:
    return {cand.id: sim for cand, sim in candidates}


async def recommend_dict(
    full_name: str,
    *,
    limit: int = 10,
    seed: int | None = None,
) -> dict[str, Any]:
    rec = await recommend(full_name, limit=limit, seed=seed)
    return {
        "source_repo": rec.source_repo,
        "repos": [repo.model_dump() for repo in rec.repos],
    }
