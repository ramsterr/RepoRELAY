# Candidate Pool Sizing Guide

## What the pool is

The candidate pool is the shortlist of repos fetched from the database for scoring and reranking. The system has ~51K repos indexed, but computing all 15 features on every repo would be wasteful. Instead:

1. **Retrieval** (cheap, indexed) — fetch plausible candidates via HNSW vector search + SQL filters
2. **Scoring** (expensive, in-memory) — compute features and weighted sum on the shortlist only
3. **Reranking** (cheap) — diversity dedup, owner limits, category grouping

The pool size is the bridge between stages 1 and 2.

## Current configuration (July 2026)

| Parameter | Value | Location | Purpose |
|---|---|---|---|
| `vector_k` | **200** | `candidates.py:37` | Top-K per vector ANN query (2 pools → 2 × 200) |
| `pool_size` | **400** | `candidates.py:36` | Max results from SQL filter (language/topic) |
| `_MIN_DB_POOL_FOR_SKIP` | **300** | `recommend.py:98` | Skip GitHub API expansion if DB pool ≥ this |
| `hnsw.ef_search` | **max(limit × 2, 100)** | `data.py` | HNSW search effort, set per-query via `SET LOCAL` |
| HNSW index `m` | **16** | migration `mvp_001` | Connections per layer in HNSW graph |
| HNSW index `ef_construction` | **200** | migration `mvp_001` | Build-time candidate list size |
| Output slots | **~45–70** (soft-cap) | `categorize_results()` | Final repos shown across 7 categories |
| Corpus size | **~51K repos** | DB | Total indexed repos with embeddings |

### Expected pool composition at these settings

| Source | Count | Notes |
|---|---|---|
| Vector pool 1 (README-vs-desc) | 200 | HNSW on `embedding` column (if populated) |
| Vector pool 2 (desc-vs-desc) | 200 | Sequential scan on `description_embedding`† |
| After vector dedup | ~280 | ~60% overlap between the two pools |
| SQL pool (language/topic) | 400 | Btree + GIN indexes, sorted by stars |
| After full dedup | ~600 | SQL and vector pools are largely disjoint |
| After owner dedup | ~200-250 | One repo per owner max |
| **Final output (soft-cap)** | **~45–70** | Spread across 7 categories |

### Category breakdown at default pool size (200 unique owners)

| Category | Soft cap formula | Typical count | Signal |
|---|---|---|---|
| Top Picks | `pool × 0.06` (4–10) | 8–10 | Highest composite score |
| Per-topic groups | `pool × 0.025` (2–5) each | 2–5 × N topics | Exact topic match |
| Same Stack | `pool × 0.025` (2–5) | 4–5 | Shared dependencies |
| Hidden Gems | `pool × 0.025` (2–5) | 3–5 | High sim + low stars |
| Cross-Language | `pool × 0.035` (3–8) | 5–7 | Different language, related |
| Trending | `pool × 0.02` (1–4) | 2–4 | Rising popularity |
| Also Good | `pool × 0.10` (5–15) | 10–15 | Above quality threshold |

† **Note:** `description_embedding` currently has no HNSW index (only `embedding` does). This means the primary vector path does a full sequential scan over ~51K vectors (~100 MB read) per query. See "Recommended DB changes" below.

## How to size your pool

### Rule 1: `vector_k` should be 0.3–0.5% of the corpus

The HNSW index returns neighbors in descending cosine similarity. The marginal quality of each additional neighbor drops as you go deeper. At 0.5% of the corpus, you're still well within the "clearly relevant" band.

| Corpus size | `vector_k` (recommended) | Top % of corpus |
|---|---|---|
| 5,000 | 100 | 2.0% |
| 10,000 | 120 | 1.2% |
| 25,000 | 150 | 0.6% |
| 50,000 | **200** | 0.4% |
| 100,000 | 250 | 0.25% |
| 250,000 | 300 | 0.12% |
| 500,000 | 400 | 0.08% |
| 1,000,000 | 500 | 0.05% |

The relationship follows a square-root curve: doubling the corpus warrants a ~40% increase in `vector_k`.

### Rule 2: `pool_size` should be 1.5–2× `vector_k`

The SQL pool (`ORDER BY stars DESC`) and the vector pool (`ORDER BY cosine_sim`) rank repos on fundamentally different axes. This means they have low overlap, so the SQL pool complements the vector pool rather than duplicating it.

### Rule 3: Keep `ef_search` at 2× `LIMIT` minimum

pgvector's default `ef_search = max(limit, 40)` is too low for high-recall ANN search. Setting it to `2 × limit` ensures the HNSW algorithm explores enough of the graph to find the true nearest neighbors.

Our implementation sets this automatically:
```python
_ef_search = max(limit * 2, 100)
await session.execute(text("SET LOCAL hnsw.ef_search = :ef"), {"ef": _ef_search})
```

### Rule 4: Ensure post-owner-dedup candidates exceed output slots by 5–8×

After owner deduplication (one repo per owner), you need at least 5× your output slots in unique-owner candidates. This gives the reranker enough choice per slot. For ~33 output slots, you need ~165 unique-owner candidates. With 50% unique-owner rate, you need ~330 raw candidates. With the SQL + vector pool overlap, ~600 raw candidates is comfortable.

## When to adjust

### Increase `vector_k` when:

- **Corpus grows** — follow the table in Rule 1
- **Precision drops** — if users report "repetitive" recommendations (same topic/owner dominating), you may need more candidates to fill category slots with variety
- **Owner dedup is too aggressive** — if the same org (e.g., `microsoft/`, `google/`) owns a large fraction of top results, more raw candidates means more unique owners survive dedup
- **Adding new vector dimension** — if you switch from 512d to 768d or 1024d, the HNSW recall curve changes; you may need to adjust `ef_search` and/or `vector_k`
- **New embedding model** — different embedding models produce different similarity distributions. Monitor the cosine sim of candidate #200 to determine if you're scraping noise.

### Decrease `vector_k` when:

- **Latency budget shrinks** — each additional 100 candidates costs ~15ms in feature computation
- **Model produces overly-correlated vectors** — if all repos are tightly clustered (cosine sim >0.95 for the 200th neighbor), you're not gaining variety
- **Compute costs matter more than recall** — the scoring step is the bottleneck, not the ANN query

### Increase `pool_size` when:

- **New languages/topics** are added to the seed config, expanding the ecosystem coverage
- **SQL pool consistently has post-dedup counts under the threshold** — more breadth needed

### Never increase past the point where:

- Cosine similarity of the last candidate drops below 0.5 (by that point, you're pulling noise)
- The 200th SQL result from `ORDER BY stars DESC` has <10 stars (irrelevant repos)

## Performance budget

Measured on the `score_many` function, which accounts for >90% of post-retrieval time:

| Vector pool candidates | Approx. score_many time | Memory (Python objects) |
|---|---|---|
| 200 (old: 2 × 100) | ~10ms | ~4 MB |
| 400 (current: 2 × 200) | ~20ms | ~8 MB |
| 600 (2 × 300) | ~30ms | ~12 MB |
| 800 (2 × 400) | ~40ms | ~16 MB |
| 1000 (2 × 500) | ~50ms | ~20 MB |

Add ~20% overhead for SQL pool candidates. The ANN query itself is sub-millisecond (HNSW) to ~50ms (sequential scan on unindexed `description_embedding`).

## Recommended DB changes

### Create HNSW index on `description_embedding`

The description_embedding column (11K+ populated vectors, used as the primary search target) currently has no HNSW index. Every vector query against it does a full sequential scan. Create the index:

```sql
CREATE INDEX IF NOT EXISTS ix_mvp_repos_desc_embedding_hnsw
ON mvp_repos
USING hnsw (description_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
```

This reduces per-query time from ~50ms (seq scan of 51K × 512d) to <1ms (HNSW), and enables the `SET LOCAL hnsw.ef_search` parameter to take effect for the primary search path.

**Migration considerations:**
- Index build time: ~2-5 minutes for 51K vectors at 512d
- Index size: ~8-12 MB
- Zero downtime: `CREATE INDEX CONCURRENTLY` avoids table locks
- Downside: incremental write cost for new embeddings (HNSW insertion is O(log N))

### Increase HNSW `m` for higher-dimensional vectors

If you upgrade to 768d vectors, consider increasing `m` from 16 to 24-32. Higher dimensions need more graph connections per layer for equivalent recall.

```sql
-- After migrating columns to vector(768):
DROP INDEX IF EXISTS ix_mvp_repos_desc_embedding_hnsw;
CREATE INDEX IF NOT EXISTS ix_mvp_repos_desc_embedding_hnsw
ON mvp_repos
USING hnsw (description_embedding vector_cosine_ops)
WITH (m = 24, ef_construction = 200);
```

## Monitoring to add

For data-driven tuning, instrument the following:

```python
# Log at the end of score_many:
logger.info(
    "pool_stats: raw=%d deduped=%d unique_owners=%d "
    "min_cosine=%.3f p50_cosine=%.3f p95_cosine=%.3f",
    len(candidates),
    len(unique_ids),
    len(unique_owners),
    min_cosine_sim,
    median_cosine_sim,
    p95_cosine_sim,
)
```

Watch these metrics over time:
- **min_cosine_sim of scored candidates**: if it stays above 0.6, your pool size is conservative (safe to increase if needed)
- **unique_owners / raw**: if <40%, owner diversity is a problem (increase pool)
- **p50_cosine_sim**: if this drops below 0.4, you're pulling noise (decrease pool or tighten scoring)
- **score_many latency**: set a p99 budget (e.g., 100ms) and cap pool size accordingly
