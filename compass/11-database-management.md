# 11 — Database Management: Postgres + pgvector + Apache AGE

How RepoRelay stores, indexes, and retrieves data at scale. Why we chose one database with three capabilities instead of three separate databases. What each piece does, when to use which one, and how to make them fast.

---

## The Problem: Three Data Shapes, Three Access Patterns

Recommendation systems deal with three fundamentally different kinds of data. You cannot use the same storage engine for all three and expect good latency.

| Data shape | Example | Access pattern | Needs |
|---|---|---|---|
| **Rows and columns** | Repo metadata, star events, user profiles | Exact lookups, range scans, aggregations | Indexed tables, ACID |
| **Vectors** | README embeddings (768 numbers per repo) | Nearest neighbor search ("find the 100 closest vectors") | ANN index, sub-20ms |
| **Graphs** | Dependency edges, contributor edges, co-star patterns | Traversal ("walk 2 hops from this node") | Edge index, fast neighbor expansion |

If you use Postgres tables for graph traversal, you write 5-table JOINs. Slow.

If you use a vector database for repo metadata, you pay per query. Expensive.

If you use a graph database for embeddings, it does not do vector search. Wrong tool.

**The solution:** Postgres as the core, with pgvector and Apache AGE as extensions. One server. Three capabilities. No data sync between systems.

---

## Component 1: Postgres — The Foundation

### What it stores

```
Postgres (relational)
 ├── repos                        (dimension: every GitHub repo)
 ├── users                        (dimension: every GitHub user)
 ├── topics                       (dimension: every GitHub topic)
 ├── languages                    (dimension: every programming language)
 ├── star_events                  (fact: user starred a repo)
 ├── fork_events                  (fact: user forked a repo)
 ├── contributor_edges            (fact: user contributed to repo)
 ├── dependency_edges             (fact: repo depends on a package)
 ├── workflow_cooccurrence        (fact: two repos appear in same CI)
 ├── readme_texts                 (content: raw README markdown)
 └── user_blend_states            (hot: per-user blend weights)
```

### How to make it fast

**1. Index what you query. Not what you store.**

Bad: index every column.
Good: index columns that appear in WHERE, JOIN, and ORDER BY of your top 5 queries.

```sql
-- Query: find repos by language + topic
CREATE INDEX idx_repos_language ON repos (language);
CREATE INDEX idx_repos_topics ON repos USING GIN (topics);

-- Query: co-star aggregation
CREATE INDEX idx_stars_user_repo ON star_events (user_id, repo_id);

-- Query: contributor overlap
CREATE INDEX idx_contributor_repo ON contributor_edges (repo_id, user_id);
```

**2. Partial indexes for active data only.**

Why scan 100 million stars when 95% of traffic is on repos active in the last year?

```sql
CREATE INDEX idx_star_events_recent
  ON star_events (repo_id, starred_at DESC)
  WHERE starred_at > NOW() - INTERVAL '2 years';
```

**3. Materialized views for expensive aggregations.**

Co-star counts change all day. But you do not need real-time accuracy on them. Refresh every hour.

```sql
CREATE MATERIALIZED VIEW co_star_counts AS
SELECT s1.repo_id AS repo_a,
       s2.repo_id AS repo_b,
       COUNT(*) AS co_star_count
FROM star_events s1
JOIN star_events s2 ON s1.user_id = s2.user_id AND s1.repo_id != s2.repo_id
GROUP BY s1.repo_id, s2.repo_id;

CREATE INDEX idx_co_star_a ON co_star_counts (repo_a, co_star_count DESC);

-- Refresh every hour
REFRESH MATERIALIZED VIEW CONCURRENTLY co_star_counts;
```

**4. Partition fact tables by time.**

The `star_events` table will grow to hundreds of millions of rows. Partition by month.

```sql
CREATE TABLE star_events (
    user_id BIGINT NOT NULL,
    repo_id BIGINT NOT NULL,
    starred_at TIMESTAMPTZ NOT NULL
) PARTITION BY RANGE (starred_at);

CREATE TABLE star_events_2026_06 PARTITION OF star_events
  FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');
```

Why: queries with `WHERE starred_at > ...` only scan relevant partitions. Old partitions can be moved to slower storage. Dropping old data is instant (DROP TABLE instead of DELETE).

**5. Connection pooling.**

Every request opens a connection to Postgres. Bad. Use a pool of persistent connections.

```python
# reporelay_core/db.py already does this
create_async_engine(
    settings.database_url,
    pool_size=10,        # 10 persistent connections
    max_overflow=20,     # allow up to 20 extra under burst
    pool_pre_ping=True,  # test connection before using it
)
```

---

## Component 2: pgvector — Vector Search

### What it stores

```
pgvector (vector index)
 └── readme_texts.embedding        (768-dim vector per repo)
     indexed via HNSW or IVFFlat
```

### What it does

When a user looks at `next.js`, you want repos with similar README content. You encode the query into a vector and find the closest neighbors.

```sql
SELECT repo_id, full_name, 1 - (embedding <=> query_embedding) AS similarity
FROM readme_texts
JOIN repos ON readme_texts.repo_id = repos.id
ORDER BY embedding <=> query_embedding
LIMIT 200;
```

The `<=>` operator is cosine distance. The `1 - distance` is cosine similarity. 1.0 = identical. 0.0 = opposite.

### How to make it fast (the indexing choice)

**HNSW (Hierarchical Navigable Small World)**

```
Build time:    slow (but offline, you do not care)
Search speed:  very fast (~5ms for 1M vectors)
Memory:        higher (full graph in RAM)
Accuracy:      95-99% recall
Tuning knobs:  m (connections per node, default 16)
               ef_construction (build quality, trade time for accuracy)
               ef_search (query quality, trade speed for accuracy)
```

Use HNSW when: your catalog is <10M repos and you want sub-10ms queries. This is Query 1 in the serving path. Speed is everything.

```sql
CREATE INDEX ON readme_texts
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);
```

**IVFFlat (Inverted File Flat)**

```
Build time:    faster
Search speed:  good (~20ms for 1M vectors)
Memory:        lower
Accuracy:      90-95% recall
Tuning knobs:  lists (number of clusters, sqrt(rows) is a good start)
```

Use IVFFlat when: your catalog is >10M repos and memory is the bottleneck. Also requires a training step.

```sql
CREATE INDEX ON readme_texts
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 1000);
```

**The rule:** Start with HNSW. If memory pressure grows beyond your instance, fall back to IVFFlat. Most projects never need the fallback.

### How to improve pgvector beyond defaults

**1. Reduce vector dimensions with PCA.**

768 dimensions from an embedding model is overkill. Most variance is captured in far fewer dimensions. Run PCA on your corpus and reduce to 256 dimensions — 3x smaller index, barely any recall loss.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=256)
reduced_embeddings = pca.fit_transform(original_768d_embeddings)
# Store the reduced 256d vectors in pgvector
```

**2. Product quantization (when catalog is truly massive).**

Above 10M vectors, even HNSW slows down. Product Quantization splits each vector into chunks and quantizes each chunk into a codebook. 1000x compression. Does not exist natively in pgvector — this is where you would consider Qdrant or Milvus as an external vector store, or implement PQ yourself.

**3. Partition embeddings by ecosystem.**

A Python developer browsing a Python repo rarely needs to see Ruby alternatives. Partition your vector index by language.

```sql
-- Separate HNSW indexes per language
CREATE INDEX ON readme_texts
  USING hnsw (embedding vector_cosine_ops)
  WHERE language = 'JavaScript';

CREATE INDEX ON readme_texts
  USING hnsw (embedding vector_cosine_ops)
  WHERE language = 'Python';
```

Now searches within an ecosystem are 50x faster (fewer vectors to search).

---

## Component 3: Apache AGE — Graph Database

### What it stores

```
Apache AGE (graph)
 ├── Nodes:        Repo, User, Topic, Language, Dependency
 └── Edges:        DEPENDS_ON (weighted)
                   STARRED_BY (with timestamp)
                   CONTRIBUTED_TO (with commit count)
                   HAS_TOPIC
                   WRITTEN_IN
                   CO_OCCURS_IN_WORKFLOW (weighted)
                   IS_ALTERNATIVE_TO (inferred)
```

### What it does

When a user looks at `next.js`, the graph answers: "what is 1-2 hops away?"

```
next.js
  │
  ├── [1-hop] DEPENDS_ON ──────▶ react
  │                              styled-components
  │                              webpack
  │
  ├── [1-hop] STARRED_BY ──────▶ user_123
  │                              user_456
  │                              ├── STARRED_BY ──▶ remix      (2-hop)
  │                              └── STARRED_BY ──▶ vite       (2-hop)
  │
  └── [1-hop] CONTRIBUTED_TO ──▶ user_789
                                  ├── CONTRIBUTED_TO ──▶ turbo (2-hop)
                                  └── CONTRIBUTED_TO ──▶ swc   (2-hop)
```

### How to make graph traversal fast

**1. Create proper graph indexes in AGE.**

AGE translates Cypher queries to SQL under the hood. Without indexes, every traversal becomes a full table scan.

```sql
-- Index nodes by ID
CREATE INDEX ON ag_label('Repo') USING btree (id);

-- Index edges by source + type (for fast neighbor expansion)
CREATE INDEX ON ag_label('DEPENDS_ON')
  USING btree (start_id, end_id);

-- Composite index for weighted traversal
CREATE INDEX ON ag_label('STARRED_BY')
  USING btree (start_id, properties->>'starred_at' DESC);
```

**2. Limit traversal depth.**

A graph with millions of nodes explodes combinatorially if you walk too far.

```
1-hop from next.js:    ~50 neighbors
2-hop from next.js:    ~50 x 50 = 2,500 neighbors
3-hop from next.js:    ~125,000 neighbors (too many, too slow)
```

**The rule:** Walk 2 hops max. If 2-hop neighbors are too few, you have a sparse graph — walk 3 hops on specific high-weight edges only.

```cypher
MATCH (r:Repo {full_name: 'vercel/next.js'})
MATCH (r)-[e:DEPENDS_ON*1..2]->(neighbor:Repo)
WHERE (ALL(e IN e WHERE e.weight > 0.5))
RETURN neighbor.full_name
LIMIT 200;
```

**3. Precompute 2-hop neighbors offline.**

Walking the graph at query time is fast for 1-hop. For 2-hop, precompute and store.

```sql
-- Run offline, store in a regular table for online lookup
CREATE TABLE two_hop_neighbors (
    source_repo_id BIGINT NOT NULL,
    neighbor_repo_id BIGINT NOT NULL,
    path_type TEXT,        -- 'dep', 'star', 'contributor'
    hop_count INT,
    combined_weight FLOAT,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source_repo_id, neighbor_repo_id, path_type)
);

CREATE INDEX ON two_hop_neighbors (source_repo_id, combined_weight DESC);

-- Online query: simple lookup, no traversal
SELECT neighbor_repo_id, combined_weight
FROM two_hop_neighbors
WHERE source_repo_id = $1
ORDER BY combined_weight DESC
LIMIT 200;
```

Now the online serving path never runs Cypher traversal. It reads a precomputed table in <5ms.

---

## The Hot-Cold Data Gradient

Not all data is equally important at request time. You split data across tiers based on access frequency.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │ TIER 1: REDIS (in-memory, <2ms)                                 │
 │                                                                 │
 │  ├── user_blend_states     per-user weights (fetched every req) │
 │  ├── feature_vectors       precomputed repo towers (read-heavy) │
 │  ├── popular_repo_metadata stars, desc, lang (top 10K repos)    │
 │  └── rate_limit_counters   per-user, per-IP                     │
 │                                                                 │
 │  TTL: blend states = 24h, features = 6h, metadata = 1h         │
 │  Eviction: LRU (least recently used)                            │
 └─────────────────────────────────────────────────────────────────┘
                              │
                              │ cache miss? fall through
                              ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ TIER 2: POSTGRES WITH INDEXES (disk + memory, 5-50ms)          │
 │                                                                 │
 │  ├── pgvector HNSW index    README embeddings for ANN          │
 │  ├── materialized views     co-star counts, co-contributor      │
 │  ├── two_hop_neighbors      precomputed graph walks            │
 │  └── repos (recent)         active repos (<2yr inactive)        │
 │                                                                 │
 │  Indexed, partitioned by time, refreshed hourly                 │
 └─────────────────────────────────────────────────────────────────┘
                              │
                              │ cold data? fall through
                              ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ TIER 3: POSTGRES RAW + ARCHIVE (disk, 50-200ms)                │
 │                                                                 │
 │  ├── old star events        2018-2024 (partitioned, compressed) │
 │  ├── archived repos         deleted, transferred, renamed       │
 │  ├── old contributor data   historical only                     │
 │  └── raw ingest logs        debugging data                      │
 │                                                                 │
 │  Rarely queried. Kept for backtesting and offline training.     │
 └─────────────────────────────────────────────────────────────────┘
```

### The caching strategy in detail

**What gets cached in Redis:**

```python
# Request flow with Redis caching

async def recommend(repo: str, user_id: str | None) -> Response:
    cache_key = f"rec:{repo}:{user_id or 'anon'}"

    # 1. Try cache first
    cached = await redis.get(cache_key)
    if cached:
        return deserialize(cached)  # <2ms

    # 2. Cache miss — compute
    result = await engine.recommend(repo, user_id)

    # 3. Store in cache
    await redis.setex(cache_key, ttl=300, value=serialize(result))
    return result
```

**What gets TTL (time-to-live) and why:**

| Data | TTL | Why |
|---|---|---|
| Recommendation results | 5 min | User might re-request. Browsing session is ~30s per repo page |
| Repo feature vectors | 6 hours | Model staleness target. Re-embedded on README change |
| User blend state | 24 hours | User accumulates a few interactions per session |
| Popular repo metadata | 1 hour | Stars change daily, descriptions rarely |
| Rate limit counters | Sliding window | Per-minute counts, not TTL-based |

**Cache invalidation — the hard part:**

```
Common mistake: cache blames the algorithm when it is a staleness problem

Strategy: Write-through for critical data
  Update DB → update cache → never serve stale

Strategy: TTL + lazy refresh for non-critical data
  Serve cache entry → if TTL is close, queue a background refresh → serve fresh next time

Strategy: For recommendations: invalidate by repo event
  New star on a repo → invalidate all cached recommendations involving that repo
  New README → re-embed → invalidate ANN similarity cache
```

---

## How the Three Components Work Together (One Request Trace)

```
User requests recommendations for "vercel/next.js"

 ┌─────────────────────────────────────────────────────────────────┐
 │ STEP 1: Redis check (2ms)                                      │
 │                                                                 │
 │  redis.get("blend:user_123") → user blend weights              │
 │  redis.get("repo:next.js:features") → precomputed features     │
 └────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ STEP 2: pgvector ANN (20ms)                                    │
 │                                                                 │
 │  SELECT repo_id FROM readme_texts                              │
 │  ORDER BY embedding <=> query_embedding                        │
 │  LIMIT 200;                                                     │
 │                                                                 │
 │  → 200 semantically similar repos                              │
 └────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ STEP 3: AGE graph traversal or precomputed table (10ms)        │
 │                                                                 │
 │  SELECT neighbor_repo_id, combined_weight                      │
 │  FROM two_hop_neighbors                                         │
 │  WHERE source_repo_id = $1                                     │
 │  ORDER BY combined_weight DESC                                  │
 │  LIMIT 200;                                                     │
 │                                                                 │
 │  → 200 graph-proximate repos (from precomputed table)          │
 └────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ STEP 4: Postgres repo metadata (5ms)                           │
 │                                                                 │
 │  SELECT full_name, description, stars, language, topics        │
 │  FROM repos                                                     │
 │  WHERE id IN (candidate_ids)                                    │
 │  AND archived = false                                           │
 │  ORDER BY stars DESC;                                           │
 │                                                                 │
 │  → Enriched repo objects with metadata                         │
 └────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ STEP 5: Back to Redis (2ms)                                    │
 │                                                                 │
 │  redis.setex("rec:next.js:user_123", 300, serialized_result)   │
 │                                                                 │
 │  → Cache the result for the next request                       │
 └─────────────────────────────────────────────────────────────────┘

Total: ~40ms (within the 150ms budget, leaves room for scoring + reranking)
```

---

## Operational Checklist

### What to monitor

| Metric | Alert threshold | Why |
|---|---|---|
| Postgres query latency p99 | >100ms | Indicates missing index or table bloat |
| pgvector ANN latency p99 | >30ms | HNSW index might need rebuilding |
| AGE traversal latency p99 | >50ms | Too many hops or missing edge index |
| Redis hit rate | <80% | Too many cache misses, TTL too short |
| Redis memory usage | >80% of max | Need bigger instance or shorter TTLs |
| Postgres connection pool wait | >10ms | Too many concurrent requests |
| Materialized view staleness | >2x refresh interval | Refresh job is failing |
| Disk usage growth | >10% week-over-week | Partition old data, archive |

### Maintenance tasks

```
Daily:
  - VACUUM ANALYZE on hot tables (repos, star_events_recent)
  - REFRESH MATERIALIZED VIEW co_star_counts

Weekly:
  - REINDEX CONCURRENTLY on fragmented indexes
  - Check HNSW index health (pgvector has a check function)
  - Check AGE edge consistency (no dangling edges)

Monthly:
  - Partition new month for time-partitioned event tables
  - PCA recompute for embedding dimension reduction
  - Review slow query logs, add missing indexes
```

---

## When to Upgrade from This Stack

| Signal | What to do |
|---|---|
| pgvector ANN >50ms sustained | Move vectors to Qdrant or Milvus |
| AGE traversal >100ms sustained | Move graph to Neo4j (managed) or NebulaGraph |
| Redis memory >16GB | Add Redis Cluster (sharding) |
| Postgres >500 concurrent connections | Add PgBouncer in front |
| >10M repos in catalog | Partition pgvector by ecosystem. Consider separate serving DB |
| Global users, latency >150ms | Multi-region read replicas + geo-routing |

---

## Summary

| Layer | Tool | What it stores | How to make it fast |
|---|---|---|---|
| **Cache** | Redis | Blend states, feature vectors, popular metadata, response cache | In-memory, TTL, LRU eviction |
| **Vectors** | pgvector | README embeddings indexed with HNSW | Dimension reduction, partition by language |
| **Graph** | Apache AGE | Traversal edges, precomputed 2-hop neighbors | Precompute walks, limit depth, index edges |
| **Relational** | Postgres | Repo metadata, event facts, aggregated views | Partial indexes, partitioning, materialized views |

One database. Three capabilities. Tiered by access frequency. Optimized per data shape.
