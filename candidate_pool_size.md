# Candidate Pool Sizing Guide

Practical reference for tuning the candidate pool as the repo corpus grows.
Updated July 2026.

---

## Quick overview

The system stores **51K+ repos** in PostgreSQL with pgvector. For each
recommendation request, it fetches a shortlist of candidates from the DB,
scores them, and picks the best. The pool size controls how many candidates
are fetched.

**Three parallel queries make up the pool:**

| Pool | What it does | Controlled by | Index used |
|---|---|---|---|
| SQL filter | Same language OR topic overlap | `pool_size` (default: 250) | btree + GIN |
| Vector (README-vs-desc) | Source README vs DB descriptions | `vector_k` (default: 100) | HNSW on `embedding` |
| Vector (desc-vs-desc) | Source description vs DB descriptions | `vector_k` (default: 100) | **_no HNSW index yet_** |

The two vector pools overlap ~60% (same DB column, different source vector).
After dedup and SQL merge, the pool is ~300–400 candidates. After owner
dedup in categorization, ~150–200 unique owners remain.

---

## How pool size relates to corpus size

As you embed more repos, the HNSW index (or sequential scan) has to search
a larger space. A pool that was enough at 10K repos becomes too narrow at 100K.

**Rule of thumb: `vector_k` should cover ~0.4% of the corpus.**

| Corpus size | `vector_k` | `pool_size` | Total raw candidates | Unique owners (est.) |
|---|---|---|---|---|
| 5,000 | 100 | 200 | ~300 | ~120 |
| 10,000 | 120 | 250 | ~370 | ~150 |
| 25,000 | 150 | 300 | ~450 | ~180 |
| 50,000 (current) | **200** | **400** | ~600 | ~240 |
| 75,000 | 220 | 450 | ~660 | ~260 |
| 100,000 | 250 | 500 | ~750 | ~300 |
| 150,000 | 300 | 550 | ~900 | ~360 |
| 200,000 | 350 | 600 | ~1,050 | ~420 |

The relationship follows a square-root curve: doubling the corpus warrants
a ~40% increase in `vector_k`.

---

## Where to change the values

**File:** `packages/mvp/src/reporelay_mvp/candidates.py`  
**Lines:** the `generate_candidates()` function signature (~line 36)

```python
async def generate_candidates(
    session: AsyncSession,
    source: Repo,
    *,
    pool_size: int = 400,   # ← SQL filter limit
    vector_k: int = 200,    # ← per-vector-query limit
    seed: int | None = None,
    tags: list[str] | None = None,
) -> list[tuple[Repo, float]]:
```

Also bump this threshold in `packages/mvp/src/reporelay_mvp/recommend.py`
(~line 98) to match:

```python
_MIN_DB_POOL_FOR_SKIP = 300  # skip GitHub search if DB pool ≥ this
```

Set it to roughly `pool_size × 0.75`.

---

## Performance reference

Measured on the scoring step (`score_many`), which accounts for >90% of
post-retrieval CPU time:

| `vector_k` | Score time | RAM (Python objects) |
|---|---|---|
| 100 | ~10ms | ~4 MB |
| 200 | ~20ms | ~8 MB |
| 300 | ~30ms | ~12 MB |
| 400 | ~40ms | ~16 MB |
| 500 | ~50ms | ~20 MB |

The DB queries themselves are fast — HNSW is sub-millisecond, and the SQL
filter (btree + GIN) is also sub-ms.

---

## Prerequisite: HNSW index on `description_embedding`

**The description_embedding column currently has NO HNSW index.** Every
vector query against it does a full sequential scan of all vectors (~100 MB
read per query for 51K repos). This is the #1 bottleneck.

**Before increasing pool sizes,** create this index:

```sql
CREATE INDEX IF NOT EXISTS ix_mvp_repos_desc_embedding_hnsw
ON mvp_repos
USING hnsw (description_embedding vector_cosine_ops)
WITH (m = 24, ef_construction = 300);
```

| Parameter | Value | Why |
|---|---|---|
| `m` | 24 | More connections per graph layer — better recall at 150K scale |
| `ef_construction` | 300 | Higher build-time search effort — better graph quality |

**Build time:**
- 50K repos: ~1–2 minutes
- 150K repos: ~3–5 minutes
- 300K repos: ~8–12 minutes

**Memory:**
- 50K repos: ~10–15 MB
- 150K repos: ~40–60 MB

Once this index exists, the `SET LOCAL hnsw.ef_search` hint becomes
effective. Add it inside `_fetch_vector_neighbors()` in
`packages/mvp/src/reporelay_mvp/data.py` (~line 620):

```python
_ef_search = max(limit * 2, 100)
await session.execute(
    text("SET LOCAL hnsw.ef_search = :ef"),
    {"ef": _ef_search},
)
```

This ensures the HNSW algorithm probes enough graph nodes to find the true
nearest neighbors at your chosen `vector_k`. Without it, pgvector defaults to
`ef_search = max(limit, 40)`, which causes recall degradation above ~100
neighbors on large corpora.

---

## Step-by-step: increase pool size

When you've embedded more repos and want to scale the pool:

### Step 1: Check current corpus size

```bash
just mvp count
```

### Step 2: Create the HNSW index (first time only)

Run the SQL above, or create an Alembic migration (recommended for repeatable deploys):

```python
# packages/mvp/src/reporelay_mvp/migrations/versions/mvp_008_xxx.py

"""add HNSW index on description_embedding"""

revision = "mvp_008"
down_revision = "mvp_007"

from alembic import op

def upgrade() -> None:
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_mvp_repos_desc_embedding_hnsw
        ON mvp_repos
        USING hnsw (description_embedding vector_cosine_ops)
        WITH (m = 24, ef_construction = 300)
    """)

def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_mvp_repos_desc_embedding_hnsw")
```

Then run:

```bash
alembic -c packages/mvp/alembic.ini upgrade head
```

### Step 3: Update `vector_k` and `pool_size`

Use the table above to find the right values for your corpus size, then update
`candidates.py` and `recommend.py`.

### Step 4: Deploy and monitor

Watch the `score_many` latency logs. If they exceed your budget, cap the
values. The sweet spot is where per-category caps in `categorize_results()`
are comfortably filled without scraping low-quality candidates.

---

## Signs you need a larger pool

- **Repetitive recommendations** — same few repos show up regardless of source
- **Empty categories** — some category groups (Same Stack, Hidden Gems) are
  frequently empty
- **Owner dedup drains the pool** — if `microsoft/` or `google/` owns 30% of
  top vector results, you need more raw candidates to find unique owners
- **Corpus has grown** — check `just mvp count` and compare against the table

## Signs you need a smaller pool

- **Latency is too high** — each 100 extra vector candidates costs ~10ms
- **Low-quality candidates** — the last candidates in the pool have cosine
  similarity below 0.5 (you're hitting noise)
- **SQL pool reaches noise** — the 400th result by stars has <10 stars
  (irrelevant repos)

---

## Quick reference card

```
Corpus     vector_k   pool_size   _MIN_DB_POOL_FOR_SKIP
──────     ────────   ─────────   ──────────────────────
  5,000        100         200              150
 10,000        120         250              190
 25,000        150         300              225
 50,000        200         400              300
 75,000        220         450              340
100,000        250         500              375
150,000        300         550              410
200,000        350         600              450
```
