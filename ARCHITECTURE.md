# Architecture

There's no training, no ML pipeline, no model that learns. The system has three pieces that already know how to do their job:

- **GitHub API** — tells you what a repo *is* (language, topics, stars, README)
- **BAAI/bge-small-en-v1.5** — a pre-trained sentence transformer that turns any text into 384 numbers (an embedding). It already knows that "Python web framework" and "async Python API" are similar concepts.
- **Postgres + pgvector** — stores the data and finds repos with nearby embeddings

You never train anything. You fetch data, embed text, run queries. That's it.

---

## System diagram

```
                    ┌──────────────┐
                    │  GitHub API  │
                    └──────┬───────┘
                           │  fetch repo metadata + README
                           ▼
┌──────────────────────────────────────────────────────┐
│                  Ingestion (just mvp save)            │
│                                                      │
│  github.py        httpx → GitHub API                 │
│  embedding.py     sentence-transformers → 384 floats │
│  data.py          INSERT INTO mvp_repos              │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
             ┌────────────────┐
             │   Postgres     │
             │  mvp_repos     │
             │  + pgvector    │
             └───────┬────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
     ▼               ▼               ▼
┌─────────┐   ┌───────────┐   ┌──────────┐
│ SQL     │   │ pgvector  │   │ raw row  │
│ filter  │   │ ANN       │   │ lookup   │
│ (lang,  │   │ (cosine   │   │ (by name)│
│ topics) │   │ distance) │   │          │
└────┬────┘   └─────┬─────┘   └────┬─────┘
     │              │              │
     └──────┬───────┘              │
            ▼                      │
     ┌──────────────┐              │
     │  candidates  │◄─────────────┘
     │  (150-250)   │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │   features   │  language match, topic overlap,
     │   (5 floats) │  cosine sim, dep overlap, stars sim
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │    score     │  weighted sum → single number
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │   rerank     │  drop same-owner, one-per-owner
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  top-N list  │  ← returned to user
     └──────────────┘
```

---

## The mvp_repos table

One table. No joins to other tables, no foreign keys, no materialized views.

```
mvp_repos
├── id            BIGINT        GitHub repo ID
├── owner         VARCHAR       e.g. "fastapi"
├── name          VARCHAR       e.g. "fastapi"
├── full_name     VARCHAR       "fastapi/fastapi"
├── description   TEXT          from GitHub
├── language      VARCHAR       "Python", "TypeScript", etc.
├── topics        TEXT[]        ["python", "json", "swagger-ui"]
├── stars         INTEGER       stargazer count
├── dependencies  TEXT[]        package names from dependency graph
├── embedding     VECTOR(384)   384 floats from BAAI/bge-small-en-v1.5
├── trending_score FLOAT        velocity signal from github.com/trending (0..1)
├── created_at    TIMESTAMPTZ
├── updated_at    TIMESTAMPTZ
├── embedded_at   TIMESTAMPTZ   when embedding was computed
├── search_fetched_at TIMESTAMPTZ  when repo was indexed via GitHub Search
└── trending_fetched_at TIMESTAMPTZ  when trending signal was last updated

Indexes:
  ix_mvp_repos_language           btree on language
  ix_mvp_repos_topics             GIN on topics (for && overlap queries)
  ix_mvp_repos_stars              btree on stars DESC
  ix_mvp_repos_full_name          btree on full_name (unique)
  ix_mvp_repos_embedding_hnsw     HNSW on embedding (for ANN search)
```

---

## The 5 stages in detail

### Stage 1 — Data

`data.py` reads from and writes to `mvp_repos`. Four queries matter:

| Function | SQL | Purpose |
|---|---|---|
| `get_repo` | `SELECT ... WHERE full_name = ?` | Look up the source repo |
| `fetch_filtered_pool` | `SELECT ... WHERE language = ? OR topics && ?` | "Give me repos in the same ecosystem" |
| `fetch_vector_neighbors` | `CROSS JOIN ... ORDER BY embedding <=> src.embedding` | "Give me repos with similar READMEs" |
| `upsert_repo` | `INSERT ... ON CONFLICT DO UPDATE` | Save a repo (used during ingestion) |

---

### Stage 2 — Features

`features.py` computes up to 7 numbers for each (source, candidate) pair. All values are in [0, 1].

```
language_match:
  1.0 if source.language == candidate.language
  0.0 otherwise

topic_overlap:
  Jaccard(set(source.topics), set(candidate.topics))
  = |intersection| / |union|
  Captures: "do these repos live in the same category?"

cosine_sim:
  1 - (source.embedding <=> candidate.embedding)
  The pgvector <=> operator computes cosine distance
  Captures: "do the READMEs talk about similar things?"

dep_overlap:
  Jaccard(set(source.dependencies), set(candidate.dependencies))
  Captures: "do they use the same libraries?"

popularity_sim:
  1 - |log1p(source.stars) - log1p(candidate.stars)| / log1p(500000)
  Captures: "are they at a similar scale?"
  (Log scale because 100 vs 200 stars matters more than 50k vs 51k)

trending_boost:
  Velocity signal scraped from github.com/trending.
  min(1.0, stars_period / 100) — catches repos with viral growth.

filter_cosine_sim (only when tags are provided):
  Semantic similarity between the user's tag text and each candidate's
  README embedding. Gives semantic tag matching — "machine learning"
  matches ML repos even without that exact tag.
```

---

### Stage 3 — Candidate Generation

`candidates.py` runs two queries and merges:

1. **SQL filter** — `WHERE language = ? OR topics && ?` — uses the btree and GIN indexes. Fast, returns repos in the same ecosystem regardless of README similarity. Max 250.

2. **pgvector ANN** — `ORDER BY embedding <=> source.embedding` — uses the HNSW index. Returns the most semantically similar repos by README content. Max 150.

The two pools are deduplicated by repo ID. Vector pool repos take priority (they carry accurate cosine similarity). SQL-only repos get a neutral cosine sim of 0.5.

Result: ~150-250 candidates, down from potentially thousands.

---

### Stage 4 — Scoring

`score.py` multiplies each feature by a weight and sums. There are two weight schemes:

**Default (no tag filter):**
```
score = 0.25 × language_match
      + 0.25 × topic_overlap
      + 0.20 × cosine_sim
      + 0.15 × dep_overlap
      + 0.05 × popularity_sim
      + 0.10 × trending_boost
```

**Tag-filtered (when user provides tags):**
```
score = 0.10 × language_match
      + 0.20 × topic_overlap
      + 0.10 × cosine_sim
      + 0.30 × filter_cosine_sim
      + 0.10 × dep_overlap
      + 0.10 × popularity_sim
      + 0.10 × trending_boost
```

The weights are hand-picked and live in dicts at the top of the file. No ML, no training. Change the numbers and the ranking changes immediately.

```python
WEIGHTS = {
    "language_match": 0.25,
    "topic_overlap": 0.25,
    "cosine_sim": 0.20,
    "dep_overlap": 0.15,
    "popularity_sim": 0.05,
    "trending_boost": 0.10,
}

TAG_WEIGHTS = {
    "language_match": 0.10,
    "topic_overlap": 0.20,
    "cosine_sim": 0.10,
    "filter_cosine_sim": 0.30,
    "dep_overlap": 0.10,
    "popularity_sim": 0.10,
    "trending_boost": 0.10,
}
```

When a `seed` is provided, each weight is jittered by +/-10% deterministically (same seed = same weights) and `popularity_sim` is boosted 3x to surface "cooler" repos.

---

### Stage 5 — Reranking

`rerank.py` applies three rules after scoring:

1. **Drop same-owner** — if source is `fastapi/fastapi`, don't recommend `fastapi/fastapi` (defensive) or `fastapi/typer` (same org, probably not useful)

2. **One per owner** — at most one repo per GitHub org/user in the final list. Prevents "here are 10 Facebook repos"

3. **Cap at limit** — return the top N that survive the filters

---

## The embedding model

The model is `BAAI/bge-small-en-v1.5` from HuggingFace. It was trained on large-scale retrieval datasets and produces 384-dimensional embeddings optimized for semantic similarity. It converts text into 384 numbers where semantically similar texts get nearby vectors.

The model is loaded once at API startup (takes ~11 seconds on a Mac) and cached in memory. Embedding computation runs in a thread pool via `asyncio.to_thread()` so it doesn't block the async event loop.

The README text fed to the model is truncated to the first 8000 characters. For most repos, the first 8000 chars of README capture the project's purpose, installation, and basic usage — enough for semantic similarity.

Cosine similarity is computed using numpy (vectorized matrix operations) for performance. When tag filtering is active, all candidate embeddings are compared against the filter embedding in a single batched matrix multiply.

**You never train, fine-tune, or update this model.** It's a fixed building block, like `import json`.

---

## What IS in this system

| Feature | How it works |
|---|---|
| Content-based recommendations | 6 features weighted and scored, no user data needed |
| Semantic tag filtering | Embed tag text, compare against candidate README embeddings |
| Trending signal | Scrapes github.com/trending for viral repos |
| GitHub webhooks | `POST /webhooks/github` clears embeddings on push for re-embedding |
| Seed-based variation | Same seed = same results, different seed = different ranking |
| Candidate growth | GitHub Search results are persisted back to the DB on every request |
| Web UI | Astro frontend with search, explore, tag filtering, reroll |
| CLI | Full CLI for save, recommend, seed, embed, trending |

## What's NOT in this system

| Not present | Why it was skipped |
|---|---|
| Training / ML pipeline | Weights are hand-tuned, model is pre-trained |
| User data / profiles | No personalization needed for content-based recs |
| Collaborative filtering | Needs real star/fork events from users |
| Redis / caching | GitHub search cached in-memory (5min TTL), queries are fast at current scale |
| Graph traversal (2-hop) | SQL filter + pgvector covers the same ground simpler |
| Feedback loop | Not needed to demonstrate the core loop |
| Multiple strategies / blending | One strategy with 6 features is enough |
| Co-star / workflow signals | These need data the MVP doesn't collect |

---

## The data you need to put in

The system only recommends repos you've ingested. If you have 3 repos in `mvp_repos`, it can only recommend among those 3 (minus the source). To get good recommendations:

1. **Ingest repos in related domains** — if you want Python web framework recs, ingest a bunch of Python web repos
2. **Aim for 20-50 repos** — that gives the candidate pool enough variety
3. **Ingest repos with READMEs** — repos without READMEs get a zero embedding and only match on structured features

```bash
# Example: build a Python web ecosystem
just mvp save fastapi/fastapi
just mvp save django/django
just mvp save pallets/flask
just mvp save encode/starlette
just mvp save psf/requests
just mvp save urllib3/urllib3
just mvp save aio-libs/aiohttp
just mvp save tiangolo/sqlmodel
just mvp save sqlalchemy/sqlalchemy
```
