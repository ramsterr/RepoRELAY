# Architecture

There's no training, no ML pipeline, no model that learns. The system has four pieces that already know how to do their job:

- **GitHub API** — tells you what a repo *is* (language, topics, stars, README)
- **Gemini `gemini-embedding-001`** — Google's embedding API that turns text into 512 numbers. Triaged via MRL (Matryoshka Representation Learning) to fit pgvector. Knows that "Python web framework" and "async Python API" are similar concepts.
- **Keyword extractor** — extracts domain-specific terms from descriptions + READMEs (e.g. "machine learning", "computer vision", "shader")
- **Postgres + pgvector + tsvector** — stores the data, finds repos with nearby embeddings, runs full-text search, and matches keyword arrays

You never train anything. You fetch data, embed text, extract keywords, run queries. That's it.

---

## System diagram

```
                    ┌──────────────┐
                    │  GitHub API  │
                    │ (dual token, │
                    │  10k req/hr) │
                    └──────┬───────┘
                           │  fetch repo metadata + README
                           ▼
┌──────────────────────────────────────────────────────┐
│                  Ingestion (just mvp save)            │
│                                                      │
│  github.py           httpx → GitHub API              │
│  embedding.py        Gemini API → 512 floats         │
│  keyword_extractor.py  description → domain keywords │
│  purpose.py          README → extracted purpose       │
│  data.py             INSERT/UPDATE mvp_repos         │
└─────────────────────┬────────────────────────────────┘
                      │
                      ▼
             ┌────────────────┐
             │   Postgres     │
             │  mvp_repos     │
             │  + pgvector    │
             │  + tsvector    │
             │  + GIN (topics,│
             │    keywords)   │
             └───────┬────────┘
                     │
     ┌───────────────┼───────────────────┐
     │               │                   │
     ▼               ▼                   ▼
┌─────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────┐
│ SQL     │   │ pgvector  │   │ full-text    │   │ keyword  │
│ filter  │   │ ANN       │   │ tsvector     │   │ GIN      │
│ (lang,  │   │ (cosine   │   │ (ts_rank)    │   │ overlap  │
│ topics) │   │ distance) │   │              │   │          │
└────┬────┘   └─────┬─────┘   └──────┬───────┘   └────┬─────┘
     │              │               │                 │
     └──────┬───────┴───────┬───────┴────────┬────────┘
            │               │                │
            ▼               ▼                ▼
     ┌──────────────┐ ┌───────────┐  ┌─────────────┐
     │  candidates  │ │ raw row   │  │  keyword-   │
     │  (150-250)   │ │ lookup    │  │  based proxy│
     │              │ │ (by name) │  │  (cold repos)│
     └──────┬───────┘ └─────┬─────┘  └──────┬──────┘
            │               │              │
            ▼               │              │
     ┌──────────────┐       │              │
     │   features   │◄──────┘──────────────┘
     │  (15 floats) │  language_match, topic_overlap,
     │              │  description_cosine_sim, readme_topic_sim,
     │              │  readme_vs_desc_cosine_sim, keyword_match,
     │              │  keyword_topic_match, dep_overlap,
     │              │  star_ratio, language_diversity,
     │              │  quality_signal, trending_boost,
     │              │  filter_cosine_sim, cosine_sim, description_sim
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │    score     │  adaptive weighted sum → single number
     │              │  (77% semantic, auto-redistributes missing signals)
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │   rerank     │  drop same-owner, one-per-owner, group by signal
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  grouped     │  ← CategorizedRecommendation with groups + flat list
     └──────────────┘
```

---

## The mvp_repos table

One table. No joins to other tables, no foreign keys, no materialized views.

```
mvp_repos
├── id                   BIGINT        GitHub repo ID
├── owner                VARCHAR       e.g. "fastapi"
├── name                 VARCHAR       e.g. "fastapi"
├── full_name            VARCHAR       "fastapi/fastapi"
├── description          TEXT          from GitHub
├── language             VARCHAR       "Python", "TypeScript", etc.
├── topics               TEXT[]        ["python", "json", "swagger-ui"]
├── stars                INTEGER       stargazer count
├── dependencies         TEXT[]        package names from dependency graph
├── embedding            VECTOR(512)   512 floats — README embedding (Gemini or BAAI/bge-small)
├── description_embedding VECTOR(512)  512 floats — description embedding (primary signal)
├── keywords             TEXT[]        domain keywords ["machine learning", "data science"]
├── search_vector        TSVECTOR      weighted tsvector (name + description + topics)
├── trending_score       FLOAT         velocity signal from github.com/trending (0..1)
├── created_at           TIMESTAMPTZ
├── updated_at           TIMESTAMPTZ
├── embedded_at          TIMESTAMPTZ   when embedding was computed
├── search_fetched_at    TIMESTAMPTZ   when repo was indexed via GitHub Search
└── trending_fetched_at  TIMESTAMPTZ   when trending signal was last updated

Indexes:
  ix_mvp_repos_language             btree on language
  ix_mvp_repos_topics               GIN on topics (for && overlap queries)
  ix_mvp_repos_keywords             GIN on keywords (for ARRAY overlap)
  ix_mvp_repos_search_vector        GIN on search_vector (for full-text ts_rank)
  ix_mvp_repos_stars                btree on stars DESC
  ix_mvp_repos_full_name            btree on full_name (unique)
  ix_mvp_repos_embedding_hnsw       HNSW on embedding (for ANN search)
  ix_mvp_repos_desc_embedding_hnsw  HNSW on description_embedding (for ANN search)
```

---

## The stages in detail

### Stage 1 — Data

`data.py` reads from and writes to `mvp_repos`. Key queries:

| Function | SQL | Purpose |
|---|---|---|
| `get_repo` | `SELECT ... WHERE full_name = ?` | Look up the source repo |
| `fetch_filtered_pool` | `SELECT ... WHERE language = ? OR topics && ?` | "Give me repos in the same ecosystem" |
| `fetch_vector_neighbors` | `CROSS JOIN ... ORDER BY description_embedding <=> src.desc_emb` | "Give me repos with similar descriptions" |
| `search_keywords` | `WHERE keywords && ARRAY[?]::text[]` | Keyword-based semantic search (GIN overlap) |
| `search_fulltext` | `WHERE search_vector @@ plainto_tsquery(?)` | Full-text search (tsvector) |
| `search_hybrid` | Combined keyword + full-text + topic overlap | Multi-strategy cold-repo search |
| `upsert_repo` | `INSERT ... ON CONFLICT DO UPDATE` | Save a repo (used during ingestion) |

---

### Stage 2 — Features

`features.py` computes up to 15 numbers for each (source, candidate) pair. All values are in [0, 1].

**Primary semantic signals (embedding-based):**

```
readme_vs_desc_cosine_sim (weight: 0.25):
  Cosine similarity between source README embedding and
  candidate description embedding. Captures: "does the candidate's
  description match what the source's README is about?"

description_cosine_sim (weight: 0.20):
  Cosine similarity between source and candidate description embeddings.
  Primary signal (94% of corpus has description embedding).
  Captures: "do these two projects describe themselves similarly?"

readme_topic_sim (weight: 0.12):
  Overlap between source README tokens and candidate topic tags.
  Uses substring matching: "neural" matches topic "neural-network".
  Captures: "do the candidate's curated topics reflect what the
  source README discusses?"

keyword_match (weight: 0.08):
  Jaccard similarity of extracted keyword sets.
  Captures domain overlap: both repos mention "data science",
  "machine learning" — same purpose, different GitHub topics.

keyword_topic_match (weight: 0.05):
  Overlap between extracted keywords and candidate GitHub topics.
  Bridges keyword extraction with topic matching.
```

**Secondary signals:**

```
topic_overlap (weight: 0.07):
  IDF-weighted Jaccard of topic sets. Rare topics ("compiler", "verilog")
  count more than common ones ("python", "javascript").

language_diversity (weight: 0.07):
  1.0 if source and candidate are in different languages, 0.0 if same.
  Bonus for cross-language discoveries (e.g. a Rust port of a Python library).

star_ratio (weight: 0.06):
  Exponential decay based on log10 star distance.
  Prevents a niche 200-star repo from recommending React (200k stars)
  as its top match.

dep_overlap (weight: 0.05):
  Jaccard similarity of dependency names.
  Captures: "do they use the same libraries?"

language_match (weight: 0.03):
  1.0 if same language, 0.0 otherwise.

quality_signal (weight: 0.02):
  Maintenance proxy: rewards repos with descriptions >60 chars, >2 topics,
  >5 deps, a language, and an embedding.

trending_boost:
  Velocity signal scraped from github.com/trending.
  min(1.0, stars_period / 100).

filter_cosine_sim (only when tags are provided, weight: 0.28):
  Semantic similarity between user's tag text embedding and each candidate's
  description embedding. Gives semantic tag matching — "machine learning"
  matches ML repos even without that exact tag.
```

### Stage 3 — Candidate Generation

`candidates.py` runs multiple queries and merges:

1. **SQL filter** — `WHERE language = ? OR topics && ?` — uses the btree and GIN indexes. Fast, returns repos in the same ecosystem regardless of README similarity. Max 250.

2. **pgvector ANN** — `ORDER BY description_embedding <=> source.desc_emb` — uses the HNSW index. Returns the most semantically similar repos by description content. Max 150.

3. **Keyword-based proxy** (cold repos) — When the source repo has keywords but no description embedding, keyword overlap + full-text search + topic overlap generate an instant proxy candidate pool. Gemini embedding runs in background and the embedded results are used on the next visit.

The pools are deduplicated by repo ID. Vector pool repos take priority (they carry accurate cosine similarity). SQL-only repos get a neutral cosine sim of 0.5.

Result: ~150-250 candidates, down from potentially thousands.

**Proxy-first architecture:** Newly saved repos get instant results. No need to wait for the embedding cron (up to 30 min). The proxy uses extracted keywords + full-text search as a temporary signal while Gemini computes the real embedding for next time.

---

### Stage 4 — Scoring

`score.py` uses adaptive weight redistribution. If the source repo is missing a signal (e.g. no README embedding), that feature's weight is redistributed to other features. All weights are in [0, 1] and normalize automatically.

**Primary weights (77% semantic):**
```
readme_vs_desc_cosine_sim = 0.25  ──┐
description_cosine_sim    = 0.20    │
readme_topic_sim          = 0.12    │  77% semantic
keyword_match             = 0.08    │  (description + README + keywords)
keyword_topic_match       = 0.05  ──┤
topic_overlap             = 0.07  ──┘
language_diversity        = 0.07
star_ratio                = 0.06
dep_overlap               = 0.05
language_match            = 0.03
quality_signal            = 0.02
```

**Tag-filtered (when user provides tags):**
```
filter_cosine_sim         = 0.28  ← tag matching takes priority
readme_vs_desc_cosine_sim = 0.16
description_cosine_sim    = 0.14
readme_topic_sim          = 0.08
keyword_match             = 0.06
...
```

When a `seed` is provided, each weight is jittered by +/-10% deterministically (same seed = same weights) and `star_ratio` is boosted 1.3x to surface repos at different popularity levels.

---

### Stage 5 — Reranking & Grouping

`rerank.py` applies diversity rules after scoring:

1. **Drop same-owner** — if source is `fastapi/fastapi`, don't recommend `fastapi/fastapi` (defensive) or `fastapi/typer` (same org, probably not useful)

2. **One per owner** — at most one repo per GitHub org/user in the final list. Prevents "here are 10 Facebook repos"

3. **Cap at limit** — return the top N that survive the filters

4. **Group by signal** — results are categorized into groups (Description Similarity, README Topic Matches, Language-Based, etc.) for the frontend's categorized display

---

## The embedding model

**Production (Render):** Gemini `gemini-embedding-001` via API. Outputs 768-dim vectors, truncated to 512 via MRL (Matryoshka Representation Learning). No local RAM cost — ideal for Render's free tier (512MB). Configured via `EMBEDDING_API=gemini` + `GEMINI_API_KEY`.

**Local dev:** `BAAI/bge-small-en-v1.5` from HuggingFace, loaded in-process. Produces 384-dim vectors natively. Configured via `EMBEDDING_API=local`.

**Description embedding is the primary signal.** It's the most purpose-dense text available and works for 94% of the corpus. README embeddings are used for the source repo only (the `readme_vs_desc_cosine_sim` feature). This means candidates only need a description embedding, not a README embedding — which is far cheaper to maintain.

**You never train, fine-tune, or update this model.** It's a fixed building block, like `import json`.

---

## Keyword extraction

`keyword_extractor.py` extracts domain-specific technical keywords from repo descriptions and READMEs. Extracted keywords are stored in a PostgreSQL TEXT[] column for fast GIN ARRAY overlap queries.

**Prioritizes:**
- Multi-word technical terms ("machine learning", "game development")
- Hyphenated compounds ("open-source", "deep-learning")
- CamelCase tokens ("OpenGL", "DataFrame")
- Domain-specific nouns (pixel, shader, chess, compiler)

**Filters out:**
- Common English stopwords
- Single-letter tokens
- Numbers
- Common generic terms ("library", "tool", "project")

Keywords are limited to 30 per repo. 20% of corpus (10K+ repos) has keywords extracted, powering the `keyword_match` and `keyword_topic_match` features.

---

## Purpose extraction

`purpose.py` extracts a project's purpose from its README when the description is too sparse (<50 chars). Falls back to the GitHub description if the README doesn't yield a clear purpose statement. Catches repos that have detailed READMEs but no GitHub description.

---

## Topic inference

`topic_inference.py` scans each repo's README and description against a 260+ topic vocabulary to suggest additional GitHub-style topic tags. Improves topic-based candidate generation and `topic_overlap` for repos without owner-assigned topics. Precision-first: avoids hallucinated topics on generic text.

---

## Relevance feedback (Show more)

`feedback.py` provides `more_like_these(source_emb, selected_repos)` — given a source embedding and a list of user-selected repos, it builds a combined embedding (weighted average) and searches for repos similar to the combined profile. Groups results by source repo with sub-categories.

`POST /recommend/more` accepts `{repo, selected_ids, topic_area}` and returns `{merged, picked}` — merged results first, then per-selected-repo breakdowns.

---

## What IS in this system

| Feature | How it works |
|---|---|
| Content-based recommendations | 12+ features weighted and scored, no user data needed |
| Hybrid search | SQL filter + pgvector ANN + full-text + keyword GIN |
| Semantic tag filtering | Embed tag text, compare against candidate description embeddings |
| Keyword-based proxy | Instant results for cold repos via keyword + full-text, geen embedding runs later |
| Relevance feedback | "Show more like these" uses weighted embedding of selected repos |
| Trending signal | Scrapes github.com/trending for viral repos |
| GitHub webhooks | `POST /webhooks/github` clears embeddings on push for re-embedding |
| Seed-based variation | Same seed = same results, different seed = different ranking |
| Dual GitHub tokens | Auto-rotation on rate limit, 10,000 req/hr combined |
| Candidate growth | GitHub Search results are persisted back to the DB on every request |
| Web UI | Astro frontend with search, explore, tag filtering, show-more, 12 art themes |
| CLI | Full CLI for save, recommend, seed, embed, extract-keywords, trending |

## What's NOT in this system

| Not present | Why it was skipped |
|---|---|
| Training / ML pipeline | Weights are hand-tuned, model is pre-trained |
| User data / profiles | No personalization needed for content-based recs |
| Collaborative filtering | Needs real star/fork events from users |
| Redis / caching | GitHub search cached in-memory (5min TTL), queries are fast at current scale |
| Graph traversal (2-hop) | SQL filter + pgvector + full-text covers the same ground simpler |
| Feedback loop | Not needed to demonstrate the core loop |
| Multiple strategies / blending | One strategy with 12+ features is enough |
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
