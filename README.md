# RepoRelay

<img width="1944" height="624" alt="image" src="https://github.com/user-attachments/assets/a63c73fb-8453-4eb6-be4b-963f52296d65" />


[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3.12/)
[![Neon](https://img.shields.io/badge/DB-Neon-00E599?logo=neon&logoColor=white)](https://neon.tech)
[![pgvector](https://img.shields.io/badge/vector-pgvector-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Vercel](https://img.shields.io/badge/frontend-Vercel-black?logo=vercel)](https://reporelay-site.vercel.app)
[![Render](https://img.shields.io/badge/API-Render-46E3B7?logo=render)](https://reporelay-mvp-api-0w1k.onrender.com)
[![Embeddings](https://img.shields.io/badge/embeddings-Gemini-4285F4?logo=google)](https://ai.google.dev/gemini-api/docs/embeddings)

A GitHub repo recommender. Give it a repo, get back similar repos.

No ML training. No user data. Postgres + pgvector + Gemini embeddings (512-dim).




# visit the site 

<img width="2858" height="1800" alt="image" src="https://github.com/user-attachments/assets/7df50eac-2147-44f4-a584-beea8938173c" />



```
fastapi/fastapi  →  pallets/flask, django/django, psf/requests, encode/starlette ...
```

Live: **[reporelay-site.vercel.app](https://repo-relay-olv2-71av5mdje-rams-projects-3ee6f183.vercel.app/)** (frontend on Vercel, API on Render).

---

## What's here

- **5-stage hybrid recommender** — SQL filter + pgvector ANN + full-text search + keyword overlap → 12 weighted features → score → rerank for diversity
- **Gemini embeddings (512-dim)** — description embeddings as primary signal, with README embeddings for source repos
- **Keyword-based semantic search** — domain keywords extracted from descriptions + READMEs, matched via GIN ARRAY overlap and tsvector full-text search. Covers cold repos with no embeddings instantly
- **Proxy-first architecture** — new repos get instant results via keyword + full-text search; Gemini embedding runs in background for next visit
- **Relevance feedback** — "Show more like these" picks repos similar to ones you've liked, merges results across multiple sources
- **Seed-based variation** — deterministic jittering of scoring weights; same seed = same results
- **Purpose extraction** — extracts project purpose from READMEs when description is sparse
- **Topic inference** — 260+ topic vocabulary inferred from README text for repos with sparse GitHub topics
- **Dual GitHub token auto-rotation** — transparent switch on rate limit, 10,000 req/hr combined
- **Keyword extraction** — 20% of corpus (10K+ repos) has domain keywords extracted, powering keyword_match scoring feature

- **Trending signal** — scrapes [github.com/trending](https://github.com/trending) daily to surface viral repos
<img width="2188" height="1526" alt="image" src="https://github.com/user-attachments/assets/8318ec07-d66b-4da0-a7b6-25fc6fc6fb0c" />



- **Web UI** — Astro 5 + vanilla JS, dark/light theme, glass cards over a generative commit-graph art backdrop
- **12 grid-art patterns** — aurora, ocean, stars, peaks, gems, cracks, ripples, forest, matrix, comb, bloom, contour — pickable from the nav, persists per user

  
- **explore topics**
  <img width="2180" height="1686" alt="image" src="https://github.com/user-attachments/assets/e3d66a67-9306-49d7-8da9-f8a177278498" />
  explore over 70+ topics

  <img width="2312" height="962" alt="image" src="https://github.com/user-attachments/assets/eae8cbc9-8019-4286-a55c-37d155e5b630" />
  
  <img width="1950" height="404" alt="image" src="https://github.com/user-attachments/assets/e9c5af66-574b-4b48-a469-9de8d2708eb5" />
  
**[→ Architecture & data flow](ARCHITECTURE.md)** · **[→ Deploy notes](DEPLOY.md)** · **[→ CLI reference](cli.md)** · **[→ Indexing guide](seed.md)** · **[→ Improvements](improvements.md)**

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/ramsterr/RepoRELAY.git && cd RepoRELAY
just sync

# 2. Set up .env
echo 'DATABASE_URL=postgresql+psycopg://reporelay:reporelay@localhost:5439/reporelay' > .env
echo 'GITHUB_TOKEN=ghp_your_token' >> .env
echo 'EMBEDDING_API=local' >> .env

# 3. Start Postgres & run migrations
just up && just migrate

# 4. Save some repos
just mvp save fastapi/fastapi
just mvp save django/django
just mvp save pallets/flask

# 5. Get recommendations
just mvp recommend fastapi/fastapi
```

**Prerequisites:** Docker, Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js 20+, pnpm

For production (Render free tier, no local model): set `EMBEDDING_API=gemini` and `GEMINI_API_KEY`.

---

## Run it

```bash
just api          # API server on :8001
just site         # Web UI on :4321  (Astro dev — first request compiles, then fast)
just site-fast    # Built site on :4321  (prebuilt via Node adapter, always fast)
just dev          # Both at once
```

---

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /popular?limit=N&topic=X` | Top repos by stars, optionally filtered by topic |
| `GET /topics?limit=N` | Top topics in the database |
| `GET /recommend?repo=owner/name&limit=N&seed=N&tags=a,b` | Ranked recommendations for a repo |
| `GET /explore?seed=N&limit=N` | Random repo + its recommendations |
| `POST /recommend/more` | Show more repos like selected candidates (relevance feedback) |

**Params:**
- `limit` — result count (default 10, max 50)
- `seed` — integer, gives deterministic variation in the ranking (same seed = same order)
- `tags` — comma-separated list; embeds the tag text and matches semantically against candidate descriptions

**Response shape** (`/recommend`):
```json
{
  "groups": [
    {
      "name": "description_cosine_sim",
      "label": "Description similarity",
      "icon": "📝",
      "repos": [
        {
          "full_name": "pallets/flask",
          "description": "The Python micro framework for building web applications.",
          "language": "Python",
          "stars": 67000,
          "score": 0.84,
          "features": {
            "language_match": 1.0,
            "topic_overlap": 0.6,
            "description_cosine_sim": 0.78,
            "dep_overlap": 0.3,
            "star_ratio": 0.95,
            "trending_boost": 0.1
          }
        }
      ]
    }
  ],
  "flat_repos": ["same repos flattened for convenience"]
}
```

---

## CLI

```bash
just mvp save owner/name           # fetch + embed + store a repo
just mvp recommend owner/name      # get recommendations (CLI)
just mvp count                     # how many repos in the DB
just mvp seed --per-language 1000 --languages python,rust  # bulk-index by language
just mvp seed-topics --per-topic 200              # bulk-index by topic (260+ topics)
just mvp embed --limit 1000                      # embed repos missing vectors
just mvp extract-keywords --limit 5000            # extract domain keywords from descriptions
just mvp trending --since daily                  # scrape github.com/trending
just mvp infer-topics                            # backfill inferred topics from READMEs
just mvp register-webhooks                       # register GitHub push webhooks on all repos
just mvp explore                                 # random repo + its recs
```

---

## How it works

```
GitHub API ──► Postgres + pgvector + tsvector ──► Candidates ──► Features ──► Score ──► Rerank ──► Results
  (fetch)      (store + ANN + full-text)           (150-250)    (12 floats) (1 float)  (diverse)  (top N, grouped)
```

1. **Fetch** — repo metadata + README from GitHub API (dual token auto-rotation)
2. **Embed** — description → 512-dim vector via Gemini `gemini-embedding-001` (or local `BAAI/bge-small-en-v1.5` with MRL truncation)
3. **Candidates** — SQL filter (language/topics) ∪ pgvector ANN ∪ full-text search ∪ keyword overlap, deduplicated
4. **Features** — 12 signals: description cosine sim, README vs desc cosine sim, README topic overlap, keyword match, keyword-topic match, topic overlap, language match, language diversity, dep overlap, star ratio, quality signal, trending boost
5. **Score** — weighted sum with adaptive redistribution (hand-tuned, no training)
6. **Rerank** — drop same-owner, enforce one-per-owner diversity, group by dominant signal

[→ Detailed architecture](ARCHITECTURE.md)

---

## Tech

| Layer | Stack |
|---|---|
| Embedding | Gemini `gemini-embedding-001` (512 dims via MRL, production) / `BAAI/bge-small-en-v1.5` (384 dims, local dev) |
| Database | Postgres 16 + pgvector (HNSW index) + tsvector (full-text) + GIN (keyword arrays) |
| API | FastAPI (async) |
| Frontend | Astro 5 (static, Vercel) |
| Packages | `uv` (Python) · `pnpm` (Node) |
| Deploy | Vercel (frontend) + Render (API) |
| Cron | GitHub Actions (embed, seed, trending — every 30 min) |

---

## Repo layout

```
.
├── apps/
│   ├── site/                     # Astro frontend (Vercel)
│   │   └── src/
│   │       ├── layouts/Base.astro    # grid art, theme toggle, art picker
│   │       ├── pages/               # index, explore, repo
│   │       └── styles/global.css
│   └── mvp_api/                  # FastAPI service (Render)
│       └── src/reporelay_mvp_api/
│           ├── main.py               # endpoints + show-more + cache
│           └── webhooks.py           # GitHub push handler
├── packages/
│   └── mvp/                      # core library (shared between CLI + API)
│       └── src/reporelay_mvp/
│           ├── recommend.py          # 5-stage pipeline + proxy cold-repo flow
│           ├── candidates.py         # SQL filter + pgvector ANN
│           ├── features.py           # 12 signal extractors + IDF-weighted topics
│           ├── score.py              # adaptive weighted sum (77% semantic)
│           ├── rerank.py             # diversity rules
│           ├── embedding.py          # Gemini + local + none (260 lines)
│           ├── feedback.py           # show-more: merge + more_like_these
│           ├── keyword_extractor.py  # domain keyword extraction from desc+README
│           ├── purpose.py            # purpose extraction from READMEs
│           ├── topic_inference.py    # 260+ topic vocabulary inference
│           ├── seed_topics.py        # uniform + weighted bulk indexing
│           ├── github.py             # dual token + httpx client
│           ├── trending.py           # github.com/trending scraper
│           ├── seed.py               # bulk indexer by language
│           ├── embed_pass.py         # backfill missing embeddings
│           └── cli.py                # typer CLI
├── .github/workflows/
│   ├── embed.yml                # cron: embed 3000 repos every 30 min
│   ├── seed.yml                 # cron: seed topics every 6 hours
│   ├── trending.yml             # cron: scrape trending daily
│   └── ping.yml                 # cron: keep Render container warm
├── infra/
│   └── docker-compose.yml       # local Postgres + pgvector
├── Dockerfile.api               # Render API image
├── render.yaml                  # Render service config
├── pyproject.toml               # uv workspace
└── justfile                     # task runner
```

# matrix
<img width="2858" height="1800" alt="image" src="https://github.com/user-attachments/assets/3d4ebccb-db42-4f18-b1d2-6363ab25338a" />


# coral

<img width="2848" height="1800" alt="image" src="https://github.com/user-attachments/assets/b49d0abe-07de-471b-9289-f77b7526e944" />

# desert

<img width="2854" height="1800" alt="image" src="https://github.com/user-attachments/assets/b03e9f51-6482-4c74-a6d2-3ebadd9b13b7" />

# waves 

<img width="2856" height="1796" alt="image" src="https://github.com/user-attachments/assets/2a0daf97-615a-46a6-855c-4aef97d0b782" />

# gems

<img width="2854" height="1800" alt="image" src="https://github.com/user-attachments/assets/10b8ada5-51dd-4947-b9a1-c879df2de976" />

# sakura 

<img width="2854" height="1796" alt="image" src="https://github.com/user-attachments/assets/cf481a28-6c04-4cf7-8459-3cf4a6b6fa23" />



---

## License

MIT
