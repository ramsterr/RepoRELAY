# RepoRelay

<img width="1944" height="624" alt="image" src="https://github.com/user-attachments/assets/a63c73fb-8453-4eb6-be4b-963f52296d65" />




A GitHub repo recommender. Give it a repo, get back similar repos.

No ML training. No user data. Postgres + pgvector + a pre-trained sentence-transformer.


# visit the site 

<img width="2858" height="1800" alt="image" src="https://github.com/user-attachments/assets/7df50eac-2147-44f4-a584-beea8938173c" />



```
fastapi/fastapi  →  pallets/flask, django/django, psf/requests, encode/starlette ...
```

Live: **[reporelay-site.vercel.app](https://repo-relay-olv2-71av5mdje-rams-projects-3ee6f183.vercel.app/)** (frontend on Vercel, API on Render).

---


## What's here

- **5-stage content-based recommender** — SQL filter + pgvector ANN → 6 hand-tuned features → weighted score → rerank for diversity
- **Semantic tag filtering** — embed the user's tag text and match against README embeddings (no exact-tag match needed)


- **Trending signal** — scrapes [github.com/trending](https://github.com/trending) daily to surface viral repos
<img width="2188" height="1526" alt="image" src="https://github.com/user-attachments/assets/8318ec07-d66b-4da0-a7b6-25fc6fc6fb0c" />



- **Web UI** — Astro 5 + vanilla JS, dark/light theme, glass cards over a generative commit-graph art backdrop
- **12 grid-art patterns** — aurora, ocean, stars, peaks, gems, cracks, ripples, forest, matrix, comb, bloom, contour — pickable from the nav, persists per user


  
# explore topics
<img width="2180" height="1686" alt="image" src="https://github.com/user-attachments/assets/e3d66a67-9306-49d7-8da9-f8a177278498" />
explore over 70+ topics

# random picks
 <img width="2312" height="962" alt="image" src="https://github.com/user-attachments/assets/eae8cbc9-8019-4286-a55c-37d155e5b630" />
 
 <img width="1950" height="404" alt="image" src="https://github.com/user-attachments/assets/e9c5af66-574b-4b48-a469-9de8d2708eb5" />


  
**[→ Architecture & data flow](ARCHITECTURE.md)** · **[→ Deploy notes](DEPLOY.md)**

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/ramsterr/RepoRELAY.git && cd RepoRELAY
just sync

# 2. Set up .env
echo 'DATABASE_URL=postgresql+psycopg://reporelay:reporelay@localhost:5439/reporelay' > .env
echo 'GITHUB_TOKEN=ghp_your_token' >> .env

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

---

## Run it

```bash
just api          # API server on :8001
just site         # Web UI on :4321  (Astro dev — first request compiles, then fast)
just site-fast    # Built site on :4321  (prebuilt via Node adapter, always fast)
just dev          # Both at once
```

`just site` uses Astro's Vite dev server, which compiles each page on first request. `just site-fast` builds the site once with the Node adapter and serves the prebuilt output — use it for instant first-paint locally.

---

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /popular?limit=N&topic=X` | Top repos by stars, optionally filtered by topic |
| `GET /topics?limit=N` | Top topics in the database |
| `GET /recommend?repo=owner/name&limit=N&seed=N&tags=a,b` | Ranked recommendations for a repo |
| `GET /explore?seed=N&limit=N` | Random repo + its recommendations |

**Params:**
- `limit` — result count (default 10, max 50)
- `seed` — integer, gives deterministic variation in the ranking (same seed = same order)
- `tags` — comma-separated list; embeds the tag text and matches semantically against candidate READMEs

**Response shape** (`/recommend`):
```json
{
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
        "cosine_sim": 0.78,
        "dep_overlap": 0.3,
        "popularity_sim": 0.95,
        "trending_boost": 0.1
      }
    }
  ]
}
```

---

## CLI

```bash
just mvp save owner/name         # fetch + embed + store a repo
just mvp recommend owner/name    # get recommendations (CLI)
just mvp count                   # how many repos in the DB
just mvp seed --per-language 1000 --languages python,rust  # bulk-index
just mvp embed --limit 1000      # embed repos missing vectors
just mvp trending --since daily  # scrape github.com/trending
just mvp register-webhooks       # register GitHub push webhook on all repos
just mvp explore                 # random repo + its recs
```

---

## How it works

```
GitHub API ──► Postgres + pgvector ──► Candidates ──► Features ──► Score ──► Rerank ──► Results
  (fetch)        (store + ANN)        (150-250)    (6 floats)   (1 float)  (diverse)  (top N)
```

1. **Fetch** — repo metadata + README from GitHub API
2. **Embed** — README → 384-dim vector via `BAAI/bge-small-en-v1.5` (loaded once at API startup)
3. **Candidates** — SQL filter (language/topics) ∪ pgvector ANN search, deduplicated
4. **Features** — 6 signals: language match, topic overlap, cosine sim, dep overlap, popularity, trending
5. **Score** — weighted sum (hand-tuned, no training)
6. **Rerank** — drop same-owner, enforce one-per-owner diversity

[→ Detailed architecture](ARCHITECTURE.md)

---

## Tech

| Layer | Stack |
|---|---|
| Embedding | `BAAI/bge-small-en-v1.5` (384 dims, pre-trained, no fine-tuning) |
| Database | Postgres 16 + pgvector (HNSW index) |
| API | FastAPI (async) |
| Frontend | Astro 5 (static, Vercel) |
| Packages | `uv` (Python) · `pnpm` (Node) |
| Deploy | Vercel (frontend) + Render (API) |

---

## Repo layout

```
.
├── apps/
│   ├── site/                 # Astro frontend (Vercel)
│   │   └── src/
│   │       ├── layouts/Base.astro    # grid art, theme toggle, art picker
│   │       ├── pages/               # index, explore, repo
│   │       └── styles/global.css
│   └── mvp_api/              # FastAPI service (Render)
│       └── src/reporelay_mvp_api/
│           ├── main.py               # endpoints
│           └── webhooks.py           # GitHub push handler
├── packages/
│   └── mvp/                  # core library (shared between CLI + API)
│       └── src/reporelay_mvp/
│           ├── recommend.py          # 5-stage pipeline
│           ├── candidates.py         # SQL filter + pgvector ANN
│           ├── features.py           # 6 signal extractors
│           ├── score.py              # weighted sum
│           ├── rerank.py             # diversity rules
│           ├── embedding.py          # sentence-transformer wrapper
│           ├── github.py             # httpx client + search
│           ├── trending.py           # github.com/trending scraper
│           ├── seed.py               # bulk indexer
│           ├── embed_pass.py         # backfill missing embeddings
│           └── cli.py                # typer CLI
├── infra/
│   └── docker-compose.yml    # local Postgres + pgvector
├── Dockerfile.api            # Render API image
├── render.yaml               # Render service config
├── pyproject.toml            # uv workspace
└── justfile                  # task runner
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
