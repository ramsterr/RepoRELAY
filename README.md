# RepoRelay

A GitHub repo recommender. Give it a repo, get back similar repos.

No ML training. No user data. Just Postgres + pgvector + a pre-trained embedding model.

```
fastapi/fastapi  →  pallets/flask, django/django, psf/requests, encode/starlette ...
```

**[→ Full architecture & how it works](ARCHITECTURE.md)**

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

**Prerequisites:** Docker, Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js

---

## Run it

```bash
just api          # API server on :8001
just site         # Web UI on :4321
just dev          # Both at once
```

---

## API

| Endpoint | Description |
|---|---|
| `GET /recommend?repo=owner/name` | Ranked recommendations |
| `GET /explore?seed=42` | Random repo + its recs |
| `GET /popular` | Top repos by stars |
| `GET /topics` | Top topics in the database |
| `GET /health` | Liveness check |

**Optional params:** `limit`, `seed` (deterministic shuffle), `tags` (semantic filter — e.g. `tags=react,typescript`)

---

## CLI

```bash
just mvp save owner/name         # fetch + embed + store a repo
just mvp recommend owner/name    # get recommendations
just mvp seed                    # bulk-index repos by language
just mvp trending                # scrape github.com/trending
just mvp embed                   # embed repos missing vectors
just mvp count                   # how many repos in the DB
```

---

## How it works

1. **Fetch** — repo metadata + README from GitHub API
2. **Embed** — README → 384-dim vector via `BAAI/bge-small-en-v1.5`
3. **Candidates** — SQL filter (language/topics) + pgvector ANN search
4. **Features** — 6 signals: language match, topic overlap, cosine sim, dep overlap, popularity, trending
5. **Score** — weighted sum (hand-tuned, no training)
6. **Rerank** — drop same-owner, enforce diversity

[→ Detailed architecture](ARCHITECTURE.md)

---

## Tech

| Layer | Stack |
|---|---|
| Embedding | `BAAI/bge-small-en-v1.5` (384 dims) |
| Database | Postgres 16 + pgvector |
| API | FastAPI (async) |
| Frontend | Astro 5 (vanilla JS) |
| Packages | `uv` (Python) · `pnpm` (Node) |

---

## License

MIT
