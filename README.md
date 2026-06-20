# RepoRelay MVP (v0.1)

A dead-simple GitHub repo recommender. Give it a repo, get back repos you might also care about. No Redis, no user data, no graph walking — just Postgres + pgvector + five features weighted together.

**This branch is the simplest version that works end-to-end. The main branch has the full engine (4 strategies, blender, user profiles, etc).**

[→ Full architecture doc (how each stage works, the database schema, what the embedding model does)](ARCHITECTURE.md)

## How it works

Five steps, each one a single file:

| Step | File | What it does |
|---|---|---|
| 1. Data | `data.py` | Reads/writes a single `mvp_repos` table in Postgres |
| 2. Features | `features.py` | Computes 5 signals: language match, topic overlap, cosine sim, dep overlap, popularity sim |
| 3. Candidates | `candidates.py` | Gathers a pool via SQL filter (same language/topics) + pgvector ANN |
| 4. Score | `score.py` | Weighted sum of the 5 features |
| 5. Rerank | `rerank.py` | Kicks out same-owner repos, enforces one-per-owner diversity |

## Getting Started (step by step)

You need three things on your machine: **Docker**, **Python 3.12**, and **uv** (pip install uv).

### Step 1 — Get a GitHub token

The data comes from the **GitHub API**. It's free. Go to https://github.com/settings/tokens, create a "classic" token, no scopes needed, and copy it.

```bash
echo 'DATABASE_URL=postgresql+psycopg://reporelay:reporelay@localhost:5439/reporelay' > .env
echo 'GITHUB_TOKEN=ghp_your_token_here' >> .env
```

That's where the repo data comes from. When you run `just mvp save`, it calls GitHub's API and asks "tell me about this repo" — then stores the answer in Postgres.

### Step 2 — Start Postgres

```bash
docker compose -f infra/docker-compose.yml up -d postgres
```

This starts a Postgres 16 container with the pgvector extension. One container, that's it.

### Step 3 — Install dependencies

```bash
uv sync
```

Installs FastAPI, SQLAlchemy, sentence-transformers, httpx — everything the project needs.

### Step 4 — Create the database table

```bash
just migrate
```

Runs a migration that creates the `mvp_repos` table in Postgres. You only do this once.

### Step 5 — Put some repos in the database

```bash
just mvp save fastapi/fastapi
just mvp save django/django
just mvp save pallets/flask
just mvp save psf/requests
```

Each `save` command does three things:
1. Calls GitHub's API for the repo's metadata (language, topics, stars, README)
2. Runs the README through a pre-trained model to get 384 numbers (an embedding)
3. Stores everything in the `mvp_repos` table

The model downloads the first time (~11 seconds), then stays in memory.

### Step 6 — Ask for recommendations

```bash
just mvp recommend fastapi/fastapi --limit 5
```

This looks up fastapi in the database, finds the most similar repos using 5 different signals, scores them, and prints a ranked list. [→ How it actually works](ARCHITECTURE.md)

```
recommendations for fastapi/fastapi

   1. pallets/flask  (Python, 71680 stars, topics: python, flask, wsgi)
   2. psf/requests   (Python, 54043 stars, topics: python, http, forhumans)
   3. django/django  (Python, 87917 stars, topics: python, django, web)
```

### Optional — Run as an API

```bash
just api                                         # starts on port 8001
curl "localhost:8001/recommend?repo=django/django&limit=5"
```

---

## Tuning

The weights live in `packages/mvp/src/reporelay_mvp/score.py`:

```python
WEIGHTS = {
    "language_match": 0.30,
    "topic_overlap": 0.30,
    "cosine_sim": 0.20,
    "dep_overlap": 0.15,
    "popularity_sim": 0.05,
}
```

Change them, save, and the next query uses the new weights. No retraining, no pipeline restart.

## What this doesn't do

- No user tracking or personalization
- No collaborative filtering (needs real star data)
- No graph-based traversal
- No Redis caching
- No feedback loop

All of those are on `main`. This branch is for understanding and demonstrating the core recommendation loop.

## Files

```
packages/mvp/                          # the recommender
  src/reporelay_mvp/
    data.py          queries + upserts
    features.py      the 5 signals
    candidates.py    pool generation
    score.py         weighted sum
    rerank.py        dedup + diversity
    recommend.py     orchestrator
    embedding.py     sentence-transformers wrapper
    github.py        fetch repos from GitHub API
    cli.py           typer CLI
    db.py            SQLAlchemy session management
    models.py        Pydantic models
    settings.py      env-based config
    migrations/      single mvp_repos table

apps/mvp_api/                         # FastAPI server
  src/reporelay_mvp_api/
    main.py         /health + /recommend
```

## License

MIT
