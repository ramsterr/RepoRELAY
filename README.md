# RepoRelay MVP (v0.1)

A dead-simple GitHub repo recommender. Give it a repo, get back repos you might also care about. No Redis, no user data, no graph walking — just Postgres + pgvector + five features weighted together.

**This branch is the simplest version that works end-to-end. The main branch has the full engine (4 strategies, blender, user profiles, etc).**

## How it works

Five steps, each one a single file:

| Step | File | What it does |
|---|---|---|
| 1. Data | `data.py` | Reads/writes a single `mvp_repos` table in Postgres |
| 2. Features | `features.py` | Computes 5 signals: language match, topic overlap, cosine sim, dep overlap, popularity sim |
| 3. Candidates | `candidates.py` | Gathers a pool via SQL filter (same language/topics) + pgvector ANN |
| 4. Score | `score.py` | Weighted sum of the 5 features |
| 5. Rerank | `rerank.py` | Kicks out same-owner repos, enforces one-per-owner diversity |

## Setup

You need Docker running and a GitHub token:

```bash
# 1. Create a .env file (copy from main branch .env.example)
echo 'GITHUB_TOKEN=ghp_your_token_here' > .env

# 2. Start Postgres
docker compose -f infra/docker-compose.yml up -d postgres

# 3. Install deps
uv sync

# 4. Run the migration (creates mvp_repos table)
just mvp-migrate
```

## Usage

CLI:

```bash
# Save a repo to the DB
just mvp save fastapi/fastapi
just mvp save django/django

# Check how many repos you have
just mvp count

# Get recommendations
just mvp recommend fastapi/fastapi --limit 5
just mvp recommend fastapi/fastapi --limit 5 --json   # machine-readable
```

API:

```bash
just mvp-api                                    # starts on port 8001

curl localhost:8001/health
curl "localhost:8001/recommend?repo=django/django&limit=5"
```

## What you'll see

```
recommendations for fastapi/fastapi

   1. pallets/flask  (Python, 71680 stars, topics: python, flask, wsgi)
   2. psf/requests   (Python, 54043 stars, topics: python, http, forhumans)
   3. django/django  (Python, 87917 stars, topics: python, django, web)
   4. nestjs/nest    (TypeScript, 75917 stars, topics: nest, javascript, typescript)
```

Python repos come first (language match), web frameworks cluster together (topic overlap), and the ranking makes intuitive sense even with 8 repos in the DB.

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
