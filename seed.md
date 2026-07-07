# Indexing & Embedding

How to fill the database with repos and compute their embeddings.

## Quick start

```bash
just seed-and-embed
```

This seeds 260+ broad software topics (130 repos each ≈ 30K+ total), then embeds them with both README and description vectors. Takes ~35 minutes (local mode).

## Commands

| Command | What it does |
|---|---|
| `just seed-and-embed` | Seed + embed in one shot (defaults: 260+ topics, 130/topic, embed 7000) |
| `just seed-and-embed "ai,rust,kubernetes" 200 3000` | Custom topics, 200/topic, embed 3000 |
| `just seed-and-embed "" 50 500` | All default topics, 50/topic, embed 500 |
| `just mvp seed-topics --per-topic 200` | Seed only (all 260+ topics, 200 each) |
| `just mvp seed-topics --topics "react,vue,nextjs" --per-topic 100` | Seed specific topics |
| `just mvp embed --limit 5000` | Embed only — backfills missing README + description vectors |
| `just mvp extract-keywords --limit 5000` | Extract domain keywords from descriptions |
| `just mvp count` | How many repos are in the DB |

## What happens under the hood

### 1. Seed (`just mvp seed-topics`)

For each topic (e.g. "machine-learning", "kubernetes", "react"):
- Calls GitHub search API: `topic:ml stars:>20 sort:stars`
- Fetches up to 200 repos per topic (2 pages × 100 results)
- Bulk-upserts into `mvp_repos` — metadata only (name, description, language, topics, stars)
- No README fetched, no embeddings computed — that's the next step

**Rate limit:** GitHub search API allows 30 requests/minute. The seeder paces itself at 2.2s between topics (~27 req/min) to stay under the budget. If a topic fails (429 / rate limit), it retries 3× with 12s backoff. Dual GitHub tokens double the available budget to 60 req/min.

**Weighted mode:** `seed_topics_weighted()` supports per-topic counts via `topics_config.py` for distributing 50K repos across ~500+ topics — popular topics get more repos, niche topics get fewer.

### 2. Embed (`just mvp embed`)

For repos with `embedded_at IS NULL` (any repo from seed that hasn't been embedded yet):

**Local mode** (`EMBEDDING_API=local`):
- Downloads the README from GitHub (1 API call per repo)
- Truncates to 8000 characters
- Runs `BAAI/bge-small-en-v1.5` locally on your machine (~300MB RAM)
- Writes 384-dim vector to `embedding` column
- Also embeds the repo's description (if non-empty) → `description_embedding` column
- Concurrency: 4 parallel fetches, 0.1s pause between launches

**Gemini mode** (`EMBEDDING_API=gemini`):
- No local model needed — all embeddings via Gemini API
- Sends description text (or README fallback) to `gemini-embedding-001`
- Returns 512-dim vectors (truncated from 768 via MRL)
- Batch size: 50 per API call
- Timeout: 45s per attempt, 3 retries
- Ideal for Render's free tier (no RAM cost)
- Run via GitHub Actions cron (embed.yml: 3000 repos every 30 min)

**The model runs on YOUR machine or through Gemini API, never on the deployed site.** Render runs in lightweight mode (`REPORE_LAY_LIGHTWEIGHT=1`), so the 512MB free tier is enough. The vectors are stored in Neon's pgvector column and read by the API — no model needed at query time.

### 3. How the site uses them

- **Repo with embeddings:** pgvector ANN search finds content-similar repos via description cosine similarity (0.20 weight) + README-vs-description cosine similarity (0.25 weight). Description embedding is the primary signal — it's purpose-dense and works for 94% of corpus.
- **Repo without description embedding:** keyword-based proxy kicks in — extracted keywords + full-text search + topic overlap generate instant candidates. Gemini embedding runs in background; results use the real vector on next visit.
- **Keyword features:** `keyword_match` (0.08 weight) and `keyword_topic_match` (0.05 weight) capture domain overlap from extracted keywords.

## Scaling up

```bash
# Light run: 500 repos across 10 topics
just seed-and-embed "python,rust,go,react,security,ai,kubernetes,docker,database,cli" 50 500

# Medium run: 3000 repos
just seed-and-embed "" 60 3000

# Heavy run: 15000 repos — increase per-topic count
just seed-and-embed "" 300 15000
```

## Running periodically

Re-run `just seed-and-embed` once a week. New repos from GitHub trending become available. Already-embedded repos are skipped (ON CONFLICT DO UPDATE). The embed pass only processes `embedded_at IS NULL` rows, so it's idempotent.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Multiple head revisions` on migrate | Alembic chain broke. `just migrate` should work after the latest fix. |
| Topic fails silently | Search API rate limit. The seeder retries 3×. If still failing, check `GITHUB_TOKEN` in `.env`. |
| Embed hangs (local) | Model download from HuggingFace is slow on first run. Be patient — it's 130MB. |
| Embed returns zeros | `REPORE_LAY_LIGHTWEIGHT` is set. Remove it or set to `0` for local embedding. |
| Gemini embed times out | Check `GEMINI_API_KEY` in `.env`. Gemini has a 45s timeout with 3 retries. |
| `sentence_transformers` import error | Run `uv sync` — the extra dependency isn't installed in lightweight mode. |
