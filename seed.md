# Indexing & Embedding

How to fill the database with repos and compute their embeddings.

## Quick start

```bash
just seed-and-embed
```

This seeds 54 broad software topics (130 repos each ≈ 7000 total), then embeds them all with both README and description vectors. Takes ~35 minutes.

## Commands

| Command | What it does |
|---|---|
| `just seed-and-embed` | Seed + embed in one shot (defaults: 54 topics, 130/topic, embed 7000) |
| `just seed-and-embed "ai,rust,kubernetes" 200 3000` | Custom topics, 200/topic, embed 3000 |
| `just seed-and-embed "" 50 500` | All default topics, 50/topic, embed 500 |
| `just mvp seed-topics --per-topic 200` | Seed only (all 54 topics, 200 each) |
| `just mvp seed-topics --topics "react,vue,nextjs" --per-topic 100` | Seed specific topics |
| `just mvp embed --limit 5000` | Embed only — backfills missing README + description vectors |
| `just mvp count` | How many repos are in the DB |

## What happens under the hood

### 1. Seed (`just mvp seed-topics`)

For each topic (e.g. "machine-learning", "kubernetes", "react"):
- Calls GitHub search API: `topic:ml stars:>20 sort:stars`
- Fetches up to 200 repos per topic (2 pages × 100 results)
- Bulk-upserts into `mvp_repos` — metadata only (name, description, language, topics, stars)
- No README fetched, no embeddings computed — that's the next step

**Rate limit:** GitHub search API allows 30 requests/minute. The seeder paces itself at 2.2s between topics (~27 req/min) to stay under the budget. If a topic fails (429 / rate limit), it retries 3× with 12s backoff.

### 2. Embed (`just mvp embed`)

For repos with `embedded_at IS NULL` (any repo from seed that hasn't been embedded yet):
- Downloads the README from GitHub (1 API call per repo)
- Truncates to 8000 characters
- Runs `BAAI/bge-small-en-v1.5` locally on your machine (~300MB RAM)
- Writes 384-dim vector to `embedding` column
- Also embeds the repo's description (if non-empty) → `description_embedding` column
- Concurrency: 4 parallel fetches, 0.1s pause between launches

**The model runs on YOUR machine, never on the deployed site.** Render runs in lightweight mode (`REPORE_LAY_LIGHTWEIGHT=1`), so the 512MB free tier is enough. The vectors are stored in Neon's pgvector column and read by the API — no model needed at query time.

### 3. How the site uses them

- **Repo with embeddings:** pgvector ANN search finds content-similar repos via README cosine similarity (0.15 weight) + description cosine similarity (0.15 weight). Both features have real vectors.
- **Repo without embeddings:** system borrows a proxy vector from a topic-sibling repo. Falls back to structured features (topic overlap, language match, description token similarity, dependency overlap, popularity, trending).

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
| Embed hangs | Model download from HuggingFace is slow on first run. Be patient — it's 130MB. |
| Embed returns zeros | `REPORE_LAY_LIGHTWEIGHT` is set. Remove it or set to `0` for local embedding. |
| `sentence_transformers` import error | Run `uv sync` — the extra dependency isn't installed in lightweight mode. |
