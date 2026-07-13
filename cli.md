# RepoRelay CLI Reference

The `reporelay-mvp` CLI is the command-line interface for managing the
recommendation engine. All commands are run via the `just mvp` task runner
or directly with `uv run --package reporelay-mvp reporelay-mvp <command>`.

```bash
# via just
just mvp <command> [options]

# directly
uv run --package reporelay-mvp reporelay-mvp <command> [options]
```

---

## Table of contents

- [Setup](#setup)
- [Commands](#commands)
  - [`save`](#save) — fetch + embed + store a single repo
  - [`count`](#count) — show how many repos are in the DB
  - [`recommend`](#recommend) — get ranked recommendations for a repo
  - [`explore`](#explore) — random repo + its recommendations
  - [`seed`](#seed) — bulk-index the corpus by language
  - [`seed-topics`](#seed-topics) — bulk-index the corpus by topic
  - [`embed`](#embed) — compute embeddings for un-embedded repos
  - [`infer-topics`](#infer-topics) — backfill inferred topics from READMEs
  - [`extract-keywords`](#extract-keywords) — extract domain keywords from descriptions
  - [`trending`](#trending) — scrape github.com/trending
  - [`register-webhooks`](#register-webhooks) — register GitHub push webhooks
- [Common workflows](#common-workflows)

---

## Setup

The CLI requires:
- A running Postgres instance with the `mvp_repos` table (via `just up && just migrate`)
- A `.env` file with `DATABASE_URL` and `GITHUB_TOKEN` set
- The `uv` package manager

```bash
# 1. Start Postgres
just up

# 2. Run migrations
just migrate

# 3. Verify the CLI is wired up
just mvp --help
```

---

## Commands

### `save`

Fetch a single repo from GitHub, persist it to the DB, embed its README,
and infer additional topics if the repo has fewer than 3 topics.

```bash
just mvp save owner/name
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `repo` | string | yes | GitHub repo as `owner/name` |

**Examples:**
```bash
just mvp save fastapi/fastapi
just mvp save pallets/flask
just mvp save django/django
```

**What it does:**
1. Fetches metadata + topics + README from the GitHub API
2. Infers additional topics from the README if the repo has <3 topics
3. Computes a 512-dim embedding of the README (Gemini) or 384-dim (local BAAI/bge-small)
4. Upserts everything into `mvp_repos`

---

### `count`

Print how many repos are currently stored.

```bash
just mvp count
```

**Output:**
```
1234 repos in mvp_repos
```

---

### `recommend`

Run the 5-stage recommendation pipeline against a stored repo and print
the results to the terminal.

```bash
just mvp recommend owner/name
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `repo` | string | yes | GitHub repo as `owner/name` |
| `--limit` | int | no | Number of recommendations to return (default: 10) |
| `--seed` | int | no | Deterministic variation seed (same seed = same order) |
| `--json` | flag | no | Emit JSON output instead of a table |

**Examples:**
```bash
# Default — 10 recommendations
just mvp recommend fastapi/fastapi

# 20 recommendations
just mvp recommend fastapi/fastapi --limit 20

# Deterministic variation
just mvp recommend fastapi/fastapi --seed 42

# JSON for piping into other tools
just mvp recommend fastapi/fastapi --json | jq '.repos[].full_name'
```

**Output (table):**
```
recommendations for fastapi/fastapi

   1. pallets/flask  (Python, 67000 stars, topics: python, web, framework)
   2. django/django  (Python, 78000 stars, topics: python, web, framework)
   3. encode/starlette  (Python, 9500 stars, topics: python, asyncio, web)
   ...
```

---

### `explore`

Pick a random repo from the DB and show its recommendations. Useful for
discovering new repos you haven't seen before.

```bash
just mvp explore --seed <int>
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--seed` | int | yes | Deterministic seed for the random repo pick |
| `--limit` | int | no | Number of recommendations (default: 10) |
| `--json` | flag | no | Emit JSON output |

**Examples:**
```bash
# Pick a random repo deterministically
just mvp explore --seed 1

# Different random repo, same seed = same result
just mvp explore --seed 42
```

---

### `seed`

Bulk-index the corpus from GitHub search by language. Each language
costs 1-3 search API calls. Default is 300 repos × 10 languages ≈ 3,000
target rows, ~30 search requests total.

```bash
just mvp seed --per-language 300 --languages python,rust
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--per-language` | int | 300 | Repos to index per language |
| `--languages` | string | top 10 by repo count | Comma-separated list of languages |
| `--min-stars` | int | 100 | GitHub stars floor for search results |
| `--page-delay` | float | 2.0 | Seconds between search API calls (2.0 = 30 req/min) |

**Examples:**
```bash
# Default — 3,000 repos across top 10 languages
just mvp seed

# Specific languages only
just mvp seed --languages python,rust,typescript

# More repos per language
just mvp seed --per-language 1000

# Lower star floor to catch more repos
just mvp seed --min-stars 50
```

---

### `seed-topics`

Bulk-index the corpus by topic instead of language. This catches repos
that share a domain (e.g. `machine-learning`, `kubernetes`, `web3`)
even if they're written in different languages.

The default topic list includes 260+ topics across ML, web, mobile,
DevOps, databases, security, blockchain, game dev, and more.

```bash
just mvp seed-topics --per-topic 200
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--topics` | string | 260+ built-in defaults | Comma-separated topic list |
| `--per-topic` | int | 200 | Repos per topic |
| `--min-stars` | int | 20 | GitHub stars floor |

**Examples:**
```bash
# Default — all 260+ topics
just mvp seed-topics

# Specific topics
just mvp seed-topics --topics machine-learning,llm,generative-ai,rag

# More repos per topic
just mvp seed-topics --per-topic 500

# Combine with the full seed-and-embed justfile recipe
just seed-and-embed "" 130 7000
```

**Notes:**
- GitHub's search API only supports one `topic:X` qualifier per query
- The CLI makes one search per topic and merges results
- This is idempotent — re-running upserts and refreshes metadata

---

### `embed`

Compute and store description + README embeddings for repos that were
indexed from search but not yet embedded. Uses Gemini API in production
(512-dim vectors, zero local RAM) or local BAAI/bge-small (384-dim).
Embeddings unlock the pgvector ANN search that powers semantic recommendations.

```bash
just mvp embed --limit 1000
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--limit` | int | 1000 | How many top-by-stars repos to embed |
| `--concurrency` | int | 4 | Parallel README fetches (local mode only) |
| `--batch-size` | int | 50 | Repos per Gemini API call (Gemini mode only) |

**Examples:**
```bash
# Embed top 1000 repos by stars
just mvp embed

# Embed more repos
just mvp embed --limit 5000

# Higher concurrency (local mode — faster, but more API calls)
just mvp embed --limit 1000 --concurrency 8
```

**Notes:**
- Local mode: first run downloads the embedding model (~80MB, ~11s)
- Gemini mode: no local model — just API calls (needs `GEMINI_API_KEY`)
- Each repo = 1 description embed call (94% of corpus) + optional README fetch
- Description embedding is the primary signal; README embedding only for source repos
- Batch size: 50 per Gemini API call
- Paced to stay under the 5,000 req/hr GitHub REST limit (dual tokens = 10,000 req/hr)
- Run after `seed` and `seed-topics` to backfill embeddings

---

### `infer-topics`

Backfill inferred topics on repos with sparse topic lists. Scans each
repo's README and description against a 250+ keyword vocabulary to
suggest additional GitHub-style topic tags.

This dramatically improves topic-based candidate generation and the
`topic_overlap` feature for repos that didn't have topics assigned by
their owners.

```bash
just mvp infer-topics
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--limit` | int | 1000 | Max repos to process |
| `--min-topics` | int | 3 | Repos with fewer than this many topics get inference |
| `--refetch` | flag | false | Re-fetch README from GitHub for better inference (uses API rate limit) |

**Examples:**
```bash
# Fast mode — uses stored description only (no API calls)
just mvp infer-topics

# Better mode — re-fetches README for more accurate inference
just mvp infer-topics --refetch --limit 500

# Aggressive — processes repos with up to 5 existing topics
just mvp infer-topics --min-topics 5
```

**Output:**
```
inferring topics for 847 repos (min_topics < 3)

  [1/847] langchain-ai/langchain: +llm, rag, langchain
  [2/847] milvus-io/milvus: +vector-database, machine-learning
  [3/847] solidity-docs: +blockchain, ethereum, smart-contracts
  ...

done — 312/847 repos got new topics
```

**Notes:**
- Without `--refetch`: only the stored description is used (fast, no API calls)
- With `--refetch`: the full README is re-fetched from GitHub for higher accuracy
- Capped at 15 inferred topics per repo
- Precision-first: avoids hallucinated topics on generic text

---

### `extract-keywords`

Extract domain-specific technical keywords from repo descriptions
and READMEs. Keywords are stored in a TEXT[] column for fast GIN ARRAY
overlap queries. Powers the `keyword_match` and `keyword_topic_match`
scoring features.

```bash
just mvp extract-keywords --limit 5000
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--limit` | int | 5000 | Max repos to process |
| `--min-chars` | int | 50 | Only extract keywords from repos with descriptions >= this length |
| `--from-github` | flag | false | Also download README from GitHub for richer keyword extraction (uses API rate limit) |
| `--since` | string | none | Process repos updated after this timestamp (ISO 8601) |

**Examples:**
```bash
# Default — extract keywords from top 5000 repos by stars
just mvp extract-keywords

# Process 10000 repos with richer README-based extraction
just mvp extract-keywords --limit 10000 --from-github

# Only repos with substantive descriptions
just mvp extract-keywords --min-chars 100
```

**Notes:**
- Extracts up to 30 keywords per repo
- Filters out stopwords, single letters, numbers, and generic terms
- Prioritizes multi-word terms, hyphenated compounds, CamelCase tokens
- Without `--from-github`: uses stored description only (fast, no API calls)
- With `--from-github`: re-fetches README from GitHub for richer extraction

---

### `trending`

Scrape [github.com/trending](https://github.com/trending) for viral repos
and update the `trending_score` column on existing rows. Catches repos
that the search API misses because their total stars are still low.

```bash
just mvp trending --since daily
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--languages` | string | top 10 + "all languages" | Comma-separated languages to scrape |
| `--since` | string | `daily` | Time window: `daily`, `weekly`, or `monthly` |
| `--delay-s` | float | 10.0 | Seconds between language scrapes (be polite) |

**Examples:**
```bash
# Daily trending (default)
just mvp trending

# Weekly trending for Python and Rust
just mvp trending --since weekly --languages python,rust

# Monthly, all languages
just mvp trending --since monthly
```

**Notes:**
- This is HTML scraping, not API calls — no rate limit
- Only updates repos that already exist in `mvp_repos` (run `save` or `seed` first to discover new trending repos)
- The `trending_score` is `min(1.0, stars_period / 100)` so it sits in [0, 1]

---

### `register-webhooks`

Register GitHub push webhooks on top-starred repos so the API receives
a notification when a watched repo's README changes. This triggers
automatic re-embedding.

Run once after deploying the API. Safe to re-run — duplicate
registrations return 422 and are reported as already-exists.

```bash
just mvp register-webhooks \
  --callback-url https://your-api.onrender.com \
  --secret your-webhook-secret
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--min-stars` | int | 1000 | Only register webhooks for repos with at least this many stars |
| `--callback-url` | string | required | Public URL of the deployed API |
| `--secret` | string | required | `GITHUB_WEBHOOK_SECRET` value (must match the API's env) |

**Examples:**
```bash
# Register on all repos with 1000+ stars
just mvp register-webhooks \
  --callback-url https://reporelay-mvp-api-0w1k.onrender.com \
  --secret $GITHUB_WEBHOOK_SECRET

# Lower threshold
just mvp register-webhooks \
  --min-stars 500 \
  --callback-url https://reporelay-mvp-api-0w1k.onrender.com \
  --secret $GITHUB_WEBHOOK_SECRET
```

**Notes:**
- Iterates the top 10 pages of GitHub search (1,000 repos) by default
- Each registration costs 1 API call
- 422 responses (already has a webhook) are silently skipped

---

## Common workflows

### First-time setup (cold start)

```bash
# 1. Start Postgres and run migrations
just up && just migrate

# 2. Seed the corpus by topic (catches more repos than language-only)
just mvp seed-topics --per-topic 200

# 3. Embed the top repos
just mvp embed --limit 5000

# 4. Backfill inferred topics on sparse repos
just mvp infer-topics
```

### Test with a few repos

```bash
just mvp save fastapi/fastapi
just mvp save pallets/flask
just mvp save django/django
just mvp recommend fastapi/fastapi
```

### Refresh the corpus

```bash
# Pick up new repos in your topics
just mvp seed-topics --per-topic 200

# Embed any new ones
just mvp embed --limit 1000

# Get fresh trending signal
just mvp trending --since daily
```

### A/B test with seeds

```bash
# Same source, different seeds = different orderings
just mvp recommend fastapi/fastapi --seed 1
just mvp recommend fastapi/fastapi --seed 2
just mvp recommend fastapi/fastapi --seed 3
```

### Export recommendations as JSON

```bash
just mvp recommend fastapi/fastapi --json --limit 20 | jq '.repos[] | {name: .full_name, score: .score}'
```

### Bulk-seed with the justfile recipe

```bash
# Default: all 260+ topics, 130 repos/topic, 7000 embeds
just seed-and-embed

# Custom
just seed-and-embed "machine-learning,llm,rag,langchain" 200 5000
```

---

## Environment variables

The CLI reads these from `.env` (loaded automatically via `dotenv-load` in the justfile):

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | yes | Postgres connection string (default: `postgresql+psycopg://reporelay:reporelay@localhost:5439/reporelay`) |
| `GITHUB_TOKEN` | yes | GitHub personal access token (raises rate limit from 60 to 5,000 req/hr) |

---

## See also

- [README.md](README.md) — project overview and API reference
- [ARCHITECTURE.md](ARCHITECTURE.md) — pipeline internals
- [seed.md](seed.md) — detailed indexing and embedding guide
- [improvements.md](improvements.md) — known issues and roadmap
