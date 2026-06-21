# Seed Pipeline — Roadmap to Production

## Goal

Continuously discover, store, and embed GitHub repos so the recommender
always has fresh data and never blocks on embeddings.

## Constraints (hard)

| Resource | Limit | Notes |
|---|---|---|
| GitHub search | 30 req / min | per-user, not per-token |
| GitHub REST | 5,000 req / hour (auth) | gates README fetches |
| GitHub GraphQL | 5,000 pts / hour (auth) | 1 search ≈ 1pt, 1 README ≈ 1pt |
| Render free | 512 MB RAM | one BGE-small model instance max |
| Neon free | 0.5 GB storage | ~150k rows of mvp_repos w/ 384-dim vectors |

## Architecture (target)

```
┌──────────────────────────────────────────────────────────────────┐
│                        PRODUCER (search)                         │
│  GitHub Search @ 30/min  ──►  mvp_repos.embedding = NULL         │
│  (idempotent upsert, mark search_fetched_at)                     │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                     CONSUMER (embed) — N workers                 │
│  Claim rows: SELECT … FOR UPDATE SKIP LOCKED                     │
│  Fetch READMEs (async batch, semaphore=10)                       │
│  Batch-encode through ONE BGE-small (batch=32)                   │
│  UPDATE mvp_repos SET embedding = … , embedded_at = NOW()        │
└──────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                  RECOMMENDER (read-path, hot)                    │
│  candidate has embedding?  → pgvector ANN (cosine sim)           │
│  candidate has no embed?  → topic overlap + language + stars     │
│  (always works — no cold-start gap)                              │
└──────────────────────────────────────────────────────────────────┘
```

Key invariants:

- **Search outruns embed by ~35×** — that's fine. Tag-fallback bridges the gap.
- **Re-runnable end-to-end** — every step is idempotent (upsert by `full_name`,
  skip-if-not-null for embeddings, `search_fetched_at` for staleness).
- **Bounded** — never exceeds GitHub limits, never runs out of RAM.
- **Resumable** — worker crash = next worker picks up unlocked rows.

---

## Phase 1 — Cron-based (this week, ~3 hours)

Get a working end-to-end loop. Boring, but unblocks everything else.

- [ ] **GitHub Actions: `.github/workflows/seed.yml`** — cron every 6h, runs `just mvp seed --per-language 1000`, uses `DATABASE_URL` + `GITHUB_TOKEN` secrets
- [ ] **GitHub Actions: `.github/workflows/embed.yml`** — cron every 1h (offset by 30min from seed), runs `just mvp embed --limit 500`
- [ ] **Add structured logging** to `seed_corpus` and `embed_pass` — emit per-language counts, repos/sec, ETA, rate-limit remaining
- [ ] **Add a `--dry-run` flag** to both commands — estimate work without doing it
- [ ] **README in seed.py** documenting the math: "1000/lang × 10 langs = 30 search calls = 1 minute at 30/min"

**Acceptance:** DB has ≥ 5,000 repos and ≥ 1,000 embeddings after 24h of cron runs.

---

## Phase 2 — Query diversity (next week, ~4 hours)

Star-sorted search returns the same 1,000 popular repos forever. Need
multiple query strategies to discover diverse + new repos.

- [ ] **Multi-strategy seeder** — `seed.py` runs N passes, each with a different sort/filter:
  - sort=stars, order=desc (popular)
  - sort=updated, order=desc (active)
  - sort=stars, order=asc, stars:>10 (rising)
  - created:>YYYY-MM-DD (new)
  - pushed:>YYYY-MM-DD (actively maintained)
  - good-first-issues:>5 (welcoming)
  - one pass per major topic: rust, ml, web, devops, etc. (10 topics × 1000 = 10k)
- [ ] **Staleness check** — skip repos with `search_fetched_at > NOW() - INTERVAL '7 days'`
- [ ] **Deduplication on insert** — unique index on `full_name` (already there), plus an "in-flight" check to avoid race conditions
- [ ] **Throughput cap** — total API calls / run = 30 (search budget). 1 pass = 10 calls. So 3 passes per run max.

**Acceptance:** Seed run discovers ≥ 100 new repos per execution (proves it's not just re-indexing the same popular ones).

---

## Phase 3 — Concurrent embed workers (week 3, ~6 hours)

The real bottleneck. One worker can do 80–100 embeds/min (gated by
README fetch rate). Multiple workers = linear speedup to a point.

- [ ] **Postgres-based work queue** — claim rows with `SELECT id FROM mvp_repos WHERE embedding IS NULL ORDER BY id LIMIT 100 FOR UPDATE SKIP LOCKED`
- [ ] **Worker pool (N=2 on free Render, N=8 on paid)** — each runs the same loop, claims 100 rows, embeds, releases
- [ ] **Semaphore on README fetches** — `asyncio.Semaphore(10)` so we don't blow past 5,000/hour auth limit
- [ ] **Exponential backoff on 429** — read `X-RateLimit-Reset`, sleep until reset, then retry (currently we just log a warning and move on)
- [ ] **Batched model.encode** — `model.encode(texts, batch_size=32, show_progress_bar=False)` — 3–4× faster than serial
- [ ] **Idempotent claim release** — if the worker dies mid-batch, the row's `FOR UPDATE` lock is released automatically (Postgres behavior)

**Acceptance:** Embed throughput ≥ 200 repos / hour on free Render. DB rows with `embedding IS NULL` count trends down over time.

---

## Phase 4 — GraphQL migration (month 2, ~8 hours)

Combine search + README fetch into fewer API calls. GraphQL points
are cheaper per useful data unit.

- [ ] **GraphQL client wrapper** — `gql.py` with token-bucket rate limiter
- [ ] **Combined search+readme query** — `search(query:"language:python stars:>100", first:50) { nodes { ... readme: object(expression:"HEAD:README.md") { text } } }`
- [ ] **Backward-compatible fallback** — if GraphQL errors, fall back to REST search + REST readme (current path)
- [ ] **Cost model in code** — `points_used = response.rateLimit.used`; budget = 5,000/hour; auto-throttle
- [ ] **Migration switch** — `GITHUB_USE_GRAPHQL=1` env var, default off until proven

**Acceptance:** Embed throughput ≥ 1,000 repos / hour. Total API points/hour stays < 4,000 (leaves 1,000 buffer for the recommender's live searches).

---

## Phase 5 — Real-time refresh (month 3, ~1 day)

Stop polling. Get notified when a repo's README changes.

- [ ] **Webhook receiver** — small FastAPI endpoint, registers GitHub webhook for `push` events on watched repos
- [ ] **Re-embed trigger** — on push to `main` of a watched repo, enqueue re-embed
- [ ] **Trending detection** — daily job that re-ranks repos by star velocity, bumps them up the embed queue
- [ ] **Soft delete** — repos not seen in 30 days marked `stale=true`, excluded from results (still in DB for analysis)

**Acceptance:** A repo with a major README update gets re-embedded within 5 minutes of the push.

---

## Phase 6 — Self-tuning (quarter 2, ~1 week)

Stop hand-tuning query strategies. Let the system learn.

- [ ] **Track click-through rate per query strategy** — which seeds produce recommendations users actually click?
- [ ] **Bandit-style query selection** — Thompson sampling over query strategies based on CTR
- [ ] **Embedding model upgrade path** — abstract the encoder so swapping BGE-small → BGE-base → instructor is a one-line change
- [ ] **A/B test new strategies** — run 10% of seed budget on a new strategy, measure CTR lift, auto-promote

**Acceptance:** Recommendation CTR improves week-over-week with no manual intervention.

---

## Cross-cutting (do alongside any phase)

- [ ] **Metrics endpoint** — `/metrics` on the API exposing: queue depth, embed throughput, search rate-limit remaining, recommendation p50/p95 latency
- [ ] **Health check that means something** — `/health` should verify DB connection, model loadable, GitHub API reachable (not just "process is up")
- [ ] **Cost dashboard** — Grafana or just a daily log line: "used X of 5000 REST reqs, Y of 5000 GraphQL pts today"
- [ ] **Backpressure** — if `mvp_repos.embedding IS NULL` count > 50k, slow down search (don't pile up work we can't process)
- [ ] **Graceful shutdown** — SIGTERM handler in workers: finish current batch, release locks, exit
- [ ] **Replay-safe migrations** — every schema change must be idempotent (use `IF NOT EXISTS`, no destructive ops without confirm)

---

## What NOT to do (anti-patterns)

- ❌ **Don't run multiple model instances in parallel.** One BGE-small on CPU is faster than two fighting for cache.
- ❌ **Don't fetch READMEs serially.** Async + semaphore is the only way to approach the 5,000/hour limit.
- ❌ **Don't re-embed on every seed run.** Use `embedded_at` + repo `pushed_at` to decide if re-embedding is needed.
- ❌ **Don't sleep a fixed 2s between search calls.** Use `X-RateLimit-Remaining` to dynamically pace.
- ❌ **Don't store embeddings as JSON.** pgvector's `vector(384)` type is 4× smaller and 100× faster for ANN.
- ❌ **Don't trust search results for description quality.** Search truncates to 200 chars. README is the source of truth for embeddings.
- ❌ **Don't add Redis yet.** Postgres `FOR UPDATE SKIP LOCKED` is a queue. Add Redis when you have 10+ workers and need pub/sub.

---

## Immediate next step

Implement **Phase 1** — the two GitHub Actions workflows. Boring, fast, gets data flowing.
