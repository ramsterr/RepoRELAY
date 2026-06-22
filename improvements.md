# Improvements

A prioritized audit by severity, not by file. Each item has a **certainty score** (1-100) — how sure I am that implementing it makes the system strictly better than the alternative.

Certainty below 70 means "strong opinion but depends on data". Above 90 means "there is no valid counterargument."

---

## 🔴 P0 — Ship Blockers

Do these before putting the site in front of more users. Each one is a real correctness, security, or availability problem.

| # | Issue | Certainty | Effort | What to do |
|---|---|---|---|---|
| 1 | **Zero test coverage** — not one test file in the repo. A 5-stage ML pipeline, a web API, and a JS frontend, all untested. | 100 | 3 days | Golden-set test: curate 10 `(source_repo → expected_result_set)` pairs, add `pytest` tests that call `recommend()` and assert the set. Unit-test `features.py` (pure math). Contract-test the API: POST valid/invalid inputs, check status codes and response shape. |
| 2 | **`_find_cosine` returns 0.5 for no-embedding candidates** — `_cosine_lookup.get(repo.id, 0.5)` gives zero-embedding repos a neutral similarity. This 0.5 is then weighted at 30% in the score, systematically inflating repos without READMEs. | 95 | 5 min | Return `0.0` instead of `0.5`. Repos with no embedding carry NO content signal. |
| 3 | **`_cosine_lookup` is a hidden global mutable** — `_build_cosine_lookup` writes to a module-level dict and `_find_cosine` reads it. Under concurrent requests, request A can mutate the dict mid-request-B, producing wrong similarity scores in the API response. | 90 | 30 min | Delete the global. Pass `cosine_lookup` as a return value from `_build_cosine_lookup` and thread it through to the response builder as a parameter. |
| 4 | **Duplicate constant** — `_MIN_DB_POOL_FOR_SKIP = 200` is defined at line 60 and again at line 140 of `recommend.py`. The second definition silently shadows the first. | 100 | 1 min | Delete line 140. |
| 5 | **Rate-limit retry blocks the uvicorn worker for 3.5+ minutes** — `@retry(wait=wait_exponential(min=30, max=600))` on the GitHub REST call means a rate-limited `/recommend` hangs the sole worker for 30s → 60s → 120s. On Render's free plan (1 worker), that's your whole API down. | 95 | 1 hr | Replace with: fail fast (return 503 `Retry-After: 60`), set `X-RateLimit-Remaining` in the response, and let the frontend show "API busy, try again." Add a circuit-breaker on the client side. |
| 6 | **CORS `allow_origins=["*"]`** — any origin can call `/recommend` and burn your GitHub token. | 100 | 5 min | Lock to `["https://reporelay-site.vercel.app", "http://localhost:4321"]`. |
| 7 | **No request-level result caching** — every `/recommend` re-runs the full 5-stage pipeline. GitHub search results are cached, but the scored/re-ranked output is not. For a popular repo, this wastes DB queries and CPU. | 95 | 2 hr | Add `rec:{owner}/{name}:{seed}:{tags_md5}` cache dict with 10-min TTL. On hit, return the scored list directly. On miss, compute and store. Memory cost: ~1000 keys × ~20KB = 20MB. |
| 8 | **No rate limiting** — one user can hammer `/recommend` 1000 times and burn the GitHub token's 5000 req/hr for everyone. | 90 | 2 hr | Add `slowapi` per-IP token bucket: 10 req/min for `/recommend`, 30 req/min for other endpoints. |
| 9 | **Embedding model loads async, races with first request** — `asyncio.create_task(preloadModel())` in `lifespan` starts the model load in the background. The first real `/recommend` request may arrive before the model is ready, producing recommendations with `cosine_sim=0` for the source (no content signal). | 90 | 30 min | Either: (a) `await preloadModel()` in `lifespan` (block startup until ready — adds ~11s to startup, worth it), or (b) return 503 `Retry-After: 30` on `/recommend` until the model flag flips. |
| 10 | **`/explore` requires a `seed` query param** — the API returns a 422 if `seed` is missing. The frontend always passes it, but the API contract is wrong. REST APIs should handle missing optional params. | 100 | 2 min | Make `seed` optional (`int | None = Query(None)`), fall back to `random.randint()` if not provided. |
| 11 | **`db.py` engine is never disposed on shutdown** — `_engine` is created once and held forever. FastAPI's `app.on_event("shutdown")` needs to call `await _engine.dispose()` or you leak connections. | 85 | 15 min | Add a shutdown event to the FastAPI `lifespan` that calls `await _engine.dispose()`. |

---

## 🟠 P1 — Architecture & Reliability

These affect correctness at scale, cold-path latency, or code maintainability. Fix them in the next sprint.

| # | Issue | Certainty | Effort | What to do |
|---|---|---|---|---|
| 12 | **`/recommend` sync-blocks for cold repos** — when a repo is not in the DB, `quick_save()` fires a real GitHub API call (`metadata + topics`, ~2s p99) before returning anything. The proxy embedding is a good idea, but the metadata fetch still blocks the response. | 90 | 1 hr | Fire the `quick_save` in a `asyncio.create_task` and respond immediately with the proxy-embedding + existing DB candidates. Stream-enrich the results via SSE or a polling token. |
| 13 | **`recommend()` is 60 lines doing 7 things** — cache, session lifecycle, quick-save, proxy embedding, candidate gen, scoring, rerank, response shaping. Violates SRP and makes changes risky. | 80 | 2 hr | Extract `_load_or_fetch_source()`, `_build_pool()`, and `_score_and_rerank()` as separate functions. The top-level `recommend()` becomes 15 lines. |
| 14 | **`/popular` ignores `trending_score` in `ORDER BY`** — the endpoint orders by `stars DESC` and includes `trending_score` as an unused column. The frontend fetches 24, sorts client-side by `trending_score` to build the "trending" card. The backend and frontend disagree about what "popular" means. | 90 | 5 min | Change the `/popular` query to `ORDER BY COALESCE(trending_score, 0) DESC, stars DESC`. Or create a separate `/trending` endpoint. The frontend shouldn't need to hack around the backend. |
| 15 | **`quick_save` skeleton-row ID uses `hash()` which can collide** — `abs(hash("owner/name")) % (10**9)` conflates with real GitHub IDs. If the skeleton repo has the same `id` as a real GitHub repo, `upsert_repo` with `ON CONFLICT (id)` overwrites the real one. | 85 | 15 min | Use a reserved ID range (e.g., negative IDs, or a separate column like `is_skeleton BOOLEAN`) so real GitHub IDs never collide with synthetic ones. |
| 16 | **`enrich_repo` swallows errors silently** — `asyncio.create_task(enrich_repo(...))` runs in the background with no error propagation. If the GitHub call fails, the source repo stays in the DB with a zero embedding forever, and the system never retries. | 85 | 1 hr | Log the error. Add `enrichment_attempts` and `enrichment_failed_at` columns to `mvp_repos`. After 3 failed attempts, stop retrying. Add a CLI command `enrich --retry-failed` for manual recovery. |
| 17 | **Redis container in `docker-compose.yml` is never used** — the code commentary everywhere says "no Redis," but there's a Redis container defined. Either wire it up for rec-caching and rate limiting, or remove it to avoid confusion. | 95 | delete: 1 min / use: 4 hr | If keeping: use it for the rec-cache (replace in-memory dict) and rate limiting. If not: remove the service block. |
| 18 | **5 pipeline stages run sequentially when 2 can parallelize** — `generate_candidates` (DB query) and `_cached_search` (GitHub API) don't depend on each other. Running serially makes cold paths 50% slower. | 90 | 1 hr | `asyncio.gather(db_pool_task, github_search_task)`, merge results after both complete. |
| 19 | **`_load_disk_cache` re-parses the full JSON file on every miss** — after a month of searches, the cache file is large and every miss triggers a full parse. | 85 | 1 hr | Either: (a) use `sqlite3` for the search cache (atomic writes, indexed lookups, no full-parse), or (b) keep the in-memory dict and flush to disk periodically, loading only once at startup. |
| 20 | **`_cached_search_results` is unbounded** — keys are never evicted except on TTL check during access. After 10,000 unique searches, the dict holds 10,000 entries forever. | 85 | 30 min | Use a `collections.OrderedDict` with maxsize, or use `cachetools.TTLCache`. |
| 21 | **`db.py` uses `pool_recycle=60` — too aggressive for a keepalive-pinged server** — the keepalive cron pings every 10 minutes, which prevents idle-pool disconnection. A 60-second `pool_recycle` on Render (which can have network hiccups) causes unnecessary reconnects. | 80 | 2 min | Set `pool_recycle=1800` (30 min) and rely on `pool_pre_ping=True` for liveness. |
| 22 | **"one per owner" rerank is too aggressive for niche queries** — for "Python web framework," dropping 9 of 10 top candidates because they're from different owners means the user gets fewer results. The rule is correct for generic queries but harms precision for specific ones. | 75 | 1 hr | Add a `diversity` parameter: `strict` (1 per owner, current), `loose` (max 2 per owner), `none` (no limit). Default to `strict` for `/explore` (browsing), `loose` for `/recommend` (specific). |

---

## 🟡 P2 — Product & Observability

Things users see or things that help you debug when they break.

| # | Issue | Certainty | Effort | What to do |
|---|---|---|---|---|
| 23 | **No Open Graph / Twitter card meta tags** — sharing a link on Slack, Twitter, or Discord shows a blank preview. Free SEO and distribution channel. | 100 | 30 min | Add `<meta property="og:title" content="RepoRelay — find repos worth your time">` and similar in `Base.astro`. |
| 24 | **`cachedFetch` is duplicated across 3 `.astro` pages** — ~100 lines of identical JS for fetch/timer/retry/sessionStorage logic in `index.astro`, `explore.astro`, and `repo.astro`. | 95 | 1 hr | Extract into `/src/lib/fetch.js` (or embed in `Base.astro` as a shared inline script) and reference from each page. |
| 25 | **No metrics endpoint** — no `/metrics` for Prometheus, no request counts, no latency histograms. When a user reports slowness, you have no data to confirm or disprove. | 90 | 2 hr | Add `prometheus-fastapi-instrumentator` middleware. Expose `/metrics` behind a simple auth header. At minimum: request count, p50/p90/p99 latency, error rate per endpoint. |
| 26 | **No error tracking** — no Sentry or Rollbar. When `/recommend` throws a 500, you never know until a user reports it. | 90 | 30 min | Add `sentry-sdk` in `main.py` with `SENTRY_DSN` env var. Free tier covers your volume. |
| 27 | **No structured logging or request IDs** — when a request is slow, you can't correlate the DB query, GitHub call, and scoring stages to find the bottleneck. | 85 | 2 hr | Add `structlog` (or `logger.info("stage", extra={"stage": "candidates", "duration_ms": dt})`). Assign a `request_id` UUID in middleware, pass through log context. |
| 28 | **No database backup** — Render free tier has no automated backups for Postgres. A bad `alembic` migration or an `ALTER TABLE` mistake = irreversible data loss. | 90 | 1 hr | Add `pg_dump -Fc` cron (nightly) that uploads to a private S3 bucket or Render disk, with a `restore-db` justfile recipe. |
| 29 | **Search input doesn't autocomplete from the DB** — typing "fast" could show "fastapi/fastapi," "fastify/fastify," etc. from the local index. Massive UX win. | 85 | 2 hr | Add a `/search?q=fast` endpoint that does `SELECT full_name FROM mvp_repos WHERE full_name ILIKE 'fast%' ORDER BY stars DESC LIMIT 5`. Wire it into the search-bar input as a `<datalist>` or custom dropdown. |
| 30 | **No "why was this recommended?" expanded view** — the `matchLabel` badge is shown but tiny. Clicking a card to expand the feature breakdown (the 6-bar chart) educates users AND increases trust in the recs. | 80 | 2 hr | Add a `[+] why?` button on each card. On click, expand an inline panel showing the 6 feature bars (language, topics, cosine, deps, popularity, trending) as horizontal green bars with labels. |
| 31 | **Art picker renders 12 × 64 = 768 SVG rects on page load** — fine on desktop, janky on mid-range mobile. | 80 | 1 hr | Lazy-render the previews only when the art picker panel is opened. Generate the preview SVG in the `open()` function. |
| 32 | **First-paint flash on theme load** — the pre-paint `<script>` sets `data-theme` before the CSS loads, but CSS transitions run on the first frame, producing a visible flash. | 85 | 30 min | Add a `class="no-transition"` on `<html>` that's removed after the first `requestAnimationFrame`. The CSS rule `.no-transition *, .no-transition *::before { transition: none !important; }` disables all transitions on first paint. |
| 33 | **No PWA / service worker** — a "trending repos" app is a perfect PWA. Users on slow networks get blank screens while waiting for the API. | 75 | 4 hr | Add a `manifest.json` + basic service worker that caches the HTML shell (`index.html`, `explore.html`, CSS) and fetches API data on top. Adds offline-first feel. |
| 34 | **No JSON-LD structured data** — a recommender could expose `ItemList` schema. Google sees nothing semantic about your content. | 70 | 30 min | Add `application/ld+json` script tag on homepage with `ItemList` schema listing the trending repos. |
| 35 | **No `/trending` endpoint exists** — the frontend's "trending on github" card fetches `/popular`, sorts client-side by `trending_score`, and takes the top 8. That's a client-side hack around a missing backend feature. | 85 | 15 min | Add a `/trending?limit=8` endpoint that orders by `trending_score DESC NULLS LAST, stars DESC`. Make `/popular` mean "popularity" and `/trending` mean "velocity." |
| 36 | **Keepalive cron has hardcoded Render URL** — `.github/workflows/ping.yml` hardcodes `reporelay-mvp-api-0w1k.onrender.com`. If Render rotates the URL, the keepalive breaks silently. | 80 | 5 min | Move the URL to a GitHub Secret (`RENDER_URL`) or pass it from the Render env. |

---

## 🔵 P3 — Code Quality & Polish

Not urgent, but each one makes the codebase easier to onboard onto, test, and ship from.

| # | Issue | Certainty | Effort | What to do |
|---|---|---|---|---|
| 37 | **Module-level mutable state everywhere** — `_cached_search_results`, `_SEARCH_CACHE`, `_cosine_lookup`, `_idf`, the model singleton, the engine singleton, `_sessionmaker`. Each is a hidden dependency; each makes testing require global reset. | 80 | 1 day | Encapsulate state in classes with explicit lifecycle: `SearchCache` class, `ScoreService` class, `EmbeddingService` class. Instantiate in `lifespan` and inject via `request.app.state`. |
| 38 | **Type hints are inconsistent** — `list[str] \| None`, `Optional[List[str]]`, bare `list[str]` with `if x is not None` all appear in the same file. | 85 | 1 hr | Run `ruff check --select UP037` and standardise on `list[str] \| None`. |
| 39 | **GitHub topics use a deprecated API preview** — `application/vnd.github.mercy-preview+json` was deprecated in 2020. Topics are now in the main `/repos/{owner}/{name}` response under `"topics"`. | 90 | 30 min | Drop the preview header. Read topics from `metadata["topics"]` in `fetch_repo_metadata` instead of making a separate round-trip. Save one GitHub API call per repo. |
| 40 | **No API versioning** — `/recommend` vs `/v1/recommend`. If you change the response shape, existing clients break. | 80 | 1 hr | Prefix all routes with `/v1` and add a redirect from the old paths. |
| 41 | **No shutdown cleanup** — the `AsyncEngine` created in `db.py` is never `await engine.dispose()`. On Render restarts, you leak idle connections. | 85 | 15 min | Add `await _engine.dispose()` in a `shutdown` event in the FastAPI `lifespan`. |
| 42 | **No slow query log** — pgvector ANN queries, topic unnest, and full-table scans have no monitoring. | 80 | 30 min | Set `log_min_duration_statement = 100ms` on the Render-hosted Postgres. |
| 43 | **No request body size limit on FastAPI** — fine for GETs, but the webhook endpoint reads `await request.body()` unbounded. | 75 | 5 min | Add `request.body(max_length=10*1024*1024)` to the webhook handler. |
| 44 | **No A/B testing infra around the `seed`** — seeds are the perfect A/B mechanism (deterministic, per-user, reproducible), but there's no tracking of which seeds lead to clicks. | 70 | 1 day | Log `seed` + `repo` + `user_id` (anonymous session) on every `/recommend` call. Track click-through on results. After 1000 requests, you can measure whether jittered weights produce more diverse click distributions. |
| 45 | **No negative feedback tracking** — a repo that's been recommended 100 times and clicked 0 times should eventually be penalized. | 70 | 2 days | Add a `clicked` event to the frontend. Log `source_repo, candidate_repo, clicked_bool` to a DB table. After enough data, feed it into the scoring loop as a dampening factor. |
| 46 | **The `/recommend` response ships the full `features` dict** — each response carries language_match, topic_overlap, cosine_sim, dep_overlap, popularity_sim, trending_boost, quality_signal, language_diversity, and filter_cosine_sim. The frontend only reads 5 of 9 in `matchLabel`. The others are dead bytes. | 80 | 10 min | Trim to the 5 features the frontend reads, or make them `?verbose=1`. |
| 47 | **No `CONTRIBUTING.md`** — if this goes public, contributors need setup instructions, code style, and PR conventions. | 75 | 30 min | Write a one-page CONTRIBUTING.md with justfile recipes, lint commands, and PR template. |
| 48 | **The favicon is the default Astro SVG** — a custom favicon for RepoRelay would add polish. | 60 | 1 hr | Generate a simple green-tinted favicon (the repo icon or a "RR" mono). |
| 49 | **The `just mvp` CLI has 8 commands listed but 2 are undocumented** — `register-webhooks` and `explore` appear in the code but not in all documentation surfaces. | 80 | 10 min | Audit the justfile, README, and DEPLOY.md for missing CLI commands. Add docstrings in `cli.py` that `just mvp --help` displays. |

---

## Summary by certainty

| Certainty ≥ 95 | Issues |
|---|---|
| 100 | Zero test coverage, duplicate constant, CORS wildcard, `/explore` requires seed, Open Graph meta tags, cachedFetch duplication |
| 95 | `_find_cosine` returns 0.5, rate-limit retry blocks worker, no result caching, API rate limiting, model load race, Redis container unused |

These 11 items are the highest-leverage changes you can make — each one is virtually guaranteed to improve the system with no downside. If you only do 5 things from this entire document, do numbers 1, 2, 5, 7, and 8.
