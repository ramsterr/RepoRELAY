# TODO

Tracked follow-ups from the architecture review. Three groups: **CORE** addresses the data-flow issues (the recommender is starving for data), **PUBLIC-FACING** is the work needed before opening the site to real users, **ADDING NEW REPOS** is the work to grow the indexed corpus past the initial 3,000.

---

## CORE — fix the corpus

The current code uses the local DB as the candidate source, with a hand-seeded pool of 4–8 repos. GitHub search is only a fallback and its results are thrown away. Flip the architecture so GitHub is the corpus and the DB is a cache.

- [x] **Flip the corpus** — rewrite `packages/mvp/src/reporelay_mvp/recommend.py` so GitHub search is the primary candidate source, not the DB fallback
- [x] **Fix the `search_repos` query** in `packages/mvp/src/reporelay_mvp/github.py:170-181` — OR the source's topics instead of picking one, paginate to 200+ candidates, add `archived:false`, lower the `stars:>100` floor (or make it a parameter)
- [x] **Persist search hits** — add a `search_fetched_at` timestamp column to `mvp_repos` and upsert results from `search_repos` so the DB grows from real queries instead of manual `save` commands
- [x] **Drop the hardcoded `0.5` cosine** for ephemeral candidates (`recommend.py:138`) — embed them at query time on the top N, or use the search API's relevance as a proxy
- [x] **Fix the silent tag-filter fallback** in `packages/mvp/src/reporelay_mvp/candidates.py:80-84` — when the filter eliminates everything, the code currently keeps the unfiltered list with only a log line; either surface this to the caller or remove the fallback
- [x] **Collapse the duplicate buttons** in `apps/site/src/pages/repo/[owner]/[name].astro:137-145` — "rerun with new seed" and "different results" run identical handlers
- [x] **Remove the seed-time illusion** — the 8 example repos on `apps/site/src/pages/index.astro:4-13` only exist because the recommender can't function with an empty DB; once the corpus is real, replace with a "trending" list

---

## PUBLIC-FACING — needed before opening the site to real users

Treat GitHub as a **background data source**, not a real-time dependency. The request path should never block on it.

- [ ] **Swap to a GitHub App** — 12,500 req/hr per installation vs 5,000 for a personal token, plus a 30 req/sec burst on REST
- [ ] **Add Redis** as a hot cache in front of Postgres — keys: `rec:{owner}/{name}` (TTL ~12h) and `search:{lang}:{topic}` (TTL ~24h)
- [ ] **Background worker** that pre-computes recommendations for the top ~1,000 repos every 6–12 hours — this absorbs 99% of traffic before it ever hits GitHub
- [ ] **Stale-while-revalidate on cache miss** — return a fallback list (top stars in the source's language) instantly, enqueue a compute job, next request serves the real result
- [ ] **Edge-cache the HTML pages** — set `Cache-Control: public, s-maxage=300, stale-while-revalidate=3600` and put the site behind a CDN (Cloudflare / Vercel / CloudFront)
- [ ] **Per-IP rate limiting at the edge** — protect the origin from a single user spamming the recompute path
- [ ] **Graceful degradation** — if GitHub is down or the worker queue is full, the site still serves *something* (cached results, popular fallback, "try one of these" list)

---

## ADDING NEW REPOS — grow the corpus past 3,000

The seed script (`packages/mvp/src/reporelay_mvp/seed.py`) is the workhorse. Currently 3,000 repos are indexed; the steps below push that to ~20,000 with real embeddings. The single biggest quality gap right now is that **none of the 3,000 repos have embeddings** — pgvector ANN is doing nothing useful. Until the embed pass runs, the recommender leans on language/topic/popularity, not content similarity.

- [x] **Embed the top 1,000 repos** — fetch each `search_fetched_at IS NOT NULL AND embedded_at IS NULL` row, download the README, run `embed_text()`, store the 384-dim vector. Unlocks pgvector ANN. CLI: `just mvp embed --limit 1000` (DONE — 1,011 embedded, puppeteer→playwright, tensorflow→caffe, redis→valkey)
- [ ] **Scale existing 10 languages to 1,000 each** — `just mvp seed --per-language 1000` brings 3,000 → 10,000 in ~10 min (100 search calls at 30 req/min)
- [ ] **Add 10 more languages** — `just mvp seed --per-language 1000 --languages kotlin,swift,dart,scala,elixir,haskell,julia,lua,r,perl` adds another ~10,000 rows
- [ ] **Multi-topic search in `_expand_pool`** — currently uses only the source's first topic. Switch to one search per top-4 topic, merge the result lists (~30 lines in `github.py` + `recommend.py`). Uses 4 of the 30 req/min budget per request but gives proper OR-of-topics semantics
- [ ] **Schedule a background re-index** — cron or systemd timer that runs `just mvp seed --per-language 200 --page-delay 2.5` every 6 hours to refresh `search_fetched_at` and add anything newly trending
- [ ] **Quality gating in the search query** — add `pushed:>2025-01-01` and `is:public` to the `search_repositories` query in `github.py:218-248` so dead/forked repos don't enter the corpus
- [ ] **Topic-based discovery** (separate from language) — add a second `seed` mode that walks `topic:react`, `topic:web-framework`, `topic:machine-learning`, etc. instead of `language:X`. ~50–100 topics × 200 repos = 10k–20k topic-driven rows
- [ ] **Embed pass for newly-searched rows** — when `_expand_pool` persists a fresh search hit, also embed the README inline (or queue it for the next pass) so the cosine sim feature starts working for new repos within minutes of first discovery
