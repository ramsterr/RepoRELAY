# Data Engineering — Micro Task Checklist

Ordered sequence. Each task is small enough to complete in one session.

---

## Phase 1: Schema Foundation (DDL)

- [x] `01.01` Create Alembic migrations directory in `packages/core/src/reporelay_core/migrations/`
- [x] `01.02` Write `CREATE TABLE repos` DDL (id, full_name, description, language, license, stars, forks, topics, created_at, updated_at, pushed_at, archived, is_template)
- [x] `01.03` Write `CREATE TABLE users` DDL (id, login, type, created_at)
- [x] `01.04` Write `CREATE TABLE topics` DDL (name, display_name)
- [x] `01.05` Write `CREATE TABLE languages` DDL (name)
- [x] `01.06` Write `CREATE TABLE readme_texts` DDL (repo_id FK, raw_text, embedding vector(768), embedded_at)
- [x] `01.07` Write `CREATE TABLE star_events` DDL partitioned by month (user_id, repo_id, starred_at)
- [x] `01.08` Write `CREATE TABLE fork_events` DDL (user_id, repo_id, forked_at)
- [x] `01.09` Write `CREATE TABLE contributor_edges` DDL (user_id, repo_id, commit_count, first_at, last_at)
- [x] `01.10` Write `CREATE TABLE dependency_edges` DDL (repo_id, dependency_name, ecosystem, version, is_dev)
- [x] `01.11` Write `CREATE TABLE workflow_cooccurrence` DDL (repo_a, repo_b, count)
- [x] `01.12` Write `CREATE TABLE user_blend_states` DDL (user_id, weights JSONB, stage, total_interactions)
- [x] `01.13` Write `CREATE TABLE two_hop_neighbors` DDL (source_repo_id, neighbor_repo_id, path_type, weight)
- [x] `01.14` Run `alembic upgrade head` — verify all tables exist in Postgres *pending Docker*

## Phase 2: Indexing (speed)

- [x] `02.01` Add B-tree index on `repos(language)`
- [x] `02.02` Add GIN index on `repos(topics)`
- [x] `02.03` Add B-tree index on `repos(stars DESC)`
- [x] `02.04` Add composite index on `star_events(user_id, repo_id)`
- [x] `02.05` Add composite index on `star_events(repo_id, starred_at DESC)`
- [x] `02.06` Add partial index on `star_events` WHERE `starred_at > now() - interval '2 years'`
- [x] `02.07` Add composite index on `contributor_edges(repo_id, user_id)`
- [x] `02.08` Add index on `dependency_edges(repo_id, dependency_name)`
- [x] `02.09` Add composite index on `two_hop_neighbors(source_repo_id, weight DESC)`
- [x] `02.10` Create HNSW index on `readme_texts(embedding vector_cosine_ops)` with `m=16, ef_construction=200`
- [x] `02.11` Run `EXPLAIN ANALYZE` on the top 5 queries from the data design doc — verify index usage

## Phase 3: Ingest — Single Repo (GitHub API)

- [x] `03.01` Extend GitHubClient with `get_topics(owner, name)` — fetch topic list
- [x] `03.02` Extend GitHubClient with `get_contributors(owner, name)` — fetch top 100 contributors
- [x] `03.03` Extend GitHubClient with `get_languages(owner, name)` — fetch language breakdown
- [x] `03.04` Extend GitHubClient with pagination helper (`get_all_pages`) for list endpoints
- [x] `03.05` Create `ingest save-repo owner name` CLI command — fetches repo, README, topics, languages, contributors
- [x] `03.06` Write `upsert_repo()` function — INSERT INTO repos ON CONFLICT UPDATE
- [x] `03.07` Write `upsert_readme()` function — INSERT into readme_texts
- [x] `03.08` Write `insert_topics()` function — INSERT INTO topics, link to repo
- [x] `03.09` Write `insert_contributors()` function — INSERT INTO contributor_edges
- [x] `03.10` Run `just ingest save-repo vercel next.js` — verify 1 row in repos, 1 row in readme_texts
- [x] `03.11` Run `just psql` → `SELECT full_name, stars, language FROM repos;` — verify data

## Phase 4: Seed Ingest — Batch

- [ ] `04.01` Create `ingest seed-topics` command — fetch top repos per topic from GitHub search API
- [ ] `04.02` Seed top 10 topics, 20 repos each = 200 repos
- [ ] `04.03` Create `ingest seed-languages` command — fetch top repos per language
- [ ] `04.04` Seed top 5 languages, 20 repos each = 100 repos
- [ ] `04.05` Run dedup across topic and language seeds
- [ ] `04.06` Verify total unique repos in database ≥ 200

## Phase 5: Dependency Graph Ingestion

- [ ] `05.01` Create Libraries.io API client (or parse manifest files directly from GitHub)
- [ ] `05.02` Write `detect_manifest(repo)` — find `package.json`, `requirements.txt`, `Cargo.toml` in repo
- [ ] `05.03` Write `parse_manifest(filename, content)` — extract `{name, version, ecosystem}`
- [ ] `05.04` Write `insert_dependencies()` — INSERT INTO dependency_edges
- [ ] `05.05` Run `ingest deps vercel next.js` — verify dependencies stored
- [ ] `05.06` Batch: ingest dependencies for all 200 seed repos
- [ ] `05.07` Verify dependency_edges total row count > 0

## Phase 6: Star Events from GitHub Archive

- [ ] `06.01` Research GitHub Archive BigQuery schema for star events
- [ ] `06.02` Write BigQuery SQL to extract star events for our seed repos from last 6 months
- [ ] `06.03` Export results as CSV/JSON
- [ ] `06.04` Create `ingest load-stars` command — bulk INSERT into star_events from file
- [ ] `06.05` Run load-stars on exported data
- [ ] `06.06` Verify `SELECT COUNT(*) FROM star_events` returns meaningful number

## Phase 7: Co-Star Materialized View

- [ ] `07.01` Write `co_star_counts` materialized view SQL
- [ ] `07.02` Run `CREATE MATERIALIZED VIEW co_star_counts`
- [ ] `07.03` Create index on `co_star_counts(repo_a, co_star_count DESC)`
- [ ] `07.04` Create `ingest refresh-co-stars` command — `REFRESH MATERIALIZED VIEW CONCURRENTLY`
- [ ] `07.05` Verify: query co_star_counts for a known repo, check results look reasonable

## Phase 8: Graph Setup (Apache AGE)

- [ ] `08.01` Verify AGE extension loaded: `SELECT * FROM ag_catalog.ag_graph;`
- [ ] `08.02` Create graph: `SELECT create_graph('reporelay');`
- [ ] `08.03` Create node label Repo in AGE: `SELECT create_vlabel('reporelay', 'Repo');`
- [ ] `08.04` Create node label User in AGE
- [ ] `08.05` Create edge label `DEPENDS_ON` with weight property
- [ ] `08.06` Create edge label `STARRED_BY` with starred_at property
- [ ] `08.07` Create edge label `CONTRIBUTED_TO` with commit_count property
- [ ] `08.08` Create edge label `HAS_TOPIC`
- [ ] `08.09` Write `load_to_age()` function — sync nodes and edges from Postgres tables into AGE graph
- [ ] `08.10` Run load_to_age on one repo — verify nodes and edges exist via Cypher query
- [ ] `08.11` Index AGE nodes by id, edges by start_id + type

## Phase 9: Two-Hop Neighbors (Precomputed)

- [ ] `09.01` Write Cypher query for 1-hop neighbors from a repo
- [ ] `09.02` Write Cypher query for 2-hop neighbors (neighbors of neighbors)
- [ ] `09.03` Create `ingest compute-2hop repo_id` command — runs Cypher, writes to `two_hop_neighbors` table
- [ ] `09.04` Run on one repo, verify data in two_hop_neighbors table
- [ ] `09.05` Batch compute 2-hop for all seed repos
- [ ] `09.06` Create `ingest refresh-2hop` command for periodic updates

## Phase 10: Embedding Pipeline

- [ ] `10.01` Choose embedding model (sentence-transformers all-MiniLM-L6-v2 for MVP)
- [ ] `10.02` Add model dependency to `packages/engine/pyproject.toml`
- [ ] `10.03` Write `compute_embedding(text) -> list[float]` function
- [ ] `10.04` Create `ingest embed-readme repo_id` command — reads raw_text, computes embedding, stores in readme_texts
- [ ] `10.05` Run on one repo, verify `SELECT vector_dims(embedding) FROM readme_texts` returns 384
- [ ] `10.06` Batch embed all repos with READMEs
- [ ] `10.07` Create `ingest embed-all` command
- [ ] `10.08` Run ANN test query: `SELECT repo_id, 1 - (embedding <=> $vec) AS sim FROM readme_texts ORDER BY sim DESC LIMIT 10`
- [ ] `10.09` Manually verify: do the top 10 results actually make sense for the query repo?

## Phase 11: Redis Caching Layer

- [ ] `11.01` Write `cache_repo_features(repo_id)` — store feature vector in Redis with TTL=6h
- [ ] `11.02` Write `get_cached_features(repo_id)` — return from Redis or None
- [ ] `11.03` Write `cache_blend_state(user_id)` — store user blend in Redis with TTL=24h
- [ ] `11.04` Write `get_cached_blend(user_id)` — return from Redis or None with Postgres fallback
- [ ] `11.05` Write `cache_recommendations(key, result)` — store rec response with TTL=5min
- [ ] `11.06` Write `get_cached_recommendations(key)` — return from Redis or None
- [ ] `11.07` Integrate cache checks into API `/recommend` endpoint
- [ ] `11.08` Test: run same request twice, verify second call hits cache (use Redis MONITOR)

## Phase 12: Connect Engine to Real Data

- [ ] `12.01` Replace `ContentBasedStrategy._fetch_by_similarity` with real pgvector ANN query
- [ ] `12.02` Replace `ItemBasedCFStrategy._fetch_by_similarity` with real co_star_counts query
- [ ] `12.03` Replace `UserBasedCFStrategy._fetch_by_similarity` with real user similarity query
- [ ] `12.04` Replace `ExplorationStrategy._fetch_by_similarity` with real trending query (stars/time)
- [ ] `12.05` Integrate Redis caching into each strategy (check cache before DB)
- [ ] `12.06` Run `just api` → call `GET /recommend?repo=vercel/next.js` → verify non-empty slots
- [ ] `12.07` Manually review: do the returned recommedations make sense?
- [ ] `12.08` Test feedback loop: call `POST /feedback` a few times, verify blend state changes

## Phase 13: Refresh Pipeline

- [ ] `13.01` Create `ingest refresh-repos` command — re-fetch metadata for repos updated in last 24h
- [ ] `13.02` Create `ingest refresh-stars` command — fetch new star events from Archive
- [ ] `13.03` Create `ingest refresh-embeddings` — re-embed repos with changed READMEs
- [ ] `13.04` Create hourly cron / scheduled job for refresh pipeline
- [ ] `13.05` Add staleness tracking table (`refresh_log`) — last refresh time per data source
- [ ] `13.06` Add health check endpoint: `GET /health/data-freshness` — returns staleness per source

---

## Phase 14: Observability

- [ ] `14.01` Add query latency logging for all DB operations
- [ ] `14.02` Add Redis hit rate tracking (cache hits / total requests)
- [ ] `14.03` Add end-to-end latency tracking for `/recommend` (p50, p99)
- [ ] `14.04` Add embedding coverage metric (% repos with embeddings)
- [ ] `14.05` Add graph coverage metric (% repos with edges)
- [ ] `14.06` Add recommendation coverage metric (% catalog surfaced in last week)
- [ ] `14.07` Set up PgHero or similar for Postgres query monitoring
- [ ] `14.08` Write slow query alerts (any query > 100ms)

---

## How to Use This

1. Start at Phase 1, task `01.01`. Do not skip ahead.
2. Each task should be one commit.
3. Run `just test` after every Phase.
4. Phases 1-3 give you a working single-repo pipeline. Ship that first.
5. Phases 4-6 give you a seed dataset. That is enough to test recommendations.
6. Phases 7-12 connect the engine to real data. That is your first end-to-end demo.
7. Phases 13-14 are production hardening. Do them after the demo works.
