# RepoRelay — TODO

## Gonna start here

- [ ] Initialize project repository (monorepo structure)
- [ ] Choose and set up tech stack (language, framework, database)
- [ ] Configure development environment (linting, formatting, CI)
- [ ] Set up environment variables and secrets management
- [ ] Write project conventions doc (branch naming, commit style)

## Design Decisions

- [x] Add framework choice (Astro over Next.js) to considerations.md
- [ ] Document future stack/framework decisions in considerations.md as they arise

## Data Pipeline

- [ ] Set up GitHub API client (auth, rate limiting, pagination)
- [ ] Ingest seed data: top N repos per topic/language
- [ ] Ingest repo metadata (topics, stars, forks, language, license)
- [ ] Ingest README files (for embeddings / NLP)
- [ ] Ingest dependency manifests (`package.json`, `Cargo.toml`, `requirements.txt`, etc.)
- [ ] Ingest contributor data (committers, PR authors per repo)
- [ ] Ingest co-starring patterns (users starring multiple repos)
- [ ] Ingest workflow files (CI config, GitHub Actions)
- [ ] Build incremental / periodic refresh pipeline (stay fresh)
- [ ] Store raw data in a relational store (Postgres / SQLite)

## Graph Database

- [ ] Choose graph DB (Neo4j, NebulaGraph, ArangoDB, or pgRouting on Postgres)
- [ ] Design graph schema:
  - [ ] Nodes: `Repo`, `User`, `Topic`, `Language`, `Dependency`
  - [ ] Edges: `DEPENDS_ON`, `STARRED_BY`, `CONTRIBUTED_TO`, `HAS_TOPIC`, `CO_OCCURS_IN_WORKFLOW`, `IS_ALTERNATIVE_TO`, etc.
- [ ] Ingest data into graph DB from raw store
- [ ] Compute and write edge weights / strengths
- [ ] Index graph for fast traversal queries
- [ ] Verify graph consistency with smoke tests