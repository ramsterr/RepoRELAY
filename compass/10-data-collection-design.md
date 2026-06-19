# 10 — Data Collection Design

How to think about data for recommendation systems, specifically for RepoRelay. Written as if a senior ML engineer sat you down to explain what matters and what does not.

---

## The Mental Model

Forget tools for a moment. Think about what you are trying to answer:

> "Given that this user is looking at this repo *right now*, what other repos should they see?"

To answer that, you need three data classes:

| Class | The question it answers | Example |
|---|---|---|
| **What is the item?** | Does this repo smell like that one? | README text, topics, language |
| **What did users do?** | Do people who liked X also like Y? | Stars, forks, clicks |
| **How are items connected?** | Are these repos wired together? | Dependencies, contributors, workflows |

Everything else — metadata, context, timestamps — exists to make those three classes **more precise**.

---

## Data Taxonomy

### Class A — Entity Data (what things ARE)

These are your dimension tables. They describe items and users. Immutable or slowly-changing.

```
Repo
 ├── id, owner, name, full_name
 ├── created_at, updated_at, pushed_at
 ├── description
 ├── homepage
 ├── language (primary)
 ├── license
 ├── archived
 ├── fork (boolean)
 ├── is_template
 ├── default_branch
 └── raw_metadata (JSON blob for GitHub-specific fields)
```

| Field | What it enables |
|---|---|
| `created_at` | Freshness boost in re-ranking |
| `pushed_at` | Activity signal — stale repos get demoted |
| `language` | Slot filtering, ecosystem grouping |
| `license` | Business rules (some licenses restrict usage) |
| `archived` | Hard filter (never recommend) |
| `fork` | Demotion signal (forks are noise unless popular independently) |

```
User
 ├── id, login
 ├── created_at
 ├── type (User / Organization)
 └── followers_count (decays with lack of use)
```

```
Topic
 ├── name
 └── display_name
```

**How a senior engineer thinks about entity data:**

- Every field is "why do I need this?" — if a field has no downstream use in candidate gen, scoring, or re-ranking, delete it
- Nested JSON blobs (`raw_metadata`) are for **future extraction**, not current use. Extract what you need into typed columns
- Boolean fields (`archived`, `fork`) are **gates**, not features. Use them in re-ranking, not as a score weight

---

### Class B — Interaction Data (what people DID)

These are your fact tables. Every row is an event with (who, what, when). This is where **signal strength** lives.

```
StarEvent
 ├── user_id
 ├── repo_id
 ├── starred_at (timestamp)
 └── session_id (optional, gold)


ForkEvent
 ├── user_id
 ├── repo_id
 ├── forked_at
 └── downstream_repo_id (the fork itself — track if it gets commits)


ViewEvent (if you have a client that collects this)
 ├── user_id (or anonymous_id)
 ├── repo_id
 ├── source_repo_id (what page were they on when they viewed this?)
 ├── view_started_at
 ├── dwell_ms
 └── session_id
```

**Signal strength hierarchy for training:**

```
strongest ──────────────────────────────────────── weakest

   star         >    fork with commits   >   fork      >   scroll to end
   with recency      (not just clone)        event         of README
   decay                                          ↑
                                                  │
                                              cheap to collect
                                              noisy to learn from
```

**Why this hierarchy:**

A star is **declarative intent**. A fork with downstream commits is **execution intent** (they actually built something). A plain fork is weak — half are abandonments. A view is cheap but a view without follow-up is noise.

**The one signal most teams miss:**

```
session: user viewed X → starred Y within 5 minutes → starred Z within 5 minutes

This triplet (X, Y, Z) is a comparison session.
The user was shopping for a category.
Y is an alternative to X. Z is another alternative.

Treat (X, Y), (X, Z), (Y, Z) as positive training pairs.
```

---

### Class C — Relationship Data (how things are CONNECTED)

This is what makes RepoRelay special. Not just "are these similar?" but "do these live next to each other in the actual software ecosystem?"

```
DependencyEdge
 ├── repo_id (source — the repo that depends)
 ├── dependency_name (e.g., "react", "express")
 ├── ecosystem (npm, pypi, cargo, maven, go)
 ├── version_constraint (e.g., "^18.2.0")
 ├── is_dev (devDependency?)
 ├── file_path (which manifest file)
 └── extracted_at
```

**Why version constraint matters:**

If project A depends on `react@18.2.0` and project B depends on `react@18.3.0`, they are closer than if B depends on `react@16.8.0`. Version proximity encodes ecosystem maturity.

```
ContributorEdge
 ├── user_id
 ├── repo_id
 ├── commit_count
 ├── first_commit_at
 └── last_commit_at
```

| Feature derived from this | What it enables |
|---|---|
| Contributor overlap (Jaccard between two repos) | Two repos sharing maintainers are eco-neighbors |
| Active vs historical contributors | A repo with 50 historical contributors but 1 active maintainer is dying |

```
WorkflowCoOccurrence
 ├── repo_a_id
 ├── repo_b_id
 ├── co_occurrence_count
 └── workflow_file_path
```

Two repos appearing in the same `.github/workflows/ci.yml` means someone wired them together in CI. This is **production intent** — stronger than a blog post mention, weaker than a dependency.

```
ReadmeCrossReference
 ├── source_repo_id (the README that mentions)
 ├── target_repo_name (the repo being referenced)
 └── context (installed via, alternative to, inspired by)
```

Parsing READMEs for cross-references ("see also", "inspired by", "alternative to") gives you **curated relationship edges** that no algorithm can infer from stars alone.

---

## Data Structure: The Star Schema

In the warehouse, structure data as a star schema. This is how every data team at scale works.

```
                    ┌───────────────────────────────────────────────────┐
                    │                   FACT TABLES                     │
                    │                                                   │
                    │  star_events      fork_events      dep_edges      │
                    │  ┌──────────────┐ ┌──────────────┐ ┌───────────┐ │
                    │  │ user_id      │ │ user_id      │ │ repo_id   │ │
                    │  │ repo_id (FK) │ │ repo_id (FK) │ │ dep_name  │ │
                    │  │ starred_at   │ │ forked_at    │ │ ecosystem │ │
                    │  │ session_id   │ │ dest_repo_id │ │ version   │ │
                    │  └──────────────┘ └──────────────┘ └───────────┘ │
                    │                                                   │
                    │  contributor_edges      view_events               │
                    │  ┌──────────────┐    ┌──────────────────────┐    │
                    │  │ user_id      │    │ user_id              │    │
                    │  │ repo_id (FK) │    │ repo_id (FK)         │    │
                    │  │ commit_count │    │ source_repo_id (FK)  │    │
                    │  │ first/last   │    │ dwell_ms            │    │
                    │  └──────────────┘    │ session_id           │    │
                    │                      └──────────────────────┘    │
                    └──────────────────────┬────────────────────────────┘
                                           │
                    ┌──────────────────────┼────────────────────────────┐
                    │                  DIMENSION TABLES                  │
                    │                      │                            │
                    │  repos ──────────────┤────────── topics           │
                    │  ┌──────────────┐    │    ┌──────────────┐       │
                    │  │ id (PK)      │    │    │ name (PK)    │       │
                    │  │ full_name    │    │    │ display_name │       │
                    │  │ description  │    │    └──────────────┘       │
                    │  │ language     │    │                            │
                    │  │ license      │    │    users                   │
                    │  │ created_at   │    │    ┌──────────────┐       │
                    │  │ updated_at   │    │    │ id (PK)      │       │
                    │  │ pushed_at    │    │    │ login        │       │
                    │  │ stars        │    │    │ type         │       │
                    │  │ forks        │    │    │ created_at   │       │
                    │  │ archived     │    │    └──────────────┘       │
                    │  │ is_template  │    │                            │
                    │  └──────────────┘    │    readme_texts            │
                    │                      │    ┌──────────────┐       │
                    │                      │    │ repo_id (FK) │       │
                    │                      │    │ raw_text     │       │
                    │                      │    │ embedding    │       │
                    │                      │    │ (pgvector)   │       │
                    │                      │    └──────────────┘       │
                    └──────────────────────┴────────────────────────────┘
```

**Why star schema not normalized:**

- Joins on fact tables are fast (they have FK indices)
- You never join fact-to-fact (that is slow). You join fact-to-dimension
- Embedding vectors live in a separate table from repo metadata (faster ANN, cleaner separation)
- Graph edges should have their own schema in Apache AGE (not SQL JOINs for traversal)

---

## The Query Patterns (design backward from these)

Every piece of data you collect should serve at least one of these queries.

**Query 1: Semantic neighbors (candidate gen)**
```sql
SELECT repo_id, full_name, 1 - (embedding <=> query_embedding) AS similarity
FROM readme_texts
ORDER BY embedding <=> query_embedding
LIMIT 200;
```
→ Needs: `readme_texts` table with pgvector indexing

**Query 2: Graph neighbors (candidate gen)**
```cypher
MATCH (r:Repo {id: $repo_id})-[:DEPENDS_ON*1..2]-(neighbor:Repo)
RETURN neighbor.id, neighbor.full_name
LIMIT 200;
```
→ Needs: Graph edges in Apache AGE with index on edge type

**Query 3: Co-starred repos (item-based CF)**
```sql
SELECT s2.repo_id, r.full_name, COUNT(*) AS co_star_count
FROM star_events s1
JOIN star_events s2 ON s1.user_id = s2.user_id AND s1.repo_id != s2.repo_id
JOIN repos r ON s2.repo_id = r.id
WHERE s1.repo_id = $repo_id
GROUP BY s2.repo_id, r.full_name
ORDER BY co_star_count DESC
LIMIT 200;
```
→ Needs: `star_events` table indexed by (user_id, repo_id)

**Query 4: Contributor overlap (ecosystem neighbor)**
```sql
SELECT c1.repo_id AS repo_a, c2.repo_id AS repo_b,
       COUNT(DISTINCT c1.user_id) AS shared_contributors
FROM contributor_edges c1
JOIN contributor_edges c2 ON c1.user_id = c2.user_id AND c1.repo_id != c2.repo_id
WHERE c1.repo_id = $repo_id
GROUP BY c1.repo_id, c2.repo_id
ORDER BY shared_contributors DESC
LIMIT 200;
```
→ Needs: `contributor_edges` table

**Query 5: Session-based comparison pairs (gold training data)**
```sql
SELECT s1.repo_id AS anchor, s2.repo_id AS alternative, COUNT(*) AS session_count
FROM star_events s1
JOIN star_events s2 ON s1.session_id = s2.session_id
  AND s1.repo_id != s2.repo_id
  AND s2.starred_at - s1.starred_at BETWEEN '0 seconds' AND '5 minutes'
GROUP BY s1.repo_id, s2.repo_id
ORDER BY session_count DESC;
```
→ Needs: `star_events` with `session_id`

---

## How a Senior Engineer Chooses Data

### Rule 1: Collect everything. Use only what matters.

Storage is cheap. Recollection is expensive. If a data point costs you nothing to persist, keep it. You can decide what enters the model later.

```
COLLECT:       stars, forks, issues, PRs, commits, releases, READMEs, deps, contributors, topics
               (everything GitHub exposes)

USE IN MODEL:  stars (weighted), deps (weighted), contributors (Jaccard), READMEs (embedding)
               (only what downstream queries consume)

IGNORE:        watchers count, issue count, PR count (correlated with stars, adds noise)
               (let ablation testing prove you wrong later)
```

### Rule 2: Timestamps are mandatory.

Without timestamps, you cannot:
- Compute acceleration (stars/week, not total stars)
- Apply recency decay (a star from 2019 counts less than a star from yesterday)
- Detect stale repos (last pushed 3 years ago = hard filter)
- Build session windows (viewed X then starred Y within 5 minutes)

Every event table gets an `at` field.

### Rule 3: Collect negatives.

Most teams collect only what users did (stars, forks). They miss what users saw and ignored (impressions without clicks, views without follows). To train a model to rank, you need both.

```
positive: user starred repo A
negative: user viewed repo A and did NOT star it
          user viewed repo A and starred repo B instead
```

Without negatives, every pair looks good. The model cannot learn to discriminate.

### Rule 4: Session IDs unlock sequence learning.

A `session_id` on every event lets you ask:
- "What did they view before starring this?" → comparison set
- "What did they view after?" → follow-up intent
- "Did they bounce immediately?" → negative signal

Most teams skip this because it requires client-side tracking (browser extension, site JS). RepoRelay controls the browser extension — collect this.

### Rule 5: Graph data is a moat.

Anyone can collect stars and READMEs. Few collect dependency graphs, contributor overlap, and workflow co-usage. This is RepoRelay's competitive advantage. Invest disproportionately in graph data.

---

## Data Quality Gates

Before data enters the feature store, run these checks:

| Gate | What it catches | Action |
|---|---|---|
| **Staleness check** | Data older than expected refresh window | Alert, do not overwrite |  
| **Completeness check** | Missing required fields (e.g., no README) | Flag for manual review |
| **Edge consistency** | `DEPENDS_ON` edge from repo A to B exists, but no reverse edge | Auto-derive reverse edge |
| **Embedding coverage** | Repo has no embedding in vector store | Re-embed or mark as no-embedding |
| **Deduplication** | Same repo ingested twice (rename, transfer) | Merge into canonical entry |
| **Negative weight check** | Edge weight is negative | Investigate and zero out |

---

## What NOT to Collect (noise reduction)

| Do not collect | Why |
|---|---|
| Raw star count (without timestamp) | Cannot compute velocity |
| Aggregated stats (avg, max, sum per repo) | Store raw events, compute aggregates in pipeline |
| Issue count (without time-to-close) | Raw count is noise. Resolution speed is signal |
| Watchers count | 0.95 correlated with stars, adds no new information |
| PR count (without merge/close status) | Open PRs are noise. Merged PRs are signal |
| Release tags without dates | Chronology matters |

---

## Summary: Your Data Collection Checklist

```
Phase 1 (Seed — week 1):
  ├─ Repo metadata for top 100K repos (GitHub API)
  ├─ README text for those repos
  ├─ Star events for those repos (GitHub Archive)
  ├─ Topics and languages
  └─ Dependency manifests (Libraries.io)

Phase 2 (Graph — week 2-3):
  ├─ Contributor data per repo
  ├─ Build dependency edges (DEPENDS_ON)
  ├─ Build contributor edges (CONTRIBUTED_TO)
  └─ Build co-star edges (derived from star events)

Phase 3 (Rich — week 4+):
  ├─ README cross-references (parse READMEs for links)
  ├─ Workflow co-occurrence (parse .github/workflows/)
  ├─ Session-level star sequences (from GitHub Archive)
  └─ View events (from your browser extension once shipped)


Every event gets:
  ├── timestamp (always)
  ├── user_id (pseudonymized)
  ├── repo_id (normalized)
  └── source context (where did this event originate?)
```

---

## How This Feeds the Pipeline

```
DATA (what we just designed)
  │
  ├─ entity data ──────────▶ dim tables ──────▶ feature store
  │                                              (fast lookup)
  │
  ├─ interaction data ─────▶ fact tables ──────▶ offline training
  │                                              (label generation)
  │
  └─ relationship data ────▶ graph edges ──────▶ candidate gen
                                                   (graph traversal)
```

Data enters Postgres raw. Pipeline reads it, normalizes it, computes embeddings, derives edges, writes to the graph. The feature store holds the precomputed result. The serving API reads from the feature store. The raw data never touches the online serving path.
