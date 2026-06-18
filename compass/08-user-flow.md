# 08 — User Flow Diagrams

Visual trace of what happens when someone uses RepoRelay. Each diagram is shown in Mermaid (renders in GitHub/docs) and ASCII (works everywhere).

---

## 1. End-to-End User Journey

What the user does, and what happens behind the scenes at each step.

### Mermaid

```mermaid
sequenceDiagram
    participant U as User
    participant S as Discovery Site (Astro)
    participant A as Serving API (FastAPI)
    participant F as Feature Store
    participant E as Recommendation Engine
    participant DB as Postgres + pgvector + AGE

    U->>S: Enter "vercel/next.js" in search box
    S->>A: GET /recommend?repo=vercel/next.js&limit=10
    A->>F: Fetch precomputed features for next.js
    F->>DB: Query embeddings, graph edges, co-stars
    DB-->>F: Feature vectors + graph neighbors
    F-->>A: Structured features
    A->>E: Run recommendation pipeline
    E->>E: Candidate generation (ANN + graph)
    E->>E: Scoring (two-tower + cross features)
    E->>E: Re-ranking (dedup, diversity, rules)
    E-->>A: Ranked repos per slot
    A-->>S: JSON response (6 slots, ~10 repos each)
    S-->>U: Render recommendation cards
    U-->>U: Browse alternatives, addons, companions...
```

### ASCII

```
 USER                  SITE (Astro)              API (FastAPI)         FEATURE STORE         ENGINE              DB
  │                        │                         │                     │                    │                  │
  │  1. Enter repo name    │                         │                     │                    │                  │
  │───────────────────────>│                         │                     │                    │                  │
  │                        │  2. GET /recommend       │                     │                    │                  │
  │                        │────────────────────────>│                     │                    │                  │
  │                        │                         │  3. Fetch features  │                    │                  │
  │                        │                         │────────────────────>│                    │                  │
  │                        │                         │                     │  4. Query DB       │                  │
  │                        │                         │                     │──────────────────────────────────────>│
  │                        │                         │                     │                    │                  │
  │                        │                         │                     │  5. Return features│                  │
  │                        │                         │                     │<──────────────────────────────────────│
  │                        │                         │  6. Features ready  │                    │                  │
  │                        │                         │<────────────────────│                    │                  │
  │                        │                         │                     │                    │                  │
  │                        │                         │  7. Run pipeline    │                    │                  │
  │                        │                         │─────────────────────────────────────────>│                  │
  │                        │                         │                     │                    │                  │
  │                        │                         │                     │  8. Ranked results │                  │
  │                        │                         │<─────────────────────────────────────────│                  │
  │                        │  9. JSON response       │                     │                    │                  │
  │                        │<────────────────────────│                     │                    │                  │
  │  10. Render cards      │                         │                     │                    │                  │
  │<───────────────────────│                         │                     │                    │                  │
  │                        │                         │                     │                    │                  │
```

---

## 2. System Architecture

All components and how they wire together.

### Mermaid

```mermaid
graph TB
    subgraph "Data Sources"
        GH[GitHub API]
        NPM[NPM/PyPI/Cargo]
    end

    subgraph "Offline Pipeline"
        ING[Ingest CLI]
        PL[Data Pipeline<br/>batch + stream]
    end

    subgraph "Storage"
        PG[(Postgres + pgvector<br/>+ Apache AGE)]
        REDIS[(Redis Cache)]
    end

    subgraph "Feature Store"
        FS[Feature Store<br/>+ Embeddings]
    end

    subgraph "Recommendation Engine"
        CG[Candidate Generation<br/>ANN + Graph Traversal]
        SC[Scoring<br/>Two-Tower + Cross Features]
        RR[Re-ranking<br/>Dedup + Diversity + Rules]
    end

    subgraph "Serving"
        API[FastAPI<br/>/recommend]
    end

    subgraph "Product Surfaces"
        EXT[Browser Extension]
        GHA[GitHub App]
        SITE[Discovery Site]
        IDE[IDE Plugin]
    end

    GH --> ING
    NPM --> ING
    ING --> PL
    PL --> PG
    PG --> FS
    FS --> CG
    CG --> SC
    SC --> RR
    RR --> API
    REDIS -.->|cache| API
    API --> EXT
    API --> GHA
    API --> SITE
    API --> IDE
```

### ASCII

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         DATA SOURCES                                │
 │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
 │   │ GitHub   │    │ NPM      │    │ PyPI     │    │ Cargo    │    │
 │   │ API      │    │ Registry │    │ Registry │    │ Registry │    │
 │   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘    │
 └────────┼───────────────┼───────────────┼───────────────┼───────────┘
          │               │               │               │
          ▼               ▼               ▼               ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                      OFFLINE PIPELINE                               │
 │   ┌──────────────────────────────────────────────────────────┐     │
 │   │  Ingest CLI  ──▶  Data Pipeline (batch + stream)         │     │
 │   └──────────────────────────┬───────────────────────────────┘     │
 └──────────────────────────────┼──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                          STORAGE                                    │
 │   ┌──────────────────────┐        ┌──────────────────────┐        │
 │   │ Postgres + pgvector  │        │ Redis                │        │
 │   │ + Apache AGE         │        │ (cache)              │        │
 │   │ (raw + graph + vecs) │        └──────────┬───────────┘        │
 │   └──────────┬───────────┘                   │                     │
 └──────────────┼───────────────────────────────┼─────────────────────┘
                │                               │
                ▼                               │
 ┌──────────────────────────────┐               │
 │       FEATURE STORE          │               │
 │  (embeddings + precomputed   │               │
 │   graph features)            │               │
 └──────────┬───────────────────┘               │
            │                                   │
            ▼                                   │
 ┌──────────────────────────────────────────────┼─────────────────────┐
 │            RECOMMENDATION ENGINE             │                     │
 │   ┌─────────────────┐                        │                     │
 │   │ Candidate Gen   │ ANN + graph traversal  │                     │
 │   └────────┬────────┘                        │                     │
 │            ▼                                 │                     │
 │   ┌─────────────────┐                        │                     │
 │   │ Scoring         │ Two-tower + cross feat │                     │
 │   └────────┬────────┘                        │                     │
 │            ▼                                 │                     │
 │   ┌─────────────────┐                        │                     │
 │   │ Re-ranking      │ Dedup + diversity      │                     │
 │   └────────┬────────┘                        │                     │
 └────────────┼─────────────────────────────────┼─────────────────────┘
              │                                 │
              ▼                                 │
 ┌──────────────────────────────────────────────┼─────────────────────┐
 │              SERVING API                     │                     │
 │   ┌──────────────────────────────────────────┴───────────────┐    │
 │   │  FastAPI   GET /recommend?repo=owner/name&limit=10       │    │
 │   │  < 150ms p99                                              │    │
 │   └──────────────────────────┬────────────────────────────────┘    │
 └──────────────────────────────┼─────────────────────────────────────┘
                                │
          ┌─────────────┬───────┴───────┬─────────────┐
          ▼             ▼               ▼             ▼
 ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
 │  Browser   │ │  GitHub    │ │ Discovery  │ │    IDE     │
 │ Extension  │ │   App      │ │   Site     │ │   Plugin   │
 └────────────┘ └────────────┘ └────────────┘ └────────────┘
```

---

## 3. Data Pipeline Flow

How raw GitHub data gets ingested, processed, and stored.

### Mermaid

```mermaid
flowchart LR
    subgraph "Ingest"
        A[GitHub API Client] -->|fetch repo metadata| B[Raw Store]
        A -->|fetch READMEs| B
        A -->|fetch contributors| B
        A -->|fetch dependency manifests| B
        A -->|fetch workflow files| B
    end

    subgraph "Process"
        B -->|normalize + transform| C[Pipeline]
        C -->|compute embeddings| D[Feature Store]
        C -->|build graph edges| E[Graph DB]
        C -->|derive co-star patterns| E
        C -->|derive co-contributor patterns| E
        C -->|compute edge weights| E
    end

    subgraph "Refresh"
        F[Schedule / Events] -->|incremental| C
        G[README changes] -->|re-embed| D
    end
```

### ASCII

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                          INGEST                                     │
 │                                                                     │
 │   GitHub API Client                                                │
 │   ├── fetch repo metadata ──────────▶ Raw Store (Postgres)         │
 │   ├── fetch READMEs ────────────────▶ Raw Store                    │
 │   ├── fetch contributors ───────────▶ Raw Store                    │
 │   ├── fetch dependency manifests ───▶ Raw Store                    │
 │   └── fetch workflow files ─────────▶ Raw Store                    │
 │                                                                     │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                          PROCESS                                    │
 │                                                                     │
 │   Raw Store ──▶ Pipeline (normalize + transform)                   │
 │                  │                                                  │
 │                  ├──▶ compute embeddings ──▶ Feature Store          │
 │                  ├──▶ build graph edges ──▶ Graph DB (AGE)         │
 │                  ├──▶ derive co-star patterns ──▶ Graph DB          │
 │                  ├──▶ derive co-contributor patterns ──▶ Graph DB   │
 │                  └──▶ compute edge weights ──▶ Graph DB             │
 │                                                                     │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         REFRESH                                     │
 │                                                                     │
 │   Schedule / Events ──▶ incremental pipeline updates               │
 │   README changes ──▶ re-embed into Feature Store                   │
 │                                                                     │
 │   Target: model staleness < 6 hours                                │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Recommendation Engine Funnel

The 3-stage ML pipeline that turns millions of repos into ranked results.

### Mermaid

```mermaid
flowchart TB
    IN["Input: repo_id (+ optional user_id)"] --> CG

    subgraph CG["Stage 1: Candidate Generation"]
        direction TB
        ANN["ANN Search<br/>on README embeddings<br/>(semantic similarity)"]
        GRA["Graph Traversal<br/>dep graph, co-stars,<br/>co-contributors"]
    end

    CG -->|"~100s of candidates"| SC

    subgraph SC["Stage 2: Scoring"]
        direction TB
        TT["Two-Tower Model<br/>repo tower + user tower"]
        CF["Cross Features<br/>dep overlap, co-star count,<br/>temporal signals"]
        TT --> CF
    end

    SC -->|"ranked list"| RR

    subgraph RR["Stage 3: Re-ranking"]
        direction TB
        D["Deduplication"]
        F["Freshness Boost"]
        DV["Diversity Constraint"]
        BIZ["Business Rules<br/>(no NSFW, no archived)"]
    end

    RR --> OUT["Ranked output: 6 slots × N repos"]
```

### ASCII

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  INPUT: repo_id (owner/name), optional user_id, context            │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STAGE 1: CANDIDATE GENERATION                                     │
 │  ┌──────────────────────────┐  ┌──────────────────────────┐       │
 │  │ ANN Search               │  │ Graph Traversal           │       │
 │  │ on README embeddings     │  │ dep graph, co-stars,      │       │
 │  │ (semantic similarity)    │  │ co-contributors           │       │
 │  └──────────────┬───────────┘  └──────────────┬────────────┘       │
 │                 └──────────────┬──────────────┘                     │
 └────────────────────────────────┼────────────────────────────────────┘
                                  │
                                  │  ~100s of candidates
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STAGE 2: SCORING                                                   │
 │  ┌──────────────────────────┐  ┌──────────────────────────┐       │
 │  │ Two-Tower Model          │  │ Cross Features            │       │
 │  │ repo tower + user tower  │──│ dep overlap, co-star      │       │
 │  │                          │  │ count, temporal signals   │       │
 │  └──────────────────────────┘  └──────────────────────────┘       │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                │  ranked list
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STAGE 3: RE-RANKING                                                │
 │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │
 │  │ Dedup    │ │ Freshness│ │ Diversity│ │ Business │             │
 │  │          │ │ Boost    │ │          │ │ Rules    │             │
 │  └──────────┘ └──────────┘ └──────────┘ └──────────┘             │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  OUTPUT: 6 slots × N ranked repos                                   │
 │  ┌────────────┬──────────┬───────────┬──────────┬─────────┬─────┐ │
 │  │Alternatives│ Addons   │Companions │ Starters │Trending │Maint│ │
 │  └────────────┴──────────┴───────────┴──────────┴─────────┴─────┘ │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Request/Response Trace

Exact code path from a user's browser action to the API response.

### Mermaid

```mermaid
sequenceDiagram
    participant U as User Browser
    participant A as apps/site/src/pages/index.astro
    participant L as apps/site/src/lib/api.ts
    participant API as apps/api/src/reporelay_api/main.py
    participant CORE as packages/core/src/reporelay_core/

    U->>A: Visit site/?repo=vercel/next.js
    A->>L: getRecommendations("vercel/next.js", 5)
    L->>API: GET /recommend?repo=vercel/next.js&limit=5
    API->>API: Validate repo format (owner/name)
    API->>API: _stub_recommend(repo, limit)
    Note over API: TODO: connect to real engine
    API-->>L: RecommendResponse (empty slots)
    L-->>A: JSON data
    A-->>U: Render HTML with slot list
```

### ASCII

```
 USER BROWSER                SITE                          API
      │                       │                             │
      │  1. Visit             │                             │
      │  site/?repo=          │                             │
      │  vercel/next.js       │                             │
      │──────────────────────>│                             │
      │                       │  2. getRecommendations()    │
      │                       │  (from lib/api.ts)          │
      │                       │                             │
      │                       │  3. GET /recommend?         │
      │                       │  repo=vercel/next.js        │
      │                       │  &limit=5                   │
      │                       │────────────────────────────>│
      │                       │                             │
      │                       │     4. Validate format      │
      │                       │     (owner/name check)      │
      │                       │                             │
      │                       │     5. _stub_recommend()    │
      │                       │     (TODO: real engine)     │
      │                       │                             │
      │                       │  6. JSON response           │
      │                       │  { source_repo, slots: [] } │
      │                       │<────────────────────────────│
      │                       │                             │
      │  7. Render HTML       │                             │
      │  with slot list       │                             │
      │<──────────────────────│                             │
      │                       │                             │

 CODE FILES INVOLVED:
 ┌─────────────────────────────────────────────────────────────┐
 │ apps/site/src/pages/index.astro    — page entry point       │
 │ apps/site/src/lib/api.ts           — API client function    │
 │ apps/api/src/reporelay_api/main.py — FastAPI endpoints      │
 │ packages/core/src/reporelay_core/  — shared settings + db   │
 └─────────────────────────────────────────────────────────────┘
```

---

## 6. Data Model (Graph Schema)

The graph structure that powers recommendations.

### Mermaid

```mermaid
erDiagram
    REPO {
        int id PK
        string full_name
        string description
        int stars
        string language
        string[] topics
        datetime created_at
        boolean archived
    }

    USER {
        int id PK
        string login
        datetime created_at
    }

    TOPIC {
        string name PK
    }

    LANGUAGE {
        string name PK
    }

    DEPENDENCY {
        string ecosystem
        string name
        string version
    }

    REPO ||--o{ DEPENDENCY : "depends on"
    USER ||--o{ REPO : "stars"
    USER ||--o{ REPO : "contributes to"
    REPO ||--o{ TOPIC : "has topic"
    REPO ||--o{ LANGUAGE : "written in"
    REPO ||--o{ REPO : "co-occurs in workflow"
    REPO ||--o{ REPO : "is alternative to"
    USER ||--o{ USER : "co-starred"
    USER ||--o{ USER : "co-contributed"
```

### ASCII

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                     GRAPH NODES                                  │
 └─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │    REPO      │      │    USER      │      │    TOPIC     │
  ├──────────────┤      ├──────────────┤      ├──────────────┤
  │ id (PK)      │      │ id (PK)      │      │ name (PK)    │
  │ full_name    │      │ login        │      └──────────────┘
  │ description  │      │ created_at   │
  │ stars        │      └──────────────┘      ┌──────────────┐
  │ language     │                            │  LANGUAGE    │
  │ topics      │                            ├──────────────┤
  │ created_at   │                            │ name (PK)    │
  │ archived     │                            └──────────────┘
  └──────────────┘
         │
         │         ┌──────────────┐
         │         │  DEPENDENCY  │
         │         ├──────────────┤
         │         │ ecosystem    │
         │         │ name         │
         │         │ version      │
         │         └──────────────┘

 ┌─────────────────────────────────────────────────────────────────┐
 │                     GRAPH EDGES                                  │
 └─────────────────────────────────────────────────────────────────┘

  REPO ──────depends on──────▶ DEPENDENCY
  DEPENDENCY ──reverse dep───▶ REPO          (drives "Starters" slot)

  USER ───────stars──────────▶ REPO
  USER ──────contributes to──▶ REPO

  REPO ──────has topic───────▶ TOPIC
  REPO ──────written in──────▶ LANGUAGE

  REPO ◀────co-occurs in────▶ REPO          (same workflow file)
  REPO ◀────is alternative──▶ REPO          (inferred similarity)

  USER ◀────co-starred──────▶ USER          (derived: star same repos)
  USER ◀────co-contributed──▶ USER          (derived: work on same repos)

 ┌─────────────────────────────────────────────────────────────────┐
 │  EDGE WEIGHTS                                                   │
 │  ┌─────────────────────────┬───────────────────────────────┐   │
 │  │ Edge                    │ Weight signal                 │   │
 │  ├─────────────────────────┼───────────────────────────────┤   │
 │  │ DEPENDS_ON              │ 1 per occurrence              │   │
 │  │ STARRED_BY              │ 1 (or 0.5 if unstarred)      │   │
 │  │ CONTRIBUTED_TO          │ 1 per commit/PR               │   │
 │  │ CO_OCCURS_IN_WORKFLOW   │ count of co-occurrences       │   │
 │  │ CO_STARRED              │ Jaccard or PMI                │   │
 │  │ CO_CONTRIBUTED          │ Jaccard or PMI                │   │
 │  └─────────────────────────┴───────────────────────────────┘   │
 └─────────────────────────────────────────────────────────────────┘
```

---

## How It All Fits Together

```
   USER JOURNEY          SYSTEM ARCHITECTURE         DATA PIPELINE
   (Diagram 1)           (Diagram 2)                (Diagram 3)
        │                      │                         │
        │  user visits site    │  components wire up     │  data flows in
        ▼                      ▼                         ▼
   ┌─────────┐           ┌─────────┐              ┌─────────┐
   │ Search  │──────────▶│  API    │◀─────────────│Feature  │
   │ results │           │  + Engine│              │ Store   │
   └─────────┘           └─────────┘              └─────────┘
        │                      │                         │
        │                      ▼                         │
        │                 ┌─────────┐                    │
        │                 │ Rec     │◀───────────────────┘
        │                 │ Engine  │
        │                 └─────────┘
        │                      │
        ▼                      ▼
   ┌─────────┐           ┌─────────┐
   │  6 Slot │◀──────────│   ML    │
   │  Cards  │           │ Funnel  │
   └─────────┘           └─────────┘
```
