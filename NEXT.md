# NEXT — bring the embedding model back into the API

## Why

Currently the API runs in **lightweight mode** — no model loaded. This saves 200-300MB RAM
but kills semantic tag filtering and forces cold repos to borrow embeddings from neighbors.

The original model (BAAI/bge-small-en-v1.5, 130MB disk, ~250MB RAM) was too heavy for
Render's 512MB limit. There is a smaller alternative.

## The model: all-MiniLM-L6-v2

|                   | BGE (current)      | MiniLM (target)     |
|-------------------|--------------------|--------------------|
| Dimensions        | 384                | 384 (same)         |
| Disk              | 130 MB             | 80 MB              |
| RAM               | ~250 MB            | ~120 MB            |
| Vocabulary        | General + academic | General + technical |
| RAM budget        | ❌ over 512MB       | ✅ ~300MB total     |

Same dimensions — no schema changes. Just swap the model and re-embed.

## RAM budget with MiniLM

```
Python + FastAPI      100 MB
MiniLM model          120 MB
DB pool (5+3)          50 MB
Misc                   30 MB
────────────────────────────
Total                ~300 MB  (190 MB headroom)
```

## What we get back

- **Semantic tag filtering** — `?tags=machine-learning` matches repos that talk about ML
  in their README, not just repos tagged "machine-learning"
- **Cold repos get real embeddings** — no more borrowing from neighbors
- **API is self-contained** — no lightweight mode hack

## What needs to change

### 1. Swap the model

```python
# packages/mvp/src/reporelay_mvp/embedding.py
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
```

### 2. Remove lightweight mode

```python
# packages/mvp/src/reporelay_mvp/embedding.py — delete _LIGHTWEIGHT and all its checks
```

### 3. Re-add model to Dockerfile

```dockerfile
# Dockerfile.api
ENV HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

# Move it back to main deps
# packages/mvp/pyproject.toml
dependencies = [
    ...
    "sentence-transformers>=3.0.0",  # back from optional
]
```

### 4. Re-embed the corpus

```bash
# Null all existing embeddings
psql $DATABASE_URL -c "UPDATE mvp_repos SET embedding = NULL, embedded_at = NULL;"

# Let the hourly cron rebuild them (~500 per hour, 10 hours for 5000 repos)
# Or run manually:
reporelay-mvp embed --limit 5000 --concurrency 4
```

### 5. Put the model back in deps

```diff
# packages/mvp/pyproject.toml
dependencies = [
    ...
-   "numpy>=2.0.0",                     # keep
+   "sentence-transformers>=3.0.0",     # back
]

- [project.optional-dependencies]
- embed = ["sentence-transformers>=3.0.0"]
```

Then `uv lock` to regenerate the lock file.

### 6. Remove proxy embedding

```python
# packages/mvp/src/reporelay_mvp/recommend.py
# Delete _find_proxy_embedding() — no longer needed
```

### 7. Enable model download at build time (optional)

If Render's network cooperates, bake the model into the Docker image for faster cold starts.
Otherwise it downloads at runtime (~30s first request, cached on disk after).

## Rollout

1. Merge the changes above
2. Push → Render rebuilds (model now loads in API)
3. Run `UPDATE mvp_repos SET embedding = NULL` on the DB
4. The hourly embed cron rebuilds embeddings over ~10 hours
5. During migration, recommendations still work (topic/language matching falls back)
6. Once complete, all 5000 repos have MiniLM embeddings + semantic tagging is live

## Risks

- MiniLM is slightly lower quality than BGE for niche technical vocabulary.
  Mitigation: the IDF-weighted topic overlap and pgvector ANN still carry strong signals.
- Re-embedding takes ~10 hours. Recommendations during migration use topic matching.
- If Render's network drops the model download, it retries at first request (30s delay).
