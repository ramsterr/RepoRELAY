# NEXT — swap to a smaller embedding model

Swap from BAAI/bge-small-en-v1.5 to all-MiniLM-L6-v2 (80MB disk, ~120MB RAM).

Same 384 dimensions — no schema changes. Different vector space — need to re-embed.

## Steps

1. Change model in `packages/mvp/src/reporelay_mvp/embedding.py`:

   ```python
   MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
   ```

2. Move `sentence-transformers` back to main deps in `packages/mvp/pyproject.toml`
   and run `uv lock`.

3. Remove lightweight mode checks from `embedding.py`.

4. Re-add `HF_HOME` and `SENTENCE_TRANSFORMERS_HOME` to `Dockerfile.api`.

5. Null all existing embeddings (different model = incompatible vectors):

   ```sql
   UPDATE mvp_repos SET embedding = NULL, embedded_at = NULL;
   ```

6. Push. The hourly embed cron rebuilds ~500 per hour. 5000 repos = ~10 hours.

   Or run manually: `reporelay-mvp embed --limit 5000 --concurrency 4`
