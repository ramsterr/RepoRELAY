# next-project

Swap embedding model from BAAI/bge-small-en-v1.5 → all-MiniLM-L6-v2.

Different model = different vector space. Need to re-embed all 5000 repos.

Null all embeddings, let the hourly embed cron rebuild them over ~10 hours.
