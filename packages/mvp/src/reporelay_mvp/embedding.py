"""
Embedding for the MVP. Uses the same model as the main app
(`all-MiniLM-L6-v2`, 384 dims) so the embedding space is comparable.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_model: Any = None
_cached_dim: int | None = None


def _get_model() -> Any:
    global _model, _cached_dim
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("loading embedding model...")
        start = time.monotonic()
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _cached_dim = _model.get_embedding_dimension()
        logger.info("model loaded in %.1fs (dim=%d)", time.monotonic() - start, _cached_dim)
    return _model


def embed_text(text_value: str) -> list[float]:
    """Compute a 384-dim embedding for a piece of text. Returns zeros for empty input."""
    if not text_value or not text_value.strip():
        return [0.0] * 384
    model = _get_model()
    vector = model.encode(text_value, show_progress_bar=False, normalize_embeddings=False)
    return vector.tolist()
