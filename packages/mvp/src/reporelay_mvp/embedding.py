"""
Embedding for the MVP. Uses BAAI/bge-small-en-v1.5 (384 dims,
trained on the Pile + C4 + arXiv + StackExchange — strong
technical vocabulary for README-vs-README similarity).

Vectors are L2-normalized (required by BGE's contrastive loss and
recommended for all cosine-similarity use cases).

Set REPORE_LAY_LIGHTWEIGHT=1 to skip loading the model (~200-300MB
RAM savings). In lightweight mode embed_text() returns zeros and
tag filtering falls back to exact topic matching. Useful for
deployments on constrained hardware (e.g. Render free tier 512MB).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_model: Any = None
_model_lock = threading.Lock()
DIMENSION = 384

MODEL_NAME = "BAAI/bge-small-en-v1.5"
_LIGHTWEIGHT = os.environ.get("REPORE_LAY_LIGHTWEIGHT", "").lower() in ("1", "true", "yes")


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer

        logger.info("loading embedding model %s ...", MODEL_NAME)
        start = time.monotonic()
        _model = SentenceTransformer(MODEL_NAME)
        logger.info(
            "model loaded in %.1fs (dim=%d)", time.monotonic() - start, DIMENSION
        )
    return _model


async def preloadModel() -> None:
    if _LIGHTWEIGHT:
        logger.info("lightweight mode — skipping model load (saves ~200-300MB)")
        return
    await asyncio.to_thread(_load_model)
    logger.info("embedding model preloaded and ready")


async def embed_text(text_value: str) -> list[float]:
    """Compute a 384-dim embedding for a piece of text. Returns zeros for empty input or lightweight mode."""
    if _LIGHTWEIGHT or not text_value or not text_value.strip():
        return [0.0] * DIMENSION
    model = _load_model()
    vector = await asyncio.to_thread(
        model.encode,
        text_value,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vector.tolist()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalized vectors. Returns 0 for zero vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    na = np.asarray(a, dtype=np.float32)
    nb = np.asarray(b, dtype=np.float32)
    dot = float(np.dot(na, nb))
    norm_a = float(np.linalg.norm(na))
    norm_b = float(np.linalg.norm(nb))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_batch_one_vs_many(one: list[float], many: list[list[float]]) -> list[float]:
    """Vectorized cosine similarity: one vector against N vectors."""
    if not many:
        return []
    n = np.asarray(many, dtype=np.float32)
    o = np.asarray(one, dtype=np.float32)
    dot = n @ o
    norms = np.linalg.norm(n, axis=1) * float(np.linalg.norm(o))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(norms > 1e-9, dot / norms, 0.0)
    return result.tolist()
