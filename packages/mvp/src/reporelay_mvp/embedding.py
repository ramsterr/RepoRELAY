"""
Embedding for the MVP.

Three modes:
  1. LOCAL — loads BAAI/bge-small-en-v1.5 (384 dims) in-process.
     Uses ~200MB RAM. Best for dev machines and bulk embedding.

  2. GEMINI — calls Google Gemini's embedding-001 (512 dims via MRL).
     Zero local RAM. Best for production (Render free tier + paid API).

  3. NONE — returns zeros. Fallback if no API key is configured.

The mode is controlled by the EMBEDDING_API env var:
  - "local"  → loads the local model
  - "gemini" → calls Gemini API (requires GEMINI_API_KEY)
  - "none"   → returns zeros

Mode detection is lazy (resolved on first call) so pydantic settings
can load the .env file first. The module imports cleanly without
requiring any env vars to be set.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_model: Any = None
_model_lock = threading.Lock()
_gemini_configured: bool = False
_gemini_lock = threading.Lock()
_mode: str | None = None
_mode_lock = threading.Lock()

DIMENSION = 512

MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "models/gemini-embedding-001"


def _resolve_mode() -> str:
    """Determine embedding mode. Lazy — reads settings on first call.

    Priority:
      1. Explicit EMBEDDING_API env var
      2. If REPORE_LAY_LIGHTWEIGHT=1: "gemini" (with key) or "none"
      3. Otherwise "local"
    """
    global _mode
    if _mode is not None:
        return _mode
    with _mode_lock:
        if _mode is not None:
            return _mode

        from reporelay_mvp.settings import get_mvp_settings

        settings = get_mvp_settings()
        api_mode = (settings.embedding_api or "").lower() or None
        lightweight = settings.lightweight
        has_gemini = bool(settings.gemini_api_key)

        if api_mode is None:
            if lightweight:
                api_mode = "gemini" if has_gemini else "none"
            else:
                api_mode = "local"

        if api_mode not in ("local", "gemini", "none"):
            raise ValueError(
                f"EMBEDDING_API must be 'local', 'gemini', or 'none', got {api_mode!r}"
            )

        _mode = api_mode
        logger.info(
            "embedding mode resolved: %s (lightweight=%s, gemini_key=%s)",
            _mode, lightweight, "yes" if has_gemini else "no",
        )
        return _mode


def _configure_gemini() -> None:
    """Configure the Gemini SDK with the API key. Idempotent."""
    global _gemini_configured
    if _gemini_configured:
        return
    with _gemini_lock:
        if _gemini_configured:
            return

        from reporelay_mvp.settings import get_mvp_settings

        settings = get_mvp_settings()
        api_key = settings.gemini_api_key
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_API=gemini but no Gemini API key found. "
                "Set GEMINI_API_KEY in your env or .env file. "
                "Get a free key at https://aistudio.google.com/"
            )

        # Patch BEFORE importing google.generativeai — the deprecated
        # SDK v0.8.6 has a bug where strip_oneof(docstring) calls
        # docstring.splitlines() without checking for None. This
        # crashes during module import (class __doc__ assignments).
        _patch_string_utils_before_import()

        import google.generativeai as genai

        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        _gemini_configured = True


def _patch_string_utils_before_import() -> None:
    """Patch strip_oneof in google.generativeai.string_utils.

    This MUST run before the first `import google.generativeai` because
    the package's __init__.py imports types/safety_types/citation_types
    which call strip_oneof() at module load time on protobuf __doc__
    attributes that may be None.

    We use importlib to load only string_utils.py (avoiding the full
    package import chain), patch it, then register it in sys.modules
    so the subsequent full import picks up our patched version.
    """
    import importlib.util
    import sys

    try:
        spec = importlib.util.find_spec("google.generativeai.string_utils")
        if spec is None or spec.origin is None:
            logger.warning("could not find string_utils spec — Gemini embed may fail")
            return

        # Load just string_utils, bypassing google.generativeai.__init__
        module = importlib.util.module_from_spec(spec)
        sys.modules["google.generativeai.string_utils"] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        _original_strip_oneof = module.strip_oneof

        def _patched_strip_oneof(docstring):
            if docstring is None:
                return ""
            return _original_strip_oneof(docstring)

        module.strip_oneof = _patched_strip_oneof  # type: ignore[attr-defined]
        logger.debug("patched google.generativeai.string_utils.strip_oneof")
    except Exception:
        logger.warning("failed to patch string_utils — Gemini embed may fail", exc_info=True)


def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        logger.info("loading embedding model %s ...", MODEL_NAME)
        start = time.monotonic()
        _model = SentenceTransformer(MODEL_NAME)
        logger.info(
            "model loaded in %.1fs (dim=%d)", time.monotonic() - start, DIMENSION
        )
    return _model


async def preloadModel() -> None:
    """Startup hook. In API mode this is a no-op (zero RAM)."""
    mode = _resolve_mode()
    if mode == "gemini":
        logger.info("embedding mode=gemini — zero local RAM, using API")
        return
    if mode == "none":
        logger.info("embedding mode=none — returning zeros (no API key configured)")
        return
    # local mode
    await asyncio.to_thread(_load_model)
    logger.info("embedding model preloaded and ready")


async def _embed_via_gemini(text_value: str) -> list[float]:
    """Call Google Gemini gemini-embedding-001 with output_dimensionality=512.

    Uses 45s timeout with 3 retries and brief backoff. Accepts both
    {'embedding': [...]} and {'embeddings': [[...]]} response shapes.
    """
    if not text_value or not text_value.strip():
        return [0.0] * DIMENSION
    _configure_gemini()

    import google.generativeai as genai

    def _call() -> list[float]:
        result = genai.embed_content(  # type: ignore[attr-defined]
            model=GEMINI_MODEL,
            content=text_value,
            task_type="retrieval_document",
            output_dimensionality=DIMENSION,
        )
        if isinstance(result, dict):
            if "embedding" in result and result["embedding"] is not None:
                vec = result["embedding"]
            elif "embeddings" in result and result["embeddings"]:
                vec = result["embeddings"][0]
            else:
                raise RuntimeError(
                    f"Gemini embed_content returned no embedding key; keys={list(result.keys())}"
                )
        else:
            vec = getattr(result, "embedding", None) or getattr(result, "embeddings", [[]])[0]
        if not vec:
            raise RuntimeError("Gemini returned empty embedding vector")
        return [float(x) for x in vec]

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return await asyncio.wait_for(asyncio.to_thread(_call), timeout=45.0)
        except asyncio.TimeoutError:
            last_exc = RuntimeError(f"Gemini embed timed out after 45s (attempt {attempt + 1}/3)")
            logger.warning("Gemini embed attempt %d timed out", attempt + 1)
        except Exception as exc:
            last_exc = exc
            logger.warning("Gemini embed attempt %d failed: %s", attempt + 1, exc)
        await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Gemini embed failed after 3 attempts: {last_exc}")


async def _embed_via_local_model(text_value: str) -> list[float]:
    """Run BAAI/bge-small-en-v1.5 locally. Uses ~200MB RAM."""
    if not text_value or not text_value.strip():
        return [0.0] * DIMENSION
    model = _load_model()
    vector = await asyncio.to_thread(
        model.encode,
        text_value,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [float(x) for x in vector.tolist()]


async def embed_text(text_value: str) -> list[float]:
    """Compute a 512-dim embedding for a piece of text.

    Returns zeros for empty input. In 'none' mode, always returns zeros.
    In 'local' or 'gemini' mode, returns a real vector.

    For document indexing (README, description). For query-time
    embedding (tag filter), use embed_text_query() instead.
    """
    return await embed_text_query(text_value, input_type="document")


async def embed_text_query(
    text_value: str,
    input_type: str = "query",
) -> list[float]:
    """Compute a 512-dim embedding for a query (vs document).

    For Gemini, uses task_type='RETRIEVAL_QUERY' when input_type is 'query'.
    For local mode, input_type is ignored.
    """
    if not text_value or not text_value.strip():
        return [0.0] * DIMENSION
    mode = _resolve_mode()
    if mode == "none":
        return [0.0] * DIMENSION
    if mode == "gemini":
        if input_type == "query":
            _configure_gemini()
            import google.generativeai as genai

            def _call() -> list[float]:
                result = genai.embed_content(  # type: ignore[attr-defined]
                    model=GEMINI_MODEL,
                    content=text_value,
                    task_type="retrieval_query",
                    output_dimensionality=DIMENSION,
                )
                return [float(x) for x in result["embedding"]]

            return await asyncio.to_thread(_call)
        return await _embed_via_gemini(text_value)
    # local
    return await _embed_via_local_model(text_value)


def embedding_mode() -> str:
    """Return the current embedding mode: 'local', 'gemini', or 'none'."""
    return _resolve_mode()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalized vectors."""
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
    """Vectorized cosine similarity: one vector against N vectors.

    Returns 0.0 for any pair where either vector is zero or has a
    near-zero norm. This prevents NaN from propagating into the scorer.
    """
    if not many:
        return []
    n = np.asarray(many, dtype=np.float32)
    o = np.asarray(one, dtype=np.float32)
    dot = n @ o
    norms = np.linalg.norm(n, axis=1) * float(np.linalg.norm(o))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(norms > 1e-9, dot / norms, 0.0)
    result = np.where(np.isfinite(result), result, 0.0)
    return [float(x) for x in result.tolist()]
