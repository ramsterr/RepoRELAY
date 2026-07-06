"""
Embedding for the MVP.

Five modes:
  1. LOCAL — loads BAAI/bge-small-en-v1.5 (384 dims) in-process.
     Uses ~200MB RAM. Best for dev machines and bulk embedding.

  2. OPENAI — calls OpenAI's text-embedding-3-small with dimensions=512.
     Zero local RAM. Best general-purpose API option.

  3. VOYAGE — calls Voyage AI's voyage-code-3 (code-specialized, 512 dims).
     Zero local RAM. Best quality for code/README/description text.

  4. GEMINI — calls Google Gemini's embedding-001 (512 dims via MRL).
     1000 req/day free tier (restrictive, not recommended for bulk).

  5. COHERE — calls Cohere's embed-v4.0 (512 dims, configurable).
     500 RPM trial key. Best free provider. No daily cap. Recommended.

  6. NONE — returns zeros. Fallback if no API key is configured.

The mode is controlled by the EMBEDDING_API env var:
  - "local"  → loads the local model
  - "openai" → calls OpenAI API (requires OPENAI_API_KEY)
  - "voyage" → calls Voyage API (requires VOYAGE_API_KEY + payment method for high RPM)
  - "cohere" → calls Cohere API (requires COHERE_API_KEY)
  - "gemini" → calls Gemini API (requires GEMINI_API_KEY)
  - "none"   → returns zeros

The REPORE_LAY_LIGHTWEIGHT=1 env var, when set, suppresses the local
model load regardless of EMBEDDING_API. Combined with EMBEDDING_API=gemini,
this is the recommended Render deploy config (best free tier, zero RAM).

Mode detection is lazy (resolved on first call) so pydantic settings
can load the .env file first. The module imports cleanly without
requiring any env vars to be set.

Vectors are L2-normalized (required by BGE's contrastive loss and
recommended for all cosine-similarity use cases).
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
_openai_client: Any = None
_openai_client_lock = threading.Lock()
_voyage_client: Any = None
_voyage_client_lock = threading.Lock()
_gemini_configured: bool = False
_gemini_lock = threading.Lock()
_cohere_client: Any = None
_cohere_client_lock = threading.Lock()
_mode: str | None = None
_mode_lock = threading.Lock()

DIMENSION = 512

MODEL_NAME = "BAAI/bge-small-en-v1.5"
OPENAI_MODEL = "text-embedding-3-small"
VOYAGE_MODEL = "voyage-code-3"
GEMINI_MODEL = "models/gemini-embedding-001"
COHERE_MODEL = "embed-v4.0"


def _resolve_mode() -> str:
    """Determine embedding mode. Lazy — reads settings on first call.

    Priority:
      1. Explicit EMBEDDING_API env var (set via pydantic settings or os.environ)
      2. If REPORE_LAY_LIGHTWEIGHT=1:
         - "cohere" if COHERE_API_KEY is set (preferred — 500 RPM, no daily cap)
         - else "gemini" if GEMINI_API_KEY is set
         - else "voyage" if VOYAGE_API_KEY is set
         - else "openai" if OPENAI_API_KEY is set
         - else "none"
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
        has_cohere = bool(settings.cohere_api_key)
        has_gemini = bool(settings.gemini_api_key)
        has_voyage = bool(settings.voyage_api_key)
        has_openai = bool(settings.openai_api_key)

        if api_mode is None:
            if lightweight:
                if has_cohere:
                    api_mode = "cohere"
                elif has_gemini:
                    api_mode = "gemini"
                elif has_voyage:
                    api_mode = "voyage"
                elif has_openai:
                    api_mode = "openai"
                else:
                    api_mode = "none"
            else:
                api_mode = "local"

        if api_mode not in ("local", "openai", "voyage", "gemini", "cohere", "none"):
            raise ValueError(
                f"EMBEDDING_API must be 'local', 'openai', 'voyage', 'gemini', 'cohere', or 'none', got {api_mode!r}"
            )

        _mode = api_mode
        logger.info(
            "embedding mode resolved: %s (lightweight=%s, cohere_key=%s, gemini_key=%s, voyage_key=%s, openai_key=%s)",
            _mode, lightweight,
            "yes" if has_cohere else "no",
            "yes" if has_gemini else "no",
            "yes" if has_voyage else "no",
            "yes" if has_openai else "no",
        )
        return _mode


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    with _openai_client_lock:
        if _openai_client is not None:
            return _openai_client
        from openai import AsyncOpenAI

        from reporelay_mvp.settings import get_mvp_settings

        settings = get_mvp_settings()
        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_API=openai but no OpenAI API key found. "
                "Set OPENAI_API_KEY in your env or .env file."
            )
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


def _get_voyage_client() -> Any:
    global _voyage_client
    if _voyage_client is not None:
        return _voyage_client
    with _voyage_client_lock:
        if _voyage_client is not None:
            return _voyage_client
        import voyageai

        from reporelay_mvp.settings import get_mvp_settings

        settings = get_mvp_settings()
        api_key = settings.voyage_api_key
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_API=voyage but no Voyage API key found. "
                "Set VOYAGE_API_KEY in your env or .env file."
            )
        _voyage_client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
    return _voyage_client


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

        import google.generativeai as genai

        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        _gemini_configured = True


def _get_cohere_client() -> Any:
    global _cohere_client
    if _cohere_client is not None:
        return _cohere_client
    with _cohere_client_lock:
        if _cohere_client is not None:
            return _cohere_client
        import cohere

        from reporelay_mvp.settings import get_mvp_settings

        settings = get_mvp_settings()
        api_key = settings.cohere_api_key
        if not api_key:
            raise RuntimeError(
                "EMBEDDING_API=cohere but no Cohere API key found. "
                "Set COHERE_API_KEY in your env or .env file. "
                "Get a free trial key at https://dashboard.cohere.com/"
            )
        _cohere_client = cohere.ClientV2(api_key=api_key)
    return _cohere_client


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
    if mode in ("openai", "voyage", "gemini", "cohere"):
        logger.info("embedding mode=%s — zero local RAM, using API", mode)
        return
    if mode == "none":
        logger.info("embedding mode=none — returning zeros (no API key configured)")
        return
    # local mode
    await asyncio.to_thread(_load_model)
    logger.info("embedding model preloaded and ready")


async def _embed_via_openai(text_value: str) -> list[float]:
    """Call OpenAI text-embedding-3-small with dimensions=512."""
    if not text_value or not text_value.strip():
        return [0.0] * DIMENSION
    client = _get_openai_client()
    response = await client.embeddings.create(
        model=OPENAI_MODEL,
        input=text_value,
        dimensions=DIMENSION,
        encoding_format="float",
    )
    return [float(x) for x in response.data[0].embedding]


async def _embed_via_voyage(text_value: str) -> list[float]:
    """Call Voyage voyage-code-3 with output_dimension=512 (code-specialized).

    voyage-code-3 is purpose-built for code retrieval — best quality for
    README/description text. Uses output_dimension=512 (one of the
    model's supported dims: 256, 512, 1024, 2048). 512 was chosen as
    the closest match to the prior 384-dim local model while staying
    within Voyage's supported dimensions.

    input_type='document' is the appropriate type for indexing README
    text into the corpus. For query-time embedding (tags filter), use
    embed_text with input_type='query' — see embed_text_query().
    """
    if not text_value or not text_value.strip():
        return [0.0] * DIMENSION
    client = _get_voyage_client()

    # voyageai.Client.embed is sync — run in thread to keep the event loop free
    def _call() -> list[float]:
        result = client.embed(
            texts=[text_value],
            model=VOYAGE_MODEL,
            input_type="document",
            output_dimension=DIMENSION,
            truncation=True,
        )
        return [float(x) for x in result.embeddings[0]]

    return await asyncio.to_thread(_call)


async def _embed_via_gemini(text_value: str) -> list[float]:
    """Call Google Gemini gemini-embedding-001 with output_dimensionality=512.

    Uses Matryoshka Representation Learning (MRL) to shrink the native
    3072-dim output to 512. This matches the current DB schema without
    requiring a migration.

    For document indexing (README/description), use task_type=
    'RETRIEVAL_DOCUMENT'. For query-time embedding, use 'RETRIEVAL_QUERY'
    — see embed_text_query().

    The google-generativeai SDK has shipped two response shapes:
      - newer (>=0.5): `{"embedding": [...]}` for single input
      - older:         `{"embeddings": [[...]]}`
    We accept both so a package upgrade can't silently break us.
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
        # Accept both response shapes
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
            # Some SDK versions return an EmbeddingResponse object
            vec = getattr(result, "embedding", None) or getattr(result, "embeddings", [[]])[0]
        if not vec:
            raise RuntimeError("Gemini returned empty embedding vector")
        return [float(x) for x in vec]

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            last_exc = exc
            # Brief backoff — 0.5s, 1s. Don't hammer the API on transient errors.
            await asyncio.sleep(0.5 * (attempt + 1))
            logger.warning("Gemini embed attempt %d failed: %s", attempt + 1, exc)
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
    In 'local', 'openai', 'voyage', or 'gemini' mode, returns a real vector.

    For document indexing (README, description). For query-time
    embedding (tag filter), use embed_text_query() instead.
    """
    return await embed_text_query(text_value, input_type="document")


async def embed_text_query(
    text_value: str,
    input_type: str = "query",
) -> list[float]:
    """Compute a 512-dim embedding for a query (vs document).

    For Voyage, uses input_type='query' which prepends a prompt template
    optimized for retrieval. For Gemini, uses task_type='RETRIEVAL_QUERY'.
    For OpenAI/local, input_type is ignored.
    """
    if not text_value or not text_value.strip():
        return [0.0] * DIMENSION
    mode = _resolve_mode()
    if mode == "none":
        return [0.0] * DIMENSION
    if mode == "openai":
        return await _embed_via_openai(text_value)
    if mode == "voyage":
        if input_type == "query":
            client = _get_voyage_client()

            def _call() -> list[float]:
                result = client.embed(
                    texts=[text_value],
                    model=VOYAGE_MODEL,
                    input_type="query",
                    output_dimension=DIMENSION,
                    truncation=True,
                )
                return [float(x) for x in result.embeddings[0]]

            return await asyncio.to_thread(_call)
        return await _embed_via_voyage(text_value)
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
    if mode == "cohere":
        # Cohere's embed_v4.0 with output_dimension=512
        # Cohere API is sync — run in thread to keep event loop free
        client = _get_cohere_client()
        it = "search_query" if input_type == "query" else "search_document"

        def _call_cohere() -> list[float]:
            result = client.embed(
                model=COHERE_MODEL,
                texts=[text_value],
                input_type=it,
                embedding_types=["float"],
                output_dimension=DIMENSION,
            )
            return [float(x) for x in result.embeddings.float_[0]]

        return await asyncio.to_thread(_call_cohere)
    # local
    return await _embed_via_local_model(text_value)


def embedding_mode() -> str:
    """Return the current embedding mode."""
    return _resolve_mode()


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
    return [float(x) for x in result.tolist()]
