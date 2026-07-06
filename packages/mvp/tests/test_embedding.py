"""Tests for embedding module — local, gemini, none modes."""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _reset_mode(monkeypatch: pytest.MonkeyPatch, **env: str) -> object:
    """Reset the embedding module's mode cache with controlled env vars."""
    for key in ("EMBEDDING_API", "REPORE_LAY_LIGHTWEIGHT", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    import reporelay_mvp.settings
    import reporelay_mvp.embedding

    reporelay_mvp.settings.MvpSettings.model_config["env_file"] = None
    reporelay_mvp.settings.get_mvp_settings.cache_clear()
    importlib.reload(reporelay_mvp.embedding)
    reporelay_mvp.embedding._mode = None
    reporelay_mvp.embedding._gemini_configured = False
    return reporelay_mvp.embedding


class TestModeSelection:
    def test_default_is_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch)
        assert emb.embedding_mode() == "local"

    def test_explicit_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, EMBEDDING_API="local")
        assert emb.embedding_mode() == "local"

    def test_explicit_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, EMBEDDING_API="gemini", GEMINI_API_KEY="AIza-test")
        assert emb.embedding_mode() == "gemini"

    def test_explicit_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, EMBEDDING_API="none")
        assert emb.embedding_mode() == "none"

    def test_lightweight_without_key_falls_back_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, REPORE_LAY_LIGHTWEIGHT="1")
        assert emb.embedding_mode() == "none"

    def test_lightweight_with_gemini_key_uses_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, REPORE_LAY_LIGHTWEIGHT="1", GEMINI_API_KEY="AIza-test")
        assert emb.embedding_mode() == "gemini"


class TestEmbedText:
    @pytest.mark.asyncio
    async def test_empty_text_returns_zeros(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch)
        for val in ("", "   ", "\n\n"):
            result = await emb.embed_text(val)
            assert result == [0.0] * emb.DIMENSION
            assert len(result) == 512

    @pytest.mark.asyncio
    async def test_none_mode_returns_zeros(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, EMBEDDING_API="none")
        result = await emb.embed_text("hello world")
        assert result == [0.0] * emb.DIMENSION

    @pytest.mark.asyncio
    async def test_gemini_mode_calls_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, EMBEDDING_API="gemini", GEMINI_API_KEY="AIza-test")
        fake_vector = [0.4] * 512
        mock_genai = MagicMock()
        mock_genai.embed_content = MagicMock(return_value={"embedding": fake_vector})

        import sys

        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            result = await emb.embed_text("chess library")
            assert result == fake_vector
            assert len(result) == 512
            mock_genai.embed_content.assert_called_once()
            call_kwargs = mock_genai.embed_content.call_args.kwargs
            assert call_kwargs["model"] == "models/gemini-embedding-001"
            assert call_kwargs["task_type"] == "retrieval_document"
            assert call_kwargs["output_dimensionality"] == 512

    @pytest.mark.asyncio
    async def test_gemini_query_uses_query_task_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, EMBEDDING_API="gemini", GEMINI_API_KEY="AIza-test")
        fake_vector = [0.5] * 512
        mock_genai = MagicMock()
        mock_genai.embed_content = MagicMock(return_value={"embedding": fake_vector})

        import sys

        with patch.dict(sys.modules, {"google.generativeai": mock_genai}):
            result = await emb.embed_text_query("chess library", input_type="query")
            assert result == fake_vector
            call_kwargs = mock_genai.embed_content.call_args.kwargs
            assert call_kwargs["task_type"] == "retrieval_query"

    @pytest.mark.asyncio
    async def test_gemini_mode_empty_text_skips_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, EMBEDDING_API="gemini", GEMINI_API_KEY="AIza-test")
        result = await emb.embed_text("")
        assert result == [0.0] * 512

    @pytest.mark.asyncio
    async def test_no_gemini_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch, EMBEDDING_API="gemini", GEMINI_API_KEY="")
        emb._gemini_configured = False
        with pytest.raises(RuntimeError, match="no Gemini API key"):
            await emb.embed_text("test")


class TestDimensionAndMode:
    def test_dimension_is_512(self, monkeypatch: pytest.MonkeyPatch) -> None:
        emb = _reset_mode(monkeypatch)
        assert emb.DIMENSION == 512
        for mode in ("local", "gemini", "none"):
            emb2 = _reset_mode(monkeypatch, EMBEDDING_API=mode)
            assert emb2.DIMENSION == 512

    def test_embedding_mode_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for mode in ("local", "gemini", "none"):
            emb = _reset_mode(monkeypatch, EMBEDDING_API=mode)
            assert emb.embedding_mode() == mode


class TestCosineAndBatch:
    def test_cosine_identical(self) -> None:
        from reporelay_mvp.embedding import cosine

        v = [0.5, 0.5, 0.5, 0.5]
        assert abs(cosine(v, v) - 1.0) < 1e-6

    def test_cosine_orthogonal(self) -> None:
        from reporelay_mvp.embedding import cosine

        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine(a, b)) < 1e-6

    def test_cosine_zero_vector(self) -> None:
        from reporelay_mvp.embedding import cosine

        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert cosine(a, b) == 0.0

    def test_cosine_batch(self) -> None:
        from reporelay_mvp.embedding import cosine_batch_one_vs_many

        one = [1.0, 0.0]
        many = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        scores = cosine_batch_one_vs_many(one, many)
        assert len(scores) == 3
        assert abs(scores[0] - 1.0) < 1e-6
        assert abs(scores[1]) < 1e-6
        assert abs(scores[2] - (-1.0)) < 1e-6
