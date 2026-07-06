"""Tests for the paste-a-new-GitHub-URL embedding flow.

These exercise the live embed path on a Repo model:
  - quick_save leaves the embedding columns NULL (no zero-vector poison)
  - generate_candidates uses an in-memory embedding even when the DB
    row is still NULL
  - fetch_vector_neighbors rejects a zero/empty source embedding
    rather than producing NaN cosines
  - _embed_source_live generates an effective description from the
    README when the GitHub description is missing/short
  - _embed_source_live raises EmbedError (not silent return) when
    the embedding API fails or returns zeros
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reporelay_mvp import data, embedding
from reporelay_mvp.candidates import generate_candidates
from reporelay_mvp.data import _to_pgvector, fetch_vector_neighbors
from reporelay_mvp.github import quick_save
from reporelay_mvp.models import Repo
from reporelay_mvp.recommend import EmbedError, _embed_source_live

# The package's __init__ shadows `recommend` with the orchestrator
# function, so we import the module directly for patch.object.
recommend_module = importlib.import_module("reporelay_mvp.recommend")


# ── 1. quick_save must not poison the embedding column ──────────────


@pytest.mark.asyncio
async def test_quick_save_leaves_embedding_null(monkeypatch):
    """After quick_save, embedding and description_embedding should be NULL
    so the live-embed path is clearly triggered and the vector search
    is never run against a zero-vector source."""
    fake_metadata = {
        "id": 12345,
        "language": "Python",
        "stargazers_count": 100,
        "description": "A test repo",
    }

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, path, **kw):
            if path.endswith("/topics"):
                return MagicMock(status_code=200, json=lambda: {"names": ["python"]})
            return MagicMock(status_code=200, json=lambda: fake_metadata)

    fake_session = AsyncMock()
    fake_session.execute = AsyncMock()
    fake_session.commit = AsyncMock()

    with patch("reporelay_mvp.github.httpx.AsyncClient", return_value=_FakeClient()):
        with patch("reporelay_mvp.github.data.get_session", AsyncMock(return_value=fake_session)):
            with patch("reporelay_mvp.github.data.upsert_repo", AsyncMock()):
                # set_embedding must NOT be called — quick_save should not
                # write a zero-vector placeholder
                with patch("reporelay_mvp.github.data.set_embedding", AsyncMock()) as set_emb:
                    await quick_save("owner", "repo")

    assert set_emb.await_count == 0, (
        "quick_save must not call set_embedding — leave the column NULL "
        "so the live-embed path is the single source of truth"
    )


# ── 2. fetch_vector_neighbors rejects empty/zero source embeddings ──


@pytest.mark.asyncio
async def test_fetch_vector_neighbors_rejects_zero_source(monkeypatch):
    """Passing a zero source embedding must return {} (not execute a query
    that would yield NaN cosines)."""
    fake_session = AsyncMock()
    fake_session.execute = AsyncMock()

    # All-zero source embedding → empty result, no SQL executed
    result = await fetch_vector_neighbors(
        fake_session,
        source_embedding=[0.0] * 512,
        exclude_id=1,
        limit=10,
    )
    assert result == {}
    fake_session.execute.assert_not_called()

    # Empty source embedding → same
    result = await fetch_vector_neighbors(
        fake_session,
        source_embedding=[],
        exclude_id=1,
        limit=10,
    )
    assert result == {}
    fake_session.execute.assert_not_called()


# ── 3. _to_pgvector formats correctly ───────────────────────────────


def test_to_pgvector_format():
    s = _to_pgvector([0.1, 0.2, 0.3])
    assert s.startswith("[")
    assert s.endswith("]")
    assert "0.1" in s and "0.2" in s and "0.3" in s


# ── 4. generate_candidates uses in-memory embedding first ──────────


@pytest.mark.asyncio
async def test_generate_candidates_uses_in_memory_embedding():
    """generate_candidates should call fetch_vector_neighbors with the
    in-memory source.embedding (not re-read from DB)."""
    source = Repo(
        id=99,
        owner="o",
        name="n",
        full_name="o/n",
        description="x",
        language="Python",
        topics=["python"],
        stars=10,
        embedding=[0.5] * 512,
    )

    fake_session = AsyncMock()

    with patch.object(data, "fetch_filtered_pool", AsyncMock(return_value=[])) as fp:
        with patch.object(data, "fetch_vector_neighbors", AsyncMock(return_value={})) as fvn:
            await generate_candidates(fake_session, source, pool_size=10, vector_k=10)

    # Confirm we passed the in-memory embedding, not a DB read
    fvn.assert_awaited_once()
    kwargs = fvn.await_args.kwargs
    assert kwargs["source_embedding"] == [0.5] * 512
    assert kwargs["exclude_id"] == 99


@pytest.mark.asyncio
async def test_generate_candidates_skips_vector_pool_when_source_has_no_embedding():
    """If the source has no real embedding, the vector pool is skipped
    and a clear log is emitted (we don't run a query with a zero vector)."""
    source = Repo(
        id=99,
        owner="o",
        name="n",
        full_name="o/n",
        description="x",
        language="Python",
        topics=["python"],
        stars=10,
        embedding=None,  # not yet embedded
    )

    fake_session = AsyncMock()

    with patch.object(data, "fetch_filtered_pool", AsyncMock(return_value=[])):
        with patch.object(data, "fetch_vector_neighbors", AsyncMock()) as fvn:
            with patch.object(data, "get_embedding", AsyncMock(return_value=None)):
                await generate_candidates(fake_session, source, pool_size=10, vector_k=10)

    fvn.assert_not_called()  # ← critical: no NaN query


# ── 5. _embed_source_live generates effective description ───────────


@pytest.mark.asyncio
async def test_embed_source_live_generates_effective_description():
    """When the GitHub description is missing/short, _embed_source_live
    must extract a purpose statement from the README and embed that
    (and persist it)."""
    r = recommend_module

    source = Repo(
        id=1,
        owner="foo",
        name="bar",
        full_name="foo/bar",
        description=None,  # GitHub description is missing
        language="Python",
        topics=["python"],
        stars=10,
    )

    readme = (
        "# foo/bar\n\n"
        "foo/bar is a fast, dependency-free Python library for parsing "
        "TOML configuration files. It is designed to be embedded into CLI "
        "tools and build systems. The parser handles all standard TOML 1.0 "
        "features including multi-line strings, inline tables, and datetimes.\n\n"
        "## Installation\n\n"
        "`pip install foo-bar`\n"
    )

    fake_session = AsyncMock()
    fake_session.commit = AsyncMock()
    fake_session.execute = AsyncMock()

    captured: dict = {}

    async def fake_set_desc(session, *, repo_id, description_embedding):
        captured["desc_emb_len"] = len(description_embedding)
        captured["desc_emb_nonzero"] = any(v != 0.0 for v in description_embedding)

    async def fake_set_readme(session, *, repo_id, embedding):
        captured["readme_emb_len"] = len(embedding)
        captured["readme_emb_nonzero"] = any(v != 0.0 for v in embedding)

    async def fake_embed(text):
        return [0.01 * (i % 100 + 1) for i in range(512)]

    async def fake_upsert(session, **kw):
        captured["persisted_description"] = kw.get("description")

    fake_http = MagicMock()
    fake_http.__aenter__ = AsyncMock(return_value=fake_http)
    fake_http.__aexit__ = AsyncMock(return_value=False)

    with patch.object(r, "_auth_client", return_value=fake_http):
        with patch.object(r, "fetch_readme", AsyncMock(return_value=readme)):
            with patch.object(r, "fetch_dependencies", AsyncMock(return_value=[])):
                with patch.object(r, "embed_text", side_effect=fake_embed):
                    with patch.object(r.data, "set_description_embedding", fake_set_desc):
                        with patch.object(r.data, "set_embedding", fake_set_readme):
                            with patch.object(r.data, "upsert_repo", fake_upsert):
                                updated, tokens, deps, status = await _embed_source_live(
                                    source, "foo", "bar", fake_session,
                                )

    assert captured.get("desc_emb_len") == 512
    assert captured.get("desc_emb_nonzero") is True
    assert captured.get("readme_emb_len") == 512
    assert captured.get("readme_emb_nonzero") is True
    assert captured.get("persisted_description"), "should persist an effective description"
    assert "foo/bar" in captured["persisted_description"].lower() or "toml" in captured["persisted_description"].lower()
    assert updated.description_embedding and any(v != 0.0 for v in updated.description_embedding)
    assert updated.embedding and any(v != 0.0 for v in updated.embedding)
    assert status["desc_emb"] == "ok"
    assert status["readme_emb"] == "ok"


# ── 6. _embed_source_live fails loudly on embed error ──────────────


@pytest.mark.asyncio
async def test_embed_source_live_raises_on_zero_vector_response():
    """If the embed API returns zeros, _embed_source_live must raise
    EmbedError, not silently leave a zero vector behind."""
    r = recommend_module

    source = Repo(
        id=2, owner="x", name="y", full_name="x/y",
        description="", language="Python", topics=[], stars=0,
    )

    fake_session = AsyncMock()
    fake_session.commit = AsyncMock()
    fake_session.execute = AsyncMock()

    readme = "# x/y\n\nThis is a real readme with enough content to embed. " * 10

    fake_http = MagicMock()
    fake_http.__aenter__ = AsyncMock(return_value=fake_http)
    fake_http.__aexit__ = AsyncMock(return_value=False)

    with patch.object(r, "_auth_client", return_value=fake_http):
        with patch.object(r, "fetch_readme", AsyncMock(return_value=readme)):
            with patch.object(r, "fetch_dependencies", AsyncMock(return_value=[])):
                with patch.object(r, "embed_text", AsyncMock(return_value=[0.0] * 512)):
                    with pytest.raises(EmbedError, match="zero vector"):
                        await _embed_source_live(source, "x", "y", fake_session)


@pytest.mark.asyncio
async def test_embed_source_live_raises_on_no_readme():
    """A repo with no README must raise EmbedError, not return silently."""
    r = recommend_module

    source = Repo(
        id=3, owner="x", name="y", full_name="x/y",
        description="x", language="Python", topics=[], stars=0,
    )

    fake_session = AsyncMock()

    fake_http = MagicMock()
    fake_http.__aenter__ = AsyncMock(return_value=fake_http)
    fake_http.__aexit__ = AsyncMock(return_value=False)

    with patch.object(r, "_auth_client", return_value=fake_http):
        with patch.object(r, "fetch_readme", AsyncMock(return_value="")):
            with patch.object(r, "fetch_dependencies", AsyncMock(return_value=[])):
                with pytest.raises(EmbedError, match="no README"):
                    await _embed_source_live(source, "x", "y", fake_session)
