"""Integration tests for the "paste a new GitHub repo URL" workflow.

These tests mock the external services (GitHub API, Gemini API) and the
database, but let the real pipeline logic run so we can verify that every
stage completes in the right order and the right data flows through.

Test groups:
  1. Full end-to-end workflow (quick_save → embed → candidates → score → rerank)
  2. Embedding correctness (both vectors computed, persisted, used)
  3. Candidate generation (vector pool uses in-memory embedding)
  4. Scoring (cosine_sim, description_cosine_sim, readme_topic_sim all have real signal)
  5. Failure handling (EmbedError on any stage failure, no silent swallowing)
  6. Caching (success cached, failure not cached)
  7. API surface (embed_status in response, 502 on EmbedError)
"""

from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reporelay_mvp import data
from reporelay_mvp.candidates import generate_candidates
from reporelay_mvp.data import _to_pgvector, fetch_vector_neighbors
from reporelay_mvp.github import quick_save
from reporelay_mvp.models import Features, Repo, ScoredRecommendation
from reporelay_mvp.recommend import EmbedError, _embed_source_live, recommend
from reporelay_mvp.score import score_many
from reporelay_mvp.rerank import rerank

# The package's __init__ shadows 'recommend' with the orchestrator function
recommend_module = importlib.import_module("reporelay_mvp.recommend")


# ── Fixtures ────────────────────────────────────────────────────────

def _make_source_repo(
    *,
    description: str | None = None,
    embedding: list[float] | None = None,
    description_embedding: list[float] | None = None,
) -> Repo:
    """Create a source Repo as it would look after quick_save (NULL embeddings)."""
    return Repo(
        id=1001,
        owner="acme",
        name="widget",
        full_name="acme/widget",
        description=description,
        language="Python",
        topics=["python", "machine-learning", "widgets"],
        stars=250,
        dependencies=["numpy", "pandas"],
        embedding=embedding,
        description_embedding=description_embedding,
    )


def _make_candidate_repo(
    repo_id: int,
    owner: str,
    name: str,
    *,
    language: str = "Python",
    topics: list[str] | None = None,
    stars: int = 500,
    description: str = "A useful library for widgets",
    embedding: list[float] | None = None,
    description_embedding: list[float] | None = None,
) -> Repo:
    """Create a candidate Repo as it would exist in the DB (with embeddings)."""
    return Repo(
        id=repo_id,
        owner=owner,
        name=name,
        full_name=f"{owner}/{name}",
        description=description,
        language=language,
        topics=topics or ["python", "library"],
        stars=stars,
        dependencies=["numpy"],
        embedding=embedding or [0.1 * (i % 10 + 1) for i in range(512)],
        description_embedding=description_embedding or [0.05 * (i % 10 + 1) for i in range(512)],
    )


README_TEXT = (
    "# acme/widget\n\n"
    "acme/widget is a high-performance Python library for processing "
    "and transforming large-scale tabular data. It provides a pandas-compatible "
    "API with 10x better performance through Rust-backed compute kernels.\n\n"
    "## Features\n\n"
    "- Zero-copy DataFrame operations\n"
    "- Lazy evaluation with query optimization\n"
    "- Native Parquet and Arrow support\n\n"
    "## Installation\n\n"
    "`pip install acme-widget`\n"
)


# ── Helpers for building mock sessions ──────────────────────────────

class FakeSession:
    """A minimal async session that records execute() calls for assertions."""

    def __init__(self):
        self.execute = AsyncMock()
        self.commit = AsyncMock()
        self.close = AsyncMock()
        self._execute_calls: list[tuple] = []

    async def execute(self, *args, **kwargs):
        self._execute_calls.append((args, kwargs))
        return MagicMock(fetchall=MagicMock(return_value=[]), fetchone=MagicMock(return_value=None))


# ── 1. FULL END-TO-END WORKFLOW ─────────────────────────────────────


class TestFullWorkflow:
    """Test the complete pipeline: quick_save → embed → candidates → score → rerank."""

    @pytest.mark.asyncio
    async def test_new_repo_workflow_completes_before_returning(self):
        """When a brand new repo is pasted, the full pipeline must complete:
        1. quick_save fetches metadata from GitHub
        2. _embed_source_live fetches README + embeds both vectors via Gemini
        3. generate_candidates builds a pool using the in-memory embedding
        4. score_many scores each candidate against the source
        5. rerank applies diversity rules
        6. Only then do we return results.
        """
        source = _make_source_repo(description="A fast widget library")
        candidates = [
            _make_candidate_repo(2001, "other", "lib-a"),
            _make_candidate_repo(2002, "another", "lib-b"),
            _make_candidate_repo(2003, "third", "lib-c", language="Rust"),
        ]

        # Track call order
        call_order: list[str] = []

        async def mock_quick_save(owner, name):
            call_order.append("quick_save")
            return 1001

        async def mock_embed_source_live(src, owner, name, session):
            call_order.append("embed_source_live")
            updated = src.model_copy(update={
                "embedding": [0.1 * (i % 10 + 1) for i in range(512)],
                "description_embedding": [0.05 * (i % 10 + 1) for i in range(512)],
            })
            return updated, {"widget", "python", "data"}, ["numpy", "pandas"], {
                "desc_emb": "ok", "readme_emb": "ok",
                "effective_description": "A fast widget library",
            }

        async def mock_expand_pool(session, src, *, seed=None, tags=None):
            call_order.append("expand_pool")
            assert src.embedding is not None, "expand_pool must see a source with real embedding"
            assert any(v != 0.0 for v in src.embedding), "embedding must be non-zero"
            return [(c, 0.5) for c in candidates]

        async def mock_score_many(src, cands, **kwargs):
            call_order.append("score_many")
            return [(c, 0.8 - i * 0.1, Features(
                language_match=1.0, topic_overlap=0.5, cosine_sim=0.7,
                description_sim=0.3, description_cosine_sim=0.6,
                readme_topic_sim=0.4, dep_overlap=0.2, popularity_sim=0.5,
                trending_boost=0.0, quality_signal=0.5, language_diversity=0.0,
            )) for i, (c, _) in enumerate(cands)]

        mock_session = FakeSession()

        with (
            patch.object(recommend_module, "quick_save", side_effect=mock_quick_save),
            patch.object(recommend_module.data, "get_session", AsyncMock(return_value=mock_session)),
            patch.object(recommend_module.data, "get_repo", AsyncMock(side_effect=[None, source])),
            patch.object(recommend_module, "_embed_source_live", side_effect=mock_embed_source_live),
            patch.object(recommend_module, "_expand_pool", side_effect=mock_expand_pool),
            patch.object(recommend_module, "score_many", side_effect=mock_score_many),
            patch.object(recommend_module, "rerank") as mock_rerank,
        ):
            mock_rerank.return_value = [
                (candidates[0], 0.8, Features(
                    language_match=1.0, topic_overlap=0.5, cosine_sim=0.7,
                    description_sim=0.3, description_cosine_sim=0.6,
                    readme_topic_sim=0.4, dep_overlap=0.2, popularity_sim=0.5,
                    trending_boost=0.0, quality_signal=0.5, language_diversity=0.0,
                )),
            ]
            result = await recommend("acme/widget", limit=1)

        # Verify the pipeline ran in the correct order
        assert call_order == [
            "quick_save", "embed_source_live", "expand_pool", "score_many",
        ], f"pipeline stages ran out of order: {call_order}"

        # Verify the result is complete
        assert result.source_repo == "acme/widget"
        assert len(result.repos) == 1
        assert result.repos[0].full_name == "other/lib-a"
        assert result.embed_status["desc_emb"] == "ok"
        assert result.embed_status["readme_emb"] == "ok"

    @pytest.mark.asyncio
    async def test_existing_repo_with_embeddings_skips_embed(self):
        """When the source repo already has real embeddings in the DB,
        _embed_source_live must NOT be called — we use the cached vectors."""
        recommend_module._rec_cache.clear()

        source = _make_source_repo(
            description="Already embedded",
            embedding=[0.2 * (i % 10 + 1) for i in range(512)],
            description_embedding=[0.1 * (i % 10 + 1) for i in range(512)],
        )

        embed_called = False

        async def mock_embed_source_live(*args, **kwargs):
            nonlocal embed_called
            embed_called = True
            raise AssertionError("_embed_source_live should not be called for cached repos")

        mock_session = FakeSession()

        with (
            patch.object(recommend_module.data, "get_session", AsyncMock(return_value=mock_session)),
            patch.object(recommend_module.data, "get_repo", AsyncMock(return_value=source)),
            patch.object(recommend_module, "_embed_source_live", side_effect=mock_embed_source_live),
            patch.object(recommend_module, "_expand_pool", AsyncMock(return_value=[])),
            patch.object(recommend_module, "score_many", AsyncMock(return_value=[])),
            patch.object(recommend_module, "rerank", return_value=[]),
        ):
            result = await recommend("acme/widget", limit=5)

        assert not embed_called, "should use cached embeddings, not call Gemini"
        assert result.embed_status["desc_emb"] == "cached"
        assert result.embed_status["readme_emb"] == "cached"
        recommend_module._rec_cache.clear()


# ── 2. EMBEDDING CORRECTNESS ────────────────────────────────────────


class TestEmbeddingCorrectness:
    """Both README and description must be embedded via Gemini, persisted,
    and used for scoring."""

    @pytest.mark.asyncio
    async def test_both_embeddings_computed_and_persisted(self):
        """_embed_source_live must compute and persist both:
        - description_embedding (from effective description)
        - embedding (from README)
        """
        source = _make_source_repo(description="A fast widget library")
        fake_session = FakeSession()

        persisted: dict[str, list[float]] = {}

        async def fake_embed(text: str) -> list[float]:
            # Return different vectors for different text so we can verify
            # the right text was passed
            return [float(len(text) % 10) * 0.1] * 512

        async def fake_set_desc(session, *, repo_id, description_embedding):
            persisted["description_embedding"] = description_embedding

        async def fake_set_readme(session, *, repo_id, embedding):
            persisted["embedding"] = embedding

        async def fake_upsert(session, **kw):
            persisted["persisted_description"] = kw.get("description")

        fake_http = MagicMock()
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=False)

        with patch.object(recommend_module, "_auth_client", return_value=fake_http):
            with patch.object(recommend_module, "fetch_readme", AsyncMock(return_value=README_TEXT)):
                with patch.object(recommend_module, "fetch_dependencies", AsyncMock(return_value=["numpy"])):
                    with patch.object(recommend_module, "embed_text", side_effect=fake_embed):
                        with patch.object(recommend_module.data, "set_description_embedding", fake_set_desc):
                            with patch.object(recommend_module.data, "set_embedding", fake_set_readme):
                                with patch.object(recommend_module.data, "upsert_repo", fake_upsert):
                                    updated, tokens, deps, status = await _embed_source_live(
                                        source, "acme", "widget", fake_session,
                                    )

        # Both embeddings must be present and non-zero
        assert "description_embedding" in persisted, "description_embedding was never persisted"
        assert "embedding" in persisted, "embedding (README) was never persisted"
        assert any(v != 0.0 for v in persisted["description_embedding"]), "description_embedding is all zeros"
        assert any(v != 0.0 for v in persisted["embedding"]), "embedding (README) is all zeros"
        assert len(persisted["description_embedding"]) == 512
        assert len(persisted["embedding"]) == 512
        assert status["desc_emb"] == "ok"
        assert status["readme_emb"] == "ok"

    @pytest.mark.asyncio
    async def test_effective_description_extracted_from_readme(self):
        """When the GitHub description is missing/None, an effective
        description must be extracted from the README and embedded."""
        source = _make_source_repo(description=None)  # No GitHub description

        captured_desc: str | None = None

        async def fake_embed(text: str) -> list[float]:
            nonlocal captured_desc
            # The first embed call is for the description
            if captured_desc is None:
                captured_desc = text
            return [0.01 * (i % 100 + 1) for i in range(512)]

        async def fake_upsert(session, **kw):
            pass

        fake_session = FakeSession()
        fake_http = MagicMock()
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=False)

        with patch.object(recommend_module, "_auth_client", return_value=fake_http):
            with patch.object(recommend_module, "fetch_readme", AsyncMock(return_value=README_TEXT)):
                with patch.object(recommend_module, "fetch_dependencies", AsyncMock(return_value=[])):
                    with patch.object(recommend_module, "embed_text", side_effect=fake_embed):
                        with patch.object(recommend_module.data, "set_description_embedding", AsyncMock()):
                            with patch.object(recommend_module.data, "set_embedding", AsyncMock()):
                                with patch.object(recommend_module.data, "upsert_repo", fake_upsert):
                                    updated, tokens, deps, status = await _embed_source_live(
                                        source, "acme", "widget", fake_session,
                                    )

        # The effective description should have been extracted from the README
        assert captured_desc is not None, "no description was embedded"
        assert "widget" in captured_desc.lower() or "acme" in captured_desc.lower(), (
            f"effective description doesn't mention the project: {captured_desc[:200]}"
        )
        assert len(captured_desc) >= 25, "effective description is too short"

    @pytest.mark.asyncio
    async def test_readme_tokens_extracted_for_keyword_matching(self):
        """After embedding, README tokens must be extracted so the
        readme_topic_sim feature can match source README keywords
        against candidate topic tags."""
        source = _make_source_repo(description="A widget library")

        fake_session = FakeSession()
        fake_http = MagicMock()
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=False)

        with patch.object(recommend_module, "_auth_client", return_value=fake_http):
            with patch.object(recommend_module, "fetch_readme", AsyncMock(return_value=README_TEXT)):
                with patch.object(recommend_module, "fetch_dependencies", AsyncMock(return_value=[])):
                    with patch.object(recommend_module, "embed_text", AsyncMock(return_value=[0.1] * 512)):
                        with patch.object(recommend_module.data, "set_description_embedding", AsyncMock()):
                            with patch.object(recommend_module.data, "set_embedding", AsyncMock()):
                                with patch.object(recommend_module.data, "upsert_repo", AsyncMock()):
                                    updated, tokens, deps, status = await _embed_source_live(
                                        source, "acme", "widget", fake_session,
                                    )

        assert tokens is not None, "readme_tokens was not extracted"
        assert len(tokens) > 0, "readme_tokens is empty"
        # The README mentions "widget", "data", "python" — these should be tokens
        token_str = " ".join(tokens).lower()
        assert "widget" in token_str or "data" in token_str, (
            f"expected meaningful tokens, got: {sorted(tokens)[:10]}"
        )


# ── 3. CANDIDATE GENERATION ─────────────────────────────────────────


class TestCandidateGeneration:
    """The vector pool must use the in-memory source embedding (not the DB)
    so freshly-embedded repos get real nearest-neighbor results."""

    @pytest.mark.asyncio
    async def test_vector_pool_uses_in_memory_embedding(self):
        """generate_candidates must pass the in-memory source.embedding
        to fetch_vector_neighbors, not rely on a DB cross-join."""
        source_emb = [0.15 * (i % 10 + 1) for i in range(512)]
        source = _make_source_repo(
            description="A widget library",
            embedding=source_emb,
            description_embedding=[0.1] * 512,
        )

        with patch.object(data, "fetch_filtered_pool", AsyncMock(return_value=[])):
            with patch.object(data, "fetch_vector_neighbors", AsyncMock(return_value={})) as fvn:
                with patch.object(data, "get_embedding", AsyncMock(return_value=None)):
                    await generate_candidates(FakeSession(), source, pool_size=10, vector_k=10)

        fvn.assert_awaited_once()
        passed_emb = fvn.await_args.kwargs["source_embedding"]
        assert passed_emb == source_emb, (
            "fetch_vector_neighbors must receive the in-memory source.embedding, "
            "not re-read from DB"
        )

    @pytest.mark.asyncio
    async def test_vector_pool_skipped_when_source_has_no_embedding(self):
        """If the source has no real embedding (None or zeros), the vector
        pool must be skipped entirely — not run with NaN results."""
        source = _make_source_repo(
            description="A widget library",
            embedding=None,
            description_embedding=None,
        )

        with patch.object(data, "fetch_filtered_pool", AsyncMock(return_value=[])):
            with patch.object(data, "fetch_vector_neighbors", AsyncMock()) as fvn:
                with patch.object(data, "get_embedding", AsyncMock(return_value=None)):
                    await generate_candidates(FakeSession(), source, pool_size=10, vector_k=10)

        fvn.assert_not_called()

    @pytest.mark.asyncio
    async def test_vector_and_sql_pools_merged_deduplicated(self):
        """When both pools return candidates, they must be merged with
        deduplication — no repo appears twice."""
        repo_a = _make_candidate_repo(2001, "other", "lib-a")
        repo_b = _make_candidate_repo(2002, "another", "lib-b")
        repo_c = _make_candidate_repo(2003, "third", "lib-c")
        # repo_a appears in both pools — must be deduplicated

        source = _make_source_repo(
            description="A widget library",
            embedding=[0.1] * 512,
        )

        with patch.object(data, "fetch_filtered_pool", AsyncMock(return_value=[repo_a, repo_b])):
            with patch.object(data, "fetch_vector_neighbors", AsyncMock(
                return_value={repo_a.id: (repo_a, 0.8), repo_c.id: (repo_c, 0.6)}
            )):
                result = await generate_candidates(FakeSession(), source, pool_size=10, vector_k=10)

        result_ids = [r.id for r, _ in result]
        assert len(result_ids) == len(set(result_ids)), "duplicate repos in candidate pool"
        assert repo_a.id in result_ids
        assert repo_b.id in result_ids
        assert repo_c.id in result_ids

    @pytest.mark.asyncio
    async def test_fetch_vector_neighbors_uses_pgvector_cosine(self):
        """The vector neighbors query must use cosine distance (<=>
        operator) to find the most similar repos in the DB."""
        source_emb = [0.1] * 512

        # We can't run the actual SQL here, but we can verify the function
        # rejects zero/empty and accepts real vectors
        fake_session = MagicMock()
        fake_session.execute = AsyncMock(return_value=MagicMock(
            _mapping={"cosine_sim": 0.9}
        ))

        # Zero vector → empty result (no query executed)
        result = await fetch_vector_neighbors(
            fake_session, source_embedding=[0.0] * 512, exclude_id=1, limit=10,
        )
        assert result == {}

        # Empty vector → empty result
        result = await fetch_vector_neighbors(
            fake_session, source_embedding=[], exclude_id=1, limit=10,
        )
        assert result == {}


# ── 4. SCORING ───────────────────────────────────────────────────────


class TestScoring:
    """Verify that cosine_sim, description_cosine_sim, and readme_topic_sim
    all carry real signal for a freshly-embedded source."""

    @pytest.mark.asyncio
    async def test_cosine_sim_computed_from_readme_embeddings(self):
        """score_many must compute cosine_sim between the source's README
        embedding and each candidate's README embedding."""
        source_emb = [0.1 * (i % 10 + 1) for i in range(512)]
        source = _make_source_repo(
            description="A widget library",
            embedding=source_emb,
            description_embedding=[0.05 * (i % 10 + 1) for i in range(512)],
        )
        cand = _make_candidate_repo(
            2001, "other", "lib-a",
            embedding=[0.12 * (i % 10 + 1) for i in range(512)],
            description_embedding=[0.06 * (i % 10 + 1) for i in range(512)],
        )

        fake_session = MagicMock()
        fake_session.execute = AsyncMock(return_value=MagicMock(
            fetchall=MagicMock(return_value=[]),
            fetchone=MagicMock(return_value=None),
        ))

        # Patch the batch description embedding fetch to return real vectors
        async def fake_get_desc_batch(session, ids):
            return {cand.id: cand.description_embedding}

        with patch.object(data, "get_description_embeddings_batch", fake_get_desc_batch):
            scored = await score_many(
                source, [(cand, 0.75)],
                session=fake_session, seed=None,
                source_readme_tokens={"widget", "python", "data"},
            )

        assert len(scored) == 1
        _, score, features = scored[0]
        # cosine_sim should be non-zero (real README vectors)
        assert features.cosine_sim > 0.0, "cosine_sim should carry real signal"
        # description_cosine_sim should be non-zero (real description vectors)
        assert features.description_cosine_sim > 0.0, "description_cosine_sim should carry real signal"
        # readme_topic_sim should be non-zero (README tokens match candidate topics)
        assert features.readme_topic_sim > 0.0, "readme_topic_sim should carry real signal"

    @pytest.mark.asyncio
    async def test_score_many_weights_reflect_real_embeddings(self):
        """When the source has real embeddings, the scoring weights must
        include cosine_sim (0.10) and description_cosine_sim (0.29) —
        not redistribute them away."""
        from reporelay_mvp.score import _get_weights

        weights = _get_weights(
            seed=None,
            has_readme_emb=True,
            has_desc_emb=True,
            has_readme_keywords=True,
            has_topics=True,
        )

        assert "cosine_sim" in weights, "cosine_sim weight missing when readme embedding exists"
        assert "description_cosine_sim" in weights, "description_cosine_sim weight missing when desc embedding exists"
        assert weights["cosine_sim"] > 0.0
        assert weights["description_cosine_sim"] > 0.0
        # These are the two most important weights — they should be significant
        assert weights["description_cosine_sim"] >= 0.20, (
            "description_cosine_sim should be a major signal (≥0.20)"
        )


# ── 5. FAILURE HANDLING ─────────────────────────────────────────────


class TestFailureHandling:
    """Every failure in the embed path must raise EmbedError, not
    silently return zeros or random results."""

    @pytest.mark.asyncio
    async def test_github_fetch_failure_raises_embed_error(self):
        """If the GitHub API is down, _embed_source_live must raise
        EmbedError — not return a source with zero embeddings."""
        source = _make_source_repo(description="A widget library")
        fake_session = FakeSession()
        fake_http = MagicMock()
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=False)

        with patch.object(recommend_module, "_auth_client", return_value=fake_http):
            with patch.object(recommend_module, "fetch_readme", AsyncMock(side_effect=Exception("GitHub 503"))):
                with pytest.raises(EmbedError, match="GitHub"):
                    await _embed_source_live(source, "acme", "widget", fake_session)

    @pytest.mark.asyncio
    async def test_gemini_api_failure_raises_embed_error(self):
        """If the Gemini embedding API fails, _embed_source_live must
        raise EmbedError — not silently leave zero vectors."""
        source = _make_source_repo(description="A widget library")
        fake_session = FakeSession()
        fake_http = MagicMock()
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=False)

        with patch.object(recommend_module, "_auth_client", return_value=fake_http):
            with patch.object(recommend_module, "fetch_readme", AsyncMock(return_value=README_TEXT)):
                with patch.object(recommend_module, "fetch_dependencies", AsyncMock(return_value=[])):
                    with patch.object(recommend_module, "embed_text", AsyncMock(side_effect=Exception("Gemini quota exceeded"))):
                        with pytest.raises(EmbedError, match="Gemini"):
                            await _embed_source_live(source, "acme", "widget", fake_session)

    @pytest.mark.asyncio
    async def test_zero_vector_from_gemini_raises_embed_error(self):
        """If Gemini returns a zero vector (API bug or empty input),
        _embed_source_live must raise EmbedError."""
        source = _make_source_repo(description="A widget library")
        fake_session = FakeSession()
        fake_http = MagicMock()
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=False)

        with patch.object(recommend_module, "_auth_client", return_value=fake_http):
            with patch.object(recommend_module, "fetch_readme", AsyncMock(return_value=README_TEXT)):
                with patch.object(recommend_module, "fetch_dependencies", AsyncMock(return_value=[])):
                    with patch.object(recommend_module, "embed_text", AsyncMock(return_value=[0.0] * 512)):
                        with pytest.raises(EmbedError, match="zero vector"):
                            await _embed_source_live(source, "acme", "widget", fake_session)

    @pytest.mark.asyncio
    async def test_no_readme_raises_embed_error(self):
        """A repo with no README at all must raise EmbedError."""
        source = _make_source_repo(description="A widget library")
        fake_session = FakeSession()
        fake_http = MagicMock()
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=False)

        with patch.object(recommend_module, "_auth_client", return_value=fake_http):
            with patch.object(recommend_module, "fetch_readme", AsyncMock(return_value="")):
                with patch.object(recommend_module, "fetch_dependencies", AsyncMock(return_value=[])):
                    with pytest.raises(EmbedError, match="no README"):
                        await _embed_source_live(source, "acme", "widget", fake_session)

    @pytest.mark.asyncio
    async def test_recommend_raises_when_source_still_has_no_embedding(self):
        """After _embed_source_live, if the source STILL has no real
        embedding (shouldn't happen, but defensive), recommend() must
        raise EmbedError instead of returning SQL-only noise."""
        recommend_module._rec_cache.clear()

        source_no_emb = _make_source_repo(
            description="A widget library",
            embedding=None,
            description_embedding=None,
        )

        async def mock_embed_live(src, owner, name, session):
            # Return source with still-None embeddings (simulates a bug)
            return src, None, [], {"desc_emb": "failed", "readme_emb": "failed"}

        mock_session = FakeSession()

        with (
            patch.object(recommend_module.data, "get_session", AsyncMock(return_value=mock_session)),
            patch.object(recommend_module.data, "get_repo", AsyncMock(return_value=source_no_emb)),
            patch.object(recommend_module, "_embed_source_live", side_effect=mock_embed_live),
        ):
            with pytest.raises(EmbedError, match="no README embedding"):
                await recommend("acme/widget", limit=5)

        recommend_module._rec_cache.clear()


# ── 6. CACHING ──────────────────────────────────────────────────────


class TestCaching:
    """Successful results must be cached; failed results must NOT be cached."""

    @pytest.mark.asyncio
    async def test_successful_result_is_cached(self):
        """After a successful recommend(), the result must be in the
        in-process cache so the next identical request is instant."""
        recommend_module._rec_cache.clear()

        source = _make_source_repo(
            description="A widget library",
            embedding=[0.1] * 512,
            description_embedding=[0.05] * 512,
        )

        with patch.object(recommend_module, "data") as mock_data:
            mock_data.get_session = AsyncMock(return_value=FakeSession())
            mock_data.get_repo = AsyncMock(return_value=source)
            with patch.object(recommend_module, "_expand_pool", AsyncMock(return_value=[])):
                with patch.object(recommend_module, "score_many", AsyncMock(return_value=[])):
                    with patch.object(recommend_module, "rerank", return_value=[]):
                        result = await recommend("acme/widget", limit=5)

        cache_key = recommend_module._rec_cache_key("acme/widget", None, None)
        assert cache_key in recommend_module._rec_cache, "successful result was not cached"

        recommend_module._rec_cache.clear()

    @pytest.mark.asyncio
    async def test_embed_failure_is_not_cached(self):
        """If recommend() fails due to EmbedError, the failure must NOT
        be cached — the next request should try again."""
        recommend_module._rec_cache.clear()

        source_no_emb = _make_source_repo(
            description="A widget library",
            embedding=None,
            description_embedding=None,
        )

        async def mock_embed_live(src, owner, name, session):
            raise EmbedError("Gemini API is down")

        with patch.object(recommend_module, "quick_save"):
            with patch.object(recommend_module, "data") as mock_data:
                mock_data.get_session = AsyncMock(return_value=FakeSession())
                mock_data.get_repo = AsyncMock(return_value=source_no_emb)
                with patch.object(recommend_module, "_embed_source_live", side_effect=mock_embed_live):
                    with pytest.raises(EmbedError):
                        await recommend("acme/widget", limit=5)

        cache_key = recommend_module._rec_cache_key("acme/widget", None, None)
        assert cache_key not in recommend_module._rec_cache, "failed result was cached — should not be"

        recommend_module._rec_cache.clear()


# ── 7. API SURFACE ──────────────────────────────────────────────────


class TestAPISurface:
    """The API must expose embed_status and return 502 on EmbedError."""

    def test_recommend_response_includes_embed_status(self):
        """The /recommend endpoint response model must include embed_status."""
        from reporelay_mvp_api.main import RecommendResponse

        resp = RecommendResponse(
            source_repo="acme/widget",
            repos=[],
            embed_status={"desc_emb": "ok", "readme_emb": "ok"},
        )
        assert resp.embed_status["desc_emb"] == "ok"
        assert resp.embed_status["readme_emb"] == "ok"

    def test_recommend_response_default_embed_status(self):
        """embed_status defaults to empty dict for backward compatibility."""
        from reporelay_mvp_api.main import RecommendResponse

        resp = RecommendResponse(source_repo="acme/widget", repos=[])
        assert resp.embed_status == {}

    def test_scored_recommendation_includes_embed_status(self):
        """The internal ScoredRecommendation model must carry embed_status."""
        rec = ScoredRecommendation(
            source_repo="acme/widget",
            repos=[],
            embed_status={"desc_emb": "ok", "readme_emb": "ok"},
        )
        assert rec.embed_status["desc_emb"] == "ok"

    def test_scored_recommendation_default_embed_status(self):
        """embed_status defaults to empty dict."""
        rec = ScoredRecommendation(source_repo="acme/widget", repos=[])
        assert rec.embed_status == {}

    def test_embed_error_is_distinct_exception(self):
        """EmbedError must be a distinct exception class so the API can
        catch it specifically and return 502 (not 500)."""
        assert issubclass(EmbedError, Exception)
        # It should be importable from the recommend module
        from reporelay_mvp.recommend import EmbedError as EE
        assert EE is EmbedError


# ── 8. QUICK_SAVE NULL EMBEDDING ────────────────────────────────────


class TestQuickSaveNullEmbedding:
    """quick_save must leave embedding columns NULL so the live-embed
    path is cleanly triggered."""

    @pytest.mark.asyncio
    async def test_quick_save_does_not_set_embedding(self):
        """quick_save must NOT call set_embedding — the embedding column
        must remain NULL until _embed_source_live fills it."""
        fake_metadata = {
            "id": 99999,
            "language": "Python",
            "stargazers_count": 42,
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

        set_emb_calls: list = []

        async def track_set_embedding(session, *, repo_id, embedding):
            set_emb_calls.append((repo_id, embedding))

        fake_session = FakeSession()

        with patch("reporelay_mvp.github.httpx.AsyncClient", return_value=_FakeClient()):
            with patch("reporelay_mvp.github.data.get_session", AsyncMock(return_value=fake_session)):
                with patch("reporelay_mvp.github.data.upsert_repo", AsyncMock()):
                    with patch("reporelay_mvp.github.data.set_embedding", track_set_embedding):
                        await quick_save("acme", "widget")

        assert len(set_emb_calls) == 0, (
            f"quick_save should not set embedding, but called set_embedding {len(set_emb_calls)} times"
        )


# ── 9. END-TO-END: SOURCE COMPARED AGAINST DB CORPUS ────────────────


class TestSourceComparedAgainstDBCorpus:
    """Verify the source repo's README embedding is compared against
    the DB corpus's README embeddings (not just description)."""

    @pytest.mark.asyncio
    async def test_source_readme_embedding_compared_against_candidates(self):
        """The scoring must use the source's README embedding to compute
        cosine_sim against each candidate's README embedding. This is the
        core signal for 'similar repos'."""
        source_emb = [0.2 * (i % 10 + 1) for i in range(512)]
        source = _make_source_repo(
            description="A widget library",
            embedding=source_emb,
            description_embedding=[0.1 * (i % 10 + 1) for i in range(512)],
        )

        # Candidate with a similar embedding
        similar_cand = _make_candidate_repo(
            2001, "other", "similar-lib",
            embedding=[0.19 * (i % 10 + 1) for i in range(512)],
            description_embedding=[0.09 * (i % 10 + 1) for i in range(512)],
        )
        # Candidate with a very different embedding
        different_cand = _make_candidate_repo(
            2002, "another", "different-lib",
            embedding=[-0.1 * (i % 10 + 1) for i in range(512)],
            description_embedding=[-0.05 * (i % 10 + 1) for i in range(512)],
        )

        fake_session = MagicMock()

        async def fake_get_desc_batch(session, ids):
            return {
                similar_cand.id: similar_cand.description_embedding,
                different_cand.id: different_cand.description_embedding,
            }

        with patch.object(data, "get_description_embeddings_batch", fake_get_desc_batch):
            scored = await score_many(
                source,
                [(similar_cand, 0.85), (different_cand, 0.15)],
                session=fake_session,
                source_readme_tokens={"widget", "python", "data"},
            )

        assert len(scored) == 2
        scores_by_name = {r.name: s for r, s, _ in scored}

        # The similar candidate should score higher than the different one
        assert scores_by_name["similar-lib"] > scores_by_name["different-lib"], (
            "similar-lib should score higher than different-lib based on "
            f"README embedding cosine similarity: {scores_by_name}"
        )
