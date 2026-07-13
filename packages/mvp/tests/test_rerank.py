from reporelay_mvp.models import Repo
from reporelay_mvp.rerank import rerank


def _repo(id, owner, stars=0):
    return Repo(
        id=id, owner=owner, name=f"r{id}", full_name=f"{owner}/r{id}",
        description="", language=None, topics=[], stars=stars,
        dependencies=[], embedding=None,
    )


def test_rerank_drops_source():
    source = _repo(1, "source")
    candidates = [
        (_repo(1, "source"), 0.9, None),
        (_repo(2, "other"), 0.8, None),
    ]
    result = rerank(source, candidates, limit=10)
    assert len(result) == 1
    assert result[0][0].id == 2


def test_rerank_drops_same_owner():
    source = _repo(1, "facebook")
    candidates = [
        (_repo(2, "facebook"), 0.9, None),
        (_repo(3, "google"), 0.8, None),
    ]
    result = rerank(source, candidates, limit=10)
    assert len(result) == 1
    assert result[0][0].owner == "google"


def test_rerank_one_per_owner():
    source = _repo(1, "source")
    candidates = [
        (_repo(2, "meta"), 0.9, None),
        (_repo(3, "meta"), 0.85, None),
        (_repo(4, "google"), 0.8, None),
    ]
    result = rerank(source, candidates, limit=10)
    assert len(result) == 2
    owners = {r.owner for r, _, _ in result}
    assert owners == {"meta", "google"}


def test_rerank_respects_limit():
    source = _repo(1, "source")
    candidates = [
        (_repo(i, f"org{i}"), 1.0 - i * 0.05, None)
        for i in range(2, 14)
    ]
    result = rerank(source, candidates, limit=3)
    assert len(result) == 3


def test_rerank_sorts_by_score_without_seed():
    source = _repo(1, "source")
    candidates = [
        (_repo(2, "b"), 0.5, None),
        (_repo(3, "a"), 0.9, None),
        (_repo(4, "c"), 0.3, None),
    ]
    result = rerank(source, candidates, limit=10)
    scores = [s for _, s, _ in result]
    assert scores == [0.9, 0.5, 0.3]
