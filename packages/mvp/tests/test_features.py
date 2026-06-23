import math

from reporelay_mvp.features import (
    _clamp,
    _jaccard,
    _popularity_sim,
    _quality_signal,
    tag_match,
)
from reporelay_mvp.models import Repo


def _repo(**kwargs):
    defaults = {
        "id": 1, "owner": "test", "name": "test", "full_name": "test/test",
        "description": None, "language": None, "topics": [], "stars": 0,
        "dependencies": [], "embedding": None,
    }
    return Repo(**{**defaults, **kwargs})


def test_jaccard_identical():
    assert math.isclose(_jaccard(["a", "b"], ["a", "b"]), 1.0)


def test_jaccard_no_overlap():
    assert _jaccard(["a"], ["b"]) == 0.0


def test_jaccard_partial():
    result = _jaccard(["a", "b", "c"], ["a", "c"])
    assert math.isclose(result, 2 / 3)


def test_jaccard_empty():
    assert _jaccard([], []) == 0.0


def test_popularity_sim_equal():
    result = _popularity_sim(1000, 1000)
    assert math.isclose(result, 1.0)


def test_popularity_sim_bigger_candidate():
    result = _popularity_sim(100, 1000)
    assert result == 1.0


def test_popularity_sim_smaller_candidate():
    result = _popularity_sim(1000, 100)
    assert 0.0 < result < 1.0


def test_popularity_zero_source():
    result = _popularity_sim(0, 100)
    assert result == 1.0


def test_clamp_normal():
    assert _clamp(0.5) == 0.5
    assert _clamp(-0.1) == 0.0
    assert _clamp(2.0) == 1.0


def test_tag_match_full():
    assert tag_match(["react", "typescript"], ["react", "typescript", "jest"]) == 1.0


def test_tag_match_partial():
    assert tag_match(["react", "python"], ["react", "typescript"]) == 0.5


def test_tag_match_none():
    assert tag_match(["react"], ["python", "django"]) == 0.0


def test_tag_match_empty():
    assert tag_match([], ["react"]) == 0.0


def test_quality_signal_empty():
    assert _quality_signal(_repo()) == 0.2


def test_quality_signal_has_language():
    assert math.isclose(_quality_signal(_repo(language="Python")), 0.3)


def test_quality_signal_well_documented():
    r = _repo(
        description="A very detailed description that is definitely longer than sixty characters to pass the threshold check",
        language="Python",
        topics=["web", "framework", "api"],
        dependencies=["a", "b", "c", "d", "e", "f"],
    )
    assert math.isclose(_quality_signal(r), 1.0)
