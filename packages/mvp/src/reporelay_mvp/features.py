"""
Stage 2 of the MVP pipeline: feature engineering.

For each (source, candidate) pair, we compute seven features:

  - language_match  : 1.0 if same language, 0.0 otherwise
  - topic_overlap   : IDF-weighted Jaccard similarity of topic sets.
                       Rare topics (compiler, verilog) count more than
                       common ones (python, javascript).
  - cosine_sim      : 1 - cosine_distance from pgvector (README similarity)
  - dep_overlap     : Jaccard similarity of dependency names
  - popularity_sim  : log-ratio of star counts (source = ceiling)
  - trending_boost  : velocity signal from github.com/trending (0..1)
  - quality_signal  : 1.0 if repo has an embedding (had a README worth
                       embedding), 0.2 otherwise. Proxy for maintenance.

All features are in [0, 1]. The scorer is a fixed weighted sum.
"""

from __future__ import annotations

import math
import threading

from reporelay_mvp.models import Features, Repo

EPS = 1e-9

# ── global IDF cache (computed once from topic distribution) ───────
_idf: dict[str, float] = {}
_idf_lock = threading.Lock()
_idf_loaded = False


def load_topic_idf(topic_counts: dict[str, int]) -> None:
    """Compute IDF weights from corpus topic frequencies. Call once at startup."""
    global _idf, _idf_loaded
    if _idf_loaded:
        return
    with _idf_lock:
        if _idf_loaded:
            return
        if not topic_counts:
            _idf = {}
        else:
            total = sum(topic_counts.values()) or 1
            _idf = {
                topic: math.log(total / max(count, 1))
                for topic, count in topic_counts.items()
            }
        _idf_loaded = True


def _weighted_jaccard(a: list[str], b: list[str]) -> float:
    """IDF-weighted Jaccard. Rare topics contribute more to similarity."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    inter = sa & sb
    if not inter:
        return 0.0
    if not _idf:
        # fallback: unweighted Jaccard
        return len(inter) / (len(union) + EPS)
    inter_weight = sum(_idf.get(t, 0.0) for t in inter)
    union_weight = sum(_idf.get(t, 0.0) for t in union)
    if union_weight < EPS:
        return 0.0
    return inter_weight / union_weight


def compute_features(source: Repo, candidate: Repo, *, cosine_sim: float, filter_cosine_sim: float = 0.0) -> Features:
    src_lang = source.language
    cand_lang = candidate.language
    same_lang = 1.0 if (src_lang and cand_lang and src_lang == cand_lang) else 0.0
    divers_lang = 1.0 if (src_lang and cand_lang and src_lang != cand_lang) else 0.0

    return Features(
        language_match=same_lang,
        topic_overlap=_weighted_jaccard(source.topics, candidate.topics),
        cosine_sim=_clamp(cosine_sim),
        dep_overlap=_jaccard(source.dependencies, candidate.dependencies),
        popularity_sim=_popularity_sim(source.stars, candidate.stars),
        trending_boost=_clamp(candidate.trending_score),
        filter_cosine_sim=_clamp(filter_cosine_sim),
        quality_signal=1.0 if candidate.embedding is not None else 0.2,
        language_diversity=divers_lang,
    )


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    inter = sa & sb
    return len(inter) / (len(union) + EPS)


def _popularity_sim(a: int, b: int) -> float:
    """Log-scaled star comparison. Source's stars set the ceiling; candidate
    gets full score if it meets or exceeds that level. Smaller repos with
    fewer stars than the source are penalized gracefully on a log scale."""
    log_a = math.log1p(max(a, 1))
    log_b = math.log1p(max(b, 1))
    if log_a <= 0:
        return 0.5
    ratio = min(log_b / log_a, 1.0)
    return ratio


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
