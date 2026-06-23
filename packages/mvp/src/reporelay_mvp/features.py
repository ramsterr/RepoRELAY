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
import re
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


def compute_features(source: Repo, candidate: Repo, *, cosine_sim: float, filter_cosine_sim: float = 0.0, description_cosine_sim: float = 0.0, readme_topic_sim: float = 0.0) -> Features:
    src_lang = source.language
    cand_lang = candidate.language
    same_lang = 1.0 if (src_lang and cand_lang and src_lang == cand_lang) else 0.0
    divers_lang = 1.0 if (src_lang and cand_lang and src_lang != cand_lang) else 0.0

    return Features(
        language_match=same_lang,
        topic_overlap=_weighted_jaccard(source.topics, candidate.topics),
        cosine_sim=_clamp(cosine_sim),
        description_sim=_description_sim(source.description, candidate.description),
        description_cosine_sim=_clamp(description_cosine_sim),
        readme_topic_sim=_clamp(readme_topic_sim),
        dep_overlap=_jaccard(source.dependencies, candidate.dependencies),
        popularity_sim=_popularity_sim(source.stars, candidate.stars),
        trending_boost=_clamp(candidate.trending_score),
        filter_cosine_sim=_clamp(filter_cosine_sim),
        quality_signal=_quality_signal(candidate),
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


def tag_match(user_tags: list[str], candidate_topics: list[str]) -> float:
    """Exact tag matching — fallback when embeddings are unavailable.
    1.0 = all user tags present in candidate topics.
    0.0 = no overlap."""
    if not user_tags or not candidate_topics:
        return 0.0
    ut = {t.lower() for t in user_tags}
    ct = {t.lower() for t in candidate_topics}
    inter = ut & ct
    if not inter:
        return 0.0
    return len(inter) / len(ut)


_DESC_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "for", "with", "this", "that",
    "to", "of", "in", "is", "it", "on", "by", "as", "at", "be",
    "from", "not", "are", "was", "your", "all", "can", "has",
    "its", "use", "you", "but", "we", "no", "so", "if",
    "one", "which", "also", "more", "just", "been", "will",
})
_DESC_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize_desc(text: str) -> set[str]:
    tokens: set[str] = set()
    for m in _DESC_TOKEN_RE.finditer(text.lower()):
        t = m.group()
        if len(t) > 1 and t not in _DESC_STOPWORDS and not t.isdigit():
            tokens.add(t)
    return tokens


def _tokenize_bigrams(text: str) -> set[str]:
    words = [
        m.group()
        for m in _DESC_TOKEN_RE.finditer(text.lower())
        if len(m.group()) > 1
        and m.group() not in _DESC_STOPWORDS
        and not m.group().isdigit()
    ]
    bigrams: set[str] = set()
    for i in range(len(words) - 1):
        bigrams.add(words[i] + " " + words[i + 1])
    return bigrams


def _description_sim(source_desc: str | None, candidate_desc: str | None) -> float:
    if not source_desc or not candidate_desc:
        return 0.0
    src = _tokenize_desc(source_desc)
    cand = _tokenize_desc(candidate_desc)
    if not src or not cand:
        return 0.0
    inter = src & cand
    if not inter:
        return 0.0
    unigram = len(inter) / len(src | cand)

    src_bi = _tokenize_bigrams(source_desc)
    cand_bi = _tokenize_bigrams(candidate_desc)
    if src_bi and cand_bi:
        bi_inter = src_bi & cand_bi
        bi_union = src_bi | cand_bi
        bigram = len(bi_inter) / len(bi_union) if bi_union else 0.0
    else:
        bigram = 0.0

    return unigram * 0.55 + bigram * 0.45


def _quality_signal(repo: Repo) -> float:
    """Quality proxy when no embedding model is available.
    Rewards repos that are well-documented in their metadata."""
    s = 0.2
    if repo.description and len(repo.description) > 60:
        s += 0.3
    if repo.topics and len(repo.topics) > 2:
        s += 0.2
    if repo.dependencies and len(repo.dependencies) > 5:
        s += 0.2
    if repo.language:
        s += 0.1
    return min(1.0, s)


_readme_cache: dict[str, set[str]] = {}


def _tokenize_readme(full_name: str, text: str) -> set[str]:
    if full_name in _readme_cache:
        return _readme_cache[full_name]
    tokens = _tokenize_desc(text[:5000])
    _readme_cache[full_name] = tokens
    return tokens


def readme_topic_sim(source_tokens: set[str], candidate_topics: list[str]) -> float:
    """Match source README tokens against candidate topic tags.
    Topics are curated labels with near-zero ambiguity — 'neural-network'
    matches README token 'neural' unambiguously, while 'linux-kernel'
    matches 'kernel' only in the OS domain.
    
    Uses substring matching so compound topics (deep-learning, neural-network)
    match partial tokens from the README (learning, neural)."""
    if not source_tokens or not candidate_topics:
        return 0.0
    lower_topics = {t.lower() for t in candidate_topics}
    matches = 0
    for token in source_tokens:
        for topic in lower_topics:
            if token in topic:
                matches += 1
                break
    return matches / len(source_tokens)
