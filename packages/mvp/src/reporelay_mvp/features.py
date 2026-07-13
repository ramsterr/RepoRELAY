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


def compute_features(source: Repo, candidate: Repo, *, cosine_sim: float, filter_cosine_sim: float = 0.0, description_cosine_sim: float = 0.0, readme_topic_sim: float = 0.0, readme_vs_desc_cosine_sim: float = 0.0) -> Features:
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
        readme_vs_desc_cosine_sim=_clamp(readme_vs_desc_cosine_sim),
        keyword_match=_keyword_jaccard(source.keywords, candidate.keywords),
        keyword_topic_match=_keyword_topic_overlap(source.keywords, candidate.topics),
        dep_overlap=_jaccard(source.dependencies, candidate.dependencies),
        popularity_sim=_popularity_sim(source.stars, candidate.stars),
        star_ratio=_star_ratio(source.stars, candidate.stars),
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


def _star_ratio(source_stars: int, cand_stars: int) -> float:
    """How close is the candidate's popularity to the source's?

    Returns 1.0 when the two repos are in the same popularity tier
    (within 1 order of magnitude), and decays smoothly as the gap
    widens. A 100-star source and a 500-star candidate → ~0.8. A
    100-star source and a 1M-star candidate → ~0.1.

    This prevents the common failure mode where a niche 200-star repo
    recommends React (200k stars) as its top match — they share the
    "javascript" topic, but they're not similar projects.
    """
    a = max(source_stars, 1)
    b = max(cand_stars, 1)
    # log10 ratio — 0 when equal, grows as the gap widens
    log_ratio = abs(math.log10(a) - math.log10(b))
    # Map to [0, 1]: log_ratio=0 → 1.0, log_ratio=3 (1000x gap) → 0.0
    # Using exponential decay: score = exp(-log_ratio)
    return max(0.0, min(1.0, math.exp(-log_ratio)))


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
    """Quality proxy — rewards repos that are well-maintained and documented.

    Signals:
    - Has a substantive description (>60 chars)
    - Has curated topic tags (>2 topics)
    - Has dependency metadata (>5 deps → actively maintained)
    - Has a language specified
    - Has an embedding (had a README worth embedding → non-trivial project)
    """
    s = 0.1
    if repo.description and len(repo.description) > 60:
        s += 0.25
    if repo.topics and len(repo.topics) > 2:
        s += 0.2
    if repo.dependencies and len(repo.dependencies) > 5:
        s += 0.15
    if repo.language:
        s += 0.1
    if repo.embedding and any(v != 0.0 for v in repo.embedding):
        s += 0.2
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


# ── Keyword-based features ──────────────────────────────────────────

def _keyword_jaccard(src_keywords: list[str], cand_keywords: list[str]) -> float:
    """Jaccard similarity of extracted keyword sets.

    Captures domain overlap: both repos mention "data science",
    "machine learning", "curriculum" — they serve the same purpose
    even if they have different GitHub topics.
    """
    if not src_keywords or not cand_keywords:
        return 0.0
    a = {k.lower() for k in src_keywords}
    b = {k.lower() for k in cand_keywords}
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    return len(inter) / len(union)


def _keyword_topic_overlap(keywords: list[str], topics: list[str]) -> float:
    """What fraction of extracted keywords match GitHub topic tags?

    Bridges keyword extraction with topic matching. If the source
    keywords include "data science" and the candidate has topics
    ["data-science", "machine-learning"], this feature captures
    that overlap even when the exact string forms differ.
    """
    if not keywords or not topics:
        return 0.0
    kw_set = {k.lower() for k in keywords}
    topic_set = {t.lower() for t in topics}

    # Direct match
    direct = kw_set & topic_set

    # Substring match: "data science" matches topic "data-science"
    # or "data-science-education"
    substring_matches = 0
    for kw in kw_set - direct:
        for topic in topic_set - direct:
            # Normalize: remove hyphens for comparison
            kw_norm = kw.replace("-", " ").replace(" ", "")
            topic_norm = topic.replace("-", "").replace(" ", "")
            if kw_norm in topic_norm or topic_norm in kw_norm:
                substring_matches += 1
                break

    total_matches = len(direct) + substring_matches
    return min(1.0, total_matches / len(kw_set))
