"""
Keyword extraction for repo descriptions and READMEs.

Extracts domain-specific technical keywords that are meaningful for
search and matching. Prioritizes:
  - Multi-word technical terms ("machine learning", "game development")
  - Hyphenated compounds ("open-source", "deep-learning")
  - CamelCase tokens ("OpenGL", "DataFrame", "CLI")
  - Domain-specific nouns (pixel, shader, chess, compiler)

Filters out:
  - Common English stopwords
  - Single-letter tokens
  - Numbers
  - Common generic terms ("library", "tool", "project")

The extracted keywords are stored in a PostgreSQL TEXT[] column for
fast ARRAY overlap queries and full-text search (tsvector).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ── Tokenization ────────────────────────────────────────────────────

# Split on non-alphanumeric except hyphens (preserve compound terms)
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*")

# CamelCase splitter: "GameDev" → ["Game", "Dev", "GameDev"]
_CAMEL_RE = re.compile(r"([A-Z][a-z]+|[A-Z]{2,}(?=[A-Z]|$))")

# Common stopwords to exclude
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "are", "was",
    "were", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "not", "no", "nor", "so", "if", "than", "then", "that", "this",
    "these", "those", "which", "who", "whom", "whose", "what", "when",
    "where", "how", "all", "both", "each", "every", "any", "few",
    "more", "most", "other", "some", "such", "only", "own", "same",
    "into", "over", "under", "up", "down", "out", "off", "about",
    "just", "now", "also", "too", "very", "well", "new", "old",
    "your", "my", "its", "our", "their", "i", "we", "you", "he",
    "she", "they", "me", "us", "him", "her", "them",
})

# Generic terms that are common across repos (low signal)
_GENERIC_TERMS: frozenset[str] = frozenset({
    "library", "tool", "project", "package", "module", "framework",
    "application", "app", "api", "data", "code", "file", "use",
    "using", "support", "provide", "based", "simple", "fast",
    "easy", "simple", "lightweight", "powerful", "flexible",
    "efficient", "modern", "minimal", "high", "low", "real",
    "time", "open", "source", "written", "built", "implementation",
    "example", "test", "tests", "version", "release", "guide",
    "documentation", "docs", "readme", "install", "installation",
    "usage", "getting", "started", "quick", "setup", "build",
    "run", "running", "feature", "features", "available", "include",
    "including", "contains", "includes", "following",
    "com", "org", "net", "http", "https", "www", "sup", "br",
    "pdf", "png", "jpg", "svg", "github", "git", "io",
})


def extract_keywords(
    text: str,
    *,
    min_length: int = 2,
    max_results: int = 30,
) -> list[str]:
    """Extract meaningful domain keywords from text.

    Args:
        text: Description or README text to extract from.
        min_length: Minimum keyword length (chars).
        max_results: Maximum number of keywords to return.

    Returns:
        List of keywords, ordered by importance (compound first, then freq).
    """
    if not text or not text.strip():
        return []

    lower_text = text.lower()
    tokens = _TOKEN_RE.findall(lower_text)
    if not tokens:
        return []

    # Phase 1: collect all tokens, counting frequency
    token_freq: Counter[str] = Counter()
    bigram_candidates: list[tuple[str, str]] = []

    for i, token in enumerate(tokens):
        # Skip stopwords at the token level
        if token in _STOPWORDS:
            continue
        # Skip generic terms
        if token in _GENERIC_TERMS:
            continue
        # Skip short tokens
        if len(token) < min_length:
            continue
        # Skip pure numbers
        if token.isdigit():
            continue
        # Skip tokens that are just a version number pattern
        if re.match(r"^v?\d+(\.\d+)*$", token):
            continue
        # Skip numeric/currency patterns (e.g., "199 mo", "499 mo", "$12")
        if re.match(r"^\d{2,}\s*(mo|month|yr|year|week|day|hr|hour|min|sec)?$", token):
            continue
        if re.match(r"^\$\d+$", token):
            continue
        # Skip UUIDs
        if len(token) > 30 and "-" in token and re.match(r"^[0-9a-f\-]{30,}$", token):
            continue
        # Skip URLs
        if token.startswith(("http", "https", "www", "ftp")):
            continue
        # Skip file references
        if re.search(r"\.(pdf|png|jpg|svg)$", token):
            continue

        token_freq[token] += 1

        # Collect bigram pairs for compound term detection
        if i > 0 and tokens[i - 1] not in _STOPWORDS and tokens[i - 1] not in _GENERIC_TERMS:
            if len(tokens[i - 1]) >= min_length:
                bigram_candidates.append((tokens[i - 1], token))

    # Phase 2: detect compound terms (bigrams that appear together)
    compound_keywords: list[str] = []
    bigram_freq: Counter[str] = Counter()
    for a, b in bigram_candidates:
        if a in _STOPWORDS or b in _STOPWORDS:
            continue
        if a in _GENERIC_TERMS and b in _GENERIC_TERMS:
            continue
        compound = f"{a} {b}"
        if len(compound) >= min_length + 2:
            bigram_freq[compound] += 1

    # Phase 3: prioritize hyphens and technical patterns
    scored: list[tuple[str, float]] = []
    for token, freq in token_freq.items():
        score = float(freq)

        # Bonus for hyphenated compounds (e.g., "deep-learning", "open-source")
        if "-" in token and len(token) > 5:
            score *= 2.5

        # Bonus for likely technical terms (not all lowercase)
        original_tokens = _TOKEN_RE.findall(token)
        for ot in original_tokens:
            if ot[0].isupper() and len(ot) > 1:
                score *= 1.5
                break

        scored.append((token, score))

    # Add compounds with high scores
    for compound, freq in bigram_freq.items():
        score = float(freq) * 1.8
        scored.append((compound, score))

    # Sort by score descending
    scored.sort(key=lambda x: (-x[1], x[0]))

    # De-duplicate: if a compound contains a single token, prefer the compound
    compounds = {c for c, _ in scored if " " in c}
    result: list[str] = []
    for keyword, score in scored:
        if keyword in result:
            continue
        # If we have a compound "game development", skip the individual "game" and "development"
        # unless they score very high on their own
        if " " not in keyword:
            # Check if this token appears inside any higher-ranked compound
            contained = False
            for c in compounds:
                if keyword in c.split():
                    contained = True
                    break
            if contained and score < 2.0:
                continue
        result.append(keyword)
        if len(result) >= max_results:
            break

    return result


def extract_keywords_from_repo(
    description: str | None,
    readme: str | None = None,
    *,
    max_results: int = 30,
) -> list[str]:
    """Extract keywords from a repo's description + README.

    Priority: description keywords get weight 2.0, README keywords get 1.0.
    """
    if not description and not readme:
        return []

    keywords: list[tuple[str, float]] = []

    if description:
        desc_keywords = extract_keywords(description, max_results=50)
        for kw in desc_keywords:
            keywords.append((kw, 2.0))

    if readme:
        readme_keywords = extract_keywords(readme[:5000], max_results=50)
        for kw in readme_keywords:
            keywords.append((kw, 1.0))

    # Deduplicate: keep the highest weight
    seen: dict[str, float] = {}
    for kw, weight in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen or weight > seen[kw_lower]:
            seen[kw_lower] = weight

    # Sort by weight (desc), then alphabetically
    sorted_kw = sorted(seen.items(), key=lambda x: (-x[1], x[0]))
    return [kw for kw, _ in sorted_kw[:max_results]]


def keywords_to_search_vector(keywords: list[str]) -> str:
    """Convert a list of keywords into a tsvector-compatible search string."""
    # Join keywords with spaces, PostgreSQL's to_tsvector handles the rest
    return " ".join(keywords)
