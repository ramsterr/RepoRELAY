"""
Purpose statement extraction from README text.

GitHub repo descriptions are usually 1-2 sentences and extremely
purpose-dense ("A pure Python chess library with move generation").
But many repos have:
  - null description
  - empty description
  - generic/filler description ("A cool project", "My first repo")
  - too-short description (< 20 chars)

This module extracts a purpose statement from the README as a fallback.
The README's first 1-3 sentences after the title heading almost always
describe the project's purpose, before diving into install/API/docs.

Pure string processing — no API calls, no model. Fast and deterministic.
"""

from __future__ import annotations

import re

# Minimum length for a description to be considered "real"
_MIN_DESC_LEN = 25

# Generic/filler patterns that indicate a low-quality description
_GENERIC_PATTERNS = re.compile(
    r"^(a\s+(cool|good|nice|simple|basic|small|little|great|new)\s+"
    r"(project|library|tool|app|package|module|repo|repository|script|program)\s*"
    r"[.!]?\s*$"
    r"|"
    r"my\s+(first|new|own)\s+(project|repo|repository|library|tool|app|script)\s*"
    r"[.!]?\s*$"
    r"|"
    r"(work\s+in\s+progress|wip|todo|tbd|placeholder|test|hello\s+world|"
    r"coming\s+soon|under\s+construction)\s*"
    r"[.!]?\s*$"
    r"|"
    r"^\s*[\.\-_\*]+\s*$"
    r")",
    re.IGNORECASE,
)


def is_good_description(description: str | None) -> bool:
    """Check if a description is substantive enough to use as-is."""
    if not description or not description.strip():
        return False
    stripped = description.strip()
    if len(stripped) < _MIN_DESC_LEN:
        return False
    return not _GENERIC_PATTERNS.match(stripped)


# ── README extraction ────────────────────────────────────────────────

# Match badges/HTML comments/images at the start of a line
_BADGE_OR_HTML_RE = re.compile(
    r"^\s*"
    r"(<!\-\-.*?\-\->"  # HTML comment
    r"|"
    r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)"  # badge with link
    r"|"
    r"!\[[^\]]*\]\([^)]*\)"  # image
    r"|"
    r"\[!\[[^\]]*\]\]\([^)]*\)"  # badge without link
    r"|"
    r"<[a-zA-Z][^>]*>"  # any HTML tag
    r")\s*$",
    re.MULTILINE,
)

# Match a markdown heading (# Title, ## Section, etc.)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)

# Match a code fence line (``` or ```python)
_CODE_FENCE_LINE_RE = re.compile(r"^\s*```.*$", re.MULTILINE)

# Match an entire code block (```...```)
_CODE_BLOCK_RE = re.compile(
    r"```[^\n]*\n.*?```",
    re.DOTALL,
)

# Match inline code
_INLINE_CODE_RE = re.compile(r"`[^`]+`")

# Match markdown links [text](url) → text
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Match bold/italic markers
_FORMAT_RE = re.compile(r"[*_]{1,3}([^*_]+)[*_]{1,3}")

# Sentence boundary (period, exclamation, question mark followed by space/newline)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# Section headings that are likely boilerplate (not the purpose statement)
_BOILERPLATE_HEADINGS = frozenset({
    "install", "installation", "usage", "getting started", "quick start",
    "requirements", "license", "licence", "contributing", "support",
    "screenshots", "demo", "configuration", "config", "setup",
    "build", "testing", "tests", "api reference", "documentation",
    "docs", "table of contents", "contents", "features", "todo",
    "status", "badges", "overview", "introduction", "examples",
    "api", "methods", "options", "arguments", "parameters",
    "changelog", "history", "roadmap", "acknowledgments", "credits",
    "authors", "citation", "sponsors", "related",
})


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks and their content."""
    return _CODE_BLOCK_RE.sub("", text)


def _split_at_subheading(text: str) -> str:
    """Stop at the next markdown heading (## or lower) after the current section."""
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        # Stop at subheadings (## or deeper) but NOT at the H1 we already passed
        if line.strip().startswith("##"):
            break
        result.append(line)
    return "\n".join(result)


def _clean_markdown(text: str) -> str:
    """Strip markdown formatting from a text block."""
    # Remove badges/HTML/images
    text = _BADGE_OR_HTML_RE.sub("", text)
    # Convert links [text](url) → text
    text = _LINK_RE.sub(r"\1", text)
    # Remove inline code
    text = _INLINE_CODE_RE.sub("", text)
    # Remove bold/italic markers
    text = _FORMAT_RE.sub(r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_purpose_block(readme: str) -> str:
    """
    Extract the purpose text block from a README.

    Steps:
    1. Remove all code blocks (so their content doesn't pollute extraction)
    2. Find the first heading (H1, H2, or H3)
    3. Take text after it, stopping at the next subheading (##)
    4. Strip badges, HTML, images from the block
    5. Return the cleaned block

    If the first heading is a section (## or ###) and there's meaningful
    prose before it, use the prose before instead.
    """
    if not readme or not readme.strip():
        return ""

    # Remove code blocks first
    no_code = _strip_code_blocks(readme)

    # Find the first heading
    heading_match = _HEADING_RE.search(no_code)
    if heading_match:
        heading_text = heading_match.group(1).strip().lower()
        heading_level = len(heading_match.group(0)) - len(heading_match.group(1).strip()) - 1
        # If the first heading is a section (## or deeper) and looks like
        # boilerplate, check if there's meaningful prose before it
        if heading_level >= 2 and heading_text in _BOILERPLATE_HEADINGS:
            before = no_code[:heading_match.start()].strip()
            if before and len(before) >= _MIN_DESC_LEN:
                return _clean_markdown(before)
        # Otherwise, take text after the heading
        after_heading = no_code[heading_match.end():]
    else:
        # No heading at all — use the whole README
        after_heading = no_code

    # Stop at the next subheading (##)
    after_heading = _split_at_subheading(after_heading)

    # Strip markdown formatting
    return _clean_markdown(after_heading)


def _first_meaningful_sentences(text: str, max_sentences: int = 3, max_chars: int = 400) -> str:
    """Take the first N meaningful sentences from cleaned text."""
    if not text:
        return ""
    # Split on sentence boundaries
    sentences = _SENTENCE_END_RE.split(text)
    result_parts: list[str] = []
    char_count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Skip very short fragments (likely noise) at the start
        if len(sentence) < 10 and not result_parts:
            continue
        result_parts.append(sentence)
        char_count += len(sentence) + 1  # +1 for the space
        if len(result_parts) >= max_sentences or char_count >= max_chars:
            break
    result = " ".join(result_parts).strip()
    # Hard cap on total length in case there are no sentence boundaries
    if len(result) > max_chars:
        result = result[:max_chars].rsplit(" ", 1)[0].strip()
    return result


def extract_purpose_from_readme(readme: str) -> str:
    """
    Extract a 1-3 sentence purpose statement from a README.

    Strategy:
    1. Remove all code blocks (so ```pip install``` etc. don't pollute)
    2. Find the first H1/H2/H3 heading
    3. Take text after it, stopping at the next ## subheading
    4. Strip markdown formatting
    5. Return the first 1-3 meaningful sentences (max 400 chars)
    """
    block = _extract_purpose_block(readme)
    if not block:
        return ""
    return _first_meaningful_sentences(block, max_sentences=3, max_chars=400)


def get_effective_description(
    description: str | None,
    readme: str | None = None,
) -> str:
    """
    Return the best available description for a repo.

    Priority:
    1. The repo's own description (if it passes `is_good_description`)
    2. The purpose extracted from the README (if available)
    3. Empty string

    This is what should be used for embedding and for the
    description_cosine_sim feature.
    """
    if is_good_description(description):
        assert description is not None
        return description.strip()
    if readme:
        extracted = extract_purpose_from_readme(readme)
        if extracted and len(extracted) >= _MIN_DESC_LEN:
            return extracted
    return ""


# ── Description cleaning for embedding ──────────────────────────────
# Strips AI-brand buzzwords and instructional fluff that dominate
# semantic vectors without adding domain signal. Used before embedding
# descriptions at both storage time and query time.

_BUZZ_PATTERNS = [
    # AI brand names / model IDs
    r"\bClaude\b",        r"\bGemini\b",
    r"\bOpenAI\b",        r"\bChatGPT\b",
    r"\bGPT[-\s]*\d*\b",  r"\bLLM[s]?\b",
    # Filler implementation phrases
    r"\bAI[-\s]*powered\b",  r"\bpowered by\b",
    r"\bbuilt (on|with|using)\b\s*\S+",
    r"\bwritten in\s+\S+",
    # Command verbs in descriptions
    r"\bFork it,?\s*",
    r"\bclone (this|the) repo,?\s*",
]
_INSTRUCTIONAL = [
    r",?\s*fill in your\s*\S*",
    r",?\s*let \S+ (evaluate|do|handle|manage|process).*?(?=[,.]|$)",
    r",?\s*check (out|it out).*?(?=[,.]|$)",
    r",?\s*(please |feel free to |be sure to |don't forget to ).*?(?=[,.]|$)",
    r",?\s*(contributions? |star[s]? |fork[s]? ).*?(?=[,.]|$)",
]


def clean_description(text: str) -> str:
    """Strip buzzwords and fluff; keep domain signal for embedding."""
    for pat in _BUZZ_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    for pat in _INSTRUCTIONAL:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"^\W+|\W+$", "", text)
    text = re.sub(r"\s*,\s*,?\s*", ", ", text).strip(" ,")
    return text.strip() if len(text.strip()) > 10 else text
