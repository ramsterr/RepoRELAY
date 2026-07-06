"""Tests for purpose extraction module."""

from __future__ import annotations

from reporelay_mvp.purpose import (
    extract_purpose_from_readme,
    get_effective_description,
    is_good_description,
)


class TestIsGoodDescription:
    def test_substantive_description_passes(self):
        assert is_good_description("A pure Python chess library with move generation") is True

    def test_none_description_fails(self):
        assert is_good_description(None) is False

    def test_empty_description_fails(self):
        assert is_good_description("") is False
        assert is_good_description("   ") is False

    def test_too_short_description_fails(self):
        assert is_good_description("chess lib") is False

    def test_generic_filler_fails(self):
        assert is_good_description("A cool project.") is False
        assert is_good_description("My first project") is False
        assert is_good_description("Work in progress") is False
        assert is_good_description("TODO") is False
        assert is_good_description("coming soon") is False

    def test_just_punctuation_fails(self):
        assert is_good_description("---") is False
        assert is_good_description("...") is False


class TestExtractPurposeFromReadme:
    def test_standard_chess_library(self):
        readme = """# python-chess
[![Build Status](https://img.shields.io/badge.svg)](https://example.com)
[![PyPI](https://img.shields.io/pypi.svg)](https://pypi.org)

A pure Python chess library with move generation, move validation,
and support for common chess variants. It also provides an engine
communication protocol (UCI) and Polyglot opening book support.

## Installation

```
pip install chess
```

## Features

- Standard chess rules
- Chess variants
- UCI engine support
"""
        purpose = extract_purpose_from_readme(readme)
        assert "chess library" in purpose.lower()
        assert "move generation" in purpose.lower()
        assert "pip install" not in purpose  # should not include boilerplate
        assert "Features" not in purpose  # should not include subheading

    def test_web_framework(self):
        readme = """# FastAPI

FastAPI framework, high performance, easy to learn, fast to code, ready for production

## Installation

```
pip install fastapi
```
"""
        purpose = extract_purpose_from_readme(readme)
        assert "fastapi" in purpose.lower()
        assert "framework" in purpose.lower() or "production" in purpose.lower()

    def test_no_heading_falls_back_to_first_text(self):
        readme = """This is a chess engine library written in Rust. It implements
the alpha-beta pruning algorithm and supports UCI protocol.

## Installation
cargo install my-chess
"""
        purpose = extract_purpose_from_readme(readme)
        assert "chess engine" in purpose.lower()
        assert "alpha-beta" in purpose.lower() or "rust" in purpose.lower()

    def test_strips_badges_and_html(self):
        readme = """# MyProject
<img src="badge1.png" />
<img src="badge2.png" />
<!-- This is a comment -->

A machine learning library for natural language processing tasks
like sentiment analysis and named entity recognition.

## Install
```
pip install myproject
```
"""
        purpose = extract_purpose_from_readme(readme)
        assert "machine learning" in purpose.lower()
        assert "natural language" in purpose.lower() or "nlp" in purpose.lower()
        assert "<img" not in purpose
        assert "<!--" not in purpose

    def test_stops_at_boilerplate_sections(self):
        readme = """# Tool

A command-line tool for converting JSON to CSV.

## Installation

```
npm install tool
```

## Usage

```
tool convert input.json output.csv
```
"""
        purpose = extract_purpose_from_readme(readme)
        assert "command-line tool" in purpose.lower() or "json to csv" in purpose.lower()
        assert "npm install" not in purpose
        assert "tool convert" not in purpose

    def test_max_three_sentences(self):
        readme = """# Library

First sentence about the library. Second sentence about features. Third sentence about performance. Fourth sentence about history. Fifth sentence about future plans.

## Install
```
pip install lib
```
"""
        purpose = extract_purpose_from_readme(readme)
        # Should not include the 4th and 5th sentences
        assert "fourth sentence" not in purpose.lower()
        assert "fifth sentence" not in purpose.lower()

    def test_max_chars_limit(self):
        readme = """# Project

""" + " ".join([f"Word{i}" for i in range(200)]) + """

## Install
```
pip install proj
```
"""
        purpose = extract_purpose_from_readme(readme)
        assert len(purpose) <= 450  # ~400 chars + some buffer

    def test_strips_markdown_links(self):
        readme = """# Project

A [link to docs](https://example.com) for a [chess library](https://chess.org) written in Python.

## Install
```
pip install proj
```
"""
        purpose = extract_purpose_from_readme(readme)
        assert "chess library" in purpose.lower()
        # The URL itself should be removed
        assert "https://" not in purpose
        assert "chess.org" not in purpose

    def test_strips_inline_code(self):
        readme = """# Tool

A `git` wrapper for managing chess databases with `python-chess` integration.

## Usage
```
tool import
```
"""
        purpose = extract_purpose_from_readme(readme)
        assert "git wrapper" in purpose.lower() or "chess databases" in purpose.lower()
        # Inline code backticks should be removed but the words kept
        assert "`" not in purpose

    def test_empty_readme_returns_empty(self):
        assert extract_purpose_from_readme("") == ""
        assert extract_purpose_from_readme("   \n\n  ") == ""

    def test_readme_with_only_code_fence_returns_empty(self):
        readme = """```
some code here
```
"""
        # Edge case: README with only code, no prose
        result = extract_purpose_from_readme(readme)
        # Should return something empty or a cleaned version
        assert isinstance(result, str)

    def test_h2_heading_works(self):
        readme = """## My Chess Engine

A high-performance chess engine implemented in C++ with bitboard representation.

### Build
```
make build
```
"""
        purpose = extract_purpose_from_readme(readme)
        assert "chess engine" in purpose.lower()

    def test_handles_special_characters(self):
        readme = """# MyProject

A library for parsing CSV files, JSON data, and XML documents. Supports streaming
I/O and handles edge cases like quoted fields & embedded delimiters.

## Install
```
pip install myproject
```
"""
        purpose = extract_purpose_from_readme(readme)
        assert "csv" in purpose.lower() or "json" in purpose.lower()


class TestGetEffectiveDescription:
    def test_good_description_returned_as_is(self):
        desc = "A pure Python chess library with move generation and validation"
        assert get_effective_description(desc, "# Title\nSome README content") == desc

    def test_null_description_falls_back_to_readme(self):
        readme = """# MyProject

A high-performance chess engine written in Rust with bitboard representation.

## Install
```
cargo install myproject
```
"""
        result = get_effective_description(None, readme)
        assert "chess engine" in result.lower()
        assert len(result) >= 25

    def test_empty_description_falls_back_to_readme(self):
        readme = """# MyProject

A machine learning library for computer vision tasks.

## Install
```
pip install myproject
```
"""
        result = get_effective_description("", readme)
        assert "machine learning" in result.lower() or "computer vision" in result.lower()

    def test_generic_description_replaced_by_readme(self):
        readme = """# MyProject

A comprehensive financial analysis tool for stock market prediction using
machine learning models like LSTM and transformer architectures.

## Install
```
pip install myproject
```
"""
        result = get_effective_description("A cool project.", readme)
        assert "cool project" not in result.lower()
        assert "financial" in result.lower() or "stock market" in result.lower()

    def test_too_short_description_replaced_by_readme(self):
        readme = """# MyProject

A web scraper for extracting product data from e-commerce websites.

## Install
```
pip install myproject
```
"""
        result = get_effective_description("scraper", readme)
        assert "scraper" != result
        assert "web scraper" in result.lower() or "e-commerce" in result.lower()

    def test_no_readme_returns_empty(self):
        assert get_effective_description(None, None) == ""
        assert get_effective_description(None, "") == ""

    def test_generic_description_with_no_good_readme(self):
        # If both description and README extraction are useless, return empty
        result = get_effective_description("A cool project.", "# Title\n\n")
        assert result == ""


class TestWeightRebalancing:
    """Tests for the weight distribution in score.py."""

    def test_weights_sum_to_one(self):
        from reporelay_mvp.score import WEIGHTS, TAG_WEIGHTS
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
        assert abs(sum(TAG_WEIGHTS.values()) - 1.0) < 1e-9

    def test_readme_vs_desc_is_primary_signal(self):
        from reporelay_mvp.score import WEIGHTS
        # readme_vs_desc_cosine_sim is the strongest individual signal
        assert WEIGHTS["readme_vs_desc_cosine_sim"] >= max(
            v for k, v in WEIGHTS.items() if k != "readme_vs_desc_cosine_sim"
        )

    def test_readme_topic_sim_is_enabled(self):
        from reporelay_mvp.score import WEIGHTS
        assert WEIGHTS["readme_topic_sim"] > 0.0

    def test_surface_signals_reduced(self):
        from reporelay_mvp.score import WEIGHTS
        assert WEIGHTS["language_match"] <= 0.05
        assert WEIGHTS["dep_overlap"] <= 0.10
        assert WEIGHTS["star_ratio"] <= 0.10

    def test_semantic_signals_dominate(self):
        from reporelay_mvp.score import WEIGHTS
        semantic_weight = (
            WEIGHTS["description_cosine_sim"]
            + WEIGHTS["topic_overlap"]
            + WEIGHTS["readme_topic_sim"]
            + WEIGHTS["readme_vs_desc_cosine_sim"]
        )
        assert semantic_weight >= 0.65

    def test_seed_does_not_triple_popularity(self):
        from reporelay_mvp.score import _get_weights
        w = _get_weights(seed=42, has_desc_emb=True, has_readme_emb=True)
        assert w["star_ratio"] <= 0.20


class TestScoreRepository:
    """Integration tests for the scoring changes."""

    def test_chess_repo_scores_higher_than_web_framework_for_chess_source(self):
        from reporelay_mvp.models import Features
        from reporelay_mvp.score import _get_weights, score_repo

        # Source: python-chess (chess library, Python)
        # Candidate A: C++ chess engine (chess, different language)
        # Candidate B: Python web framework (same language, different domain)

        features_chess_engine = Features(
            language_match=0.0,  # different language
            topic_overlap=0.8,   # high topic overlap (chess)
            cosine_sim=0.7,     # high README similarity
            description_sim=0.3,
            description_cosine_sim=0.85,  # HIGH - both are chess
            readme_topic_sim=0.6,
            dep_overlap=0.0,    # different ecosystem
            popularity_sim=0.5,
            trending_boost=0.0,
            quality_signal=0.8,
            language_diversity=1.0,
        )

        features_web_framework = Features(
            language_match=1.0,  # same Python
            topic_overlap=0.1,   # low topic overlap
            cosine_sim=0.4,      # moderate README similarity
            description_sim=0.1,
            description_cosine_sim=0.15,  # LOW - different domain
            readme_topic_sim=0.0,
            dep_overlap=0.3,     # some shared deps
            popularity_sim=0.95, # very popular
            trending_boost=0.5,
            quality_signal=0.9,
            language_diversity=0.0,
        )

        w = _get_weights(None, has_desc_emb=True, has_readme_emb=True, has_topics=True)

        score_chess_engine = sum(
            getattr(features_chess_engine, name) * weight
            for name, weight in w.items()
        )
        score_web_framework = sum(
            getattr(features_web_framework, name) * weight
            for name, weight in w.items()
        )

        # The chess engine should score higher than the web framework
        # because the description_cosine_sim signal is now dominant
        assert score_chess_engine > score_web_framework, (
            f"chess engine ({score_chess_engine:.3f}) should beat "
            f"web framework ({score_web_framework:.3f})"
        )
