"""Tests for topic_inference module."""

from __future__ import annotations

from reporelay_mvp.topic_inference import infer_topics, infer_topics_for_repo


class TestInferTopics:
    def test_empty_text_returns_empty(self):
        assert infer_topics("") == []

    def test_short_text_returns_empty(self):
        assert infer_topics("hello world") == []

    def test_ml_framework_detection(self):
        text = """
        This project uses PyTorch for deep learning. The model is trained
        on ImageNet using convolutional neural networks for image classification.
        We use transfer learning from pretrained ResNet models.
        """
        topics = infer_topics(text)
        assert "pytorch" in topics
        assert "deep-learning" in topics
        assert "cnn" in topics or "computer-vision" in topics

    def test_llm_detection(self):
        text = """
        A RAG pipeline built with LangChain and OpenAI GPT-4. Uses vector
        database for retrieval augmented generation. Supports prompt engineering
        and fine-tuning of large language models.
        """
        topics = infer_topics(text)
        assert "llm" in topics
        assert "langchain" in topics
        assert "rag" in topics

    def test_web_framework_detection(self):
        text = """
        A full-stack web application built with Next.js and React.
        Uses Tailwind CSS for styling and Prisma as the ORM.
        Deployed on Vercel with server side rendering.
        """
        topics = infer_topics(text)
        assert "nextjs" in topics or "react" in topics
        assert "prisma" in topics or "database" in topics

    def test_devops_detection(self):
        text = """
        Infrastructure as code using Terraform and Kubernetes.
        CI/CD pipeline with GitHub Actions. Monitoring with
        Prometheus and Grafana dashboards.
        """
        topics = infer_topics(text)
        assert "kubernetes" in topics
        assert "terraform" in topics
        assert "ci-cd" in topics or "github-actions" in topics

    def test_blockchain_detection(self):
        text = """
        Smart contracts written in Solidity for the Ethereum blockchain.
        DeFi protocol with NFT staking. Built with Hardhat and
        deployed to Polygon.
        """
        topics = infer_topics(text)
        assert "blockchain" in topics
        assert "solidity" in topics or "ethereum" in topics

    def test_game_dev_detection(self):
        text = """
        A 3D game built with the Godot game engine. Features
        ray tracing shaders and OpenGL rendering. Includes
        procedural world generation.
        """
        topics = infer_topics(text)
        assert "game-development" in topics
        assert "godot" in topics

    def test_mobile_detection(self):
        text = """
        Cross-platform mobile app built with Flutter and Dart.
        Uses Riverpod for state management. Supports both
        iOS and Android.
        """
        topics = infer_topics(text)
        assert "flutter" in topics
        assert "mobile" in topics

    def test_existing_topics_excluded(self):
        text = """
        Machine learning project using PyTorch for deep learning.
        """
        topics = infer_topics(text, existing_topics=["pytorch", "machine-learning"])
        assert "pytorch" not in topics
        assert "machine-learning" not in topics
        assert "deep-learning" in topics

    def test_max_topics_respected(self):
        text = """
        A comprehensive project using PyTorch, TensorFlow, Keras,
        scikit-learn, XGBoost, LightGBM, Hugging Face transformers,
        OpenAI GPT-4, LangChain, LlamaIndex, and more.
        """
        topics = infer_topics(text, max_topics=3)
        assert len(topics) <= 3

    def test_precision_over_recall(self):
        """Ambiguous text shouldn't produce spurious topics."""
        text = """
        A simple utility library for common operations.
        Provides helper functions for strings and numbers.
        """
        topics = infer_topics(text)
        # Should not hallucinate ML or blockchain topics
        assert "machine-learning" not in topics
        assert "blockchain" not in topics

    def test_phrase_match_beats_single_word(self):
        """Phrase matches should rank higher than single-word matches."""
        text = """
        This project implements natural language processing pipelines
        for text classification and sentiment analysis using NLP techniques.
        """
        topics = infer_topics(text)
        # "natural language processing" is a phrase match → nlp topic
        assert "nlp" in topics or "natural-language-processing" in topics


class TestInferTopicsForRepo:
    def test_with_description_only(self):
        topics = infer_topics_for_repo(
            "A machine learning library for deep learning with PyTorch",
            None,
        )
        assert "pytorch" in topics or "deep-learning" in topics

    def test_with_readme_only(self):
        readme = """
        # My Project
        
        This is a web scraper built with Python. It crawls websites
        and extracts data using beautiful soup and requests.
        """
        topics = infer_topics_for_repo(None, readme)
        assert "web-scraper" in topics or "crawling" in topics

    def test_with_both(self):
        desc = "A React component library"
        readme = """
        # UI Kit
        
        Built with React and TypeScript. Uses Storybook for
        component development. Styled with Tailwind CSS.
        """
        topics = infer_topics_for_repo(desc, readme, ["react"])
        assert "react" not in topics  # already existing
        assert "storybook" in topics or "tailwindcss" in topics

    def test_empty_inputs(self):
        assert infer_topics_for_repo(None, None) == []
        assert infer_topics_for_repo("", "") == []
