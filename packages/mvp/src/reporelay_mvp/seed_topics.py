"""
Seed repos by topic (not language). Pulls trending / high-star repos
across broad software categories — machine-learning, cybersecurity,
cloud, frontend, backend, database, devops, observability, design,
etc. — so the corpus represents the actual software ecosystem.

Each topic gets `per_topic` repos from a GitHub search sorted by
stars, then upserted into the DB. The next step after seeding is
`just mvp embed --limit N` to backfill README embeddings.

Supports two modes:
  - Uniform (default): same `per_topic` count for every topic
  - Weighted: per-topic counts from topics_config.py (sums to 50k)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from reporelay_mvp import data
from reporelay_mvp.github import _auth_client, search_repositories
from reporelay_mvp.settings import get_mvp_settings

logger = logging.getLogger(__name__)

DEFAULT_TOPICS: list[str] = [
    # ── Programming Languages ──────────────────────────────────────
    "python", "javascript", "typescript", "java", "c", "cpp", "csharp",
    "ruby", "php", "swift", "kotlin", "scala", "r", "julia", "haskell",
    "elixir", "erlang", "clojure", "fsharp", "objective-c", "perl",
    "lua", "dart", "zig", "nim", "ocaml", "groovy", "powershell",
    "rust", "go", "assembly", "v", "crystal", "odin",

    # ── Frontend ───────────────────────────────────────────────────
    "react", "vue", "angular", "svelte", "nextjs", "nuxt", "gatsby",
    "remix", "astro", "solidjs", "jquery", "bootstrap", "tailwindcss",
    "material-ui", "chakra-ui", "ant-design", "storybook",
    "styled-components", "emotion", "postcss", "webpack", "vite",
    "parcel", "rollup", "esbuild", "turbopack",
    "design-system", "accessibility", "css", "sass", "less",
    "html", "web-components",

    # ── Backend ────────────────────────────────────────────────────
    "fastapi", "django", "flask", "express", "nestjs", "koa", "hapi",
    "spring-boot", "spring", "gin", "echo", "fiber", "actix", "rocket",
    "axum", "phoenix", "rails", "laravel", "aspnet", "vertx",
    "micronaut", "quarkus", "graphql", "trpc", "grpc",
    "backend", "rest-api", "api",

    # ── Mobile ─────────────────────────────────────────────────────
    "react-native", "flutter", "ios", "android", "swiftui",
    "jetpack-compose", "kotlin-multiplatform", "expo", "ionic",
    "xamarin", "nativescript", "mobile", "capacitor",

    # ── Data & ML ──────────────────────────────────────────────────
    "machine-learning", "deep-learning", "pytorch", "tensorflow", "keras",
    "scikit-learn", "xgboost", "lightgbm", "huggingface", "transformers",
    "computer-vision", "nlp", "natural-language-processing",
    "reinforcement-learning", "mlops", "model-deployment",
    "feature-engineering", "neural-network", "cnn", "rnn", "gan",
    "object-detection", "image-classification", "semantic-segmentation",
    "text-classification", "sentiment-analysis", "speech-recognition",
    "ocr", "text-to-speech", "diffusion", "stable-diffusion",
    "data-science", "data-engineering", "data-pipeline",
    "apache-spark", "apache-kafka", "apache-airflow", "dbt",
    "etl", "data-warehouse", "data-lake", "business-intelligence",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "jupyter", "notebooks", "polars", "dask",

    # ── AI / LLM ───────────────────────────────────────────────────
    "ai", "llm", "chatbot", "generative-ai", "rag", "langchain",
    "openai", "gpt", "llamaindex", "semantic-kernel", "autogen",
    "prompt-engineering", "fine-tuning", "embedding",
    "vector-database", "chromadb", "pinecone", "weaviate", "qdrant",
    "milvus", "annoy", "faiss", "automation",

    # ── DevOps & Cloud ─────────────────────────────────────────────
    "kubernetes", "docker", "devops", "ci-cd", "terraform", "ansible",
    "aws", "azure", "gcp", "google-cloud",
    "infrastructure-as-code", "container-orchestration",
    "service-mesh", "istio", "helm", "argocd", "gitops",
    "github-actions", "gitlab-ci", "jenkins", "circleci",
    "prometheus", "grafana", "datadog", "cloudformation", "pulumi",
    "vagrant", "podman", "serverless", "microservices",
    "linux", "nginx", "apache", "envoy",

    # ── Databases ──────────────────────────────────────────────────
    "database", "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "elasticsearch", "cassandra", "dynamodb", "couchdb", "neo4j",
    "cockroachdb", "tidb", "planetscale", "supabase",
    "prisma", "typeorm", "sequelize", "drizzle", "sqlalchemy",
    "mariadb", "clickhouse", "timescaledb", "surrealdb",

    # ── Security ───────────────────────────────────────────────────
    "cybersecurity", "security", "penetration-testing",
    "vulnerability-scanning", "encryption", "authentication",
    "authorization", "oauth", "jwt", "sso", "zero-trust",
    "devsecops", "siem", "owasp", "cryptography",

    # ── API & Networking ───────────────────────────────────────────
    "websocket", "http", "tcp", "dns", "cdn", "load-balancing",
    "reverse-proxy", "rate-limiting", "api-gateway", "openapi",
    "swagger", "postman", "networking", "proxy",

    # ── Testing ────────────────────────────────────────────────────
    "testing", "unit-testing", "integration-testing",
    "end-to-end-testing", "test-automation", "playwright", "cypress",
    "selenium", "jest", "pytest", "junit", "mocha",
    "performance-testing", "load-testing", "chaos-engineering",
    "mocking", "tdd", "bdd",

    # ── Blockchain & Web3 ──────────────────────────────────────────
    "blockchain", "ethereum", "solidity", "smart-contracts", "defi",
    "nft", "dao", "web3", "cryptocurrency", "bitcoin", "solana",
    "polygon", "hardhat", "truffle", "ipfs",

    # ── Game Development ───────────────────────────────────────────
    "game-development", "unity", "unreal-engine", "godot",
    "game-engine", "raylib", "love2d", "pygame", "opengl", "vulkan",
    "directx", "shader", "ray-tracing",

    # ── Embedded & Systems ─────────────────────────────────────────
    "embedded", "iot", "arduino", "raspberry-pi", "esp32", "rtos",
    "firmware", "device-driver", "operating-system", "linux-kernel",
    "systems-programming", "compiler", "parser", "ast",

    # ── Observability ──────────────────────────────────────────────
    "observability", "monitoring", "logging", "tracing", "metrics",
    "opentelemetry", "apm", "alerting", "dashboard",

    # ── UI/UX & Design ─────────────────────────────────────────────
    "ui", "ux", "figma", "animation", "motion-design", "dark-mode",
    "icons", "typography", "color-system",

    # ── Documentation & Education ──────────────────────────────────
    "documentation", "api-documentation", "tutorial", "education",
    "learning", "cheat-sheet", "best-practices", "style-guide",
    "example", "awesome-list",

    # ── CLI & Terminal ─────────────────────────────────────────────
    "cli", "terminal", "shell", "bash", "zsh", "fish", "neovim",
    "vim", "emacs", "tmux",

    # ── Build Tools & Package Managers ─────────────────────────────
    "package-manager", "monorepo", "build-tool", "bundler",
    "linting", "formatting", "code-generation", "static-analysis",
    "type-checker",

    # ── Media & Content ────────────────────────────────────────────
    "image-processing", "video-processing", "audio-processing",
    "pdf", "markdown", "cms", "e-commerce", "payment", "stripe",

    # ── Other Domains ──────────────────────────────────────────────
    "analytics", "a-b-testing", "feature-flags", "workflow",
    "task-scheduling", "cron", "message-queue", "rabbitmq", "nats",
    "search-engine", "full-text-search", "web-scraper", "crawling",
    "data-extraction", "bot", "discord", "telegram", "slack",
    "email", "smtp", "notification", "geospatial", "gis",
    "robotics", "bioinformatics", "scientific-computing",
    "computer-algebra", "math", "statistics",
    "serialization", "json", "yaml", "toml", "xml",
    "localization", "i18n", "pwa", "service-worker",
    "ssr", "ssg", "edge-computing", "deno", "bun", "nodejs",
    "regex", "template-engine",
]


async def seed_topics(
    *,
    topics: list[str] | None = None,
    per_topic: int = 200,
    min_stars: int = 20,
    delay_s: float = 3.0,
) -> int:
    if topics is None:
        topics = DEFAULT_TOPICS

    pages = max(1, (per_topic + 99) // 100)
    settings = get_mvp_settings()
    total_upserted = 0

    async with _auth_client(settings.github_token) as client:
        for topic in topics:
            topic_upserted = 0
            tries = 0
            while tries < 3:
                tries += 1
                try:
                    for page in range(1, pages + 1):
                        payload = await search_repositories(
                            client,
                            topics=[topic],
                            min_stars=min_stars,
                            sort="stars",
                            per_page=100,
                            page=page,
                        )
                        items: list[dict[str, Any]] = payload.get("items", [])
                        if not items:
                            break

                        session = await data.get_session()
                        try:
                            written = await data.bulk_upsert_from_search(session, items)
                            await session.commit()
                            topic_upserted += written
                        finally:
                            await session.close()

                        if len(items) < 100:
                            break  # no more results
                    break  # success — exit retry loop
                except Exception as exc:
                    logger.warning(
                        "topic %r attempt %d/3 failed: %s", topic, tries, exc
                    )
                    if tries < 3:
                        await asyncio.sleep(12)
                    else:
                        logger.warning("topic %r gave up after 3 attempts", topic)

            logger.info(
                "topic %r: %d upserted (target=%d)", topic, topic_upserted, per_topic
            )
            total_upserted += topic_upserted

            await asyncio.sleep(delay_s)

    logger.info("seed-topics complete: %d repos across %d topics", total_upserted, len(topics))
    return total_upserted


async def seed_topics_weighted(
    *,
    topics: list[tuple[int, str]] | None = None,
    category: str | None = None,
    min_stars: int = 20,
    delay_s: float = 3.0,
    total_limit: int = 0,
) -> int:
    """Seed repos using per-topic counts (weighted distribution).

    Each topic has its own `count` value — popular topics get more
    repos, niche topics get fewer. This distributes 50k repos across
    ~500+ topics.

    Args:
        topics: List of (count, topic_name) tuples. If None, uses
                WEIGHTED_TOPICS from topics_config.
        category: If set, only seed topics from this category.
        min_stars: Minimum stars for GitHub search.
        delay_s: Seconds between topic searches (2.0 = 30 req/min).
        total_limit: Cap total repos seeded (0 = no limit). Safety valve.
    """
    if topics is None:
        from reporelay_mvp.topics_config import WEIGHTED_TOPICS, get_topics_for_category

        if category:
            topics = get_topics_for_category(category)
            logger.info("seeding category %r: %d topics", category, len(topics))
        else:
            topics = list(WEIGHTED_TOPICS)

    settings = get_mvp_settings()
    total_upserted = 0
    total_config = sum(count for count, _ in topics)

    start_time = time.monotonic()
    processed = 0

    async with _auth_client(settings.github_token) as client:
        for per_topic_count, topic in topics:
            if total_limit > 0 and total_upserted >= total_limit:
                logger.info("total_limit=%d reached, stopping", total_limit)
                break

            processed += 1
            topic_upserted = 0
            pages = max(1, (per_topic_count + 99) // 100)
            tries = 0

            while tries < 3:
                tries += 1
                try:
                    for page in range(1, pages + 1):
                        payload = await search_repositories(
                            client,
                            topics=[topic],
                            min_stars=min_stars,
                            sort="stars",
                            per_page=100,
                            page=page,
                        )
                        items: list[dict[str, Any]] = payload.get("items", [])
                        if not items:
                            break

                        session = await data.get_session()
                        try:
                            written = await data.bulk_upsert_from_search(session, items)
                            await session.commit()
                            topic_upserted += written
                        finally:
                            await session.close()

                        if len(items) < 100:
                            break

                    break  # success — exit retry loop
                except Exception as exc:
                    logger.warning("topic %r attempt %d/3 failed: %s", topic, tries, exc)
                    if tries < 3:
                        await asyncio.sleep(12)
                    else:
                        logger.warning("topic %r gave up after 3 attempts", topic)

            total_upserted += topic_upserted
            elapsed = time.monotonic() - start_time
            pct = (processed / len(topics)) * 100 if topics else 0

            # Progress log every topic
            logger.info(
                "  [%d/%d %.1f%%] %r: %d upserted (target=%d) | total=%d | elapsed=%.0fs",
                processed, len(topics), pct, topic, topic_upserted, per_topic_count,
                total_upserted, elapsed,
            )

            await asyncio.sleep(delay_s)

    elapsed = time.monotonic() - start_time
    logger.info(
        "seed-topics-weighted complete: %d repos across %d topics (config=%d) in %.0fs",
        total_upserted, processed, total_config, elapsed,
    )
    return total_upserted
