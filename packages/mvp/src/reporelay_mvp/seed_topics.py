"""
Seed repos by topic (not language). Pulls trending / high-star repos
across broad software categories — machine-learning, cybersecurity,
cloud, frontend, backend, database, devops, observability, design,
etc. — so the corpus represents the actual software ecosystem.

Each topic gets `per_topic` repos from a GitHub search sorted by
stars, then upserted into the DB. The next step after seeding is
`just mvp embed --limit N` to backfill README embeddings.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from reporelay_mvp import data
from reporelay_mvp.github import (_auth_client, search_repositories)
from reporelay_mvp.settings import get_mvp_settings

logger = logging.getLogger(__name__)

DEFAULT_TOPICS: list[str] = [
    "machine-learning", "deep-learning", "cybersecurity", "security",
    "kubernetes", "docker", "devops",
    "api", "graphql", "database", "postgresql",
    "react", "vue", "nextjs", "typescript",
    "fastapi", "spring-boot", "backend",
    "data-science", "data-engineering",
    "observability", "monitoring", "testing", "ci-cd",
    "microservices", "serverless", "blockchain",
    "cli", "terminal", "game-development",
    "ai", "llm", "chatbot", "automation",
    "rust", "go", "mobile",
    "design-system", "tailwindcss", "accessibility",
    "embedded", "networking", "compiler",
    "analytics", "visualization", "tutorial", "education",
    "terraform", "ansible",
    "generative-ai", "rag", "langchain",
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
