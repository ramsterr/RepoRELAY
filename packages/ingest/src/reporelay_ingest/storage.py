from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from reporelay_core.db import get_engine, _get_sessionmaker

logger = logging.getLogger(__name__)


async def _get_session() -> AsyncSession:
    sessionmaker = _get_sessionmaker()
    return sessionmaker()


async def upsert_repo(session: AsyncSession, repo_data: dict[str, Any]) -> int:
    await session.execute(
        text(
            """
            INSERT INTO repos (
                id, owner, name, full_name, description, homepage,
                language, license, stars, forks, topics,
                created_at, updated_at, pushed_at, archived, is_template,
                default_branch
            ) VALUES (
                :id, :owner, :name, :full_name, :description, :homepage,
                :language, :license, :stars, :forks, :topics,
                :created_at, :updated_at, :pushed_at, :archived, :is_template,
                :default_branch
            )
            ON CONFLICT (id) DO UPDATE SET
                owner = EXCLUDED.owner,
                name = EXCLUDED.name,
                full_name = EXCLUDED.full_name,
                description = EXCLUDED.description,
                homepage = EXCLUDED.homepage,
                language = EXCLUDED.language,
                license = EXCLUDED.license,
                stars = EXCLUDED.stars,
                forks = EXCLUDED.forks,
                topics = EXCLUDED.topics,
                created_at = EXCLUDED.created_at,
                updated_at = EXCLUDED.updated_at,
                pushed_at = EXCLUDED.pushed_at,
                archived = EXCLUDED.archived,
                is_template = EXCLUDED.is_template,
                default_branch = EXCLUDED.default_branch
            """
        ),
        {
            "id": repo_data["id"],
            "owner": repo_data["owner"]["login"],
            "name": repo_data["name"],
            "full_name": repo_data["full_name"],
            "description": repo_data.get("description"),
            "homepage": repo_data.get("homepage"),
            "language": repo_data.get("language"),
            "license": repo_data["license"]["spdx_id"] if repo_data.get("license") else None,
            "stars": repo_data.get("stargazers_count", 0),
            "forks": repo_data.get("forks_count", 0),
            "topics": repo_data.get("topics", []),
            "created_at": _parse_dt(repo_data.get("created_at")),
            "updated_at": _parse_dt(repo_data.get("updated_at")),
            "pushed_at": _parse_dt(repo_data.get("pushed_at")),
            "archived": repo_data.get("archived", False),
            "is_template": repo_data.get("is_template", False),
            "default_branch": repo_data.get("default_branch"),
        },
    )
    await session.flush()
    return repo_data["id"]


async def upsert_readme(session: AsyncSession, repo_id: int, raw_text: str) -> None:
    await session.execute(
        text(
            """
            INSERT INTO readme_texts (repo_id, raw_text, embedded_at)
            VALUES (:repo_id, :raw_text, NULL)
            ON CONFLICT (repo_id) DO UPDATE SET
                raw_text = EXCLUDED.raw_text,
                embedded_at = NULL
            """
        ),
        {"repo_id": repo_id, "raw_text": raw_text},
    )
    await session.flush()


async def insert_topics(session: AsyncSession, repo_id: int, topic_names: list[str]) -> None:
    for name in topic_names:
        await session.execute(
            text(
                """
                INSERT INTO topics (name)
                VALUES (:name)
                ON CONFLICT (name) DO NOTHING
                """
            ),
            {"name": name},
        )
    await session.flush()


async def insert_contributors(
    session: AsyncSession, repo_id: int, contributors: list[dict[str, Any]]
) -> None:
    for contributor in contributors:
        user_id = contributor["id"]
        login = contributor["login"]
        user_type = contributor.get("type", "User")

        await session.execute(
            text(
                """
                INSERT INTO users (id, login, type)
                VALUES (:id, :login, :type)
                ON CONFLICT (id) DO UPDATE SET
                    login = EXCLUDED.login,
                    type = EXCLUDED.type
                """
            ),
            {"id": user_id, "login": login, "type": user_type},
        )

        await session.execute(
            text(
                """
                INSERT INTO contributor_edges (user_id, repo_id, commit_count)
                VALUES (:user_id, :repo_id, :commit_count)
                ON CONFLICT (user_id, repo_id) DO UPDATE SET
                    commit_count = EXCLUDED.commit_count
                """
            ),
            {
                "user_id": user_id,
                "repo_id": repo_id,
                "commit_count": contributor.get("contributions", 0),
            },
        )
    await session.flush()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


async def insert_dependencies(
    session: AsyncSession, repo_id: int, deps: list[dict[str, Any]]
) -> None:
    for dep in deps:
        await session.execute(
            text(
                """
                INSERT INTO dependency_edges (repo_id, dependency_name, ecosystem, version_constraint, is_dev)
                VALUES (:repo_id, :name, :ecosystem, :version, :is_dev)
                ON CONFLICT DO NOTHING
                """
            ),
            {
                "repo_id": repo_id,
                "name": dep["name"],
                "ecosystem": dep.get("ecosystem", "unknown"),
                "version": dep.get("version", ""),
                "is_dev": dep.get("is_dev", False),
            },
        )
    await session.flush()


async def insert_star_events(
    session: AsyncSession, events: list[dict[str, Any]]
) -> int:
    count = 0
    for event in events:
        starred_at = _parse_starred_at(event.get("starred_at"))
        await session.execute(
            text(
                """
                INSERT INTO star_events (user_id, repo_id, starred_at)
                VALUES (:user_id, :repo_id, :starred_at)
                ON CONFLICT DO NOTHING
                """
            ),
            {
                "user_id": event["user_id"],
                "repo_id": event["repo_id"],
                "starred_at": starred_at,
            },
        )
        count += 1
    await session.flush()
    return count


def _parse_starred_at(value: Any) -> datetime | None:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value
