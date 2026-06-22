"""
Database access for the MVP.

Reuses the same Postgres instance as the main app, but reads/writes its
own `mvp_repos` table. No Redis, no co-star materialized view, no graph.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from reporelay_mvp.settings import get_mvp_settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _engine, _sessionmaker
    if _sessionmaker is None:
        settings = get_mvp_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=3,
            pool_recycle=60,
            connect_args={
                "connect_timeout": 10,
                "keepalives_idle": 30,
            },
        )
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    sessionmaker = _get_sessionmaker()
    async with sessionmaker() as session:
        yield session
