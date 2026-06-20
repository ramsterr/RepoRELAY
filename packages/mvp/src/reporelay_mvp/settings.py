"""
Settings for the MVP package.

The MVP reuses the database from the main app but with its own table
(`mvp_repos`). It does not require Redis.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class MvpSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://reporelay:reporelay@localhost:5439/reporelay"
    github_token: str = ""

    embedding_dim: int = 384


@lru_cache(maxsize=1)
def get_mvp_settings() -> MvpSettings:
    return MvpSettings()
