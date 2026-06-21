"""
Settings for the MVP package.

The MVP reuses the database from the main app but with its own table
(`mvp_repos`). It does not require Redis.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
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

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_postgres_url(cls, v: str) -> str:
        if isinstance(v, str) and v.startswith("postgresql://"):
            return "postgresql+psycopg://" + v[len("postgresql://") :]
        return v


@lru_cache(maxsize=1)
def get_mvp_settings() -> MvpSettings:
    return MvpSettings()
