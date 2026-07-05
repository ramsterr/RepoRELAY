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
        case_sensitive=False,
    )

    database_url: str = "postgresql+psycopg://reporelay:reporelay@localhost:5439/reporelay"
    github_token: str = ""

    embedding_dim: int = 512
    openai_api_key: str = ""
    voyage_api_key: str = ""
    gemini_api_key: str = ""
    cohere_api_key: str = ""
    embedding_api: str = ""
    # Backward-compat: REPORE_LAY_LIGHTWEIGHT=1 still works
    lightweight: bool = False
    # Match the legacy env var name (uppercase, with prefix)
    # Pydantic 2 with case_sensitive=False uses lowercase field name,
    # so we use an alias to also accept REPORE_LAY_LIGHTWEIGHT.

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_postgres_url(cls, v: str) -> str:
        if isinstance(v, str) and v.startswith("postgresql://"):
            return "postgresql+psycopg://" + v[len("postgresql://") :]
        return v

    @field_validator("lightweight", mode="before")
    @classmethod
    def _coerce_lightweight(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _coerce_openai_api_key(cls, v: object) -> str:
        # Make sure empty strings stay empty (not converted to None)
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("voyage_api_key", mode="before")
    @classmethod
    def _coerce_voyage_api_key(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _coerce_gemini_api_key(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v).strip()


@lru_cache(maxsize=1)
def get_mvp_settings() -> MvpSettings:
    settings = MvpSettings()
    # Backward-compat: also accept REPORE_LAY_LIGHTWEIGHT=1 directly from
    # os.environ since pydantic with case_sensitive=False uses the
    # lowercase field name (lightweight), missing the legacy env var.
    import os

    if not settings.lightweight:
        legacy = os.environ.get("REPORE_LAY_LIGHTWEIGHT", "").strip().lower()
        if legacy in ("1", "true", "yes", "on"):
            object.__setattr__(settings, "lightweight", True)
    return settings
