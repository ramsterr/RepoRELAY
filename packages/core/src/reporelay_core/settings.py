from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://reporelay:reporelay@localhost:5439/reporelay"
    redis_url: str = "redis://localhost:6379/0"
    github_token: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
