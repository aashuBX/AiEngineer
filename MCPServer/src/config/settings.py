"""
Settings for MCPServer, loading from environment variables.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for MCPServer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API & Auth ──────────────────────────────────────────────────────────
    mcp_api_key: str = Field(default="dev-key-12345", description="API key for MCP clients")
    port: int = 8002
    host: str = "0.0.0.0"

    # ── External APIs ──────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", description="Tavily Web Search API key")
    weather_api_key: str = Field(default="", description="Weather API key")

    # ── Database Connections ───────────────────────────────────────────────
    db_tool_connection_string: str = Field(default="sqlite:///./mcp_db.sqlite")

    # ── File System Permissions ────────────────────────────────────────────
    # Comma-separated absolute paths. Empty means no limits in dev (DANGEROUS IN PROD).
    allowed_fs_paths: str = Field(default="")

    @property
    def fs_allowed_dirs(self) -> list[str]:
        if not self.allowed_fs_paths:
            return []
        return [p.strip() for p in self.allowed_fs_paths.split(",") if p.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
