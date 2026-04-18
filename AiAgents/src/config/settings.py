"""
Centralized Pydantic Settings configuration for AiAgents.
All environment variables are loaded from .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Configuration ─────────────────────────────────────────────────
    default_llm_provider: Literal["openai", "anthropic", "groq", "google", "ollama"] = "groq"
    default_llm_model: str = "qwen-qwq-32b"
    default_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=4096, ge=1)

    # ── API Keys ───────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    groq_api_key: str = Field(default="", description="Groq API key")
    google_api_key: str = Field(default="", description="Google Gemini API key")

    # ── Observability ──────────────────────────────────────────────────────
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_tracing: bool = True
    langsmith_project: str = "ai-engineer"

    langfuse_public_key: str = Field(default="", description="Langfuse public key")
    langfuse_secret_key: str = Field(default="", description="Langfuse secret key")
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── Memory / Checkpointing ─────────────────────────────────────────────
    checkpoint_db_url: str = "sqlite:///./data/checkpoints.db"

    # ── External Services ──────────────────────────────────────────────────
    genai_system_url: str = "http://localhost:8001"
    mcp_server_url: str = "http://localhost:8002"
    mcp_api_key: str = Field(default="", description="MCPServer API key")

    # ── Long-term Vector Memory ────────────────────────────────────────────
    vector_memory_provider: Literal["faiss", "qdrant"] = "faiss"
    qdrant_url: str = "http://localhost:6333"

    # ── Ollama (local) ─────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings (singleton)."""
    return Settings()


# Convenience singleton
settings = get_settings()
