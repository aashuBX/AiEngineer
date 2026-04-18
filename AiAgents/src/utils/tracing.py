"""
LangSmith and Langfuse tracing integration.
Call `setup_tracing()` at application startup.
"""

import os

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_tracing() -> None:
    """Configure LangSmith and/or Langfuse tracing from settings."""
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info(f"LangSmith tracing enabled → project: {settings.langsmith_project}")
    else:
        logger.info("LangSmith tracing disabled (no API key or tracing=false)")

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse.callback import CallbackHandler as LangfuseCallback  # noqa: F401
            logger.info("Langfuse tracing available")
        except ImportError:
            logger.warning("langfuse package not installed; Langfuse tracing unavailable")


def get_langfuse_handler():
    """Return a Langfuse CallbackHandler if configured, else None."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except ImportError:
        logger.warning("langfuse not installed; returning None handler")
        return None
