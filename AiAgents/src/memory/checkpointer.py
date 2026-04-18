"""
LangGraph Checkpointer — SQLite (dev) and PostgreSQL (prod) persistence.
Provides thread-based conversation isolation and time-travel debugging.
"""

from typing import Any

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_sqlite_checkpointer(db_path: str | None = None):
    """
    Return a SQLite-backed checkpointer for development.

    Args:
        db_path: SQLite file path. Defaults to settings.checkpoint_db_url.

    Returns:
        SqliteSaver instance (use as async context manager).
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # Parse file path from sqlite:/// URL
    path = db_path or settings.checkpoint_db_url
    if path.startswith("sqlite:///"):
        path = path[len("sqlite:///"):]

    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    logger.info(f"Checkpointer: SQLite at {path}")
    return AsyncSqliteSaver.from_conn_string(path)


def get_postgres_checkpointer(conn_string: str | None = None):
    """
    Return a PostgreSQL-backed checkpointer for production.

    Args:
        conn_string: PostgreSQL connection string.
                     Defaults to settings.checkpoint_db_url.

    Returns:
        AsyncPostgresSaver instance (use as async context manager).
    """
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:
        raise ImportError(
            "langgraph-checkpoint-postgres is not installed. "
            "Run: uv add langgraph-checkpoint-postgres"
        )

    conn = conn_string or settings.checkpoint_db_url
    logger.info("Checkpointer: PostgreSQL")
    return AsyncPostgresSaver.from_conn_string(conn)


def get_checkpointer(mode: str = "auto"):
    """
    Auto-select a checkpointer based on the configured DB URL.

    Args:
        mode: "sqlite" | "postgres" | "auto" (default).
              In "auto" mode, detects provider from the URL.

    Returns:
        A LangGraph checkpointer instance.
    """
    url = settings.checkpoint_db_url

    if mode == "sqlite" or (mode == "auto" and url.startswith("sqlite")):
        return get_sqlite_checkpointer()
    elif mode == "postgres" or (mode == "auto" and url.startswith("postgresql")):
        return get_postgres_checkpointer()
    else:
        logger.warning(
            f"Could not determine checkpointer from URL '{url}' — defaulting to SQLite"
        )
        return get_sqlite_checkpointer()


def make_thread_config(session_id: str) -> dict[str, Any]:
    """
    Build a LangGraph config dict for per-thread checkpointing.

    Args:
        session_id: Unique conversation/thread ID.

    Returns:
        Config dict to pass as `config=` to graph.invoke() / .ainvoke().
    """
    return {"configurable": {"thread_id": session_id}}
