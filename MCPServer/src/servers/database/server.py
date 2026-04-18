"""
Database MCP Server.
Provides tools to query connected SQL databases safely.
"""

from mcp.server.fastmcp import FastMCP
from typing import Any

from src.config.settings import settings

mcp = FastMCP("database_server", description="Database introspection and querying")


def _get_engine():
    from sqlalchemy import create_engine
    return create_engine(settings.db_tool_connection_string)


@mcp.tool()
def list_tables() -> str:
    """
    List all tables in the connected database.
    """
    try:
        from sqlalchemy import inspect
        engine = _get_engine()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        return f"Tables in database: {', '.join(tables)}"
    except Exception as e:
        return f"Error connecting to database: {e}"


@mcp.tool()
def get_table_schema(table_name: str) -> str:
    """
    Get the schema (columns, types, foreign keys) for a specific table.

    Args:
        table_name: Name of the table to inspect.
    """
    try:
        from sqlalchemy import inspect
        engine = _get_engine()
        inspector = inspect(engine)

        if table_name not in inspector.get_table_names():
            return f"Error: Table '{table_name}' does not exist."

        columns = inspector.get_columns(table_name)
        fks = inspector.get_foreign_keys(table_name)

        schema = [f"Schema for table '{table_name}':"]
        for col in columns:
            nullable = "NULL" if col.get("nullable") else "NOT NULL"
            schema.append(f"  - {col['name']} ({col['type']}) {nullable}")

        if fks:
            schema.append("\nForeign Keys:")
            for fk in fks:
                schema.append(f"  - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")

        return "\n".join(schema)
    except Exception as e:
        return f"Error inspecting table '{table_name}': {e}"


@mcp.tool()
def query_database(sql: str) -> str:
    """
    Execute a read-only SQL query against the connected database.
    WARNING: Only SELECT statements are allowed. Data modification will be blocked.

    Args:
        sql: The SQL SELECT statement to execute.
    """
    import re
    # Basic protection against modification (ActionValidator provides deeper protection)
    if not sql.strip().upper().startswith("SELECT") and not sql.strip().upper().startswith("WITH"):
        return "Error: Only SELECT queries are permitted via this tool."

    try:
        from sqlalchemy import text
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            keys = result.keys()

            if not rows:
                return "Query executed successfully. 0 rows returned."

            # Format as simple text table or JSON-like string
            output = [", ".join(keys)]
            for row in rows[:100]:  # strict limit on result size
                output.append(", ".join(str(v) for v in row))

            if len(rows) > 100:
                output.append(f"... (truncated {len(rows)-100} more rows) ...")

            return "\n".join(output)
    except Exception as e:
        return f"SQL Query Error: {e}"
