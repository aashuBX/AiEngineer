"""Tests for Database MCP server."""
import pytest
from unittest.mock import MagicMock, patch


class TestDatabaseServer:
    """Test database introspection tools."""

    def test_database_server_import(self):
        """Verify the database server can be imported."""
        from src.servers.database.server import mcp
        assert mcp is not None
        assert mcp.name == "database_server"

    def test_query_blocks_non_select(self):
        """Verify that non-SELECT queries are blocked."""
        from src.servers.database.server import query_database
        result = query_database("DROP TABLE users")
        assert "Error" in result
        assert "SELECT" in result

    def test_query_allows_select(self):
        """Verify SELECT queries pass the guard check."""
        from src.servers.database.server import query_database
        # Will fail at DB connection but should pass the SQL guard
        result = query_database("SELECT 1")
        # Either succeeds or fails with a connection error, not a permission error
        assert "Only SELECT" not in result

    def test_query_allows_with_cte(self):
        """Verify WITH (CTE) queries pass the guard."""
        from src.servers.database.server import query_database
        result = query_database("WITH cte AS (SELECT 1) SELECT * FROM cte")
        assert "Only SELECT" not in result
