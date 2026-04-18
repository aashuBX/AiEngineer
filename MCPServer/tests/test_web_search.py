"""Tests for Web Search MCP server."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestWebSearch:
    """Test web search tool functions."""

    def test_web_search_server_import(self):
        """Verify the web search server can be imported."""
        from src.servers.web_search.server import mcp
        assert mcp is not None
        assert mcp.name == "web_search_server"

    @pytest.mark.asyncio
    async def test_duckduckgo_fallback(self):
        """Test DuckDuckGo fallback when no Tavily key configured."""
        from src.servers.web_search.server import _duckduckgo_search
        # Should not raise even if duckduckgo_search is not installed
        result = await _duckduckgo_search("test query", 3)
        assert isinstance(result, str)


class TestFetchWebpage:
    """Test webpage fetching tool."""

    @pytest.mark.asyncio
    async def test_fetch_invalid_url(self):
        """Test error handling for invalid URLs."""
        from src.servers.web_search.server import fetch_webpage
        result = await fetch_webpage("http://this-does-not-exist-12345.com")
        assert "Failed" in result or "Error" in result
