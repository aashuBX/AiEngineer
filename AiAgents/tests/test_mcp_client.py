"""Tests for MCP client module."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestMCPClientManager:
    """Test MCP client initialization and configuration."""

    def test_client_initialization(self):
        from src.mcp_client.client import MCPClientManager
        client = MCPClientManager()
        assert client is not None


class TestMCPConfig:
    """Test MCP configuration loading."""

    def test_server_config_model(self):
        from src.mcp_client.config import MCPServerConfig
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
            args=["-m", "test_server"],
        )
        assert config.name == "test"
        assert config.transport == "stdio"


class TestToolAdapter:
    """Test MCP tool adaptation to LangChain format."""

    def test_tool_adapter_import(self):
        from src.mcp_client.tool_adapter import MCPToolAdapter
        assert MCPToolAdapter is not None
