"""Tests for MCP Gateway and A2A server."""
import pytest
from unittest.mock import MagicMock


class TestUnifiedGateway:
    """Test the unified MCP gateway."""

    def test_gateway_import(self):
        from src.gateway.unified_server import unified_mcp
        assert unified_mcp is not None

    def test_gateway_name(self):
        from src.gateway.unified_server import unified_mcp
        assert unified_mcp.name == "ai_engineer_gateway"


class TestAgentCard:
    """Test A2A agent card models."""

    def test_agent_card_model(self):
        from src.gateway.agent_card import AgentCard
        card = AgentCard(
            name="TestAgent",
            description="A test agent",
            url="http://localhost:8001",
            capabilities=["test"],
        )
        assert card.name == "TestAgent"
        assert "test" in card.capabilities
