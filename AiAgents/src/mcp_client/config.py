"""
MCP Connection Configurations — predefined server connection profiles.
"""

from dataclasses import dataclass, field
from typing import Literal

from src.config.settings import settings


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection."""
    name: str
    transport: Literal["stdio", "streamable_http"]
    # For streamable_http
    url: str = ""
    headers: dict = field(default_factory=dict)
    # For stdio
    command: str = "python"
    args: list[str] = field(default_factory=list)
    env: dict | None = None

    def to_dict(self) -> dict:
        """Convert to MultiServerMCPClient-compatible dict."""
        if self.transport == "streamable_http":
            return {
                "transport": "streamable_http",
                "url": self.url,
                "headers": self.headers,
            }
        return {
            "transport": "stdio",
            "command": self.command,
            "args": self.args,
            "env": self.env,
        }


# ── Predefined configurations ──────────────────────────────────────────────────

PRODUCTION_CONFIG = MCPServerConfig(
    name="unified_mcp",
    transport="streamable_http",
    url=f"{settings.mcp_server_url}/mcp",
    headers={"X-API-Key": settings.mcp_api_key},
)

WEB_SEARCH_CONFIG = MCPServerConfig(
    name="web_search",
    transport="streamable_http",
    url=f"{settings.mcp_server_url}/mcp/web-search",
    headers={"X-API-Key": settings.mcp_api_key},
)

DATABASE_CONFIG = MCPServerConfig(
    name="database",
    transport="streamable_http",
    url=f"{settings.mcp_server_url}/mcp/database",
    headers={"X-API-Key": settings.mcp_api_key},
)

LOCAL_DEV_CONFIG = MCPServerConfig(
    name="local_dev",
    transport="stdio",
    command="python",
    args=["../../MCPServer/src/gateway/unified_server.py"],
)


def get_production_configs() -> dict[str, dict]:
    """Return all production MCP server configs as a dict for MultiServerMCPClient."""
    return {PRODUCTION_CONFIG.name: PRODUCTION_CONFIG.to_dict()}


def get_dev_configs() -> dict[str, dict]:
    """Return local dev MCP server configs."""
    return {LOCAL_DEV_CONFIG.name: LOCAL_DEV_CONFIG.to_dict()}
