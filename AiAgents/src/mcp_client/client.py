"""
MCP Client — MultiServerMCPClient wrapper for connecting to MCPServer.
Supports stdio (local dev) and streamable_http (production) transports.

NOTE: As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient can no longer
be used as a context manager. We instantiate it directly and call get_tools().
"""

from contextlib import asynccontextmanager
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCPClientManager:
    """
    Manages connections to one or more MCP servers and exposes their
    tools as LangChain-compatible BaseTool instances.
    """

    def __init__(self, server_configs: dict[str, dict] | None = None):
        """
        Args:
            server_configs: Dict of server_name → connection config.
                            If None, auto-builds from settings.
        """
        self._server_configs = server_configs or self._default_configs()
        self._client: MultiServerMCPClient | None = None
        self._tools: list[BaseTool] = []

    def _default_configs(self) -> dict[str, dict]:
        """Build default server configs from settings."""
        return {
            "unified_mcp": {
                "transport": "streamable_http",
                "url": f"{settings.mcp_server_url}/mcp",
                "headers": {"X-API-Key": settings.mcp_api_key},
            }
        }

    @asynccontextmanager
    async def connect(self):
        """Async context manager — connects to all configured MCP servers.

        Uses the langchain-mcp-adapters 0.1.0+ API:
          client = MultiServerMCPClient(...)
          tools  = await client.get_tools()
        """
        logger.info(f"Connecting to MCP servers: {list(self._server_configs.keys())}")
        self._client = MultiServerMCPClient(self._server_configs)
        try:
            self._tools = await self._client.get_tools()
            logger.info(
                f"MCP connected — {len(self._tools)} tools available: "
                f"{[t.name for t in self._tools]}"
            )
            yield self
        finally:
            # Best-effort cleanup — close underlying sessions if the client
            # exposes a close/aclose method (varies by adapter version).
            close_fn = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
            if close_fn:
                try:
                    import inspect
                    if inspect.iscoroutinefunction(close_fn):
                        await close_fn()
                    else:
                        close_fn()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"MCPClient cleanup error (non-fatal): {exc}")
            self._client = None
            self._tools = []

    @property
    def tools(self) -> list[BaseTool]:
        """Return all discovered tools from connected MCP servers."""
        if not self._tools:
            logger.warning("MCPClientManager: no tools loaded — call connect() first")
        return self._tools

    def get_tool(self, name: str) -> BaseTool | None:
        """Find a specific tool by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Directly invoke a named MCP tool."""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in connected MCP servers")
        logger.debug(f"Calling MCP tool: {tool_name}({kwargs})")
        return await tool.arun(kwargs)


def create_stdio_client(server_script: str, server_name: str = "local") -> MCPClientManager:
    """
    Create an MCPClientManager for a local stdio MCP server (dev mode).

    Args:
        server_script: Path to the MCP server Python script.
        server_name:   Logical name for this server.
    """
    configs = {
        server_name: {
            "transport": "stdio",
            "command": "python",
            "args": [server_script],
        }
    }
    return MCPClientManager(server_configs=configs)


def create_http_client(
    url: str,
    api_key: str = "",
    server_name: str = "remote",
) -> MCPClientManager:
    """
    Create an MCPClientManager for a remote streamable-http MCP server.

    Args:
        url:         Full URL of the MCP server endpoint.
        api_key:     Optional API key for authentication.
        server_name: Logical name for this server.
    """
    configs = {
        server_name: {
            "transport": "streamable_http",
            "url": url,
            "headers": {"X-API-Key": api_key} if api_key else {},
        }
    }
    return MCPClientManager(server_configs=configs)
