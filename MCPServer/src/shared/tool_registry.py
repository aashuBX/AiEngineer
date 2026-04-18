"""
Tool Registry — Dynamic MCP tool registration and discovery.
Supports loading tools from server modules and exposing them to agents.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolDefinition:
    """Definition of a registered tool."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        server: str = "",
        schema: Optional[Dict[str, Any]] = None,
        is_async: bool = False,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.server = server
        self.schema = schema or {}
        self.is_async = is_async


class ToolRegistry:
    """Central registry for all MCP tools across server modules.

    Features:
    - Manual and auto-discovery registration
    - Tool lookup by name or server
    - Tool listing for agent card generation
    - Execution proxy with error handling
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        func: Callable,
        server: str = "",
        schema: Optional[Dict[str, Any]] = None,
        is_async: bool = False,
    ):
        """Register a tool in the registry.

        Args:
            name: Unique tool name.
            description: Human-readable description.
            func: The tool function/coroutine.
            server: Name of the parent MCP server.
            schema: JSON schema for parameters.
            is_async: Whether the function is async.
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            func=func,
            server=server,
            schema=schema,
            is_async=is_async,
        )
        logger.info(f"Registered tool: {name} (server={server})")

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """List all registered tools (for agent discovery)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "server": tool.server,
            }
            for tool in self._tools.values()
        ]

    def list_by_server(self, server: str) -> List[Dict[str, str]]:
        """List tools belonging to a specific server."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
            if t.server == server
        ]

    async def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with error handling.

        Args:
            name: Tool name to execute.
            **kwargs: Tool arguments.

        Returns:
            Tool execution result.
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry.")

        try:
            if tool.is_async:
                return await tool.func(**kwargs)
            else:
                return tool.func(**kwargs)
        except Exception as e:
            logger.error(f"Tool '{name}' execution failed: {e}")
            raise

    def discover_from_servers(self):
        """Auto-discover and register all tools from known MCP server modules.

        Imports each server module and registers its MCP-decorated tools.
        """
        server_modules = [
            ("web_search", "src.servers.web_search.server"),
            ("database", "src.servers.database.server"),
            ("file_system", "src.servers.file_system.server"),
            ("calculator", "src.servers.calculator.server"),
            ("weather", "src.servers.weather.server"),
            ("api_integration", "src.servers.api_integration.server"),
        ]

        for server_name, module_path in server_modules:
            try:
                import importlib
                module = importlib.import_module(module_path)
                mcp_server = getattr(module, "mcp", None)

                if mcp_server and hasattr(mcp_server, "_tool_manager"):
                    for tool in mcp_server._tool_manager.tools.values():
                        self.register(
                            name=tool.name,
                            description=tool.description or "",
                            func=tool.fn,
                            server=server_name,
                            is_async=True,
                        )

                logger.info(f"Discovered tools from '{server_name}'")
            except Exception as e:
                logger.warning(f"Failed to discover tools from '{server_name}': {e}")

    @property
    def count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)


# Global singleton
tool_registry = ToolRegistry()
