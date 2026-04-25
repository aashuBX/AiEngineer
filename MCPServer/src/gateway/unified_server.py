"""
Unified MCP Server Gateway.
Pulls together all sub-servers (Web, Calc, DB, FS, Weather, API) into a single FastMCP instance.
"""

from mcp.server.fastmcp import FastMCP

# Import all sub-servers
from src.servers.web_search.server import mcp as web_mcp
from src.servers.database.server import mcp as db_mcp
from src.servers.file_system.server import mcp as fs_mcp
from src.servers.api_integration.server import mcp as api_mcp
from src.servers.crm.server import mcp as crm_mcp
from src.servers.faq.server import mcp as faq_mcp
from src.servers.feedback.server import mcp as feedback_mcp
from src.servers.handoff.server import mcp as handoff_mcp

# Create the unified FastMCP app
unified_mcp = FastMCP("AiEngineer_Unified_MCP", description="Gateway for all AI Engineer tools")


def _merge_tools(target: FastMCP, source: FastMCP) -> None:
    """Manually register tools from a source FastMCP into a target FastMCP."""
    # FastMCP stores registered tools in a private _tool_manager.
    # We iterate and re-register them on the unified server.
    for tool_name, tool_data in source._tool_manager.tools.items():
        # FastMCP stores tools as Tool objects, we can bind the original functions
        target.tool(
            name=tool_name,
            description=tool_data.description
        )(tool_data.fn)


# Merge all sub-server tools into the unified instance
_merge_tools(unified_mcp, web_mcp)
_merge_tools(unified_mcp, db_mcp)
_merge_tools(unified_mcp, fs_mcp)
_merge_tools(unified_mcp, api_mcp)
_merge_tools(unified_mcp, crm_mcp)
_merge_tools(unified_mcp, faq_mcp)
_merge_tools(unified_mcp, feedback_mcp)
_merge_tools(unified_mcp, handoff_mcp)

if __name__ == "__main__":
    # Provides stdio CLI if executed directly
    unified_mcp.run(transport="stdio")
