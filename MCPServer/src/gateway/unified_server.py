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

# Create the unified FastMCP app with Docker-compatible security settings.
# The MCP SDK v1.0+ validates the Host header (DNS rebinding protection).
# In Docker Compose, ai-agents calls http://mcp_server:8002/mcp, so
# the Host header is "mcp_server:8002" — we must explicitly allow it.
from mcp.server.streamable_http_manager import TransportSecuritySettings

_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "mcp_server:8002",    # Docker Compose service name (underscore)
        "mcp-server:8002",    # Docker Compose container name (hyphen)
        "localhost:8002",     # local dev
        "127.0.0.1:8002",    # local dev
    ],
    allowed_origins=[],
)

unified_mcp = FastMCP("AiEngineer_Unified_MCP", transport_security=_security)



def _merge_tools(target: FastMCP, source: FastMCP) -> None:
    """Manually register tools from a source FastMCP into a target FastMCP."""
    # FastMCP stores registered tools in a private _tool_manager._tools dict.
    for tool_name, tool_data in source._tool_manager._tools.items():
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
