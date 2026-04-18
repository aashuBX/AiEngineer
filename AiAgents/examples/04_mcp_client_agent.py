"""Example 04: Agent using MCP tools from MCPServer."""
import asyncio
from src.config.llm_providers import get_llm
from src.mcp_client.client import MCPClientManager


async def main():
    """Demonstrate an agent using MCP tools."""
    llm = get_llm(provider="groq", model="llama-3.3-70b-versatile")

    # Connect to MCP server
    client = MCPClientManager()
    print("MCP Client Agent Demo")
    print("=" * 50)
    print("Configure your MCP server connection in src/mcp_client/config.py")
    print("Then run the MCPServer unified gateway first.")


if __name__ == "__main__":
    asyncio.run(main())
