"""
REST API Integration MCP Server.
Generic tool to safely call external REST endpoints.
"""

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("api_integration_server")


@mcp.tool()
async def call_rest_api(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    json_body: dict | None = None,
) -> str:
    """
    Call a generic external REST API.

    Args:
        url: The full HTTP/HTTPS URL.
        method: HTTP method (GET, POST, PUT, DELETE).
        headers: Optional dictionary of HTTP headers.
        json_body: Optional JSON payload for POST/PUT.
    """
    allowed_methods = {"GET", "POST", "PUT", "DELETE", "PATCH"}
    _method = method.upper()

    if _method not in allowed_methods:
        return f"Error: Method {_method} not supported."

    if not url.startswith("http://") and not url.startswith("https://"):
        return "Error: URL must start with http:// or https://"

    # Force a timeout to prevent hanging the agent
    timeout = 15.0

    try:
        async with httpx.AsyncClient() as client:
            request = client.build_request(
                _method,
                url,
                headers=headers or {},
                json=json_body,
                timeout=timeout,
            )
            response = await client.send(request)
            response.raise_for_status()

            # Attempt to parse json, fallback to text
            try:
                data = response.json()
                import json
                return json.dumps(data, indent=2)[:8000] # truncate
            except ValueError:
                return response.text[:8000]

    except httpx.HTTPStatusError as e:
        return f"API Error ({e.response.status_code}): {e.response.text[:1000]}"
    except Exception as e:
        return f"API Call Failed: {e}"
