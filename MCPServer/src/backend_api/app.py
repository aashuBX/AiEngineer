"""
MCPServer Application — FastMCP Streamable HTTP transport (mcp SDK v1.0+).

Endpoint map:
  GET  http://mcp-server:8002/health  ← container healthcheck
  POST http://mcp-server:8002/mcp     ← MCP protocol (used by ai-agents)

Why we use FastMCP's Starlette app directly (NOT mounted inside FastAPI):
  FastAPI does NOT propagate its lifespan to mounted sub-apps.
  FastMCP's Starlette app has its own lifespan that initialises the
  StreamableHTTPSessionManager task group — if that lifespan is skipped
  every request fails with:
    RuntimeError: Task group is not initialized. Make sure to use run().
  Solution: use the Starlette app as the root ASGI app and inject our
  /health route directly into its router.
"""

import logging
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.config.settings import settings
from src.gateway.unified_server import unified_mcp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_backend")


# ── Health endpoint handler ────────────────────────────────────────────────────
async def health_endpoint(request: Request) -> JSONResponse:
    """Container liveness / healthcheck — always returns 200 when server is up."""
    tool_count = len(unified_mcp._tool_manager._tools)
    return JSONResponse({"status": "ok", "tools_count": tool_count})


# ── Build the FastMCP Starlette app ───────────────────────────────────────────
# streamable_http_app() returns a Starlette app whose lifespan initialises the
# session manager's task group. We use it as the ROOT ASGI app.
logger.info("MCPServer Backend starting up...")
app = unified_mcp.streamable_http_app()

# Inject /health BEFORE the /mcp route so it matches first.
app.router.routes.insert(0, Route("/health", health_endpoint, methods=["GET"]))
logger.info(f"Registered routes: {[r.path for r in app.routes]}")


# ── Auth middleware (guards /mcp/* only) ───────────────────────────────────────
class AuthMiddleware:
    """Rejects calls to /mcp without a valid X-API-Key header."""

    def __init__(self, app_inner):
        self._app = app_inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path: str = scope.get("path", "")
            if path.startswith("/mcp") and settings.mcp_api_key:
                headers = dict(scope.get("headers", []))
                api_key = headers.get(b"x-api-key", b"").decode()
                if api_key != settings.mcp_api_key:
                    response = JSONResponse(
                        {"detail": "Invalid or missing X-API-Key"}, status_code=401
                    )
                    await response(scope, receive, send)
                    return
        await self._app(scope, receive, send)


# ── CORS middleware ────────────────────────────────────────────────────────────
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(AuthMiddleware)

logger.info("FastMCP streamable-http transport ready — MCP endpoint: /mcp")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.backend_api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,  # reload=True breaks anyio task groups
    )

