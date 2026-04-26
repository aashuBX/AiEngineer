"""
MCPServer FastAPI Application.
Exposes the Unified FastMCP via Server-Sent Events (SSE) and handles A2A connections.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request
from starlette.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.gateway.unified_server import unified_mcp

# Standard logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: configure FastMCP context or connections if needed
    logger.info("MCPServer Backend starting up...")
    yield
    # Shutdown
    logger.info("MCPServer Backend shutting down...")


app = FastAPI(
    title="MCPServer Gateway",
    description="Unified API for MCP Tools via SSE and HTTP REST",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(request: Request):
    """Simple API Key auth checking X-API-Key header."""
    if not settings.mcp_api_key:
        return  # Auth disabled
    api_key = request.headers.get("X-API-Key")
    if api_key != settings.mcp_api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")


# ── FastMCP SSE Endpoints ──────────────────────────────────────────────────────

# FastMCP provides an ASGI app for SSE transport, but we need to mount it
# or wire it manually. As of mcp python SDK 1.0+, FastMCP integrates with starlette.

try:
    from mcp.server.fastmcp import create_starlette_app
    # Create the internal SSE app
    sse_app = create_starlette_app(unified_mcp, debug=True)

    # Mount the SSE app under /mcp, applying auth dependency manually below.
    # Note: Mounting completely hands over the route. Wait, FastMCP create_starlette_app
    # returns an ASGI app. If we mount it, auth will be bypassed unless handled at middleware.
    
    # Let's add an auth middleware specifically for the /mcp path
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path.startswith("/mcp"):
            if settings.mcp_api_key:
                api_key = request.headers.get("X-API-Key")
                if api_key != settings.mcp_api_key:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(status_code=401, content={"detail": "Invalid API Key"})
        return await call_next(request)

    app.mount("/mcp", sse_app)
    logger.info("Mounted FastMCP SSE transport at /mcp")
except ImportError:
    logger.error("Could not mount FastMCP SSE. Ensure mcp SDK is v1.0.0+")


# ── Health Status ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return {"status": "ok", "tools_count": len(unified_mcp._tool_manager._tools)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.backend_api.app:app", host=settings.host, port=settings.port, reload=True)
