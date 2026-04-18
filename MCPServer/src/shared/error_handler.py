"""
Shared Error Handlers and Middleware for FastMCP / FastAPI.
"""

from fastapi import Request
from fastapi.responses import JSONResponse


async def custom_exception_handler(request: Request, exc: Exception):
    """Fallback exception handler for unhandled errors in the API."""
    import logging
    logger = logging.getLogger("mcp_error_handler")
    logger.error(f"Unhandled Server Error at {request.url.path}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)},
    )
