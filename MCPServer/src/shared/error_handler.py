"""
Error Handler — Standardized MCP/API error handling with error codes and structured responses.
"""

import logging
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Optional

logger = logging.getLogger("mcp_error_handler")


# ------------------------------------------------------------------
# MCP Error Codes
# ------------------------------------------------------------------

class MCPErrorCode:
    """Standardized error codes for MCP operations."""
    # General
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"

    # Auth
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"

    # Tools
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"

    # Data
    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    FILE_ACCESS_DENIED = "FILE_ACCESS_DENIED"

    # External
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    SEARCH_FAILED = "SEARCH_FAILED"


# ------------------------------------------------------------------
# Exception Classes
# ------------------------------------------------------------------

class MCPError(Exception):
    """Base exception for MCP errors."""

    def __init__(
        self,
        message: str,
        error_code: str = MCPErrorCode.INTERNAL_ERROR,
        status_code: int = 500,
        details: Optional[dict] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ToolNotFoundError(MCPError):
    def __init__(self, tool_name: str):
        super().__init__(
            message=f"Tool '{tool_name}' not found",
            error_code=MCPErrorCode.TOOL_NOT_FOUND,
            status_code=404,
        )


class ToolExecutionError(MCPError):
    def __init__(self, tool_name: str, original_error: Exception):
        super().__init__(
            message=f"Tool '{tool_name}' execution failed: {original_error}",
            error_code=MCPErrorCode.TOOL_EXECUTION_FAILED,
            status_code=500,
            details={"original_error": str(original_error)},
        )


# ------------------------------------------------------------------
# FastAPI Exception Handlers
# ------------------------------------------------------------------

async def mcp_exception_handler(request: Request, exc: MCPError):
    """Handle MCP-specific exceptions with structured responses."""
    logger.error(
        f"MCP Error [{exc.error_code}] at {request.url.path}: {exc.message}",
        extra={"details": exc.details},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "path": str(request.url.path),
        },
    )


async def custom_exception_handler(request: Request, exc: Exception):
    """Fallback exception handler for unhandled errors in the API."""
    logger.error(
        f"Unhandled Server Error at {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_code": MCPErrorCode.INTERNAL_ERROR,
            "message": "Internal Server Error",
            "details": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "path": str(request.url.path),
        },
    )


async def validation_exception_handler(request: Request, exc):
    """Handle Pydantic/FastAPI validation errors."""
    from fastapi.exceptions import RequestValidationError
    if isinstance(exc, RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "error_code": MCPErrorCode.VALIDATION_ERROR,
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
            },
        )
    return await custom_exception_handler(request, exc)
