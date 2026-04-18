"""
Validators — Pydantic schemas for request validation across the MCPServer.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

class SearchQuerySchema(BaseModel):
    """Validate web search requests."""
    query: str = Field(..., min_length=1, max_length=500, description="The search query string")
    max_results: int = Field(5, ge=1, le=20, description="Maximum results to return (1-20)")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


# ------------------------------------------------------------------
# File System
# ------------------------------------------------------------------

class FilePathSchema(BaseModel):
    """Validate file path requests."""
    path: str = Field(..., min_length=1, description="Absolute or relative file path")

    @field_validator("path")
    @classmethod
    def path_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v.strip()


class FileWriteSchema(BaseModel):
    """Validate file write requests."""
    path: str = Field(..., min_length=1, description="Target file path")
    content: str = Field(..., description="Content to write to the file")
    overwrite: bool = Field(True, description="Whether to overwrite existing file")


# ------------------------------------------------------------------
# Database
# ------------------------------------------------------------------

class SQLQuerySchema(BaseModel):
    """Validate SQL query requests. Enforces read-only."""
    sql: str = Field(..., min_length=1, description="SQL SELECT statement")
    timeout_seconds: float = Field(10.0, ge=1.0, le=30.0, description="Query timeout")

    @field_validator("sql")
    @classmethod
    def sql_must_be_read_only(cls, v):
        stripped = v.strip().upper()
        if not stripped.startswith(("SELECT", "WITH")):
            raise ValueError("Only SELECT and WITH (CTE) queries are permitted")
        # Block common dangerous patterns
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "EXEC"]
        for keyword in dangerous:
            if f" {keyword} " in f" {stripped} ":
                raise ValueError(f"Dangerous SQL keyword '{keyword}' is not permitted")
        return v


# ------------------------------------------------------------------
# API Integration
# ------------------------------------------------------------------

class RestAPISchema(BaseModel):
    """Validate external REST API call requests."""
    url: str = Field(..., description="Full HTTP/HTTPS URL")
    method: str = Field("GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    json_body: Optional[Dict[str, Any]] = Field(None, description="JSON request body")
    timeout: float = Field(15.0, ge=1.0, le=60.0, description="Request timeout in seconds")

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @field_validator("method")
    @classmethod
    def method_must_be_valid(cls, v):
        allowed = {"GET", "POST", "PUT", "DELETE", "PATCH"}
        if v.upper() not in allowed:
            raise ValueError(f"Method must be one of: {', '.join(allowed)}")
        return v.upper()


# ------------------------------------------------------------------
# Tool Invocation
# ------------------------------------------------------------------

class ToolInvokeSchema(BaseModel):
    """Validate MCP tool invocation requests."""
    tool_name: str = Field(..., min_length=1, description="Name of the MCP tool")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

    @field_validator("tool_name")
    @classmethod
    def tool_name_alphanumeric(cls, v):
        import re
        if not re.match(r'^[a-z_][a-z0-9_]*$', v):
            raise ValueError("Tool name must be lowercase snake_case")
        return v
