"""
Request / Response models for the backend API.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime


# ------------------------------------------------------------------
# Base Response Models
# ------------------------------------------------------------------

class BaseResponse(BaseModel):
    """Standard success response wrapper."""
    status: str = "success"
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error response."""
    status: str = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ------------------------------------------------------------------
# MCP-specific Models
# ------------------------------------------------------------------

class ToolCallRequest(BaseModel):
    """Request to invoke an MCP tool through the REST API."""
    tool_name: str = Field(..., description="Name of the MCP tool to invoke")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    timeout: float = Field(15.0, ge=1.0, le=60.0, description="Timeout in seconds")


class ToolCallResponse(BaseModel):
    """Response from an MCP tool invocation."""
    status: str = "success"
    tool_name: str
    result: Any = None
    execution_time_ms: float = 0.0


# ------------------------------------------------------------------
# A2A Models
# ------------------------------------------------------------------

class AgentInvokeRequest(BaseModel):
    """Request to invoke an agent via A2A protocol."""
    query: str = Field(..., description="The task or question for the agent")
    agent_id: Optional[str] = Field(None, description="Target agent ID (auto-routes if None)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")


class AgentInvokeResponse(BaseModel):
    """Response from an agent invocation."""
    status: str = "success"
    agent_id: str = ""
    response: str = ""
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: Optional[str] = None


# ------------------------------------------------------------------
# Health & Status
# ------------------------------------------------------------------

class HealthStatus(BaseModel):
    """System health status."""
    status: str = "healthy"
    version: str = "1.0.0"
    services: Dict[str, str] = Field(default_factory=dict)
    uptime_seconds: float = 0.0
