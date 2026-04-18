"""
Agent Card — A2A protocol agent capability advertisement.
Defines the agent card schema and provides pre-configured card instances.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

class AgentCapability(BaseModel):
    """A single capability exposed by an agent."""
    name: str = Field(..., description="Capability identifier")
    description: str = Field("", description="Human-readable description")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameter schema (JSON Schema)")


class AgentCard(BaseModel):
    """A2A Agent Card — advertises agent identity and capabilities.

    Used for inter-agent discovery: agents query each other's cards
    to determine who can handle a given task.
    """
    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    description: str = Field("", description="What this agent does")
    url: str = Field("", description="Base URL to reach this agent")
    capabilities: List[AgentCapability] = Field(default_factory=list)
    version: str = Field("1.0.0", description="Agent version")
    provider: str = Field("AiEngineer", description="Provider name")
    authentication: Optional[Dict[str, str]] = Field(None, description="Auth requirements")


# ------------------------------------------------------------------
# Pre-configured Cards
# ------------------------------------------------------------------

AI_ENGINEER_AGENT_CARD = AgentCard(
    agent_id="ai-engineer-orchestrator",
    name="AI Engineer Orchestrator",
    description="Multi-agent orchestration platform with MCP tool access, RAG, and specialized agents.",
    url="http://localhost:8001",
    capabilities=[
        AgentCapability(
            name="general_query",
            description="Route and answer general questions using specialized agents",
        ),
        AgentCapability(
            name="web_search",
            description="Search the web for real-time information",
        ),
        AgentCapability(
            name="database_query",
            description="Query connected SQL databases (read-only)",
        ),
        AgentCapability(
            name="file_operations",
            description="Read and write files in permitted directories",
        ),
        AgentCapability(
            name="rag_query",
            description="Answer questions using RAG over knowledge base",
        ),
        AgentCapability(
            name="graph_query",
            description="Query the knowledge graph for entity relationships",
        ),
    ],
    version="1.0.0",
    provider="AiEngineer Platform",
    authentication={"type": "api_key", "header": "X-API-Key"},
)


def get_agent_card() -> AgentCard:
    """Return the platform's agent card for A2A discovery."""
    return AI_ENGINEER_AGENT_CARD
