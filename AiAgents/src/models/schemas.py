"""
Shared Pydantic data models used across agents, tools, and APIs.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field
from typing import List


# ── Enumerations ───────────────────────────────────────────────────────────────
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"
    AWAITING_HUMAN = "awaiting_human"


class IntentType(str, Enum):
    FAQ = "faq"
    CRM = "crm"
    RAG = "rag"
    GRAPH_RAG = "graph_rag"
    FEEDBACK = "feedback"
    HANDOFF = "handoff"
    UNKNOWN = "unknown"


class GuardrailDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"
    RETRY = "retry"   # Triggers hallucination recovery re-generation


class HallucinationVerdict(str, Enum):
    """Verdict from the post-generation hallucination detection cascade."""
    FAITHFUL = "faithful"             # All claims grounded in context
    HALLUCINATED = "hallucinated"     # Clear fabrication detected
    PARTIALLY_FAITHFUL = "partially_faithful"  # Some claims unsupported


# ── Core Models ────────────────────────────────────────────────────────────────
class AgentMessage(BaseModel):
    """A structured message between agents or from user/agent."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    agent_name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """Result of a single tool invocation."""

    tool_name: str
    tool_input: dict[str, Any]
    result: Any
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TaskResult(BaseModel):
    """Final result returned by an agent or workflow."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    status: TaskStatus
    agent_name: str
    output: str
    citations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentConfig(BaseModel):
    """Configuration for a single agent instance."""

    name: str
    description: str
    provider: str = "groq"
    model: str = "qwen-qwq-32b"
    temperature: float = 0.0
    max_tokens: int = 4096
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    max_iterations: int = 10


class GuardrailResult(BaseModel):
    """Result of a guardrail check."""

    decision: GuardrailDecision
    reason: Optional[str] = None
    sanitized_content: Optional[str] = None
    detected_issues: list[str] = Field(default_factory=list)


class HallucinationResult(BaseModel):
    """Structured result from the tiered hallucination detection cascade."""

    verdict: HallucinationVerdict
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Tier 1 — Semantic similarity grounding
    semantic_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Tier 2 — LLM-as-a-Judge
    llm_judge_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)

    # Tier 3 — Claim decomposition
    total_claims: Optional[int] = None
    supported_claims: Optional[int] = None

    # Audit trail
    detection_tiers_run: List[str] = Field(default_factory=list)
    retry_suggested: bool = False   # True → recovery should attempt re-generation
    error: Optional[str] = None     # Set if a detection tier itself failed


class HandoffRequest(BaseModel):
    """Request to escalate to a human agent."""

    session_id: str
    reason: str
    conversation_summary: str
    user_message: str
    urgency: str = "normal"  # "low" | "normal" | "high" | "critical"
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackEntry(BaseModel):
    """User feedback on an agent response."""

    session_id: str
    message_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None
    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
