"""
Safety configuration — centralized rules, thresholds, and blacklists for guardrails.
"""

from dataclasses import dataclass, field


@dataclass
class SafetyConfig:
    """Centralized configuration for all guardrail components."""

    # ── Input Guardrails ───────────────────────────────────────────────────
    block_pii: bool = True           # Block (vs sanitize) PII in inputs
    sanitize_pii: bool = False       # Sanitize instead of block if True
    block_injection: bool = True     # Always block prompt injections
    block_harmful: bool = True       # Block harmful content keywords

    # ── Output Guardrails ──────────────────────────────────────────────────
    block_toxic_output: bool = True
    block_high_hallucination: bool = False
    hallucination_threshold: float = 0.6  # Risk score above which to flag
    max_output_length: int = 4096         # Truncate outputs beyond this

    # ── Action Guardrails ──────────────────────────────────────────────────
    tools_requiring_approval: set[str] = field(default_factory=lambda: {
        "write_file", "insert_data", "send_email", "call_rest_api",
    })
    permanently_blocked_tools: set[str] = field(default_factory=lambda: {
        "delete_all_data", "drop_database", "execute_arbitrary_code",
    })

    # ── Rate Limits (tool → (max_calls, per_seconds)) ─────────────────────
    rate_limits: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "web_search": (20, 60),
        "query_database": (50, 60),
        "call_rest_api": (10, 60),
        "default": (100, 60),
    })

    # ── Topic Blocklist ────────────────────────────────────────────────────
    blocked_topics: list[str] = field(default_factory=lambda: [
        "weapons of mass destruction",
        "drug synthesis",
        "child exploitation",
        "financial fraud instructions",
    ])


# Default safety config (conservative)
DEFAULT_SAFETY_CONFIG = SafetyConfig()

# Permissive config for internal/trusted environments
PERMISSIVE_SAFETY_CONFIG = SafetyConfig(
    block_pii=False,
    sanitize_pii=True,
    block_high_hallucination=False,
    tools_requiring_approval=set(),
)

# Strict config for public-facing deployments
STRICT_SAFETY_CONFIG = SafetyConfig(
    block_pii=True,
    sanitize_pii=False,
    block_toxic_output=True,
    block_high_hallucination=True,
    hallucination_threshold=0.4,
)
