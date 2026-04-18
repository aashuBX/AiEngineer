"""
Action Validator — validates tool calls before execution.
Checks permitted actions, rate limits, and dangerous operations.
"""

import time
from collections import defaultdict
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Blocked tool actions ───────────────────────────────────────────────────────
BLOCKED_TOOLS: set[str] = {
    "delete_all_data",
    "drop_database",
    "execute_arbitrary_code",
    "system_command",
}

# ── Dangerous tool patterns that require approval ──────────────────────────────
REQUIRES_APPROVAL: set[str] = {
    "write_file",
    "insert_data",
    "send_email",
    "call_rest_api",
}

# ── Rate limiting config ───────────────────────────────────────────────────────
RATE_LIMITS: dict[str, tuple[int, int]] = {
    # tool_name: (max_calls, per_seconds)
    "web_search": (20, 60),
    "query_database": (50, 60),
    "call_rest_api": (10, 60),
    "default": (100, 60),
}


class ActionValidator:
    """Validates tool calls for safety, permissions, and rate limits."""

    def __init__(self, require_approval_for: set[str] | None = None):
        self._call_log: dict[str, list[float]] = defaultdict(list)
        self._require_approval = require_approval_for or REQUIRES_APPROVAL
        self._audit_log: list[dict] = []

    def validate(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        session_id: str = "unknown",
        skip_rate_limit: bool = False,
    ) -> dict[str, Any]:
        """
        Validate a tool action.

        Returns:
            Dict with keys: allowed (bool), requires_approval (bool), reason (str).
        """
        # ── Hard block ─────────────────────────────────────────────────────
        if tool_name in BLOCKED_TOOLS:
            logger.warning(f"ActionValidator: BLOCKED tool '{tool_name}' — session={session_id}")
            self._audit(tool_name, tool_input, session_id, "blocked")
            return {"allowed": False, "requires_approval": False, "reason": f"Tool '{tool_name}' is permanently blocked"}

        # ── Dangerous — needs human approval ──────────────────────────────
        needs_approval = tool_name in self._require_approval
        if needs_approval:
            logger.info(f"ActionValidator: tool '{tool_name}' requires human approval")

        # ── Rate limit check ───────────────────────────────────────────────
        if not skip_rate_limit:
            rate_check = self._check_rate_limit(tool_name, session_id)
            if not rate_check["allowed"]:
                self._audit(tool_name, tool_input, session_id, "rate_limited")
                return {"allowed": False, "requires_approval": False, "reason": rate_check["reason"]}

        # ── SQL injection check for DB tools ──────────────────────────────
        if tool_name in ("query_database", "execute_sql"):
            sql = str(tool_input.get("sql", tool_input.get("query", "")))
            if self._detect_sql_injection(sql):
                logger.warning(f"ActionValidator: SQL injection attempt in tool={tool_name}")
                self._audit(tool_name, tool_input, session_id, "sql_injection")
                return {"allowed": False, "requires_approval": False, "reason": "SQL injection detected"}

        self._audit(tool_name, tool_input, session_id, "approved" if not needs_approval else "pending_approval")
        return {
            "allowed": True,
            "requires_approval": needs_approval,
            "reason": "Approved" if not needs_approval else "Requires human approval",
        }

    def _check_rate_limit(self, tool_name: str, session_id: str) -> dict:
        max_calls, per_seconds = RATE_LIMITS.get(tool_name, RATE_LIMITS["default"])
        now = time.time()
        key = f"{session_id}:{tool_name}"
        # Remove old timestamps
        self._call_log[key] = [t for t in self._call_log[key] if now - t < per_seconds]
        if len(self._call_log[key]) >= max_calls:
            return {"allowed": False, "reason": f"Rate limit exceeded for '{tool_name}': {max_calls}/{per_seconds}s"}
        self._call_log[key].append(now)
        return {"allowed": True}

    def _detect_sql_injection(self, sql: str) -> bool:
        import re
        dangerous = re.compile(
            r"(?i)(DROP\s+TABLE|DELETE\s+FROM\s+\w+\s+WHERE\s+1=1|';\s*--|UNION\s+SELECT|xp_cmdshell)",
        )
        return bool(dangerous.search(sql))

    def _audit(self, tool_name: str, inputs: dict, session_id: str, decision: str) -> None:
        entry = {
            "tool": tool_name,
            "session_id": session_id,
            "decision": decision,
            "timestamp": time.time(),
            "input_keys": list(inputs.keys()),
        }
        self._audit_log.append(entry)
        logger.debug(f"ActionValidator audit: {entry}")

    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)
