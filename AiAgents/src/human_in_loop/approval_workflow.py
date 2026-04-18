"""
Human-in-the-Loop Approval Workflow.
Handles approval/rejection/modification flows for both CLI (dev) and webhook (prod).
"""

import asyncio
from typing import Any, Callable

import httpx

from src.human_in_loop.interrupt_handler import parse_human_decision
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── CLI Approval (Development) ─────────────────────────────────────────────────

def cli_approval_handler(interrupt_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Collect human approval via CLI input (for dev/testing).

    Args:
        interrupt_payload: The interrupt context from LangGraph.

    Returns:
        Human decision dict.
    """
    print("\n" + "=" * 60)
    print("🔴 HUMAN APPROVAL REQUIRED")
    print("=" * 60)
    print(f"Session: {interrupt_payload.get('session_id')}")
    print(f"Agent:   {interrupt_payload.get('current_agent')}")
    print(f"\nQuestion: {interrupt_payload.get('question')}")
    if interrupt_payload.get("context"):
        print(f"\nContext: {interrupt_payload['context']}")
    print("\nOptions:")
    print("  approve           → Approve as-is")
    print("  reject            → Reject and stop")
    print("  modify: <text>    → Approve with modification")
    print("-" * 60)

    response = input("Your decision: ").strip()
    return parse_human_decision(response)


# ── Webhook Approval (Production) ─────────────────────────────────────────────

class WebhookApprovalWorkflow:
    """
    Sends approval requests to an external webhook and polls for responses.
    Suitable for production deployments (Slack, PlatformUI, email, etc.)
    """

    def __init__(self, webhook_url: str, poll_url: str, api_key: str = ""):
        self.webhook_url = webhook_url
        self.poll_url = poll_url
        self.api_key = api_key

    async def request_approval(
        self,
        interrupt_payload: dict[str, Any],
        timeout_seconds: int = 300,
        poll_interval: float = 5.0,
    ) -> dict[str, Any]:
        """
        Send interrupt payload to webhook and poll for human response.

        Args:
            interrupt_payload: Context from LangGraph interrupt.
            timeout_seconds:   Max wait time for human response.
            poll_interval:     Seconds between status polls.

        Returns:
            Human decision dict.
        """
        session_id = interrupt_payload.get("session_id", "unknown")
        headers = {"X-API-Key": self.api_key} if self.api_key else {}

        # Notify webhook
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    self.webhook_url,
                    json=interrupt_payload,
                    headers=headers,
                )
            logger.info(f"ApprovalWorkflow: webhook notified for session {session_id}")
        except Exception as e:
            logger.error(f"ApprovalWorkflow: webhook failed: {e}")
            return {"approved": False, "comment": "Webhook notification failed"}

        # Poll for response
        elapsed = 0.0
        while elapsed < timeout_seconds:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        f"{self.poll_url}/{session_id}",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("decision"):
                            logger.info(f"ApprovalWorkflow: received decision for {session_id}")
                            return parse_human_decision(data)
            except Exception as e:
                logger.debug(f"ApprovalWorkflow: poll error: {e}")

        logger.warning(f"ApprovalWorkflow: timeout after {timeout_seconds}s — auto-rejecting")
        return {"approved": False, "comment": "Timed out waiting for human approval"}


# ── Unified Approval Entry Point ───────────────────────────────────────────────

async def process_approval(
    interrupt_payload: dict[str, Any],
    handler: Callable | None = None,
    mode: str = "cli",
    **kwargs,
) -> dict[str, Any]:
    """
    Unified approval dispatcher.

    Args:
        interrupt_payload: Context from LangGraph interrupt.
        handler:           Custom approval handler function.
        mode:              "cli" | "webhook" | "auto_approve" | "auto_reject"
        **kwargs:          Passed to WebhookApprovalWorkflow if mode="webhook".

    Returns:
        Human decision dict.
    """
    if handler:
        if asyncio.iscoroutinefunction(handler):
            return await handler(interrupt_payload)
        return handler(interrupt_payload)

    if mode == "cli":
        return cli_approval_handler(interrupt_payload)

    elif mode == "webhook":
        workflow = WebhookApprovalWorkflow(
            webhook_url=kwargs.get("webhook_url", ""),
            poll_url=kwargs.get("poll_url", ""),
            api_key=kwargs.get("api_key", ""),
        )
        return await workflow.request_approval(interrupt_payload)

    elif mode == "auto_approve":
        logger.warning("ApprovalWorkflow: auto-approving (test mode)")
        return {"approved": True, "comment": "Auto-approved"}

    elif mode == "auto_reject":
        return {"approved": False, "comment": "Auto-rejected"}

    return {"approved": False, "comment": f"Unknown mode: {mode}"}
