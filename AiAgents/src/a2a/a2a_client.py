"""
A2A Client — Agent-to-Agent communication using the A2A protocol.
Discovers remote agents via Agent Cards and delegates tasks via HTTP/JSON-RPC.
"""

import asyncio
from typing import Any
from uuid import uuid4

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

# A2A Task states
A2A_STATES = {"submitted", "working", "completed", "failed", "canceled"}


class A2AClient:
    """
    Client for interacting with A2A-compatible remote agents.

    Supports:
    - Agent Card discovery (.well-known/agent.json)
    - Task submission (non-streaming and streaming)
    - Long-running task polling
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        """
        Args:
            base_url: Base URL of the remote A2A agent server.
            timeout:  HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._agent_card: dict | None = None

    # ── Agent Card Discovery ───────────────────────────────────────────────
    async def discover(self) -> dict:
        """Fetch and cache the remote agent's Agent Card."""
        if self._agent_card:
            return self._agent_card

        url = f"{self.base_url}/.well-known/agent.json"
        logger.info(f"A2AClient: discovering agent at {url}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            self._agent_card = resp.json()
            logger.info(
                f"A2AClient: discovered agent '{self._agent_card.get('name')}' "
                f"with skills: {[s.get('id') for s in self._agent_card.get('skills', [])]}"
            )
        return self._agent_card

    # ── Task Submission ────────────────────────────────────────────────────
    async def send_task(
        self,
        message: str,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        Submit a task to the remote agent and wait for completion.

        Args:
            message:    The user message / task description.
            session_id: Optional session ID for context continuity.
            metadata:   Additional task metadata.

        Returns:
            The completed task result dict.
        """
        task_id = str(uuid4())
        _session_id = session_id or str(uuid4())

        payload = {
            "jsonrpc": "2.0",
            "id": task_id,
            "method": "tasks/send",
            "params": {
                "id": task_id,
                "sessionId": _session_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
                "metadata": metadata or {},
            },
        }

        logger.info(f"A2AClient: sending task {task_id} to {self.base_url}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/", json=payload)
            resp.raise_for_status()
            result = resp.json()

        # Poll if task is still in progress
        task = result.get("result", {})
        return await self._poll_until_done(task, _session_id)

    # ── Polling ─────────────────────────────────────────────────────────────
    async def _poll_until_done(
        self,
        task: dict,
        session_id: str,
        poll_interval: float = 1.0,
        max_polls: int = 60,
    ) -> dict:
        """Poll task status until terminal state or timeout."""
        task_id = task.get("id")
        state = task.get("status", {}).get("state", "submitted")

        for _ in range(max_polls):
            if state in ("completed", "failed", "canceled"):
                break

            await asyncio.sleep(poll_interval)

            payload = {
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "tasks/get",
                "params": {"id": task_id, "sessionId": session_id},
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/", json=payload)
                resp.raise_for_status()
                task = resp.json().get("result", task)
                state = task.get("status", {}).get("state", state)
                logger.debug(f"A2AClient: task {task_id} state={state}")

        if state not in ("completed", "failed", "canceled"):
            logger.warning(f"A2AClient: task {task_id} timed out in state={state}")

        return task

    def extract_text(self, task: dict) -> str:
        """Extract the text content from a completed A2A task result."""
        try:
            artifacts = task.get("artifacts", [])
            for artifact in artifacts:
                for part in artifact.get("parts", []):
                    if part.get("type") == "text":
                        return part.get("text", "")
        except Exception:
            pass
        return task.get("status", {}).get("message", "No response from remote agent.")


class A2ARegistry:
    """Registry of known remote A2A agents."""

    def __init__(self):
        self._agents: dict[str, A2AClient] = {}

    def register(self, name: str, base_url: str) -> A2AClient:
        """Register a remote agent by name and URL."""
        client = A2AClient(base_url)
        self._agents[name] = client
        logger.info(f"A2ARegistry: registered agent '{name}' at {base_url}")
        return client

    def get(self, name: str) -> A2AClient | None:
        """Look up a registered agent client."""
        return self._agents.get(name)

    async def discover_all(self) -> dict[str, dict]:
        """Discover (fetch Agent Cards for) all registered agents."""
        results = {}
        for name, client in self._agents.items():
            try:
                card = await client.discover()
                results[name] = card
            except Exception as e:
                logger.warning(f"A2ARegistry: failed to discover '{name}': {e}")
        return results
