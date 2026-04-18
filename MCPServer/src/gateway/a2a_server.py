"""
A2A (Agent-to-Agent) Server.
Exposes LangGraph orchestration agents as external services using HTTP.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)


class A2ARequest(BaseModel):
    agent_id: str
    message: str
    context: dict | None = None
    session_id: str


def register_a2a_endpoints(app: FastAPI):
    """
    Attach A2A routes to the FastAPI application.
    Allows external systems or other agents to call the AiAgents orchestrator over REST.
    """
    
    @app.post("/a2a/invoke")
    async def invoke_agent(request: A2ARequest):
        logger.info(f"A2A Request received for {request.agent_id} (session: {request.session_id})")
        
        # In a real environment, this would forward via HTTP or an internal queue to the AiAgents repo.
        # Since this runs in MCPServer Gateway, we mock the forward call or wire it to the URL of AiAgents.
        
        from src.config.settings import settings
        import httpx
        
        # AiAgents API assumed to run on another port, typically 8000
        ai_agents_url = "http://localhost:8000/invoke"
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    ai_agents_url,
                    json={
                        "agent_id": request.agent_id,
                        "session_id": request.session_id,
                        "message": request.message,
                        "context": request.context,
                    },
                    timeout=30.0
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.error(f"A2A Forwarding failed: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to reach AiAgents: {e}")

    @app.get("/a2a/capabilities")
    async def list_capabilities():
        """Expose the Agent Card."""
        return {
            "name": "AiEngineer Master Orchestrator",
            "capabilities": ["intent_routing", "crm_lookup", "rag_synthesis", "web_search", "db_query"],
            "protocols": ["a2a", "http"]
        }
