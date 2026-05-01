"""
AiAgents HTTP API — FastAPI server that bridges the PlatformUI to the LangGraph pipeline.

Endpoints:
  POST /chat    — Send a user message, run the full multi-agent graph, return the answer.
  POST /upload  — Proxy document uploads to GenAISystem for RAG ingestion.
  GET  /health  — Container healthcheck.
"""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Global MCP Client ──────────────────────────────────────────────────────────
# Held open for the lifetime of the process so every request reuses the same
# network connection to MCPServer.
_mcp_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: connect MCPClient and register tools on all agents."""
    global _mcp_manager

    from src.mcp_client.client import MCPClientManager
    _mcp_manager = MCPClientManager()

    logger.info("Connecting MCP Client to MCPServer...")
    async with _mcp_manager.connect() as manager:
        logger.info(f"MCP connected. {len(manager.tools)} tools available.")
        app.state.mcp_tools = manager.tools
        yield

    logger.info("AiAgents API shutting down.")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AiAgents API",
    description="Multi-agent orchestration API — powered by LangGraph + MCP",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    architecture: Optional[str] = "supervisor"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    agent_used: Optional[str] = None


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Container healthcheck endpoint."""
    tools = getattr(app.state, "mcp_tools", [])
    return {
        "status": "ok",
        "mcp_tools_loaded": len(tools),
        "mcp_server_url": settings.mcp_server_url,
        "genai_system_url": settings.genai_system_url,
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Run a user message through the full multi-agent LangGraph pipeline.

    Flow:
      1. Build conversation state with the user's message.
      2. Register MCP tools on the agents.
      3. Run through Intent → GuardrailAgent → Supervisor → Specialist Agent.
      4. Return the final synthesized answer.
    """
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"[{session_id}] Received chat: {req.message!r}")

    try:
        from langchain_core.messages import HumanMessage
        from src.agents.intent.intent_agent import IntentAgent
        from src.agents.guardrail.guardrail_agent import GuardrailAgent
        from src.agents.crm.crm_agent import CrmAgent
        from src.agents.faq.faq_agent import FAQAgent
        from src.agents.feedback.feedback_agent import FeedbackAgent
        from src.agents.handoff.handoff_agent import HandoffAgent
        from src.agents.rag.rag_agent import RagAgent
        from src.agents.graph_rag.graph_rag_agent import GraphRagAgent
        from src.graphs.multi_agent_graph import build_supervisor_graph
        from src.graphs.multi_agent_react_graph import build_multi_agent_react_graph

        tools = getattr(app.state, "mcp_tools", [])

        # ── Guardrail: Input validation ────────────────────────────────────
        guardrail = GuardrailAgent()
        guard_state = {"messages": [HumanMessage(content=req.message)], "mode": "input"}
        guard_result = await guardrail.process(guard_state)

        if guard_result.get("task_status") == "blocked":
            return ChatResponse(
                session_id=session_id,
                answer=guard_result.get("error", "Your message was blocked by the content policy."),
                agent_used="guardrail",
            )

        # ── Intent classification ──────────────────────────────────────────
        intent_agent = IntentAgent()
        intent_state = {"messages": [HumanMessage(content=req.message)]}
        intent_result = await intent_agent.process(intent_state)
        detected_intent = intent_result.get("intent", "general")
        logger.info(f"[{session_id}] Intent detected: {detected_intent}")

        # ── Build worker agents and register MCP tools ─────────────────────
        crm = CrmAgent()
        crm.register_tools(tools)

        faq = FAQAgent()
        faq.register_tools(tools)

        feedback = FeedbackAgent()
        feedback.register_tools(tools)

        handoff = HandoffAgent()
        handoff.register_tools(tools)
        
        rag = RagAgent()
        rag.register_tools(tools)
        
        graph_rag = GraphRagAgent()
        graph_rag.register_tools(tools)

        worker_nodes = {
            "crm_agent": crm.process,
            "faq_agent": faq.process,
            "feedback_agent": feedback.process,
            "handoff_agent": handoff.process,
            "rag_agent": rag.process,
            "graph_rag_agent": graph_rag.process,
        }

        # ── Run the Multi-Agent pipeline ──────────────────────────
        if req.architecture == "react":
            logger.info(f"[{session_id}] Using Multi-Agent ReAct architecture")
            graph = build_multi_agent_react_graph(worker_nodes)
        else:
            logger.info(f"[{session_id}] Using Supervisor architecture")
            graph = build_supervisor_graph(worker_nodes)
        initial_state = {
            "messages": [HumanMessage(content=req.message)],
            "session_id": session_id,
            "intent": detected_intent,
            "worker_responses": [],
            "iteration_count": 0,
        }
        final_state = await graph.ainvoke(initial_state)

        # ── Extract the final answer ────────────────────────────────────────
        messages = final_state.get("messages", [])
        answer = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and getattr(msg, "type", "") in ("ai", "assistant"):
                answer = msg.content
                break

        if not answer:
            answer = "I processed your request but could not generate a response."

        agent_used = final_state.get("routing_decision", "supervisor")
        logger.info(f"[{session_id}] Response generated by: {agent_used}")

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            agent_used=agent_used,
        )

    except Exception as e:
        logger.error(f"[{session_id}] Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(e)}")


@app.post("/upload", response_model=UploadResponse, tags=["RAG"])
async def upload_document(
    file: UploadFile = File(...),
    collection: str = "default",
):
    """
    Upload a document for RAG ingestion.
    Proxies the file to GenAISystem's /ingest/documents endpoint.
    """
    logger.info(f"Proxying document upload: {file.filename} → GenAISystem")

    try:
        content = await file.read()
        genai_url = f"{settings.genai_system_url}/ingest/documents"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                genai_url,
                files={"file": (file.filename, content, file.content_type)},
                params={"collection": collection},
            )
            response.raise_for_status()
            data = response.json()

        return UploadResponse(
            job_id=data.get("job_id", str(uuid.uuid4())),
            filename=file.filename,
            status=data.get("status", "queued"),
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"GenAISystem rejected upload: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail="GenAISystem ingestion error")
    except Exception as e:
        logger.error(f"Upload proxy error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=port, reload=True)
