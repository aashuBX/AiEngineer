"""
PlatformUI — Streamlit Chatbot for AiEngineer Platform.

Features:
  - Multi-agent chat powered by the AiAgents LangGraph pipeline.
  - Document upload for RAG ingestion (PDF, TXT, DOCX, CSV, Markdown).
  - Session tracking and conversation history.
  - Live agent routing display (which agent answered).
"""

import os
import uuid

import httpx
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────
AGENTS_API_URL = os.getenv("AGENTS_API_URL", "http://localhost:8000")
GENAI_SYSTEM_URL = os.getenv("GENAI_SYSTEM_URL", "http://localhost:8001")

st.set_page_config(
    page_title="AiEngineer — Agent Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .crm-badge { background-color: #1d4ed8; color: white; }
    .faq-badge { background-color: #065f46; color: white; }
    .feedback-badge { background-color: #92400e; color: white; }
    .handoff-badge { background-color: #7c3aed; color: white; }
    .guardrail-badge { background-color: #991b1b; color: white; }
    .supervisor-badge { background-color: #374151; color: white; }
    .upload-success { color: #065f46; font-weight: 600; }
    .upload-error { color: #991b1b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_status" not in st.session_state:
    st.session_state.upload_status = []


def get_badge(agent: str) -> str:
    """Return an HTML badge for the agent that handled the request."""
    agent = (agent or "supervisor").lower()
    css_class = f"{agent.replace('_agent', '')}-badge"
    label = agent.replace("_", " ").title()
    return f'<span class="agent-badge {css_class}">{label}</span>'


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/microsoft/generative-ai-for-beginners/main/images/ai-engineer.png",
             width=60, use_column_width=False)
    st.title("🤖 AiEngineer")
    st.caption("Multi-Agent AI Platform")

    st.markdown("---")
    st.subheader("📋 Session")
    st.code(f"ID: {st.session_state.session_id[:12]}...", language=None)
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.upload_status = []
        st.rerun()

    st.markdown("---")

    # ── Document Upload for RAG ──────────────────────────────────────────
    st.subheader("📄 RAG Document Upload")
    st.caption("Upload documents so the AI agents can answer questions from them.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "docx", "csv", "md", "json", "html"],
        label_visibility="collapsed",
    )
    collection = st.text_input("Collection name", value="default",
                                help="Logical grouping for your documents in the vector store.")

    if st.button("⬆️ Upload & Ingest", use_container_width=True, disabled=uploaded_file is None):
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            try:
                response = httpx.post(
                    f"{AGENTS_API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    params={"collection": collection},
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                st.session_state.upload_status.append({
                    "filename": data.get("filename", uploaded_file.name),
                    "job_id": data.get("job_id", ""),
                    "status": "✅ Queued",
                })
                st.success(f"✅ {uploaded_file.name} ingested into **{collection}**!")
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

    # Show upload history
    if st.session_state.upload_status:
        st.markdown("**Recent Uploads:**")
        for item in reversed(st.session_state.upload_status[-5:]):
            st.markdown(f"- `{item['filename']}` {item['status']}")

    st.markdown("---")
    st.subheader("🌐 System Status")
    if st.button("Check Health", use_container_width=True):
        try:
            r = httpx.get(f"{AGENTS_API_URL}/health", timeout=5.0)
            data = r.json()
            st.success(f"✅ AiAgents: {data.get('mcp_tools_loaded', 0)} tools loaded")
        except Exception as e:
            st.error(f"❌ AiAgents unreachable: {e}")


# ── Main Chat Area ───────────────────────────────────────────────────────────
st.markdown("## 💬 Agent Chat")
st.caption("Powered by LangGraph multi-agent pipeline → MCPServer tools → Groq LLM")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("agent"):
            st.markdown(get_badge(msg["agent"]), unsafe_allow_html=True)
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything... (e.g., 'What is order status for #1234?')"):
    # Render user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the AiAgents backend
    with st.chat_message("assistant"):
        with st.spinner("Agents are thinking..."):
            try:
                response = httpx.post(
                    f"{AGENTS_API_URL}/chat",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "I couldn't generate a response.")
                agent_used = data.get("agent_used", "supervisor")

                # Show the agent badge then the answer
                st.markdown(get_badge(agent_used), unsafe_allow_html=True)
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "agent": agent_used,
                })

            except httpx.ConnectError:
                error_msg = (
                    "⚠️ **Cannot connect to the AiAgents backend.**\n\n"
                    f"Make sure `docker compose up` is running and `ai_agents` service is healthy.\n"
                    f"Expected at: `{AGENTS_API_URL}`"
                )
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except Exception as e:
                error_msg = f"❌ **Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
