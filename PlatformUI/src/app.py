"""
PlatformUI — Streamlit Dashboard for AI Engineer 
Provides a chat interface and a Human-In-The-Loop approval view.
"""

import streamlit as st
import httpx
import uuid

# --- Page Config ---
st.set_page_config(
    page_title="Platform UI | AI Engineer",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Platform UI — Agent Interaction")

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    st.subheader("Human-In-The-Loop (Demo)")
    st.info(
        "In a full integration, LangGraph interrupts would appear here "
        "for your review before the agent proceeds with dangerous tools."
    )

# --- Chat Stream ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
if prompt := st.chat_input("Ask a question or request an action..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simulated Backend Call (In prod this goes to AiAgents A2A or GenAISystem APIs)
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            # Dummy integration for now until AiAgents API wrapper is bound
            answer = f"Echo (Backend placeholder): Received '{prompt}'. My backend connects to AiAgents orchestration logic."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
