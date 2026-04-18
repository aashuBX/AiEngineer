# AiAgents

> The brain of the platform — multi-agent orchestration with LangGraph, MCP client for tool access, A2A for inter-agent communication, memory, guardrails, and human-in-the-loop workflows.

## Features

- **7 Specialized Agents**: Intent, Guardrail, GraphRAG, CRM, FAQ, Feedback, HandOff
- **5 Graph Patterns**: ReAct, Supervisor-Worker, Hierarchical, Plan-Execute, Map-Reduce
- **MCP Client**: Connect to any MCP tool server (stdio + streamable-http)
- **A2A Protocol**: Agent-to-Agent discovery and delegation
- **Memory**: Conversation buffer, summary, long-term vector memory, checkpointing
- **Guardrails**: Input/output validation, PII detection, toxicity filtering
- **Human-in-the-Loop**: Interrupt points, approval workflows
- **Prompts**: CoT, Self-Consistency, Tree-of-Thought, Few-Shot, Structured Output

## Directory Structure

```
src/
├── config/          # Settings, LLM provider factory
├── models/          # State schemas, Pydantic models
├── agents/          # Specialized agents (intent, guardrail, CRM, etc.)
├── graphs/          # LangGraph workflows (single, multi, hierarchical, plan-execute, map-reduce)
├── mcp_client/      # MCP client, config, tool adapter
├── a2a/             # Agent-to-Agent client
├── memory/          # Checkpointer, conversation memory, long-term memory
├── prompts/         # Prompt templates, CoT, few-shot, structured output
├── guardrails/      # Input/output/action validators, safety config
├── human_in_loop/   # Interrupt handler, approval workflows
└── utils/           # Logger, tracing (LangSmith/Langfuse)
```

## Usage

```python
from src.config.llm_providers import get_llm
from src.graphs.single_agent_graph import create_react_agent_graph

llm = get_llm(provider="groq", model="llama-3.3-70b-versatile")
graph = create_react_agent_graph(llm, tools=[...])
result = graph.invoke({"messages": [("user", "What is the weather in Tokyo?")]})
```

## Installation

```bash
uv sync
cp .env.example .env
# Add your API keys to .env
```
