# AI Engineer Platform

> A production-grade AI engineering platform covering GenAI & Agentic AI — from foundational RAG to multi-agent orchestration with MCP tool integration.

## Architecture

```
AiEngineer/
├── AiAgents/        # Multi-agent orchestration (LangGraph, MCP Client, A2A)
├── MCPServer/       # MCP tool servers & backend API (FastMCP, FastAPI)
├── GenAISystem/     # RAG pipeline, vector stores, knowledge graph, evaluation
├── PlatformUI/      # Dashboard UI
├── docker-compose.yml
└── README.md
```

## Quick Start

```bash
# Clone
git clone https://github.com/your-org/AiEngineer.git
cd AiEngineer

# Install each repo (using uv)
cd AiAgents && uv sync && cd ..
cd MCPServer && uv sync && cd ..
cd GenAISystem && uv sync && cd ..

# Set environment variables
cp AiAgents/.env.example AiAgents/.env
cp MCPServer/.env.example MCPServer/.env
cp GenAISystem/.env.example GenAISystem/.env
# Edit .env files with your API keys

# Run services
docker-compose up -d  # Optional: Neo4j, PostgreSQL
```

## Repos

| Repo | Purpose | Key Tech |
|:---|:---|:---|
| **AiAgents** | Multi-agent orchestration, supervisor-worker graphs, human-in-the-loop | LangGraph, LangChain, MCP Adapters |
| **MCPServer** | Tool servers (web search, DB, filesystem, calculator, weather, API) | FastMCP, FastAPI |
| **GenAISystem** | RAG pipeline, hybrid retrieval, knowledge graph, evaluation | FAISS, Qdrant, Neo4j, RAGAS |

## Technology Stack

- **Python 3.12+** with `uv` package manager
- **LangGraph** for stateful agent workflows
- **FastMCP** for Model Context Protocol servers
- **Neo4j** for knowledge graph (GraphRAG)
- **FAISS / Qdrant / ChromaDB / Pinecone** for vector search
- **RAGAS** for RAG evaluation
- **FastAPI** for REST APIs

## License

MIT
