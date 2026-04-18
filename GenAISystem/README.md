# GenAISystem

> The knowledge layer — RAG pipeline, vector databases, knowledge graphs, hybrid retrieval, evaluation, and agentic RAG workflows.

## Features

- **Ingestion Pipeline**: Multi-format loading (PDF, TXT, CSV, DOCX, JSON, HTML, MD)
- **4 Chunking Strategies**: Recursive, Semantic, Agentic (LLM-guided), Document-Structure
- **Multi-Provider Embeddings**: OpenAI, SentenceTransformers, HuggingFace, Google, Cohere
- **4 Vector Stores**: FAISS, Qdrant, ChromaDB, Pinecone
- **Knowledge Graph**: Neo4j with entity extraction, triplet generation, graph retrieval
- **Hybrid Retrieval**: Vector + BM25 Keyword + Graph, with RRF fusion and cross-encoder reranking
- **Query Router**: LLM-based strategy selection (vector / keyword / graph / hybrid)
- **Generation**: Context-aware response with citations and streaming (SSE)
- **Evaluation**: RAGAS metrics + custom hallucination/latency/citation metrics
- **Agentic RAG**: Self-RAG, Corrective RAG (CRAG), Adaptive RAG — all as LangGraph workflows

## Quick Start

```bash
uv sync
cp .env.example .env

# Run the FastAPI server
uvicorn src.api.app:app --port 8002

# Ingest documents
curl -X POST http://localhost:8002/ingest/documents -F "file=@document.pdf"

# Query
curl -X POST http://localhost:8002/rag/query -d '{"query": "What is...", "strategy": "hybrid"}'
```

## Directory Structure

```
src/
├── config/          # Pydantic settings
├── ingestion/       # Data loader, chunking (4 strategies), preprocessing
├── embeddings/      # Factory, batch embedder, cache
├── vectorstores/    # FAISS, Qdrant, ChromaDB, Pinecone + factory
├── knowledge_graph/ # Neo4j client, entity extractor, triplets, graph retriever, visualizer
├── retrieval/       # Vector, keyword (BM25), hybrid, fusion (RRF), reranker, query router
├── generation/      # Response generator, citations, streaming
├── evaluation/      # RAGAS evaluator, custom metrics, evaluation pipeline
├── agentic_rag/     # Self-RAG, Corrective RAG, Adaptive RAG, RAG graph
└── api/             # FastAPI app, RAG API, ingestion API
```
