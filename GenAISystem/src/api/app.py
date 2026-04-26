"""
FastAPI Server for GenAISystem Pipeline.
Provides /ingest and /query endpoints.
"""

import logging
import tempfile
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from src.config.settings import settings
from src.vectorstores.factory import get_vector_store
from src.embeddings.factory import get_embeddings
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.retrieval.hybrid import HybridRetriever
from src.generation.response import ResponseGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genai_api")

# Global instances
vector_store = None
kg_builder = None
hybrid_retriever = None
response_generator = None


def _create_llm():
    """Create an LLM instance based on the configured provider."""
    provider = settings.llm_provider.lower()
    logger.info(f"Creating LLM: provider={provider}, model={settings.llm_model}")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.llm_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model,
            temperature=0,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=0,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.llm_model,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, kg_builder, hybrid_retriever, response_generator
    logger.info("Initializing GenAISystem backends...")
    
    # Init LLM via dynamic provider factory
    llm = _create_llm()
    
    # Init stores
    vector_store = get_vector_store()
    kg_builder = KnowledgeGraphBuilder(llm=llm)
    
    # Init RAG pieces
    from src.retrieval.llm_reranker import LLMReranker
    from src.retrieval.reranker import CrossEncoderReranker
    
    llm_reranker = LLMReranker(llm=llm)
    cross_encoder = CrossEncoderReranker()
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        llm=llm,
        llm_reranker=llm_reranker,
        cross_encoder=cross_encoder
    )
    response_generator = ResponseGenerator(llm=llm)
    
    yield
    logger.info("Shutting down GenAISystem...")



app = FastAPI(title="GenAISystem API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    strategy: str = "hybrid" # hybrid, vector, graph
    rerank_strategy: str = "auto" # "auto", "none", "cross_encoder", "llm_reranker"
    top_k: int = 5


class IngestRequest(BaseModel):
    directory_path: str
    strategy: str = "recursive"


@app.post("/query")
async def retrieve_and_generate(req: QueryRequest):
    """
    RAG Query endpoint.
    strategy:
      - "vector"  → Pinecone similarity search only (RagAgent)
      - "graph"   → Neo4j knowledge graph only (GraphRagAgent)
      - "hybrid"  → Both combined
    """
    from langchain_core.documents import Document

    try:
        docs = []

        if req.strategy in ("vector", "hybrid"):
            # Pinecone vector similarity search
            try:
                vector_docs = vector_store.similarity_search(req.query, k=req.top_k)
                docs.extend(vector_docs)
                logger.info(f"Vector search returned {len(vector_docs)} docs")
            except Exception as e:
                logger.error(f"Vector search failed: {e}")

        if req.strategy in ("graph", "hybrid"):
            # Neo4j knowledge graph search
            try:
                from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
                graph = Neo4jGraph(
                    url=settings.neo4j_uri,
                    username=settings.neo4j_username,
                    password=settings.neo4j_password,
                )
                chain = GraphCypherQAChain.from_llm(
                    _create_llm(), graph=graph, verbose=False, return_direct=True
                )
                graph_res = chain.invoke({"query": req.query})
                raw = graph_res.get("result", "")
                if raw:
                    docs.append(Document(page_content=str(raw), metadata={"source": "Neo4j"}))
                    logger.info("Graph search returned 1 result")
            except Exception as e:
                logger.error(f"Graph search failed: {e}")

        # Rerank if hybrid
        if req.strategy == "hybrid" and docs:
            docs = hybrid_retriever.async_retrieve(
                req.query, top_k=req.top_k, rerank_strategy=req.rerank_strategy
            )
        else:
            docs = docs[:req.top_k]

        # ── Mock fallback: if no docs found, return placeholder context ────────
        if not docs:
            logger.warning(f"No documents found for query '{req.query}' — returning mock context")
            docs = [
                Document(
                    page_content=(
                        "This is a placeholder response. The knowledge base is currently empty. "
                        "Please upload documents via the /upload endpoint to enable real RAG responses."
                    ),
                    metadata={"source": "mock", "note": "No documents ingested yet"},
                )
            ]

        # Format for the calling agent
        formatted_docs = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]

        # Generate a direct answer
        gen_res = await response_generator.generate_answer(req.query, docs)

        return {
            "query": req.query,
            "strategy": req.strategy,
            "answer": gen_res.get("answer"),
            "documents": formatted_docs,
            "citations": gen_res.get("citations", []),
        }
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        # Return a graceful mock response instead of crashing
        return {
            "query": req.query,
            "strategy": req.strategy,
            "answer": f"The knowledge base is currently unavailable. Please try again or upload documents first. (Error: {str(e)})",
            "documents": [],
            "citations": [],
        }


@app.post("/ingest")
def trigger_ingestion(req: IngestRequest):
    """
    Trigger the backend ingestion pipeline on a directory.
    """
    from src.ingestion.pipeline import IngestionPipeline
    from src.embeddings.factory import get_embeddings
    
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        knowledge_graph=kg_builder,
        embedder=get_embeddings()
    )
    
    try:
        stats = pipeline.run_directory_ingestion(
            directory=req.directory_path,
            strategy=req.strategy
        )
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error(f"Ingest endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/documents")
async def ingest_document(file: UploadFile = File(...), collection: str = "default"):
    """
    Accept a file upload, save it temporarily, and run the ingestion pipeline.
    This is the endpoint proxied by AiAgents /upload.
    """
    import uuid
    from src.ingestion.pipeline import IngestionPipeline
    from src.embeddings.factory import get_embeddings

    suffix = "." + file.filename.rsplit(".", 1)[-1] if "." in file.filename else ""
    tmp_dir = tempfile.mkdtemp()
    tmp_path = f"{tmp_dir}/{file.filename}"

    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        pipeline = IngestionPipeline(
            vector_store=vector_store,
            knowledge_graph=kg_builder,
            embedder=get_embeddings(),
        )
        stats = pipeline.run_directory_ingestion(directory=tmp_dir)

        return {
            "job_id": str(uuid.uuid4()),
            "filename": file.filename,
            "status": "ingested",
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"Document ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/health")
def health_check():
    return {"status": "up"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host=settings.host, port=settings.port, reload=True)
