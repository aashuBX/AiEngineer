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
semantic_cache = None
keyword_retriever = None


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
    global vector_store, kg_builder, hybrid_retriever, response_generator, semantic_cache, keyword_retriever
    logger.info("Initializing GenAISystem backends...")
    
    # Init LLM via dynamic provider factory
    llm = _create_llm()
    embeddings = get_embeddings()
    
    # Init stores
    vector_store = get_vector_store()
    kg_builder = KnowledgeGraphBuilder(llm=llm)
    
    # Init Cache and Keyword Retriever
    from src.retrieval.semantic_cache import SemanticCache
    from src.retrieval.keyword_retriever import KeywordRetriever
    semantic_cache = SemanticCache(
        embeddings=embeddings,
        threshold=settings.semantic_cache_threshold,
        ttl_hours=settings.semantic_cache_ttl_hours,
        redis_url=settings.redis_url
    )
    keyword_retriever = KeywordRetriever()
    
    # Init RAG pieces
    from src.retrieval.llm_reranker import LLMReranker
    from src.retrieval.reranker import CrossEncoderReranker
    
    llm_reranker = LLMReranker(llm=llm)
    cross_encoder = CrossEncoderReranker()
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        llm=llm,
        llm_reranker=llm_reranker,
        cross_encoder=cross_encoder,
        keyword_retriever=keyword_retriever
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
      - "vector"  → Pinecone + BM25
      - "graph"   → Neo4j knowledge graph only
      - "hybrid"  → Both combined
    """
    from langchain_core.documents import Document

    try:
        # ── 1. Semantic Cache Check ──
        if semantic_cache:
            cache_res = semantic_cache.get(req.query)
            if cache_res:
                return {
                    "query": req.query,
                    "strategy": req.strategy,
                    "answer": cache_res["answer"],
                    "documents": [],
                    "citations": [],
                    "cache_hit": True,
                    "cache_score": cache_res["cache_score"]
                }

        # ── 2. Unified Retrieval (Dense, Sparse, Graph, RRF, Rerank) ──
        docs = hybrid_retriever.async_retrieve(
            req.query,
            strategy=req.strategy,
            top_k=req.top_k,
            rerank_strategy=req.rerank_strategy
        )

        # ── 3. Mock fallback if no docs found ──
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

        formatted_docs = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]

        # ── 4. Extractive QA Bypass Check ──
        top_doc = docs[0]
        top_score = top_doc.metadata.get("rerank_score", 0)
        
        if top_score >= settings.extractive_qa_threshold and req.strategy != "graph":
            logger.info(f"LLM Bypass triggered (score {top_score:.3f} >= {settings.extractive_qa_threshold})")
            answer = top_doc.page_content
            citations = [{"id": 1, "source": top_doc.metadata.get("source"), "snippet": answer[:150]}]
            
            # Cache the exact match
            if semantic_cache:
                semantic_cache.set(req.query, answer)
                
            return {
                "query": req.query,
                "strategy": req.strategy,
                "answer": answer,
                "documents": formatted_docs,
                "citations": citations,
                "llm_bypassed": True,
                "bypass_reason": f"High-confidence exact match (score={top_score:.3f})"
            }

        # ── 5. LLM Synthesis ──
        gen_res = await response_generator.generate_answer(req.query, docs)
        answer = gen_res.get("answer")

        # Cache the generated response
        if semantic_cache and answer:
            semantic_cache.set(req.query, answer)

        return {
            "query": req.query,
            "strategy": req.strategy,
            "answer": answer,
            "documents": formatted_docs,
            "citations": gen_res.get("citations", []),
            "llm_bypassed": False
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
