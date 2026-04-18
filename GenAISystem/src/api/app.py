"""
FastAPI Server for GenAISystem Pipeline.
Provides /ingest and /query endpoints.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, kg_builder, hybrid_retriever, response_generator
    logger.info("Initializing GenAISystem backends...")
    
    # Init LLM (generic import for factory)
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0)
    
    # Init stores
    vector_store = get_vector_store()
    kg_builder = KnowledgeGraphBuilder(llm=llm)
    
    # Init RAG pieces
    hybrid_retriever = HybridRetriever(vector_store=vector_store, llm=llm)
    response_generator = ResponseGenerator(llm=llm)
    
    yield
    logger.info("Shutting down GenAISystem...")


app = FastAPI(title="GenAISystem API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    strategy: str = "hybrid" # hybrid, vector, graph
    top_k: int = 5


class IngestRequest(BaseModel):
    directory_path: str
    strategy: str = "recursive"


@app.post("/query")
async def retrieve_and_generate(req: QueryRequest):
    """
    RAG Query endpoint. Returns context documents only if AiAgents GraphRag needs them,
    or generates a final answer if used standalone.
    """
    try:
        # 1. Retrieve
        docs = hybrid_retriever.async_retrieve(req.query, top_k=req.top_k)
        
        # Format docs for the orchestration Agent (GraphRagAgent)
        formatted_docs = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]
        
        # We also generate a direct answer here just in case caller wants it directly
        gen_res = await response_generator.generate_answer(req.query, docs)
        
        return {
            "query": req.query,
            "answer": gen_res.get("answer"),
            "documents": formatted_docs,
            "citations": gen_res.get("citations")
        }
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/health")
def health_check():
    return {"status": "up"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host=settings.host, port=settings.port, reload=True)
