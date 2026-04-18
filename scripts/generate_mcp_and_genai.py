import os

PROJECT_ROOT = "/Users/aashu-kumar-jha/Documents/projects/AiEngineer"

FILES_CONTENT = {
    # ---------------- MCPServer Files ---------------- #
    "MCPServer/src/backend_api/routes/search.py": """from fastapi import APIRouter, Depends, HTTPException
from ...shared.validators import SearchQuerySchema
from ...servers.web_search.server import perform_search

router = APIRouter()

@router.post("/")
async def search_web(query_details: SearchQuerySchema):
    try:
        results = await perform_search(query_details.query, max_results=query_details.max_results)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
""",
    "MCPServer/src/backend_api/routes/data.py": """from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class DatabaseQuery(BaseModel):
    query: str
    db_name: str

@router.post("/query")
async def execute_query(query_data: DatabaseQuery):
    # Route to database server implementation
    return {"status": "success", "results": f"Stub for {query_data.db_name}"}
""",
    "MCPServer/src/backend_api/routes/files.py": """from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/list")
async def list_files(path: str = "."):
    # Route to file system server implementation
    return {"status": "success", "files": []}
""",
    "MCPServer/src/backend_api/middleware/auth.py": """import os
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key")
        expected_key = os.getenv("MCP_INTERNAL_API_KEY", "dev-secret-key")
        
        # Bypass auth for health/docs
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)
            
        if not api_key or api_key != expected_key:
             # Returning 401 response manually as raising HTTPException in middleware requires special handling
             from fastapi.responses import JSONResponse
             return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
             
        return await call_next(request)
""",
    "MCPServer/src/backend_api/middleware/rate_limiter.py": """from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()
        
        # Remove old requests
        if client_ip in self.clients:
            self.clients[client_ip] = [t for t in self.clients[client_ip] if t > current_time - self.window_seconds]
        else:
            self.clients[client_ip] = []
            
        if len(self.clients[client_ip]) >= self.max_requests:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})
            
        self.clients[client_ip].append(current_time)
        return await call_next(request)
""",
    "MCPServer/src/backend_api/middleware/logging_middleware.py": """from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import logging
import time

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Timing: {process_time:.4f}s")
        return response
""",
    "MCPServer/src/backend_api/models/request_response.py": """from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class BaseResponse(BaseModel):
    status: str = "success"
    data: Optional[Any] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
""",
    "MCPServer/src/shared/tool_registry.py": """from typing import Callable, Dict, Any, List

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, func: Callable, schema: Any = None):
        self._tools[name] = {
            "description": description,
            "func": func,
            "schema": schema
        }

    def get_tool(self, name: str):
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": name, "description": data["description"]} for name, data in self._tools.items()]

tool_registry = ToolRegistry()
""",
    "MCPServer/src/shared/validators.py": """from pydantic import BaseModel, Field

class SearchQuerySchema(BaseModel):
    query: str = Field(..., description="The search query string")
    max_results: int = Field(5, description="Maximum number of results to fetch")

class FilePathSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative file path")
""",
    "MCPServer/src/gateway/agent_card.py": """from pydantic import BaseModel
from typing import List, Optional

class AgentCapability(BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

class AgentCard(BaseModel):
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    version: str = "1.0.0"
""",

    # ---------------- GenAISystem Files ---------------- #
    "GenAISystem/src/ingestion/chunking/recursive_chunker.py": """from langchain.text_splitter import RecursiveCharacterTextSplitter

class AdvancedRecursiveChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )

    def split_text(self, text: str):
        return self.splitter.split_text(text)
    
    def split_documents(self, documents: list):
        return self.splitter.split_documents(documents)
""",
    "GenAISystem/src/ingestion/chunking/semantic_chunker.py": """# Stub for semantic chunker
class SemanticChunker:
    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
    
    def split_text(self, text: str):
        # Implement semantic similarity boundary detection
        pass
""",
    "GenAISystem/src/ingestion/chunking/agentic_chunker.py": """# Stub for agentic chunker leveraging LLM
class AgenticChunker:
    def __init__(self, llm):
        self.llm = llm
        
    def chunk_document(self, text: str):
        # Prompt LLM to determine boundaries based on topic shift
        pass
""",
    "GenAISystem/src/ingestion/chunking/document_chunker.py": """# Stub for structure-aware chunker
class DocumentStructureChunker:
    def __init__(self):
        pass
        
    def parse_and_chunk(self, file_path: str):
        # E.g. Split by Markdown headers
        pass
""",
    "GenAISystem/src/ingestion/preprocessing/metadata_extractor.py": """import re

class MetadataExtractor:
    @staticmethod
    def extract_basics(text: str) -> dict:
        metadata = {}
        # Naive extraction logic
        if re.search(r"Title: (.*)", text):
            metadata["title"] = re.search(r"Title: (.*)", text).group(1)
        return metadata
""",
    "GenAISystem/src/embeddings/embedding_factory.py": """from langchain_openai import OpenAIEmbeddings

class EmbeddingFactory:
    @staticmethod
    def get_embeddings(provider: str = "openai", model_name: str = "text-embedding-3-small"):
        if provider == "openai":
            return OpenAIEmbeddings(model=model_name)
        # Add HuggingFace, Cohere, etc.
        raise ValueError(f"Provider {provider} not supported.")
""",
    "GenAISystem/src/embeddings/batch_embedder.py": """class BatchEmbedder:
    def __init__(self, embedder_func, batch_size=100):
        self.embedder_func = embedder_func
        self.batch_size = batch_size

    def embed_documents(self, documents: list):
        results = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            results.extend(self.embedder_func(batch))
        return results
""",
    "GenAISystem/src/embeddings/embedding_cache.py": """import hashlib
import json

class EmbeddingCache:
    def __init__(self, cache_file: str = "embedding_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _hash_text(self, text: str):
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str):
        return self.cache.get(self._hash_text(text))

    def set(self, text: str, embedding: list):
        self.cache[self._hash_text(text)] = embedding
""",
    "GenAISystem/src/vectorstores/base_store.py": """from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Any]:
        pass
""",
    "GenAISystem/src/vectorstores/faiss_store.py": """from .base_store import BaseVectorStore

class FaissStore(BaseVectorStore):
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.index = None # Lazy load FAISS index

    def add_documents(self, documents, metadata=None):
        pass
        
    def search(self, query, top_k=5):
        return []
""",
    "GenAISystem/src/vectorstores/qdrant_store.py": """from .base_store import BaseVectorStore

class QdrantStore(BaseVectorStore):
    def __init__(self, collection_name: str, url: str):
        self.collection_name = collection_name
        self.url = url

    def add_documents(self, documents, metadata=None):
        pass
        
    def search(self, query, top_k=5):
        return []
""",
    "GenAISystem/src/vectorstores/chromadb_store.py": """from .base_store import BaseVectorStore

class ChromaDBStore(BaseVectorStore):
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory

    def add_documents(self, documents, metadata=None):
        pass
        
    def search(self, query, top_k=5):
        return []
""",
    "GenAISystem/src/vectorstores/pinecone_store.py": """from .base_store import BaseVectorStore

class PineconeStore(BaseVectorStore):
    def __init__(self, index_name: str):
        self.index_name = index_name

    def add_documents(self, documents, metadata=None):
        pass
        
    def search(self, query, top_k=5):
        return []
""",
    "GenAISystem/src/vectorstores/store_factory.py": """from .faiss_store import FaissStore
from .chromadb_store import ChromaDBStore

class VectorStoreFactory:
    @staticmethod
    def get_store(provider: str, **kwargs):
        if provider == "faiss":
            return FaissStore(kwargs.get("embedding_function"))
        elif provider == "chroma":
            return ChromaDBStore(kwargs.get("persist_directory", "./chroma_db"))
        raise ValueError("Unsupported vector store provider")
""",
    "GenAISystem/src/knowledge_graph/neo4j_client.py": """class Neo4jClient:
    def __init__(self, uri, username, password):
        self.uri = uri
        self.username = username
        self.password = password
        
    def connect(self):
        pass
        
    def execute_query(self, query: str, parameters: dict = None):
        return []
""",
    "GenAISystem/src/knowledge_graph/entity_extractor.py": """class EntityExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, text: str):
        # Uses LLM to extract NER logic
        return []
""",
    "GenAISystem/src/knowledge_graph/triplet_generator.py": """class TripletGenerator:
    def generate_triplets(self, text: str):
        # Generates Subject-Predicate-Object triplets
        return []
""",
    "GenAISystem/src/knowledge_graph/graph_retriever.py": """class GraphRetriever:
    def __init__(self, neo4j_client, llm):
        self.client = neo4j_client
        self.llm = llm

    def retrieve(self, query: str):
        # Translates query to Cypher and fetches context
        return "Graph Context"
""",
    "GenAISystem/src/knowledge_graph/graph_visualizer.py": """class GraphVisualizer:
    def render_subgraph(self, entities: list):
        pass
""",
    "GenAISystem/src/retrieval/vector_retriever.py": """class VectorRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5):
        return self.vector_store.search(query, top_k)
""",
    "GenAISystem/src/retrieval/keyword_retriever.py": """class KeywordRetriever:
    def __init__(self, documents):
        self.documents = documents
        # Build BM25 index here

    def retrieve(self, query: str, top_k: int = 5):
        # Execute BM25 ranking
        return []
""",
    "GenAISystem/src/retrieval/hybrid_retriever.py": """class HybridRetriever:
    def __init__(self, vector_retriever, keyword_retriever, fusion_algorithm):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.fusion_algorithm = fusion_algorithm

    def retrieve(self, query: str, top_k: int = 5):
        v_results = self.vector_retriever.retrieve(query, top_k*2)
        k_results = self.keyword_retriever.retrieve(query, top_k*2)
        return self.fusion_algorithm.fuse(v_results, k_results)[:top_k]
""",
    "GenAISystem/src/retrieval/fusion.py": """class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, *result_lists):
        # RRF implementation
        fused_scores = {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc.get("id", str(doc))
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {"doc": doc, "score": 0.0}
                fused_scores[doc_id]["score"] += 1.0 / (rank + self.k)
                
        sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results]
""",
    "GenAISystem/src/retrieval/query_router.py": """class QueryRouter:
    def __init__(self, llm):
        self.llm = llm

    def route_query(self, query: str) -> str:
        # Route logic: "vector", "keyword", "graph", or "hybrid"
        return "hybrid"
""",
    "GenAISystem/src/generation/response_generator.py": """class ResponseGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, query: str, context: list):
        prompt = f"Answer this query based on context:\\nContext: {context}\\nQuery: {query}"
        return self.llm.invoke(prompt)
""",
    "GenAISystem/src/generation/citation_handler.py": """class CitationHandler:
    def inject_citations(self, response_text: str, sources: list):
        # Appends citation markers mapping to sources
        return response_text
""",
    "GenAISystem/src/generation/streaming_response.py": """class StreamingResponder:
    async def stream_generate(self, llm, query: str, context: list):
        prompt = f"Context: {context}\\nQuery: {query}"
        async for chunk in llm.astream(prompt):
            yield chunk
""",
    "GenAISystem/src/evaluation/custom_metrics.py": """class CustomMetrics:
    @staticmethod
    def check_hallucination(answer: str, context: str, llm) -> float:
        # Ask LLM-as-judge if answer is supported by context
        return 0.0
""",
    "GenAISystem/src/evaluation/evaluation_pipeline.py": """class EvaluationPipeline:
    def __init__(self, evaluators: list):
        self.evaluators = evaluators

    def run_evalutaion(self, dataset: list):
        results = {}
        for entry in dataset:
            # Execute evaluations
            pass
        return results
""",
    "GenAISystem/src/agentic_rag/rag_graph.py": """from langgraph.graph import StateGraph, END

class AgenticRAGGraph:
    def __init__(self):
        self.graph = StateGraph(dict) # Replace with specific AgentState Dict
        # Build out nodes: retrieve, grade, generate
        
    def compile(self):
        return self.graph.compile()
""",
    "GenAISystem/src/agentic_rag/self_rag.py": """class SelfRAG:
    def implement_flow(self):
        # Self-reflective RAG workflow matching the implementation plan
        pass
""",
    "GenAISystem/src/agentic_rag/corrective_rag.py": """class CorrectiveRAG:
    def implement_flow(self):
        # Corrective RAG (CRAG) falling back to web search if retrieval is poor
        pass
""",
    "GenAISystem/src/api/rag_api.py": """from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class RAGQuery(BaseModel):
    query: str
    strategy: str = "hybrid"

@router.post("/query")
async def execute_rag_query(query_data: RAGQuery):
    # Route to RAG pipeline orchestration
    return {"response": "Mock answer based on retrieved context.", "sources": []}
""",
    "GenAISystem/src/api/ingestion_api.py": """from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/ingest/documents")
async def upload_document(file: UploadFile = File(...)):
    # Trigger ingestion pipeline
    return {"status": "Accepted", "filename": file.filename}
"""
}

def create_files():
    created_count = 0
    for rel_path, content in FILES_CONTENT.items():
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_count += 1
        print(f"Created/Updated: {rel_path}")
    print(f"\\nSuccessfully processed {created_count} files.")

if __name__ == "__main__":
    create_files()
