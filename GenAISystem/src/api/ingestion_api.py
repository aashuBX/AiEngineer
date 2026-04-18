"""
Ingestion API — FastAPI endpoints for document upload, text ingestion, and job tracking.
"""

import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# In-memory job tracker (replace with database in production)
_ingestion_jobs: Dict[str, Dict[str, Any]] = {}


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class TextIngestionRequest(BaseModel):
    """Direct text ingestion request."""
    text: str = Field(..., description="Text content to ingest")
    collection: str = Field("default", description="Target collection name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    chunking_strategy: str = Field("recursive", description="Chunking strategy: recursive, semantic, agentic, document")
    chunk_size: int = Field(1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(200, description="Chunk overlap in characters")


class IngestionStatus(BaseModel):
    """Ingestion job status response."""
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float = 0.0
    documents_processed: int = 0
    total_documents: int = 0
    error: Optional[str] = None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/documents", response_model=IngestionStatus)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = "default",
    chunking_strategy: str = "recursive",
):
    """Upload and process a document through the ingestion pipeline.

    Supported formats: PDF, TXT, CSV, DOCX, JSON, HTML, Markdown.
    Processing happens asynchronously — use the job_id to track progress.
    """
    job_id = str(uuid.uuid4())

    _ingestion_jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "collection": collection,
        "progress": 0.0,
        "documents_processed": 0,
        "total_documents": 1,
    }

    # Read file content
    content = await file.read()

    # Queue background processing
    background_tasks.add_task(
        _process_document,
        job_id=job_id,
        filename=file.filename,
        content=content,
        collection=collection,
        chunking_strategy=chunking_strategy,
    )

    return IngestionStatus(
        job_id=job_id,
        status="queued",
        total_documents=1,
    )


@router.post("/text", response_model=IngestionStatus)
async def ingest_text(request: TextIngestionRequest, background_tasks: BackgroundTasks):
    """Ingest raw text directly into the knowledge base."""
    job_id = str(uuid.uuid4())

    _ingestion_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "documents_processed": 0,
        "total_documents": 1,
    }

    background_tasks.add_task(
        _process_text,
        job_id=job_id,
        text=request.text,
        collection=request.collection,
        metadata=request.metadata,
        chunking_strategy=request.chunking_strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    return IngestionStatus(
        job_id=job_id,
        status="queued",
        total_documents=1,
    )


@router.get("/status/{job_id}", response_model=IngestionStatus)
async def check_ingestion_status(job_id: str):
    """Check the status of an ingestion job."""
    if job_id not in _ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _ingestion_jobs[job_id]
    return IngestionStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        documents_processed=job.get("documents_processed", 0),
        total_documents=job.get("total_documents", 0),
        error=job.get("error"),
    )


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Remove an entire knowledge base collection and its data."""
    # TODO: Wire to vector store deletion
    return {
        "status": "deleted",
        "collection": name,
        "message": f"Collection '{name}' has been removed.",
    }


# ------------------------------------------------------------------
# Background Tasks
# ------------------------------------------------------------------

async def _process_document(
    job_id: str,
    filename: str,
    content: bytes,
    collection: str,
    chunking_strategy: str,
):
    """Background task: process an uploaded document."""
    try:
        _ingestion_jobs[job_id]["status"] = "processing"
        _ingestion_jobs[job_id]["progress"] = 0.1

        # TODO: Wire to ingestion pipeline
        # 1. Detect format from filename
        # 2. Load and parse document
        # 3. Clean and preprocess text
        # 4. Chunk based on strategy
        # 5. Embed chunks
        # 6. Store in vector database

        _ingestion_jobs[job_id]["progress"] = 1.0
        _ingestion_jobs[job_id]["status"] = "completed"
        _ingestion_jobs[job_id]["documents_processed"] = 1

    except Exception as e:
        _ingestion_jobs[job_id]["status"] = "failed"
        _ingestion_jobs[job_id]["error"] = str(e)


async def _process_text(
    job_id: str,
    text: str,
    collection: str,
    metadata: Dict[str, Any],
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Background task: process raw text ingestion."""
    try:
        _ingestion_jobs[job_id]["status"] = "processing"

        # TODO: Wire to ingestion pipeline
        _ingestion_jobs[job_id]["progress"] = 1.0
        _ingestion_jobs[job_id]["status"] = "completed"
        _ingestion_jobs[job_id]["documents_processed"] = 1

    except Exception as e:
        _ingestion_jobs[job_id]["status"] = "failed"
        _ingestion_jobs[job_id]["error"] = str(e)
