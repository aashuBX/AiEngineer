from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/ingest/documents")
async def upload_document(file: UploadFile = File(...)):
    # Trigger ingestion pipeline
    return {"status": "Accepted", "filename": file.filename}
