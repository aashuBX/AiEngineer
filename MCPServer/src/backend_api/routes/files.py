from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/list")
async def list_files(path: str = "."):
    # Route to file system server implementation
    return {"status": "success", "files": []}
