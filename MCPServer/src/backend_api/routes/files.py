"""
Files Route — REST API for file system operations,
backed by the MCP file system server.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional

router = APIRouter()


class FileReadRequest(BaseModel):
    """File read request."""
    path: str = Field(..., description="Absolute path to the file")


class FileWriteRequest(BaseModel):
    """File write request."""
    path: str = Field(..., description="Absolute path to the file")
    content: str = Field(..., description="Content to write")


class DirectoryListRequest(BaseModel):
    """Directory listing request."""
    path: str = Field(".", description="Absolute path to list")


class FileResponse(BaseModel):
    """Standard file operation response."""
    status: str = "success"
    data: Any = None
    message: str = ""


@router.post("/list", response_model=FileResponse)
async def list_files(request: DirectoryListRequest):
    """List the contents of a directory.

    Only permits access to directories configured in fs_allowed_dirs settings.
    """
    try:
        from src.servers.file_system.server import list_directory

        result = list_directory(request.path)

        if result.startswith("Error"):
            raise HTTPException(status_code=403, detail=result)

        files = [f for f in result.split("\n") if f.strip()] if result != "Directory is empty." else []
        return FileResponse(data=files, message=f"Listed {len(files)} items")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Directory listing failed: {str(e)}")


@router.post("/read", response_model=FileResponse)
async def read_file(request: FileReadRequest):
    """Read the contents of a text file.

    Access is restricted to configured allowed directories.
    """
    try:
        from src.servers.file_system.server import read_file

        result = read_file(request.path)

        if result.startswith("Error"):
            status_code = 403 if "denied" in result else 404
            raise HTTPException(status_code=status_code, detail=result)

        return FileResponse(data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read failed: {str(e)}")


@router.post("/write", response_model=FileResponse)
async def write_file(request: FileWriteRequest):
    """Write content to a file.

    Access is restricted to configured allowed directories.
    Creates parent directories if they don't exist.
    """
    try:
        from src.servers.file_system.server import write_file

        result = write_file(request.path, request.content)

        if result.startswith("Error"):
            raise HTTPException(status_code=403, detail=result)

        return FileResponse(message=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File write failed: {str(e)}")
