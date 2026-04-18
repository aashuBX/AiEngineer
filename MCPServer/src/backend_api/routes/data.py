"""
Data Route — REST API for database introspection and querying,
backed by the MCP database server.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

router = APIRouter()


class DatabaseQuery(BaseModel):
    """Database query request."""
    query: str = Field(..., description="SQL SELECT query to execute")
    db_name: str = Field("default", description="Database identifier")


class TableSchemaRequest(BaseModel):
    """Table schema inspection request."""
    table_name: str = Field(..., description="Name of the table to inspect")


class DataResponse(BaseModel):
    """Standard data response."""
    status: str = "success"
    data: Any = None
    message: str = ""


@router.post("/query", response_model=DataResponse)
async def execute_query(query_data: DatabaseQuery):
    """Execute a read-only SQL query via the MCP database server.

    Only SELECT and WITH (CTE) queries are permitted.
    """
    try:
        from src.servers.database.server import query_database

        result = query_database(query_data.query)

        if result.startswith("Error"):
            raise HTTPException(status_code=400, detail=result)

        return DataResponse(data=result, message=f"Query executed on '{query_data.db_name}'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@router.get("/tables", response_model=DataResponse)
async def list_tables():
    """List all tables in the connected database."""
    try:
        from src.servers.database.server import list_tables

        result = list_tables()
        return DataResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")


@router.post("/schema", response_model=DataResponse)
async def get_schema(request: TableSchemaRequest):
    """Get the schema of a specific table."""
    try:
        from src.servers.database.server import get_table_schema

        result = get_table_schema(request.table_name)
        if result.startswith("Error"):
            raise HTTPException(status_code=404, detail=result)

        return DataResponse(data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")
