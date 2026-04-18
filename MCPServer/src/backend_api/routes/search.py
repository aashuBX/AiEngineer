"""
Search Route — Exposes web search via REST API, backed by the MCP web search server.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

router = APIRouter()


class SearchQuery(BaseModel):
    """Search request schema."""
    query: str = Field(..., description="The search query string")
    max_results: int = Field(5, ge=1, le=20, description="Maximum results (1-20)")


class SearchResult(BaseModel):
    """Individual search result."""
    title: str = ""
    url: str = ""
    content: str = ""


class SearchResponse(BaseModel):
    """Search response payload."""
    status: str = "success"
    query: str
    results: List[SearchResult] = []
    total: int = 0


@router.post("/", response_model=SearchResponse)
async def search_web(query_details: SearchQuery):
    """Execute a web search via the underlying MCP web search server.

    Uses Tavily API if configured, falls back to DuckDuckGo.
    """
    try:
        from src.servers.web_search.server import web_search

        raw_result = await web_search(query_details.query, query_details.max_results)

        # Parse the string result into structured results
        results = _parse_search_results(raw_result)

        return SearchResponse(
            query=query_details.query,
            results=results,
            total=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def _parse_search_results(raw: str) -> List[SearchResult]:
    """Parse the raw text output from the MCP web_search tool into structured results."""
    if not raw or raw == "No results found.":
        return []

    results = []
    current = {}

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("Title: "):
            if current:
                results.append(SearchResult(**current))
            current = {"title": line[7:], "url": "", "content": ""}
        elif line.startswith("URL: "):
            current["url"] = line[5:]
        elif line.startswith("Content: "):
            current["content"] = line[9:]

    if current:
        results.append(SearchResult(**current))

    return results
