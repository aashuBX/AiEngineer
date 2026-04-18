"""
Web Search MCP Server.
Provides tools to search the web using Tavily or DuckDuckGo.
"""

import httpx
from mcp.server.fastmcp import FastMCP
from typing import Any

from src.config.settings import settings

# Create a FastMCP app
mcp = FastMCP("web_search_server", description="Web search and scraping tools")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for real-time information.

    Args:
        query: The search query string.
        max_results: Maximum number of search results to return (default 5).
    """
    if settings.tavily_api_key:
        return await _tavily_search(query, max_results)
    else:
        return await _duckduckgo_search(query, max_results)


async def _tavily_search(query: str, max_results: int) -> str:
    """Search using Tavily API."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "advanced",
                    "include_raw_content": False,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for r in data.get("results", []):
                results.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}\n")

            return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Tavily search failed: {e}"


async def _duckduckgo_search(query: str, max_results: int) -> str:
    """Fallback search using DuckDuckGo (no API key required)."""
    try:
        from duckduckgo_search import DDGS
        import asyncio

        def _sync_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await asyncio.to_thread(_sync_search)

        out = []
        for r in results:
            out.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nContent: {r.get('body')}\n")

        return "\n".join(out) if out else "No results found."
    except Exception as e:
        return f"DuckDuckGo search failed: {e}"


@mcp.tool()
async def fetch_webpage(url: str) -> str:
    """
    Fetch and extract the main text content from a web page.

    Args:
        url: The URL of the webpage to scrape.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()

            # basic extraction using BeautifulSoup
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts and styles
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            text = soup.get_text(separator="\n")
            # compress whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)[:8000]  # truncate to avoid massive context
    except Exception as e:
        return f"Failed to fetch {url}: {e}"
