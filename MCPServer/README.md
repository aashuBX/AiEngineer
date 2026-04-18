# MCPServer

> The hands of the platform — tools exposed via Model Context Protocol, communicating with backend APIs. Each tool is a bounded-context micro-server.

## Features

- **6 MCP Tool Servers**: Web Search, Database, File System, API Integration, Calculator, Weather
- **Unified Gateway**: Single MCP endpoint aggregating all tools
- **Backend API**: FastAPI with auth, rate limiting, and request logging
- **A2A Server**: Exposes agents as external services
- **Security**: API key auth, rate limiting, sandboxed file operations

## Tool Servers

| Server | Tools | Backend |
|:---|:---|:---|
| **Web Search** | `web_search`, `fetch_webpage` | Tavily, DuckDuckGo |
| **Database** | `list_tables`, `get_table_schema`, `query_database` | SQLAlchemy |
| **File System** | `list_directory`, `read_file`, `write_file` | OS (sandboxed) |
| **API Integration** | `call_rest_api` | httpx |
| **Calculator** | `calculate`, `fibonacci` | Safe eval |
| **Weather** | `get_current_weather` | WeatherAPI |

## Quick Start

```bash
uv sync
cp .env.example .env

# Run unified MCP server (stdio)
python -m src.gateway.unified_server

# Run FastAPI backend
uvicorn src.backend_api.app:app --port 8001

# Test with MCP Inspector
npx @modelcontextprotocol/inspector
```

## Directory Structure

```
src/
├── config/          # Server settings
├── servers/         # Individual MCP tool servers
├── backend_api/     # FastAPI app, routes, middleware, models
├── gateway/         # Unified MCP server, A2A server, agent cards
└── shared/          # Tool registry, validators, error handler
```
