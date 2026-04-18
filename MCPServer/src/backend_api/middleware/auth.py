import os
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
