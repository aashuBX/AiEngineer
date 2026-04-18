from starlette.middleware.base import BaseHTTPMiddleware
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
