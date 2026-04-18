from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import logging
import time

logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Timing: {process_time:.4f}s")
        return response
