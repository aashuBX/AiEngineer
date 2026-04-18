from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class BaseResponse(BaseModel):
    status: str = "success"
    data: Optional[Any] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
