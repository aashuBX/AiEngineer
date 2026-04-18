from pydantic import BaseModel, Field

class SearchQuerySchema(BaseModel):
    query: str = Field(..., description="The search query string")
    max_results: int = Field(5, description="Maximum number of results to fetch")

class FilePathSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative file path")
