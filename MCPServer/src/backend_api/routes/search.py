from fastapi import APIRouter, Depends, HTTPException
from ...shared.validators import SearchQuerySchema
from ...servers.web_search.server import perform_search

router = APIRouter()

@router.post("/")
async def search_web(query_details: SearchQuerySchema):
    try:
        results = await perform_search(query_details.query, max_results=query_details.max_results)
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
