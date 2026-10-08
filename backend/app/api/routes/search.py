from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse
from app.services.rag.retrieval import search_similar_documents

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def semantic_search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    filters = {
        "label": payload.label,
        "path": payload.path,
        "module": payload.module,
        "state": payload.state,
        "start_date": payload.start_date,
        "end_date": payload.end_date,
    }
    results = search_similar_documents(
        db,
        payload.repo_id,
        payload.query,
        source_type=payload.source_type,
        limit=payload.limit,
        metadata_filters={key: value for key, value in filters.items() if value},
    )
    return SearchResponse(results=results)
