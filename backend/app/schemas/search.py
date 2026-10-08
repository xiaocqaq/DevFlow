from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    repo_id: UUID
    query: str
    source_type: str | None = None
    label: str | None = None
    path: str | None = None
    module: str | None = None
    state: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    results: list[dict]
