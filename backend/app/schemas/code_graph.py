from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CodeSymbolResponse(BaseModel):
    id: UUID
    repo_id: UUID
    path: str
    name: str
    kind: str
    language: str
    start_line: int | None = None
    end_line: int | None = None
    branch: str | None = None
    commit_sha: str | None = None
    pr_id: UUID | None = None
    pr_number: int | None = None
    metadata: dict = Field(default_factory=dict, alias="meta")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


class CodeGraphSearchResponse(BaseModel):
    query: str
    symbols: list[CodeSymbolResponse]
    relations: list[dict]
    documents: list[dict]
