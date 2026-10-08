from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.analysis import IssueAnalysisOutput


class IssueResponse(BaseModel):
    id: UUID
    repo_id: UUID
    number: int
    title: str
    body: str | None = None
    state: str
    labels: list[str] = []
    author: str | None = None
    assignees: list[str] = []
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class IssueAnalysisResponse(BaseModel):
    analysis_id: UUID | str
    result: IssueAnalysisOutput
