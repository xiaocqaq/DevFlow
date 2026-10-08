from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.analysis import PRReviewOutput


class PRFileResponse(BaseModel):
    id: UUID
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None = None

    model_config = {"from_attributes": True}


class PRReviewCommentResponse(BaseModel):
    id: UUID
    body: str | None = None
    path: str | None = None
    line: int | None = None
    author: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class PullRequestResponse(BaseModel):
    id: UUID
    repo_id: UUID
    number: int
    title: str
    body: str | None = None
    state: str
    author: str | None = None
    base_branch: str | None = None
    head_branch: str | None = None
    merged_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    files: list[PRFileResponse] = []
    review_comments: list[PRReviewCommentResponse] = []

    model_config = {"from_attributes": True}


class PRAnalysisResponse(BaseModel):
    analysis_id: UUID | str
    result: PRReviewOutput
