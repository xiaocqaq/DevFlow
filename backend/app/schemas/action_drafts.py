from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ActionDraftCreateRequest(BaseModel):
    repo_id: UUID
    draft_type: str = Field(default="issue_comment")
    target_type: str = Field(default="issue")
    target_id: UUID | None = None
    title: str = ""
    content: str
    risk_level: str = "medium"
    metadata: dict = Field(default_factory=dict)


class ActionDraftResponse(BaseModel):
    id: UUID
    repo_id: UUID
    draft_type: str
    target_type: str
    target_id: UUID | None = None
    title: str
    content: str
    risk_level: str
    status: str
    execution_result: dict | None = None
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict, alias="meta")
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True, "populate_by_name": True}


class ActionDraftConfirmResponse(BaseModel):
    draft: ActionDraftResponse
    audit_id: UUID


class AuditLogResponse(BaseModel):
    id: UUID
    user_id: UUID | None = None
    repo_id: UUID | None = None
    action: str
    target_type: str | None = None
    target_id: str | None = None
    status: str
    request_json: dict
    result_json: dict
    created_at: datetime

    model_config = {"from_attributes": True}
