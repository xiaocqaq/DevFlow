from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ConversationResponse(BaseModel):
    id: UUID
    repo_id: UUID
    title: str
    status: str = "active"
    message_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}
