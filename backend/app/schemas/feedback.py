from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


FeedbackRating = Literal["helpful", "unhelpful"]
FeedbackReason = Literal[
    "inaccurate",
    "not_relevant",
    "missing_context",
    "unreliable_citation",
    "tool_error",
    "other",
]
ReviewStatus = Literal["open", "in_review", "resolved", "dismissed"]


class ChatFeedbackCreate(BaseModel):
    repo_id: UUID
    conversation_id: UUID
    assistant_message_id: UUID
    rating: FeedbackRating
    reason: FeedbackReason | None = None
    comment: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_reason(self) -> "ChatFeedbackCreate":
        if self.rating == "helpful" and self.reason is not None:
            raise ValueError("有帮助反馈不需要填写负反馈原因")
        return self


class ChatFeedbackReview(BaseModel):
    review_status: ReviewStatus
    review_note: str | None = Field(default=None, max_length=2000)


class ChatFeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    repo_id: UUID
    conversation_id: UUID
    assistant_message_id: UUID
    run_id: UUID | None = None
    rating: FeedbackRating
    reason: FeedbackReason | None = None
    comment: str | None = None
    review_status: ReviewStatus
    review_note: str | None = None
    notification_status: str
    notification_error: str | None = None
    notified_at: datetime | None = None
    reviewed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
