from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ProjectIndexStateResponse(BaseModel):
    repo_id: UUID
    status: str
    fingerprint: str = ""
    docs_indexed: int = 0
    docs_total: int = 0
    error_message: str | None = None
    summary_json: dict[str, Any] | None = None
    snoozed_until: datetime | None = None
    last_scan_at: datetime | None = None


class ProjectIndexScanResponse(BaseModel):
    repo_id: UUID
    started: bool
    status: str = "building"


class ProjectIndexSnoozeRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=30)


class ProjectIndexSnoozeResponse(BaseModel):
    repo_id: UUID
    snoozed_until: datetime
