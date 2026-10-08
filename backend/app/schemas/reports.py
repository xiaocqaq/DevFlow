from datetime import date
from uuid import UUID

from pydantic import BaseModel


class WeeklyReportRequest(BaseModel):
    repo_id: UUID
    start_date: date
    end_date: date


class WeeklyReportResponse(BaseModel):
    report_markdown: str
