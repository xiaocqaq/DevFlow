from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


WorkflowTaskStatus = Literal["pending", "running", "success", "skipped", "error"]
WorkflowFindingSeverity = Literal["info", "warning", "blocker"]


def _workflow_id() -> str:
    return f"workflow_{uuid.uuid4().hex[:12]}"


class WorkflowClaim(BaseModel):
    target_type: str
    target_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    write_intent: bool = False
    reason: str = ""


class WorkflowTask(BaseModel):
    id: str
    agent_name: str
    task_type: str
    objective: str
    dependencies: list[str] = Field(default_factory=list)
    claim: WorkflowClaim
    input_hint: dict[str, Any] = Field(default_factory=dict)
    critical: bool = False


class WorkflowSpec(BaseModel):
    workflow_id: str = Field(default_factory=_workflow_id)
    goal: str
    trigger_message: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    tasks: list[WorkflowTask] = Field(default_factory=list)
    max_parallel_tasks: int = Field(default=3, ge=1, le=8)
    requires_human_confirmation: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowTaskResult(BaseModel):
    task_id: str
    agent_name: str
    task_type: str
    status: WorkflowTaskStatus
    summary: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = 0
    confidence: float | None = Field(default=None, ge=0, le=1)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowFinding(BaseModel):
    finding_type: str
    severity: WorkflowFindingSeverity
    message: str
    task_ids: list[str] = Field(default_factory=list)
    recommendation: str = ""


class WorkflowObservation(BaseModel):
    observer_agent: str = "observer_agent"
    findings: list[WorkflowFinding] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0, le=1)
    human_review_required: bool = False
    summary: str = ""


class WorkflowReport(BaseModel):
    spec: WorkflowSpec
    task_results: list[WorkflowTaskResult]
    observation: WorkflowObservation
    final_answer: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    agent_events: list[dict[str, Any]] = Field(default_factory=list)
