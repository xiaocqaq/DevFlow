from typing import Any

from pydantic import BaseModel, Field
from uuid import UUID


class RagEvalCaseRequest(BaseModel):
    id: str = ""
    question: str
    reference: str
    expected_source_ids: list[str] = Field(default_factory=list)
    source_type: str | None = None
    required_tools: list[str] = Field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    kind: str = "positive"
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    answer_regex: str | None = None
    source_file: str | None = None


class EvalRunRequest(BaseModel):
    name: str = "basic_eval"
    repo_id: UUID | None = None
    eval_types: list[str] = Field(default_factory=lambda: ["issue_classification", "rag", "pr_summary", "tool_calls"])
    run_judge: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    suite_id: str | None = None
    cases: list[RagEvalCaseRequest] | None = None


class EvalIssueDraftRequest(BaseModel):
    title: str | None = None


class EvalRunResponse(BaseModel):
    eval_id: str
    status: str
    result: dict
