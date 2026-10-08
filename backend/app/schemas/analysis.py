from typing import Literal

from pydantic import BaseModel, Field


class DuplicateCandidate(BaseModel):
    issue_id: str
    title: str
    score: float = Field(ge=0, le=1)


class IssueEvidence(BaseModel):
    source_type: str
    title: str
    snippet: str
    reference: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class IssueDrafts(BaseModel):
    clarification_comment: str | None = None
    task_breakdown: str | None = None


class IssueAnalysisOutput(BaseModel):
    summary: str
    conclusion: Literal["进入开发", "等待澄清", "关闭", "拆分", "合并到已有 Issue"]
    conclusion_reason: str
    category: Literal["Bug", "Feature", "Question", "Documentation", "Refactor", "Test", "Ops"]
    priority: Literal["P0", "P1", "P2", "P3"]
    complexity: Literal["S", "M", "L", "XL"]
    suggested_owner: str
    owner_reason: str
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)
    evidence: list[IssueEvidence] = Field(default_factory=list)
    checklist: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    drafts: IssueDrafts = Field(default_factory=IssueDrafts)
    confidence: float = Field(ge=0, le=1)


class PRReviewFinding(BaseModel):
    severity: Literal["P1", "P2", "P3"]
    title: str
    evidence: str
    required_action: str
    blocking: bool = False


class PRReviewOutput(BaseModel):
    summary: str
    plan: list[str] = []
    executed_steps: list[str] = []
    merge_recommendation: Literal["建议合入", "修改后合入", "暂缓", "拒绝"] = "暂缓"
    recommendation_reason: str = ""
    review_findings: list[PRReviewFinding] = Field(default_factory=list)
    key_changes: list[str] = []
    risk_points: list[str] = []
    blocking_issues: list[str] = []
    review_checklist: list[str] = []
    test_suggestions: list[str] = []
    files_need_attention: list[str] = []
    review_comments: list[str] = []
    confidence: float = Field(ge=0, le=1)


class CIDebugOutput(BaseModel):
    failure_summary: str
    failure_type: Literal["test", "build", "lint", "dependency", "permission", "environment", "unknown"]
    plan: list[str] = []
    executed_steps: list[str] = []
    first_error: str | None = None
    root_cause: str = ""
    possible_causes: list[str] = []
    fix_steps: list[str] = []
    debug_steps: list[str] = []
    related_files: list[str] = []
    is_merge_blocking: bool = True
    blocking_reason: str = ""
    confidence: float = Field(ge=0, le=1)


class SafetyOutput(BaseModel):
    requires_confirmation: bool
    risk_level: Literal["low", "medium", "high"]
    reason: str
    draft: str | None = None
