from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    repo_id: UUID
    conversation_id: UUID | None = None
    message: str
    context: dict = Field(default_factory=dict)


class ChatPlanExecuteRequest(ChatRequest):
    workflow_spec: dict


class ToolCallTrace(BaseModel):
    tool_name: str
    arguments: dict
    status: str
    error: str | None = None
    error_kind: str | None = None
    retryable: bool | None = None
    attempts: int | None = None
    tool_use_id: str | None = None
    skill_name: str | None = None
    skill_title: str | None = None
    skill_version: str | None = None
    skill_steps: list[str] = Field(default_factory=list)
    skill_activation_mode: str | None = None
    skill_activation_reason: str | None = None
    skill_instruction_digest: str | None = None


class ChatResponse(BaseModel):
    run_id: UUID | str
    message_id: UUID | str
    route: str
    tool_calls: list[ToolCallTrace]
    agent_events: list[dict] = []
    answer: str
    citations: list[dict] = []
    completion: dict = {}
    compression_stats: dict = {}
    progressive_compaction: dict = {}
    model_usage: list[dict] = []
    context_diagnostics: dict = {}


class ChatHistoryMessage(BaseModel):
    id: UUID | str
    conversation_id: UUID | str | None = None
    role: str
    content: str
    route: str | None = None
    tool_calls: list[ToolCallTrace] = []
    agent_events: list[dict] = []
    compression_stats: dict = {}
    progressive_compaction: dict = {}
    model_usage: list[dict] = []
    context_diagnostics: dict = {}
    created_at: str | None = None


class PromptRecommendationRequest(BaseModel):
    repo_id: UUID
    exclude: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=4, le=5)


class PromptRecommendationResponse(BaseModel):
    agent: str
    suggestions: list[str]
