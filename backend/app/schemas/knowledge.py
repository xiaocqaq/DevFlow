from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryNoteRequest(BaseModel):
    title: str = ""
    content: str = ""


class TeamMemberRequest(BaseModel):
    name: str
    role: str = ""
    strengths: str = ""
    tech_stack: str = Field(default="", alias="techStack")

    model_config = {"populate_by_name": True}


class TeamMemberResponse(BaseModel):
    id: UUID
    name: str
    role: str = ""
    strengths: str = ""
    tech_stack: str = Field(default="", alias="techStack")
    created_at: datetime | None = None

    model_config = {"populate_by_name": True}


class KnowledgeItemResponse(BaseModel):
    id: UUID
    document_ids: list[UUID]
    source_type: str
    title: str
    content_preview: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_count: int
    created_at: datetime | None = None


class KnowledgeUploadResponse(BaseModel):
    id: UUID
    filename: str
    chunks: int
    indexed: bool
    duplicate: bool = False


class KnowledgeGraphNodeResponse(BaseModel):
    id: str
    type: str
    title: str
    subtitle: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    highlighted: bool = False


class KnowledgeGraphEdgeResponse(BaseModel):
    id: str
    from_type: str
    from_id: str
    to_type: str
    to_id: str
    relation: str
    confidence: float = 1.0
    source: str = "explicit"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphResponse(BaseModel):
    repo_id: UUID
    center_type: str | None = None
    center_id: str | None = None
    depth: int = 1
    nodes: list[KnowledgeGraphNodeResponse]
    edges: list[KnowledgeGraphEdgeResponse]
    stats: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphRebuildResponse(BaseModel):
    repo_id: UUID
    edges: int
    nodes: int


class RecallEventResponse(BaseModel):
    id: UUID
    repo_id: UUID | None = None
    conversation_id: UUID | None = None
    tool_name: str
    query: str = ""
    scope: str = "project"
    mode: str = "hybrid"
    status: str = "success"
    result_count: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class MemorySearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
    result_count: int


class MemoryStatusResponse(BaseModel):
    repo_id: UUID
    index: dict[str, Any] = Field(default_factory=dict)
    documents: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    recall: dict[str, Any] = Field(default_factory=dict)
    candidates: dict[str, Any] = Field(default_factory=dict)
    vector: dict[str, Any] = Field(default_factory=dict)


class MemoryCandidateResponse(BaseModel):
    id: UUID
    repo_id: UUID | None = None
    conversation_id: UUID | None = None
    session_id: UUID | None = None
    kind: str
    title: str
    content: str
    status: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    reviewed_at: datetime | None = None


class MemoryCandidateActionResponse(BaseModel):
    id: UUID
    status: str
    document_id: UUID | None = None
