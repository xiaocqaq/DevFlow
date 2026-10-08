from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class KnowledgeBaseConfigUpdate(BaseModel):
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = Field(default=None, ge=64, le=4096)
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None
    rerank_enabled: bool | None = None
    rerank_provider: str | None = None
    rerank_model: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    score_threshold_enabled: bool | None = None
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    chunk_size: int | None = Field(default=None, ge=200, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


class KnowledgeBaseResponse(BaseModel):
    id: UUID
    name: str
    full_name: str
    description: str = ""
    provider: str = "github"
    document_count: int = 0
    chunk_count: int = 0
    character_count: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class KnowledgeDocumentResponse(BaseModel):
    id: UUID
    repo_id: UUID
    name: str
    source_type: str
    status: str
    content_type: str | None = None
    file_size: int = 0
    character_count: int = 0
    chunk_count: int = 0
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None


class KnowledgeChunkResponse(BaseModel):
    id: UUID
    position: int
    title: str
    content: str
    character_count: int
    enabled: bool = True


class RAGQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)
    source_types: list[str] = Field(default_factory=list, max_length=20)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class RAGSourceResponse(BaseModel):
    citation: int
    document_id: str
    source_id: str
    source_type: str
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)


class RAGSearchResponse(BaseModel):
    query: str
    results: list[RAGSourceResponse]
    result_count: int
    took_ms: int
    retrieval_config: dict[str, Any] = Field(default_factory=dict)


class RetrievalTestRequest(BaseModel):
    query: str = Field(min_length=1, max_length=250)
    top_k: int | None = Field(default=None, ge=1, le=20)
    retrieval_method: Literal["vector", "fulltext", "hybrid"] | None = None
    rerank_enabled: bool | None = None
    score_threshold_enabled: bool | None = None
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalTestResponse(BaseModel):
    id: UUID
    query: str
    results: list[RAGSourceResponse]
    result_count: int
    took_ms: int
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class RetrievalTestHistoryResponse(BaseModel):
    id: UUID
    query: str
    result_count: int
    took_ms: int
    created_at: datetime | None = None


class ModelCatalogResponse(BaseModel):
    embedding_models: list[dict[str, Any]]
    rerank_models: list[dict[str, Any]]
    deepseek: dict[str, Any]


class RAGAnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSourceResponse]
    retrieval_count: int
    generation_mode: Literal["llm", "extractive", "no_evidence"]
    model: str | None = None
    answer_gate: dict[str, Any] = Field(default_factory=dict)
    retrieval_config: dict[str, Any] = Field(default_factory=dict)
    took_ms: int


class RAGStatusResponse(BaseModel):
    repo_id: UUID
    document_count: int
    source_count: int
    source_types: dict[str, int] = Field(default_factory=dict)
    embedding: dict[str, Any] = Field(default_factory=dict)
    vector_store: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    supported_files: list[str] = Field(default_factory=list)


class RAGReindexResponse(BaseModel):
    repo_id: UUID
    indexed_documents: int
    embedding_provider: str
    embedding_model: str
