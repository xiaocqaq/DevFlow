from collections.abc import Iterable
from typing import Literal


RetrievalMode = Literal["rag", "direct"]

# RAG is reserved for reusable, unstructured project knowledge where semantic
# recall is useful. Structured live facts remain in their source tables/APIs.
RAG_SOURCE_TYPES = frozenset(
    {
        "issue",
        "pull_request",
        "workflow_run",
        "project_doc",
        "knowledge_file",
        "memory_note",
    }
)

DIRECT_SOURCE_TYPES = frozenset(
    {
        "project_overview",
        "project_manifest",
        "team_member",
        "weekly_report",
        "chat_session",
        "conversation_memory",
        "thread_memory",
    }
)


def retrieval_mode(source_type: str) -> RetrievalMode:
    if source_type in RAG_SOURCE_TYPES:
        return "rag"
    return "direct"


def normalize_source_types(source_types: Iterable[str | None] | None) -> list[str]:
    normalized = [str(source_type).strip() for source_type in (source_types or []) if source_type and str(source_type).strip()]
    return list(dict.fromkeys(normalized))


def should_vectorize(source_type: str) -> bool:
    return retrieval_mode(source_type) == "rag"


def is_searchable_source(source_type: str) -> bool:
    return retrieval_mode(source_type) == "rag"


def policy_metadata(source_type: str) -> dict[str, object]:
    mode = retrieval_mode(source_type)
    return {
        "retrieval_mode": mode,
        "rag_enabled": mode == "rag",
    }
