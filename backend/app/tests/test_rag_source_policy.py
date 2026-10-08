import os
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import (
    Base,
    ChatMessage,
    ChatSession,
    Conversation,
    ConversationMemory,
    Document,
    EvidenceItem,
    Issue,
    KnowledgeBaseConfig,
    PRFile,
    PRReviewComment,
    PullRequest,
    Repository,
    WorkflowRun,
)
from app.services.chat_memory import ContextAssembler, EvidenceStore, SessionSealer
from app.services.rag.indexing import CI_LOG_CHUNK_CHARS, index_repository_documents
from app.services.rag.knowledge_base import get_or_create_config, retrieve_knowledge
from app.services.rag.qa import rag_status, reindex_documents
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.source_policy import retrieval_mode, should_vectorize
from app.services.rag.vector_store import DocumentPayload, MilvusUnavailableError, add_documents, reset_milvus_store_cache


def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _repo(db):
    repo = Repository(owner="local", name="policy", full_name="local/policy")
    db.add(repo)
    db.flush()
    return repo


def _vector() -> list[float]:
    return [1.0, *([0.0] * (settings.embedding_dimensions - 1))]


class _FakeMilvusStore:
    def __init__(self) -> None:
        self.collection_name = "test_collection"
        self.upserts: list[tuple[list[Document], list[list[float]]]] = []

    def upsert_documents(self, documents, vectors) -> None:
        self.upserts.append((list(documents), list(vectors)))

    def delete_documents(self, **_kwargs) -> None:
        return None


def _install_fake_milvus(monkeypatch) -> _FakeMilvusStore:
    store = _FakeMilvusStore()
    monkeypatch.setattr("app.services.rag.vector_store._milvus_store", lambda: store)
    return store


@pytest.mark.parametrize(
    ("source_type", "expected"),
    [
        ("issue", "rag"),
        ("pull_request", "rag"),
        ("workflow_run", "rag"),
        ("project_doc", "rag"),
        ("knowledge_file", "rag"),
        ("memory_note", "rag"),
        ("code_file", "direct"),
        ("project_overview", "direct"),
        ("project_manifest", "direct"),
        ("team_member", "direct"),
        ("weekly_report", "direct"),
        ("chat_session", "direct"),
        ("conversation_memory", "direct"),
        ("thread_memory", "direct"),
        ("future_unknown_type", "direct"),
    ],
)
def test_source_policy_separates_rag_and_direct_data(source_type, expected) -> None:
    assert retrieval_mode(source_type) == expected
    assert should_vectorize(source_type) is (expected == "rag")


def test_knowledge_base_defaults_enable_high_score_threshold() -> None:
    db = _db_session()
    repo = _repo(db)

    config = get_or_create_config(db, repo.id)

    assert config.score_threshold_enabled is True
    assert config.score_threshold == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_default_score_threshold_filters_results_below_point_eight(monkeypatch) -> None:
    db = _db_session()
    repo = _repo(db)
    monkeypatch.setattr(
        "app.services.rag.knowledge_base.search_similar_documents",
        lambda *_args, **_kwargs: [
            {"id": "high-confidence", "score": 0.81},
            {"id": "low-confidence", "score": 0.79},
        ],
    )

    items, trace, _duration_ms = await retrieve_knowledge(
        db,
        repo.id,
        "authentication token",
        retrieval_method="fulltext",
        rerank_enabled=False,
    )

    assert [item["id"] for item in items] == ["high-confidence"]
    assert trace["score_threshold_enabled"] is True
    assert trace["score_threshold"] == pytest.approx(0.8)


def test_add_documents_only_embeds_rag_sources(monkeypatch) -> None:
    store = _install_fake_milvus(monkeypatch)
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    db = _db_session()
    repo = _repo(db)

    docs = add_documents(
        db,
        repo.id,
        [
            DocumentPayload(source_type="knowledge_file", title="Runbook", content="Rotate the token."),
            DocumentPayload(source_type="team_member", title="Alice", content="Backend engineer"),
            DocumentPayload(source_type="chat_session", title="Session", content="Raw transcript"),
        ],
    )

    by_type = {doc.source_type: doc for doc in docs}
    assert by_type["knowledge_file"].meta["retrieval_mode"] == "rag"
    assert by_type["team_member"].meta["retrieval_mode"] == "direct"
    assert not hasattr(by_type["knowledge_file"], "embedding")
    assert len(store.upserts) == 1
    assert [doc.source_type for doc in store.upserts[0][0]] == ["knowledge_file"]
    assert store.upserts[0][1] == [_vector()]


def test_add_documents_rejects_source_code_documents(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)

    with pytest.raises(ValueError, match="工作区实时搜索"):
        add_documents(
            db,
            repo.id,
            [DocumentPayload(source_type="code_file", title="auth.py", content="def rotate_token(): pass")],
        )


def test_add_documents_rejects_partial_embedding_batches(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    monkeypatch.setattr("app.services.rag.vector_store.embed_texts", lambda _texts, _dimensions: [])
    db = _db_session()
    repo = _repo(db)

    with pytest.raises(ValueError, match="embedding count"):
        add_documents(
            db,
            repo.id,
            [DocumentPayload(source_type="knowledge_file", title="Runbook", content="Rotate the token.")],
        )

    assert db.query(Document).count() == 0


def test_general_rag_search_excludes_code_and_direct_records(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    add_documents(
        db,
        repo.id,
        [
            DocumentPayload(
                source_type="knowledge_file",
                title="Authentication runbook",
                content="Rotate the access token after authentication failure.",
            ),
            DocumentPayload(
                source_type="team_member",
                title="Token owner",
                content="Maintains authentication token services.",
            ),
        ],
        vectors=[_vector(), _vector()],
        sync_vector_store=False,
    )
    monkeypatch.setattr("app.services.rag.retrieval.search_milvus_documents", lambda *_args, **_kwargs: [])

    general = search_similar_documents(
        db,
        repo.id,
        "authentication token",
        retrieval_method="fulltext",
        apply_rerank=False,
    )
    code = search_similar_documents(
        db,
        repo.id,
        "rotate_access_token",
        source_type="code_file",
        retrieval_method="hybrid",
        apply_rerank=False,
    )
    team = search_similar_documents(
        db,
        repo.id,
        "token",
        source_type="team_member",
        retrieval_method="fulltext",
        apply_rerank=False,
    )
    code_with_vector_method = search_similar_documents(
        db,
        repo.id,
        "rotate_access_token",
        source_type="code_file",
        retrieval_method="vector",
        apply_rerank=False,
    )
    mixed = search_similar_documents(
        db,
        repo.id,
        "rotate_access_token",
        source_types=["knowledge_file", "code_file"],
        retrieval_method="hybrid",
        limit=2,
        apply_rerank=False,
    )

    assert {item["source_type"] for item in general} == {"knowledge_file"}
    assert code == []
    assert code_with_vector_method == []
    assert {item["source_type"] for item in mixed} <= {"knowledge_file"}
    assert team == []


def test_vector_search_does_not_fall_back_to_database(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    db.add(
        Document(
            repo_id=repo.id,
            source_type="knowledge_file",
            source_id=uuid.uuid4(),
            title="Authentication runbook",
            content="Rotate the access token.",
            meta={"retrieval_mode": "rag"},
        )
    )
    db.commit()

    with pytest.raises(MilvusUnavailableError, match="disabled"):
        search_similar_documents(
            db,
            repo.id,
            "authentication token",
            retrieval_method="vector",
            apply_rerank=False,
        )


def test_hybrid_search_falls_back_to_fulltext_when_milvus_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    db.add(
        Document(
            repo_id=repo.id,
            source_type="knowledge_file",
            source_id=uuid.uuid4(),
            title="Authentication runbook",
            content="Rotate the access token after authentication failure.",
            meta={"retrieval_mode": "rag"},
        )
    )
    db.commit()

    results = search_similar_documents(
        db,
        repo.id,
        "authentication token",
        retrieval_method="hybrid",
        apply_rerank=False,
    )

    assert [item["title"] for item in results] == ["Authentication runbook"]


@pytest.mark.asyncio
async def test_code_file_retrieval_is_rejected_without_query_embedding(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    monkeypatch.setattr(
        "app.services.rag.knowledge_base.embed_knowledge_query",
        lambda *_args, **_kwargs: pytest.fail("code retrieval must not embed the query"),
    )
    db = _db_session()
    repo = _repo(db)
    items, trace, _duration_ms = await retrieve_knowledge(
        db,
        repo.id,
        "rotate_access_token",
        source_types=[" code_file ", "code_file"],
        rerank_enabled=False,
    )

    assert items == []
    assert trace["source_types"] == ["code_file"]


def test_reindex_uses_the_global_embedding_contract(monkeypatch) -> None:
    store = _install_fake_milvus(monkeypatch)
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda *_args, **_kwargs: pytest.fail("repository refresh must not pre-embed with global settings"),
    )
    calls: list[tuple[str, str, int, int]] = []

    def fake_embed(texts, *, provider, model, dimensions):
        calls.append((provider, model, dimensions, len(texts)))
        return [_vector() for _text in texts]

    monkeypatch.setattr("app.services.rag.qa.embed_knowledge_texts", fake_embed)
    db = _db_session()
    repo = _repo(db)
    db.add(
        KnowledgeBaseConfig(
            repo_id=repo.id,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
        )
    )
    db.add(Issue(repo_id=repo.id, number=3, title="Token expiry", body="Refresh once.", state="open"))
    db.commit()

    indexed = reindex_documents(db, repo.id)

    assert indexed == 1
    assert calls == [(settings.embedding_provider, settings.embedding_model, settings.embedding_dimensions, 1)]
    assert len(store.upserts) == 1


def test_reindex_rejects_partial_embedding_batches(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    db.add(
        Document(
            repo_id=repo.id,
            source_type="memory_note",
            source_id=uuid.uuid4(),
            title="Decision",
            content="Use short-lived tokens.",
            meta={},
        )
    )
    db.commit()
    monkeypatch.setattr("app.services.rag.qa.embed_texts", lambda _texts, _dimensions: [])

    with pytest.raises(ValueError, match="embedding count"):
        reindex_documents(db, repo.id)


def test_reindex_clears_legacy_metadata_and_status_only_counts_rag(monkeypatch) -> None:
    _install_fake_milvus(monkeypatch)
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    monkeypatch.setattr(
        "app.services.rag.qa.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    db = _db_session()
    repo = _repo(db)
    docs = add_documents(
        db,
        repo.id,
        [
            DocumentPayload(source_type="memory_note", title="Decision", content="Use short-lived tokens."),
            DocumentPayload(source_type="weekly_report", title="Weekly", content="Two pull requests merged."),
        ],
    )
    weekly = next(doc for doc in docs if doc.source_type == "weekly_report")
    weekly.meta = {
        **(weekly.meta or {}),
        "embedding_provider": "legacy",
        "embedding_model": "legacy",
        "embedding_dimensions": 2,
    }
    db.add_all(
        [
            Document(
                repo_id=repo.id,
                source_type="chat_session",
                source_id=uuid.uuid4(),
                title="Legacy raw chat",
                content="A private raw conversation transcript.",
                meta={},
            ),
            Document(
                repo_id=repo.id,
                source_type="pull_request",
                source_id=uuid.uuid4(),
                title="Legacy PR",
                content="SENSITIVE_PATCH_FROM_OLD_INDEX",
                meta={},
            ),
            Document(
                repo_id=repo.id,
                source_type="workflow_run",
                source_id=uuid.uuid4(),
                title="Legacy passing run",
                content="all tests passed",
                meta={"conclusion": "success"},
            ),
            Document(
                repo_id=repo.id,
                source_type="code_file",
                source_id=uuid.uuid4(),
                title="Legacy source chunk",
                content="def stale_handler(): pass",
                meta={"retrieval_mode": "lexical", "path": "service.py"},
            ),
        ]
    )
    db.commit()

    indexed = reindex_documents(db, repo.id)
    db.refresh(weekly)
    status = rag_status(db, repo.id)

    assert indexed == 1
    assert not hasattr(weekly, "embedding")
    assert weekly.meta["retrieval_mode"] == "direct"
    assert "embedding_provider" not in weekly.meta
    assert status["document_count"] == 1
    assert status["source_types"] == {"memory_note": 1}
    assert db.query(Document).filter(Document.repo_id == repo.id).count() == 2
    assert db.query(Document).filter(Document.title == "Legacy raw chat").count() == 0
    assert db.query(Document).filter(Document.content == "SENSITIVE_PATCH_FROM_OLD_INDEX").count() == 0
    assert db.query(Document).filter(Document.title == "Legacy passing run").count() == 0
    assert db.query(Document).filter(Document.title == "Legacy source chunk").count() == 0


def test_repository_index_only_embeds_unstructured_historical_fields(monkeypatch) -> None:
    _install_fake_milvus(monkeypatch)
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    db = _db_session()
    repo = _repo(db)
    db.add(
        Issue(
            repo_id=repo.id,
            number=7,
            title="Authentication occasionally fails",
            body="Users see an expired-session message.",
            state="open",
            labels=["backend"],
            author="alice",
            assignees=["bob"],
        )
    )
    db.add(
        PullRequest(
            repo_id=repo.id,
            number=9,
            title="Retry token refresh",
            body="Retry once after refresh.",
            state="open",
            author="carol",
            base_branch="main",
            head_branch="fix/retry",
            files=[
                PRFile(
                    filename="auth.py",
                    status="modified",
                    additions=4,
                    deletions=1,
                    patch="SENSITIVE_PATCH_SHOULD_NOT_BE_EMBEDDED",
                )
            ],
            review_comments=[
                PRReviewComment(
                    body="Keep the retry idempotent.",
                    path="auth.py",
                    line=42,
                    author="reviewer",
                )
            ],
        )
    )
    db.add_all(
        [
            WorkflowRun(
                repo_id=repo.id,
                name="passing build",
                status="completed",
                conclusion="success",
                logs_text="all tests passed",
            ),
            WorkflowRun(
                repo_id=repo.id,
                name="failing build",
                status="completed",
                conclusion="failure",
                logs_text="test_refresh_retry failed with 401",
            ),
        ]
    )
    db.flush()

    indexed = index_repository_documents(db, repo.id)
    docs = db.query(Document).filter(Document.repo_id == repo.id).all()
    by_type = {doc.source_type: doc for doc in docs}

    assert indexed == 3
    assert "state:" not in by_type["issue"].content
    assert "labels:" not in by_type["issue"].content
    assert by_type["issue"].meta["state"] == "open"
    assert "SENSITIVE_PATCH_SHOULD_NOT_BE_EMBEDDED" not in by_type["pull_request"].content
    assert "Keep the retry idempotent." in by_type["pull_request"].content
    assert by_type["pull_request"].meta["files"] == ["auth.py"]
    workflow_docs = [doc for doc in docs if doc.source_type == "workflow_run"]
    assert len(workflow_docs) == 1
    assert "test_refresh_retry failed with 401" in workflow_docs[0].content


def test_repository_sync_respects_global_embedding_contract(monkeypatch) -> None:
    store = _install_fake_milvus(monkeypatch)
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda *_args, **_kwargs: pytest.fail("sync must not use the global embedding provider"),
    )
    calls: list[tuple[str, str, int]] = []

    def fake_embed(texts, *, provider, model, dimensions):
        calls.append((provider, model, dimensions))
        return [_vector() for _text in texts]

    monkeypatch.setattr("app.services.rag.indexing.embed_knowledge_texts", fake_embed)
    db = _db_session()
    repo = _repo(db)
    db.add(
        KnowledgeBaseConfig(
            repo_id=repo.id,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
        )
    )
    db.add(Issue(repo_id=repo.id, number=5, title="Token expiry", body="Refresh once.", state="open"))
    db.flush()

    indexed = index_repository_documents(db, repo.id)
    doc = db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "issue").one()

    assert indexed == 1
    assert calls == [(settings.embedding_provider, settings.embedding_model, settings.embedding_dimensions)]
    assert not hasattr(doc, "embedding")
    assert doc.meta["embedding_provider"] == settings.embedding_provider
    assert len(store.upserts) == 1


def test_failed_ci_logs_are_redacted_and_bounded_before_embedding(monkeypatch) -> None:
    _install_fake_milvus(monkeypatch)
    monkeypatch.setattr(
        "app.services.rag.vector_store.embed_texts",
        lambda texts, dimensions: [_vector() for _text in texts],
    )
    db = _db_session()
    repo = _repo(db)
    secret = "ghp_" + "a" * 32
    db.add(
        WorkflowRun(
            repo_id=repo.id,
            name="failing build",
            status="completed",
            conclusion="failure",
            logs_text=f"Authorization: Bearer {secret}\nTOKEN={secret}\n" + ("failure context\n" * 1200),
        )
    )
    db.flush()

    index_repository_documents(db, repo.id)
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo.id, Document.source_type == "workflow_run")
        .order_by(Document.title.asc())
        .all()
    )

    assert len(docs) > 1
    assert all(secret not in doc.content for doc in docs)
    assert "[REDACTED" in "\n".join(doc.content for doc in docs)
    assert all(len(doc.content) <= CI_LOG_CHUNK_CHARS + len("failing build") + 1 for doc in docs)
    assert all(doc.meta["secrets_redacted"] is True for doc in docs)


def test_evidence_store_redacts_failed_ci_logs_before_chat_context() -> None:
    db = _db_session()
    repo = _repo(db)
    secret = "ghp_" + "a" * 32
    db.add(
        WorkflowRun(
            repo_id=repo.id,
            name="failing build",
            status="completed",
            conclusion="failure",
            logs_text=f"MY_API_KEY={secret}\nERROR authentication failed",
        )
    )
    db.flush()

    EvidenceStore(db).rebuild_repo(repo)
    evidence = (
        db.query(EvidenceItem)
        .filter(EvidenceItem.repo_id == repo.id, EvidenceItem.source_type == "workflow_run")
        .one()
    )

    assert secret not in evidence.content
    assert "MY_API_KEY=[REDACTED]" in evidence.content
    assert evidence.meta["secrets_redacted"] is True


def test_context_assembler_prefetches_rag_evidence_before_chat_agent(monkeypatch) -> None:
    db = _db_session()
    repo = _repo(db)
    queries: list[str] = []

    def fake_search(self, repo_id, conversation_id, query, limit=5):
        queries.append(query)
        return [
            {
                "title": "Authentication retry policy",
                "source_type": "project_doc",
                "source_id": "doc-1",
                "snippet": "Refresh retries must use an idempotency key.",
                "score": 0.91,
                "metadata": {},
                "retrieval": {"sources": ["vector", "keyword"]},
            }
        ]

    monkeypatch.setattr(EvidenceStore, "search", fake_search)
    assembled = ContextAssembler(db).assemble(
        repo,
        None,
        "How should refresh retries work?",
    )

    assert queries == ["How should refresh retries work?"]
    assert assembled.evidence[0]["source_id"] == "doc-1"
    assert "[Related Evidence]" in assembled.system_context
    assert "Refresh retries must use an idempotency key." in assembled.system_context


def test_context_assembler_reports_exact_recent_history_coverage() -> None:
    db = _db_session()
    repo = _repo(db)
    conversation = Conversation(repo_id=repo.id, title="Recall recent messages")
    db.add(conversation)
    db.flush()
    db.add_all(
        [
            ChatMessage(repo_id=repo.id, conversation_id=conversation.id, role="user", content="hi"),
            ChatMessage(
                repo_id=repo.id,
                conversation_id=conversation.id,
                role="assistant",
                content="你好！有什么可以帮你？",
            ),
            ChatMessage(
                repo_id=repo.id,
                conversation_id=conversation.id,
                role="user",
                content="我刚刚说了什么？",
            ),
        ]
    )
    db.flush()

    assembled = ContextAssembler(db).assemble(repo, conversation, "我刚刚说了什么？")
    coverage = assembled.compression_stats["history_coverage"]

    assert coverage["complete_and_exact"] is True
    assert coverage["prompt_messages"] == 3
    assert coverage["exact_suffix_messages"] == 3
    assert coverage["omitted_messages"] == 0
    assert coverage["has_compaction_boundary"] is False


@pytest.mark.asyncio
async def test_session_sealing_keeps_raw_memory_out_of_project_rag() -> None:
    class FakeLLM:
        async def chat_text(self, system_prompt, user_prompt):
            return "The team chose short-lived access tokens."

        async def chat_json(self, system_prompt, payload):
            return {
                "summary": "Use short-lived access tokens.",
                "facts": ["Tokens expire after fifteen minutes."],
                "decisions": ["Refresh once before asking the user to sign in."],
                "open_questions": [],
                "tasks": [],
                "user_preferences": [],
                "repo_context": [],
                "citations": [],
            }

    db = _db_session()
    repo = _repo(db)
    conversation = Conversation(repo_id=repo.id, title="Authentication")
    db.add(conversation)
    db.flush()
    user_message = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="user",
        content="How should token refresh work?",
    )
    assistant_message = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="assistant",
        content="Refresh once, then require sign-in.",
    )
    db.add_all([user_message, assistant_message])
    db.flush()

    await SessionSealer(db, llm=FakeLLM()).seal(
        repo=repo,
        conversation=conversation,
        user_message=user_message,
        assistant_message=assistant_message,
        route="chat",
        tool_calls=[],
    )
    db.commit()

    assert db.query(ChatSession).filter(ChatSession.conversation_id == conversation.id).count() == 1
    assert db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).count() == 1
    assert (
        db.query(Document)
        .filter(Document.source_type.in_(["chat_session", "conversation_memory", "thread_memory"]))
        .count()
        == 0
    )
