import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Document, Repository
from app.services.rag.answer_gate import evaluate_answer_gate
from app.services.rag.knowledge_base import retrieve_knowledge
from app.services.rag.retrieval import (
    _merge_candidates,
    expand_parent_context,
    search_similar_documents,
)


def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _repo(db):
    repo = Repository(owner="local", name="retrieval", full_name=f"local/retrieval-{uuid.uuid4()}")
    db.add(repo)
    db.flush()
    return repo


def _document(db, repo, *, title, content, source_id=None, metadata=None):
    item = Document(
        repo_id=repo.id,
        source_type="knowledge_file",
        source_id=source_id or uuid.uuid4(),
        title=title,
        content=content,
        meta=metadata or {},
    )
    db.add(item)
    db.flush()
    return item


def test_bm25_prefers_rare_error_name_and_exposes_trace() -> None:
    db = _db_session()
    repo = _repo(db)
    _document(db, repo, title="General retry notes", content="failure retry token failure retry token")
    rare = _document(
        db,
        repo,
        title="JWT failure",
        content="ExpiredSignatureError is raised when the access token has expired.",
        metadata={"path": "backend/app/auth/token.py"},
    )
    db.commit()

    results = search_similar_documents(
        db,
        repo.id,
        "ExpiredSignatureError",
        retrieval_method="fulltext",
        apply_rerank=False,
    )

    assert results[0]["id"] == str(rare.id)
    assert results[0]["bm25_score"] > 0
    assert results[0]["keyword_rank"] == 1
    assert results[0]["retrieval"]["keyword_backend"] == "bm25"
    assert "expiredsignatureerror" in results[0]["retrieval"]["bm25_matched_terms"]


def test_metadata_state_filter_is_applied_before_bm25_scoring() -> None:
    db = _db_session()
    repo = _repo(db)
    open_doc = _document(
        db,
        repo,
        title="Open authentication issue",
        content="authentication token failure",
        metadata={"state": "open"},
    )
    _document(
        db,
        repo,
        title="Closed authentication issue",
        content="authentication token failure",
        metadata={"state": "closed"},
    )
    db.commit()

    results = search_similar_documents(
        db,
        repo.id,
        "authentication token",
        retrieval_method="fulltext",
        metadata_filters={"state": "open"},
        apply_rerank=False,
    )

    assert [item["id"] for item in results] == [str(open_doc.id)]


def test_weighted_fusion_uses_project_formula_and_dual_recall_bonus() -> None:
    vector = [
        {
            "id": "shared",
            "score": 1.0,
            "vector_score": 1.0,
            "metadata": {"chunk_id": "chunk-shared"},
            "retrieval": {"sources": ["vector"], "vector_score": 1.0, "vector_rank": 1},
        },
        {
            "id": "vector-only",
            "score": 0.8,
            "vector_score": 0.8,
            "metadata": {"chunk_id": "chunk-vector"},
            "retrieval": {"sources": ["vector"], "vector_score": 0.8, "vector_rank": 2},
        },
    ]
    keyword = [
        {
            "id": "shared",
            "score": 1.0,
            "keyword_score": 1.0,
            "bm25_score": 8.2,
            "metadata": {"chunk_id": "chunk-shared"},
            "retrieval": {"sources": ["keyword"], "keyword_score": 1.0, "keyword_rank": 1},
        }
    ]

    results = _merge_candidates(vector, keyword)

    assert results[0]["id"] == "shared"
    assert results[0]["combined_score"] == pytest.approx(1.0)
    assert results[0]["retrieval"]["fusion_method"] == "weighted"
    assert results[0]["retrieval"]["fusion_weights"] == {
        "vector": 0.62,
        "keyword": 0.30,
        "dual_recall_bonus": 0.08,
    }
    assert results[1]["combined_score"] == pytest.approx(0.496)


def test_parent_expansion_reads_only_same_parent_source_and_version() -> None:
    db = _db_session()
    repo = _repo(db)
    source_id = uuid.uuid4()
    rows = [
        _document(
            db,
            repo,
            title=f"auth.md#{index + 1}",
            content=content,
            source_id=source_id,
            metadata={
                "chunk_id": f"chunk-{index}",
                "parent_id": "parent-auth",
                "child_index": index,
                "branch": "main",
            },
        )
        for index, content in enumerate(["token setup", "refresh token failure", "retry once"])
    ]
    _document(
        db,
        repo,
        title="old branch",
        content="must not be expanded",
        source_id=source_id,
        metadata={"chunk_id": "old", "parent_id": "parent-auth", "child_index": 2, "branch": "legacy"},
    )
    db.commit()

    items, stage = expand_parent_context(
        db,
        repo.id,
        [
            {
                "id": str(rows[1].id),
                "source_id": str(source_id),
                "source_type": "knowledge_file",
                "metadata": rows[1].meta,
                "_raw_content": rows[1].content,
                "retrieval": {},
            }
        ],
    )

    assert items[0]["parent_expanded"] is True
    assert items[0]["_raw_content"] == "token setup\n\nrefresh token failure\n\nretry once"
    assert "must not be expanded" not in items[0]["_raw_content"]
    assert stage["additional_chunks"] == 2


@pytest.mark.parametrize(
    ("query", "items", "expected"),
    [
        ("token refresh", [], "insufficient_evidence"),
        (
            "这个怎么办",
            [
                {"id": "a", "source_id": "one", "score": 0.9, "snippet": "第一种处理方式"},
                {"id": "b", "source_id": "two", "score": 0.88, "snippet": "第二种处理方式"},
            ],
            "ask_clarification",
        ),
        (
            "是否允许部署",
            [
                {"id": "a", "source_id": "one", "score": 0.92, "snippet": "当前版本允许部署。"},
                {"id": "b", "source_id": "two", "score": 0.9, "snippet": "当前版本禁止部署。"},
            ],
            "conflict",
        ),
        (
            "token refresh policy",
            [{"id": "a", "source_id": "one", "score": 0.92, "snippet": "Refresh the token once."}],
            "answer",
        ),
    ],
)
def test_answer_gate_exposes_all_project_decisions(query, items, expected) -> None:
    gate = evaluate_answer_gate(query, items, score_threshold=0.8, threshold_enabled=True)

    assert gate["decision"] == expected
    assert "reason" in gate


@pytest.mark.asyncio
async def test_qwen_rerank_receives_raw_content_before_compression(monkeypatch) -> None:
    db = _db_session()
    repo = _repo(db)
    candidate = {
        "id": "auth",
        "source_id": str(uuid.uuid4()),
        "source_type": "knowledge_file",
        "title": "auth.md",
        "_raw_content": "irrelevant setup. " * 80 + "Refresh the token once after ExpiredSignatureError.",
        "snippet": "short pre-rerank snippet",
        "score": 0.91,
        "metadata": {},
        "retrieval": {"fusion_method": "weighted"},
    }

    def fake_search(*_args, **kwargs):
        kwargs["pipeline_trace"].update({"fusion_method": "weighted", "stages": []})
        return [candidate]

    async def fake_qwen(_query, items, _config, _limit):
        assert items[0]["_raw_content"].startswith("irrelevant setup")
        assert "compression" not in items[0]
        return [{**items[0], "score": 0.95, "rerank_score": 0.95, "retrieval": {**items[0]["retrieval"], "rerank_mode": "qwen_api"}}]

    monkeypatch.setattr("app.services.rag.knowledge_base.search_similar_documents", fake_search)
    monkeypatch.setattr("app.services.rag.knowledge_base._qwen_rerank", fake_qwen)

    items, trace, _duration = await retrieve_knowledge(
        db,
        repo.id,
        "ExpiredSignatureError token refresh",
        retrieval_method="fulltext",
        score_threshold_enabled=False,
    )

    assert items[0]["compression"]["mode"] == "query_aware_extractive"
    assert "ExpiredSignatureError" in items[0]["snippet"]
    stage_names = [stage["name"] for stage in trace["pipeline"]["stages"]]
    assert stage_names == ["rerank", "parent_expansion", "score_threshold", "context_compression", "answer_gate"]
    assert trace["answer_gate"]["decision"] == "answer"


@pytest.mark.asyncio
async def test_qwen_failure_is_traced_and_falls_back_to_heuristic(monkeypatch) -> None:
    db = _db_session()
    repo = _repo(db)

    def fake_search(*_args, **kwargs):
        kwargs["pipeline_trace"].update({"fusion_method": "weighted", "stages": []})
        return [
            {
                "id": "auth",
                "source_id": str(uuid.uuid4()),
                "source_type": "knowledge_file",
                "title": "auth middleware",
                "_raw_content": "auth middleware refreshes the token",
                "snippet": "auth middleware refreshes the token",
                "score": 0.9,
                "metadata": {},
                "retrieval": {"sources": ["vector", "keyword"]},
            }
        ]

    async def fail_qwen(*_args, **_kwargs):
        raise TimeoutError("rerank timeout")

    monkeypatch.setattr("app.services.rag.knowledge_base.search_similar_documents", fake_search)
    monkeypatch.setattr("app.services.rag.knowledge_base._qwen_rerank", fail_qwen)

    items, trace, _duration = await retrieve_knowledge(
        db,
        repo.id,
        "auth middleware",
        retrieval_method="fulltext",
        score_threshold_enabled=False,
    )

    rerank_stage = next(stage for stage in trace["pipeline"]["stages"] if stage["name"] == "rerank")
    assert items[0]["retrieval"]["rerank_mode"] == "heuristic"
    assert rerank_stage["fallback"] is True
    assert rerank_stage["fallback_reason"] == "TimeoutError"


@pytest.mark.asyncio
async def test_retrieval_trace_contains_every_selected_pipeline_stage() -> None:
    db = _db_session()
    repo = _repo(db)
    _document(
        db,
        repo,
        title="Authentication runbook",
        content="The auth middleware refreshes expired tokens once.",
        metadata={"path": "backend/app/auth/middleware.py"},
    )
    db.commit()

    items, trace, _duration = await retrieve_knowledge(
        db,
        repo.id,
        "auth middleware token",
        retrieval_method="fulltext",
        rerank_enabled=False,
        score_threshold_enabled=False,
    )

    stage_names = [stage["name"] for stage in trace["pipeline"]["stages"]]
    assert stage_names == [
        "vector_recall",
        "keyword_recall",
        "single_retriever_selection",
        "deduplication_and_parent_aggregation",
        "rerank",
        "parent_expansion",
        "score_threshold",
        "context_compression",
        "answer_gate",
    ]
    assert trace["fusion_method"] == "fulltext"
    assert trace["answer_gate"]["decision"] == "answer"
    assert items[0]["retrieval"]["keyword_backend"] == "bm25"
    assert items[0]["retrieval"]["selected_for_context"] is True
