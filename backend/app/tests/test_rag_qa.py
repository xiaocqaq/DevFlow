import pytest

from app.core.config import settings
from app.services.rag.embeddings import deterministic_embedding
from app.services.rag.qa import answer_question, build_sources
from app.services.rag.retrieval import cosine_similarity


def test_deterministic_embedding_handles_chinese_without_spaces() -> None:
    question = deterministic_embedding("员工怎么申请年假？", 128)
    relevant = deterministic_embedding("员工年假审批流程和请假规定", 128)
    unrelated = deterministic_embedding("数据库连接池监控告警", 128)

    assert cosine_similarity(question, relevant) > cosine_similarity(question, unrelated)


def test_build_sources_hides_server_file_path() -> None:
    sources = build_sources(
        [
            {
                "id": "doc-1",
                "source_id": "source-1",
                "source_type": "knowledge_file",
                "title": "handbook.md#1",
                "snippet": "年假需要先在系统中提交审批。",
                "score": 0.91,
                "metadata": {"filename": "handbook.md", "stored_path": "C:/private/upload/handbook.md"},
            }
        ]
    )

    assert sources[0]["citation"] == 1
    assert sources[0]["metadata"]["filename"] == "handbook.md"
    assert "stored_path" not in sources[0]["metadata"]


@pytest.mark.asyncio
async def test_answer_question_uses_llm_and_returns_numbered_sources(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(
        "app.services.rag.qa.search_similar_documents",
        lambda *args, **kwargs: [
            {
                "id": "doc-1",
                "source_id": "source-1",
                "source_type": "knowledge_file",
                "title": "auth.md#1",
                "snippet": "访问令牌过期后，客户端使用刷新令牌重试一次。",
                "score": 0.95,
                "metadata": {"filename": "auth.md"},
            }
        ],
    )

    class FakeLLM:
        async def chat_text(self, system_prompt: str, user_message: str) -> str:
            assert "只能依据" in system_prompt
            assert "[资料 1]" in user_message
            return "访问令牌过期后应使用刷新令牌重试一次。[1]"

    result = await answer_question(object(), "00000000-0000-0000-0000-000000000001", "登录过期后怎么办？", llm=FakeLLM())

    assert result["generation_mode"] == "llm"
    assert result["retrieval_count"] == 1
    assert result["sources"][0]["citation"] == 1
    assert "[1]" in result["answer"]


@pytest.mark.asyncio
async def test_answer_question_has_offline_extractive_fallback(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", None)
    monkeypatch.setattr(
        "app.services.rag.qa.search_similar_documents",
        lambda *args, **kwargs: [
            {
                "id": "doc-1",
                "source_id": "source-1",
                "source_type": "knowledge_file",
                "title": "leave.md#1",
                "snippet": "员工申请年假时，需要先提交直属主管审批。审批通过后同步到考勤系统。",
                "score": 0.88,
                "metadata": {},
            }
        ],
    )

    result = await answer_question(object(), "00000000-0000-0000-0000-000000000001", "年假怎么审批？")

    assert result["generation_mode"] == "extractive"
    assert "主管审批" in result["answer"]
    assert "[1]" in result["answer"]


@pytest.mark.asyncio
async def test_answer_question_does_not_call_llm_when_gate_rejects_evidence(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "test-key")

    async def fake_retrieve(*_args, **_kwargs):
        return (
            [
                {
                    "id": "doc-1",
                    "source_id": "source-1",
                    "source_type": "knowledge_file",
                    "title": "old.md",
                    "snippet": "旧版本允许部署。",
                    "score": 0.91,
                    "metadata": {"branch": "legacy"},
                    "retrieval": {"answer_gate": "conflict"},
                }
            ],
            {"answer_gate": {"decision": "conflict", "reason": "高排名证据给出了相反结论"}},
            1,
        )

    monkeypatch.setattr("app.services.rag.qa.retrieve_knowledge", fake_retrieve)

    class FakeDB:
        def query(self):
            raise AssertionError("retrieve_knowledge is mocked")

    class FailingLLM:
        async def chat_text(self, _system_prompt: str, _user_message: str) -> str:
            raise AssertionError("LLM must not run when the answer gate rejects evidence")

    result = await answer_question(
        FakeDB(),
        "00000000-0000-0000-0000-000000000001",
        "现在允许部署吗？",
        llm=FailingLLM(),
    )

    assert result["generation_mode"] == "no_evidence"
    assert result["answer_gate"]["decision"] == "conflict"
    assert result["retrieval_count"] == 1
    assert "存在冲突" in result["answer"]
