import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import ChatMessage, ChatSession, Conversation, RecallEvent, Repository
from app.db.session import Base
from app.services.chat_memory import (
    ContextAssembler,
    MemoryTools,
    MessageStore,
    ProgressiveContextManager,
    SessionSealer,
)


def _db() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _scope(db: Session) -> tuple[Repository, Conversation]:
    repo = Repository(owner="acme", name="context", full_name="acme/context")
    db.add(repo)
    db.flush()
    conversation = Conversation(repo_id=repo.id, title="Long context")
    db.add(conversation)
    db.flush()
    return repo, conversation


def _message(
    db: Session,
    repo: Repository,
    conversation: Conversation,
    *,
    role: str,
    content: str,
    at: datetime,
    route: str | None = None,
    tool_calls: list[dict] | None = None,
    meta: dict | None = None,
) -> ChatMessage:
    row = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role=role,
        content=content,
        route=route,
        tool_calls=tool_calls or [],
        meta=meta or {},
        created_at=at,
    )
    db.add(row)
    db.flush()
    return row


def test_durable_timeline_crosses_boundary_and_hides_compatibility_copies() -> None:
    db = _db()
    repo, conversation = _scope(db)
    start = datetime.now(timezone.utc)
    first = _message(db, repo, conversation, role="user", content="first exact request", at=start)
    answer = _message(db, repo, conversation, role="assistant", content="first exact answer", at=start + timedelta(seconds=1))
    boundary = _message(
        db,
        repo,
        conversation,
        role="system",
        content="summary",
        route="context_compaction",
        meta={"subtype": "compact_boundary"},
        at=start + timedelta(seconds=2),
    )
    _message(
        db,
        repo,
        conversation,
        role="assistant",
        content=answer.content,
        meta={"subtype": "preserved_after_compact", "preserved_from": str(answer.id)},
        at=start + timedelta(seconds=3),
    )
    latest = _message(db, repo, conversation, role="user", content="latest request", at=start + timedelta(seconds=4))

    timeline = MessageStore(db).timeline(repo.id, conversation.id, limit=100)

    assert [row.id for row in timeline] == [first.id, answer.id, latest.id]
    assert boundary.id not in {row.id for row in timeline}


def test_durable_timeline_cursor_is_stable_when_timestamps_are_equal() -> None:
    db = _db()
    repo, conversation = _scope(db)
    at = datetime.now(timezone.utc)
    rows = [
        _message(db, repo, conversation, role="user", content=f"same-time-{index}", at=at)
        for index in range(3)
    ]
    expected = sorted(rows, key=lambda row: row.id)

    newest_page = MessageStore(db).timeline(repo.id, conversation.id, limit=2)
    older_page = MessageStore(db).timeline(
        repo.id,
        conversation.id,
        limit=2,
        before_message_id=newest_page[0].id,
    )

    assert [row.id for row in [*older_page, *newest_page]] == [row.id for row in expected]


@pytest.mark.asyncio
async def test_exact_thread_recall_reads_pre_boundary_text_and_is_audited() -> None:
    db = _db()
    repo, conversation = _scope(db)
    start = datetime.now(timezone.utc)
    old = _message(
        db,
        repo,
        conversation,
        role="user",
        content="UNIQUE-PRE-BOUNDARY original wording",
        at=start,
    )
    _message(
        db,
        repo,
        conversation,
        role="system",
        content="compressed summary",
        route="context_compaction",
        meta={"subtype": "compact_boundary", "preserved_message_ids": []},
        at=start + timedelta(seconds=1),
    )
    _message(db, repo, conversation, role="assistant", content="new answer", at=start + timedelta(seconds=2))

    tools = MemoryTools(db, repo, conversation)
    result = await tools.get_thread_context({"keyword": "UNIQUE-PRE-BOUNDARY", "limit": 10}, {})
    page = await tools.get_thread_context(
        {"message_id": str(old.id), "offset": 7, "max_chars": 12},
        {},
    )

    assert "UNIQUE-PRE-BOUNDARY original wording" in result["answer"]
    assert str(old.id) in result["answer"]
    assert old.content[7:19] in page["answer"]
    events = db.query(RecallEvent).filter(RecallEvent.tool_name == "devflow_get_thread_context").all()
    assert len(events) == 2
    assert all(event.mode == "exact_transcript" for event in events)


@pytest.mark.asyncio
async def test_session_event_tool_returns_transcript_and_records_recall() -> None:
    db = _db()
    repo, conversation = _scope(db)
    session = ChatSession(
        repo_id=repo.id,
        conversation_id=conversation.id,
        route="chat",
        digest="short digest",
        transcript_json={"messages": [{"role": "user", "content": "exact sealed text"}]},
        status="sealed",
        sealed_at=datetime.now(timezone.utc),
    )
    db.add(session)
    db.flush()

    result = await MemoryTools(db, repo, conversation).read_session_events(
        {"session_id": str(session.id), "include_transcript": True},
        {},
    )
    continuation = await MemoryTools(db, repo, conversation).read_session_events(
        {"session_id": str(session.id), "include_transcript": True, "offset": 5, "max_chars": 400},
        {},
    )

    assert "exact sealed text" in result["answer"]
    assert continuation["results"][0]["metadata"]["offset"] == 5
    events = db.query(RecallEvent).filter(RecallEvent.tool_name == "devflow_read_session_events").all()
    assert len(events) == 2
    assert all(event.mode == "sealed_transcript" for event in events)
    assert all(event.result_count == 1 for event in events)


@pytest.mark.asyncio
async def test_formal_compaction_reuses_original_rows_and_rehydrates_working_state() -> None:
    class FakeLLM:
        async def chat_text(self, system_prompt: str, user_prompt: str) -> str:
            return "User intent, decisions, errors, pending task and next step are preserved."

    db = _db()
    repo, conversation = _scope(db)
    start = datetime.now(timezone.utc)
    rows: list[ChatMessage] = []
    for index in range(6):
        rows.append(
            _message(
                db,
                repo,
                conversation,
                role="user" if index % 2 == 0 else "assistant",
                content=f"message {index} " + ("details " * 80),
                route="workspace.read_file" if index == 3 else None,
                tool_calls=[
                    {
                        "tool_name": "workspace.read_file",
                        "arguments": {"path": "backend/app/auth.py"},
                        "status": "success",
                    }
                ]
                if index == 3
                else [],
                at=start + timedelta(seconds=index),
            )
        )

    manager = ProgressiveContextManager(db, llm=FakeLLM())  # type: ignore[arg-type]
    result = await manager._compact(
        repo,
        conversation,
        "continue",
        {"stage": "auto_compact"},
        "auto_compact",
    )

    assert result["compacted"] is True
    assert result["rehydration"]["working_files"] == ["backend/app/auth.py"]
    assert not any(
        (row.meta or {}).get("subtype") == "preserved_after_compact"
        for row in db.query(ChatMessage).all()
    )
    boundary = db.get(ChatMessage, uuid.UUID(result["boundary_id"]))
    assert boundary is not None
    preserved_ids = [uuid.UUID(value) for value in boundary.meta["preserved_message_ids"]]
    active = MessageStore(db).active_for_context(repo.id, conversation.id, limit=100)
    assert [row.id for row in active if row.role != "system"] == preserved_ids
    assert all(row.id in {item.id for item in rows} for row in active if row.role != "system")


@pytest.mark.asyncio
async def test_session_sealer_keeps_raw_transcript_when_extraction_fails() -> None:
    class FailingLLM:
        async def chat_json(self, system_prompt: str, payload: dict) -> dict:
            raise TimeoutError("memory model timeout")

    db = _db()
    repo, conversation = _scope(db)
    now = datetime.now(timezone.utc)
    user = _message(db, repo, conversation, role="user", content="remember exact input", at=now)
    assistant = _message(db, repo, conversation, role="assistant", content="completed answer", at=now + timedelta(seconds=1))

    sealed = await SessionSealer(db, llm=FailingLLM()).seal(  # type: ignore[arg-type]
        repo=repo,
        conversation=conversation,
        user_message=user,
        assistant_message=assistant,
        route="chat",
        tool_calls=[],
    )

    assert sealed.status == "sealed_with_warnings"
    assert sealed.transcript_json["messages"][0]["content"] == "remember exact input"
    assert sealed.transcript_json["messages"][1]["content"] == "completed answer"
    assert "memory_extraction_failed" in sealed.transcript_json["sealing"]["errors"][0]


def test_context_assembler_keeps_recent_history_out_of_system_context() -> None:
    db = _db()
    repo, conversation = _scope(db)
    now = datetime.now(timezone.utc)
    _message(db, repo, conversation, role="user", content="UNIQUE-RECENT-MESSAGE", at=now)

    assembled = ContextAssembler(db).assemble(repo, conversation, "UNIQUE-RECENT-MESSAGE")

    assert "UNIQUE-RECENT-MESSAGE" not in assembled.system_context
    assert assembled.history_messages[-1]["content"] == "UNIQUE-RECENT-MESSAGE"
    assert assembled.compression_stats["pending_user_estimated_tokens"] == 0
    assert assembled.compression_stats["assembled_prompt_estimated_tokens"] >= assembled.compression_stats["history_estimated_tokens"]
