import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.api.routes import feedback as feedback_routes
from app.db.models import AgentRun, Base, ChatFeedback, ChatMessage, ChatSession, Conversation, Repository
from app.schemas.feedback import ChatFeedbackCreate, ChatFeedbackReview


def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _feedback_fixture(db):
    repo = Repository(owner="devflow", name="demo", full_name=f"devflow/demo-{uuid.uuid4().hex[:8]}")
    db.add(repo)
    db.flush()
    conversation = Conversation(repo_id=repo.id, title="反馈测试")
    db.add(conversation)
    db.flush()
    run_id = uuid.uuid4()
    user_message = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="user",
        content="为什么 CI 失败？",
    )
    assistant_message = ChatMessage(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="assistant",
        content="CI 是因为测试失败。",
        route="ci_debug",
        tool_calls=[{"tool_name": "ci_debug", "status": "success"}],
        meta={"run_id": str(run_id), "agent_events": [{"type": "tool_use", "tool_name": "ci_debug"}]},
    )
    db.add_all([user_message, assistant_message])
    db.flush()
    db.add(
        AgentRun(
            id=run_id,
            repo_id=repo.id,
            conversation_id=conversation.id,
            user_message=user_message.content,
            route="ci_debug",
            tool_calls=assistant_message.tool_calls,
            final_answer=assistant_message.content,
            status="success",
        )
    )
    db.add(
        ChatSession(
            repo_id=repo.id,
            conversation_id=conversation.id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            route="ci_debug",
            transcript_json={
                "agent_events": [{"type": "tool_use", "tool_name": "ci_debug"}],
                "tool_use_results": [{"type": "tool_result", "status": "success"}],
                "memory_update": {"facts": ["CI failed"]},
            },
            digest="排查 CI 失败",
            status="sealed",
        )
    )
    db.commit()
    return repo, conversation, assistant_message, run_id


@pytest.mark.asyncio
async def test_negative_feedback_persists_notifies_and_traces(monkeypatch):
    db = _db_session()
    repo, conversation, assistant_message, run_id = _feedback_fixture(db)
    notifications = []

    async def fake_notification(**kwargs):
        notifications.append(kwargs)
        return "sent", None

    monkeypatch.setattr(feedback_routes, "send_negative_feedback_notification", fake_notification)
    result = await feedback_routes.create_feedback(
        ChatFeedbackCreate(
            repo_id=repo.id,
            conversation_id=conversation.id,
            assistant_message_id=assistant_message.id,
            rating="unhelpful",
            reason="inaccurate",
            comment="真正失败的是类型检查。",
        ),
        db,
    )

    assert result.run_id == run_id
    assert result.notification_status == "sent"
    assert result.review_status == "open"
    assert len(notifications) == 1

    metrics = await feedback_routes.feedback_metrics(repo.id, db)
    assert metrics["assistant_messages"] == 1
    assert metrics["unhelpful"] == 1
    assert metrics["negative_feedback_rate"] == 1.0
    assert metrics["reason_counts"] == {"inaccurate": 1}

    trace = await feedback_routes.feedback_trace(run_id, db)
    assert trace["trace_id"] == str(run_id)
    assert trace["run"]["route"] == "ci_debug"
    assert trace["messages"]["assistant"]["id"] == str(assistant_message.id)
    assert trace["session"]["agent_events"][0]["tool_name"] == "ci_debug"

    reviewed = await feedback_routes.review_feedback(
        result.id,
        ChatFeedbackReview(review_status="resolved", review_note="已修复 Prompt"),
        db,
    )
    assert reviewed.review_status == "resolved"
    assert reviewed.reviewed_at is not None


@pytest.mark.asyncio
async def test_feedback_is_idempotent_per_assistant_message(monkeypatch):
    db = _db_session()
    repo, conversation, assistant_message, _run_id = _feedback_fixture(db)
    notifications = []

    async def fake_notification(**kwargs):
        notifications.append(kwargs)
        return "sent", None

    monkeypatch.setattr(feedback_routes, "send_negative_feedback_notification", fake_notification)
    payload = ChatFeedbackCreate(
        repo_id=repo.id,
        conversation_id=conversation.id,
        assistant_message_id=assistant_message.id,
        rating="unhelpful",
    )
    first = await feedback_routes.create_feedback(payload, db)
    second = await feedback_routes.create_feedback(payload, db)

    assert first.id == second.id
    assert db.query(ChatFeedback).count() == 1
    assert len(notifications) == 1

    helpful = await feedback_routes.create_feedback(
        ChatFeedbackCreate(
            repo_id=repo.id,
            conversation_id=conversation.id,
            assistant_message_id=assistant_message.id,
            rating="helpful",
        ),
        db,
    )
    assert helpful.rating == "helpful"
    assert helpful.reason is None
    assert helpful.notification_status == "not_applicable"
