import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models import AgentRun, ChatFeedback, ChatMessage, ChatSession, Conversation, Repository
from app.db.session import get_db
from app.schemas.feedback import ChatFeedbackCreate, ChatFeedbackResponse, ChatFeedbackReview
from app.services.feedback_notifications import send_negative_feedback_notification
from app.services.permissions import ensure_demo_user

router = APIRouter()


def _run_id_from_message(message: ChatMessage) -> uuid.UUID | None:
    raw_run_id = (message.meta or {}).get("run_id")
    if not raw_run_id:
        return None
    try:
        return uuid.UUID(str(raw_run_id))
    except (TypeError, ValueError):
        return None


def _message_or_404(db: Session, payload: ChatFeedbackCreate) -> ChatMessage:
    message = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.id == payload.assistant_message_id,
            ChatMessage.repo_id == payload.repo_id,
            ChatMessage.conversation_id == payload.conversation_id,
            ChatMessage.role == "assistant",
        )
        .one_or_none()
    )
    if message is None:
        raise HTTPException(status_code=404, detail="未找到对应的 Agent 回答")
    return message


def _serialize_message(message: ChatMessage | None) -> dict[str, Any] | None:
    if message is None:
        return None
    return {
        "id": str(message.id),
        "role": message.role,
        "content": message.content,
        "route": message.route,
        "tool_calls": message.tool_calls or [],
        "metadata": message.meta or {},
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


@router.get("/metrics")
async def feedback_metrics(
    repo_id: uuid.UUID = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    rows = db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo_id).all()
    assistant_messages = db.query(ChatMessage).filter(ChatMessage.repo_id == repo_id, ChatMessage.role == "assistant").count()
    helpful = sum(row.rating == "helpful" for row in rows)
    unhelpful = sum(row.rating == "unhelpful" for row in rows)
    rated = len(rows)
    reasons: dict[str, int] = {}
    for row in rows:
        if row.reason:
            reasons[row.reason] = reasons.get(row.reason, 0) + 1
    return {
        "repo_id": str(repo_id),
        "assistant_messages": assistant_messages,
        "rated_messages": rated,
        "feedback_coverage": round(rated / assistant_messages, 4) if assistant_messages else 0.0,
        "helpful": helpful,
        "unhelpful": unhelpful,
        "helpful_rate": round(helpful / rated, 4) if rated else 0.0,
        "unhelpful_rate": round(unhelpful / rated, 4) if rated else 0.0,
        "negative_feedback_rate": round(unhelpful / assistant_messages, 4) if assistant_messages else 0.0,
        "open_reviews": sum(row.rating == "unhelpful" and row.review_status in {"open", "in_review"} for row in rows),
        "reason_counts": reasons,
    }


@router.get("/trace/{run_id}")
async def feedback_trace(run_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = db.get(AgentRun, run_id)
    feedback = db.query(ChatFeedback).filter(ChatFeedback.run_id == run_id).one_or_none()
    if run is None and feedback is None:
        raise HTTPException(status_code=404, detail="未找到对应的执行轨迹")

    assistant_message = db.get(ChatMessage, feedback.assistant_message_id) if feedback else None
    if assistant_message is None and run is not None:
        candidates = (
            db.query(ChatMessage)
            .filter(ChatMessage.conversation_id == run.conversation_id, ChatMessage.role == "assistant")
            .order_by(ChatMessage.created_at.desc())
            .all()
        )
        assistant_message = next((item for item in candidates if _run_id_from_message(item) == run_id), None)

    session = (
        db.query(ChatSession).filter(ChatSession.assistant_message_id == assistant_message.id).one_or_none()
        if assistant_message
        else None
    )
    user_message = db.get(ChatMessage, session.user_message_id) if session and session.user_message_id else None
    repo_id = feedback.repo_id if feedback else run.repo_id if run else None
    conversation_id = feedback.conversation_id if feedback else run.conversation_id if run else None
    repo = db.get(Repository, repo_id) if repo_id else None
    conversation = db.get(Conversation, conversation_id) if conversation_id else None
    transcript = session.transcript_json if session else {}

    return {
        "trace_id": str(run_id),
        "repository": {"id": str(repo.id), "full_name": repo.full_name} if repo else None,
        "conversation": {"id": str(conversation.id), "title": conversation.title} if conversation else None,
        "feedback": ChatFeedbackResponse.model_validate(feedback).model_dump(mode="json") if feedback else None,
        "run": {
            "id": str(run.id),
            "status": run.status,
            "route": run.route,
            "user_message": run.user_message,
            "final_answer": run.final_answer,
            "tool_calls": run.tool_calls or [],
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        if run
        else None,
        "messages": {
            "user": _serialize_message(user_message),
            "assistant": _serialize_message(assistant_message),
        },
        "session": {
            "id": str(session.id),
            "status": session.status,
            "digest": session.digest,
            "agent_events": transcript.get("agent_events", []),
            "tool_use_results": transcript.get("tool_use_results", []),
            "memory_update": transcript.get("memory_update", {}),
            "sealed_at": session.sealed_at.isoformat() if session.sealed_at else None,
        }
        if session
        else None,
    }


@router.get("", response_model=list[ChatFeedbackResponse])
async def list_feedback(
    repo_id: uuid.UUID = Query(...),
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[ChatFeedback]:
    query = db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo_id)
    if conversation_id:
        query = query.filter(ChatFeedback.conversation_id == conversation_id)
    return query.order_by(ChatFeedback.created_at.desc()).limit(limit).all()


@router.post("", response_model=ChatFeedbackResponse)
async def create_feedback(payload: ChatFeedbackCreate, db: Session = Depends(get_db)) -> ChatFeedback:
    repo = db.get(Repository, payload.repo_id)
    conversation = db.get(Conversation, payload.conversation_id)
    if repo is None or conversation is None or conversation.repo_id != repo.id:
        raise HTTPException(status_code=404, detail="未找到对应的项目会话")
    message = _message_or_404(db, payload)
    run_id = _run_id_from_message(message)
    run = db.get(AgentRun, run_id) if run_id else None
    reporter = ensure_demo_user(db)

    feedback = db.query(ChatFeedback).filter(ChatFeedback.assistant_message_id == message.id).one_or_none()
    notify = payload.rating == "unhelpful" and (feedback is None or feedback.rating != "unhelpful" or feedback.notified_at is None)
    if feedback is None:
        feedback = ChatFeedback(
            repo_id=repo.id,
            conversation_id=conversation.id,
            assistant_message_id=message.id,
            run_id=run_id,
            user_id=reporter.id,
            rating=payload.rating,
        )
        db.add(feedback)

    previous_rating = feedback.rating
    feedback.rating = payload.rating
    feedback.reason = payload.reason if payload.rating == "unhelpful" else None
    feedback.comment = (payload.comment or "").strip() or None
    feedback.run_id = run_id
    feedback.user_id = reporter.id
    if payload.rating == "unhelpful":
        if previous_rating != "unhelpful":
            feedback.review_status = "open"
            feedback.review_note = None
            feedback.reviewed_at = None
        if notify:
            feedback.notification_status = "pending"
            feedback.notification_error = None
    else:
        feedback.review_status = "resolved"
        feedback.review_note = None
        feedback.reviewed_at = datetime.now(timezone.utc)
        feedback.notification_status = "not_applicable"
        feedback.notification_error = None
    db.commit()
    db.refresh(feedback)

    if notify:
        notification_status, notification_error = await send_negative_feedback_notification(
            feedback=feedback,
            repo=repo,
            message=message,
            run=run,
            reporter=reporter,
        )
        feedback.notification_status = notification_status
        feedback.notification_error = notification_error
        feedback.notified_at = datetime.now(timezone.utc) if notification_status == "sent" else None
        db.commit()
        db.refresh(feedback)
    return feedback


@router.patch("/{feedback_id}", response_model=ChatFeedbackResponse)
async def review_feedback(
    feedback_id: uuid.UUID,
    payload: ChatFeedbackReview,
    db: Session = Depends(get_db),
) -> ChatFeedback:
    feedback = db.get(ChatFeedback, feedback_id)
    if feedback is None:
        raise HTTPException(status_code=404, detail="未找到反馈记录")
    feedback.review_status = payload.review_status
    feedback.review_note = (payload.review_note or "").strip() or None
    feedback.reviewed_at = datetime.now(timezone.utc) if payload.review_status in {"resolved", "dismissed"} else None
    db.commit()
    db.refresh(feedback)
    return feedback
