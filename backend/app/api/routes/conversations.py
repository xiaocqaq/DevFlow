import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.conversations import ConversationCreateRequest, ConversationResponse
from app.services.conversations import create_conversation, delete_conversation, list_conversations

router = APIRouter()


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


@router.get("/{repo_id}/conversations", response_model=list[ConversationResponse])
async def get_conversations(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Any]:
    _repo_or_404(db, repo_id)
    conversations = list_conversations(db, repo_id)
    db.commit()
    return conversations


@router.post("/{repo_id}/conversations", response_model=ConversationResponse)
async def add_conversation(
    repo_id: uuid.UUID,
    payload: ConversationCreateRequest | None = None,
    db: Session = Depends(get_db),
) -> Any:
    _repo_or_404(db, repo_id)
    conversation = create_conversation(db, repo_id, payload.title if payload else None)
    db.commit()
    db.refresh(conversation)
    return conversation


@router.delete("/{repo_id}/conversations/{conversation_id}")
async def remove_conversation(
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    _repo_or_404(db, repo_id)
    replacement = delete_conversation(db, repo_id, conversation_id)
    conversations = list_conversations(db, repo_id)
    db.commit()
    return {
        "deleted": str(conversation_id),
        "active_conversation_id": str(replacement.id),
        "conversations": [
            {
                "id": str(item.id),
                "repo_id": str(item.repo_id),
                "title": item.title,
                "status": item.status,
                "message_count": item.message_count,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
            }
            for item in conversations
        ],
    }
