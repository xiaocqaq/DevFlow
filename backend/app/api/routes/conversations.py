# ==============================================================================
# 模块名称: 对话会话路由 (Conversations API)
# 架构定位: Web 接口层 (API Routes Layer)
# 核心职责:
#   1. 提供多会话历史记录查询接口（返回会话 ID、标题、活跃状态、消息条数、时间戳）。
#   2. 提供新建对话会话接口（支持指定标题或系统自动序列化递增标题）。
#   3. 提供会话删除与自动切换生效会话接口（保证前端视图总有活跃会话展示）。
# ==============================================================================

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.conversations import ConversationCreateRequest, ConversationResponse
from app.services.conversations import create_conversation, delete_conversation, list_conversations

# 创建会话专用子路由
router = APIRouter()


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """
    【通用前置校验】根据仓库 UUID 校验其是否存在，不存在则抛出 404 HTTP 异常。

    设计模式:
      - 快速失败原则 (Fail-fast): 提早拦截非法 repo_id 请求，避免后续无意义的 SQL 操作。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


@router.get("/{repo_id}/conversations", response_model=list[ConversationResponse])
async def get_conversations(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[Any]:
    """
    【API 接口】获取仓库的所有活跃会话列表。

    处理流程:
      1. 校验仓库存在性。
      2. 调用 `list_conversations`，内部会自动初始化默认会话并执行历史消息关联回填。
      3. 提交数据库事务并返回 Pydantic 校验后的实体列表。
    """
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
    """
    【API 接口】在指定仓库下新建一个独立会话。

    参数说明:
      - `payload`: 可选，允许客户端自定义传递初始标题；若为空则由系统自动指派序号标题。
    """
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
    """
    【API 接口】删除指定会话并返回最新的活跃会话上下文。

    前端无缝衔接设计:
      - 当用户在前端点击“删除当前会话”时，前端需要立刻知道接下来应该聚焦哪一个会话。
      - 接口不仅返回被删除的 `deleted` ID，还返回接替聚焦的 `active_conversation_id`
        以及刷新后的完整 `conversations` 列表，避免前端发起多次网络往返请求 (Round-trip)。
    """
    _repo_or_404(db, repo_id)
    # 软删除当前会话并自动定位或创建替代会话
    replacement = delete_conversation(db, repo_id, conversation_id)
    # 重新加载删除后的有效会话列表
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

