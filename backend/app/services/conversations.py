# ==============================================================================
# 模块名称: 会话管理核心服务 (Conversation Lifecycle Service)
# 架构定位: 业务逻辑层 (Services Layer)
# 核心职责:
#   1. 提供多会话隔离管理（类似 ChatGPT 侧边栏多窗口历史记录）。
#   2. 自动保障默认会话（保证每个仓库至少拥有一个活跃可用的会话）。
#   3. 智能命名机制：首轮对话根据用户提问内容自动更新会话标题。
#   4. 会话级联软删除与多表关联清理（高效使用 synchronize_session=False）。
#   5. 历史版本无会话字段数据的静默平滑回填迁移 (Backfill)。
# ==============================================================================

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import (
    AgentRun,
    AgentTaskRun,
    AgentWorkflowRun,
    ChatFeedback,
    ChatMessage,
    ChatSession,
    Conversation,
    ConversationMemory,
    EvidenceItem,
    MemoryCandidate,
    RecallEvent,
    Repository,
)


def _default_title(db: Session, repo_id: uuid.UUID) -> str:
    """
    【内部辅助方法】生成新会话的默认标题序列号。

    语法与逻辑说明:
      - 通过 `db.query(Conversation).filter(...).count()` 快速查询现有会话总数。
      - 若首次创建则命名为 "默认会话"，后续则生成类似 "新会话 2"、"新会话 3" 的递增序号。

    Args:
        db (Session): SQLAlchemy 会话。
        repo_id (uuid.UUID): 关联的仓库主键。

    Returns:
        str: 默认会话标题。
    """
    existing = db.query(Conversation).filter(Conversation.repo_id == repo_id).count()
    return "默认会话" if existing == 0 else f"新会话 {existing + 1}"


def ensure_default_conversation(db: Session, repo_id: uuid.UUID) -> Conversation:
    """
    【核心幂等方法】确保指定仓库至少存在一个活跃 (status='active') 的默认会话。

    设计模式:
      - 懒加载与兜底保障模式 (Ensure / Lazy Fallback)。
      - 优先按创建时间升序 (`created_at.asc()`) 获取最先创建的活跃会话。
      - 若数据库中不存在任何活跃会话，则自动创建一个 "默认会话" 并调用 `db.flush()` 立即持久化分配 ID。

    Args:
        db (Session): 数据库会话。
        repo_id (uuid.UUID): 仓库 UUID。

    Returns:
        Conversation: 活跃的默认会话实体。
    """
    conversation = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.created_at.asc())
        .first()
    )
    if conversation:
        return conversation

    # 兜底新建
    conversation = Conversation(repo_id=repo_id, title="默认会话", status="active")
    db.add(conversation)
    # db.flush() 将新建记录推送到事务缓冲区生成主键 UUID，但不提交事务
    db.flush()
    return conversation


def ensure_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID | None = None) -> Conversation:
    """
    【会话上下文路由】解析并获取当前请求所需的会话对象。

    解析优先级:
      1. 若显式传递了 `conversation_id`，查询该 ID 且状态为 active 的会话。
      2. 若未传递、已软删除或 ID 对应实体不存在，自动回退到默认活跃会话 (`ensure_default_conversation`)。

    Args:
        db (Session): 数据库会话。
        repo_id (uuid.UUID): 仓库 UUID。
        conversation_id (uuid.UUID | None): 可选的指定会话 ID。

    Returns:
        Conversation: 经过有效性验证的 Conversation 实体。
    """
    if conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
            .one_or_none()
        )
        if conversation:
            return conversation
    # 回退到默认会话
    return ensure_default_conversation(db, repo_id)


def list_conversations(db: Session, repo_id: uuid.UUID) -> list[Conversation]:
    """
    【查询方法】获取指定仓库下的所有活跃会话列表（按最近活跃时间倒序排列）。

    附加动作:
      - 自动调用 `ensure_default_conversation` 保障列表至少有一项。
      - 执行 `_backfill_repo_conversation` 将旧版本遗留未打会话标的消息自动关联。

    Args:
        db (Session): 数据库会话。
        repo_id (uuid.UUID): 仓库 UUID。

    Returns:
        list[Conversation]: 活跃会话列表。
    """
    ensure_default_conversation(db, repo_id)
    _backfill_repo_conversation(db, repo_id)
    db.flush()
    return (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .all()
    )


def create_conversation(db: Session, repo_id: uuid.UUID, title: str | None = None) -> Conversation:
    """
    【新建会话】显式为仓库创建一条新会话。

    Python 表达式巧用:
      - `(title or "").strip() or _default_title(db, repo_id)`:
        先去除首尾空格，若结果为空字符串则短路执行 `_default_title` 获取递增默认名称。

    Args:
        db (Session): 数据库会话。
        repo_id (uuid.UUID): 仓库 UUID。
        title (str | None): 可选的自定义标题。

    Returns:
        Conversation: 新建并已经分配 ID 的 Conversation ORM 对象。
    """
    conversation = Conversation(repo_id=repo_id, title=(title or "").strip() or _default_title(db, repo_id))
    db.add(conversation)
    db.flush()
    return conversation


def delete_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation:
    """
    【级联删除与替代回退】软删除会话并深度清理其关联的历史运行与记忆数据。

    性能与架构核心解析:
      - 为什么使用 `delete(synchronize_session=False)`?
        默认情况下 SQLAlchemy 会在 Python 内存中尝试同步所有加载的实体的状态，当批量删除大量消息、
        工作流、记忆候选时极耗 CPU 和内存。指定 `synchronize_session=False` 会直接下发原始 SQL DELETE，
        性能最高且完全避免死锁。
      - 清理范围:
        反馈评价 (`ChatFeedback`)、记忆萃取候选 (`MemoryCandidate`)、召回日志 (`RecallEvent`)、
        工作流任务 (`AgentTaskRun`)、工作流实例 (`AgentWorkflowRun`)、对话消息 (`ChatMessage`)、
        执行会话 (`ChatSession`)、智能体运行记录 (`AgentRun`)、会话记忆 (`ConversationMemory`)、
        佐证引用项 (`EvidenceItem`)。
      - 软删除标记: 会话本身的 `status` 标记为 'deleted'，保留主键防引用悬空。
      - 自动切换替代项 (Replacement): 删除后自动返回仓库中剩下的最近活跃会话（若全被删空则新建一个）。

    Args:
        db (Session): 数据库会话。
        repo_id (uuid.UUID): 仓库 UUID。
        conversation_id (uuid.UUID): 待删除会话 UUID。

    Returns:
        Conversation: 删除后当前仓库生效的替代活跃会话。
    """
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
        .one_or_none()
    )
    if conversation is None:
        return ensure_default_conversation(db, repo_id)

    # 1. 搜集二级关联 ID（ChatSession 与 AgentWorkflowRun）
    session_ids = [
        item.id
        for item in db.query(ChatSession.id).filter(ChatSession.conversation_id == conversation.id).all()
    ]
    workflow_ids = [
        item.id
        for item in db.query(AgentWorkflowRun.id).filter(AgentWorkflowRun.conversation_id == conversation.id).all()
    ]

    # 2. 批量快速物理清理下游叶子节点与关联表数据
    db.query(ChatFeedback).filter(ChatFeedback.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(MemoryCandidate).filter(MemoryCandidate.conversation_id == conversation.id).delete(synchronize_session=False)
    if session_ids:
        db.query(MemoryCandidate).filter(MemoryCandidate.session_id.in_(session_ids)).delete(synchronize_session=False)
    db.query(RecallEvent).filter(RecallEvent.conversation_id == conversation.id).delete(synchronize_session=False)
    if workflow_ids:
        db.query(AgentTaskRun).filter(AgentTaskRun.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
    db.query(AgentWorkflowRun).filter(AgentWorkflowRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatSession).filter(ChatSession.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(AgentRun).filter(AgentRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(EvidenceItem).filter(EvidenceItem.conversation_id == conversation.id).delete(synchronize_session=False)

    # 3. 将会话状态变更为已删除 (软删除)
    conversation.status = "deleted"
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()

    # 4. 寻找接替当前视图的下一个活跃会话
    replacement = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .first()
    )
    return replacement or create_conversation(db, repo_id, "默认会话")


def touch_conversation(
    db: Session,
    conversation: Conversation,
    *,
    user_message: str | None = None,
    increment: int = 0,
) -> None:
    """
    【活跃度与智能标题刷新】当发生新对话时更新会话统计指标并按需自动重命名。

    特性:
      - 智能标题生成 (Auto Titling):
        若该会话尚无历史消息 (`message_count == 0`) 且标题仍是默认系统名（"默认会话" 或 "新会话" 开头），
        则自动截取第一条用户消息的前 40 个字符作为会话标题，赋予用户如同 ChatGPT 的良好体验。
      - 消息计数与时间戳推进:
        原子增加 `message_count`，并将 `updated_at` 刷新为当前 UTC 时间，使该会话在列表置顶。

    Args:
        db (Session): 数据库会话。
        conversation (Conversation): 会话实体。
        user_message (str | None): 可选的首条用户提问内容。
        increment (int): 消息条数增量（通常为 1 或 2）。
    """
    # 首条消息智能生成标题
    if user_message and conversation.message_count == 0 and conversation.title.startswith(("默认会话", "新会话")):
        title = user_message.strip().replace("\n", " ")
        if title:
            conversation.title = title[:40]
    # 增加计数并更新时间戳
    conversation.message_count = (conversation.message_count or 0) + increment
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()


def backfill_all_default_conversations(db: Session) -> None:
    """
    【数据迁移运维方法】为系统中所有仓库执行遗留历史数据的会话平滑回填。

    在系统升级引入多会话模型前，历史消息未关联 `conversation_id`。
    此方法遍历所有仓库，确保默认会话存在并将孤儿消息归纳其中。
    """
    for repo in db.query(Repository).all():
        ensure_default_conversation(db, repo.id)
        _backfill_repo_conversation(db, repo.id)
    db.commit()


def _backfill_repo_conversation(db: Session, repo_id: uuid.UUID) -> None:
    """
    【内部回填执行】将指定仓库中尚未关联 conversation_id 的旧数据绑定到默认会话。
    """
    conversation = ensure_default_conversation(db, repo_id)
    # 对孤儿记录执行批量 UPDATE
    for model in [ChatMessage, ChatSession, AgentRun]:
        db.query(model).filter(model.repo_id == repo_id, model.conversation_id.is_(None)).update(
            {model.conversation_id: conversation.id},
            synchronize_session=False,
        )
    # 重新统计该默认会话的消息总数
    conversation.message_count = db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).count()

