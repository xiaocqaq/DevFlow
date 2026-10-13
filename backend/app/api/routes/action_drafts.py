# ==============================================================================
# 模块名称: 动作草案与高危审批路由 (Action Drafts & Human-in-the-Loop)
# 架构定位: 安全防御层 (Safety Layer) / Web 接口层
# 核心职责:
#   1. 提供 Human-in-the-Loop (人在回路) 机制，拦截 Agent 产生的高危外部写操作 (如 GitHub 评论、打标签、关闭 Issue)。
#   2. 动作草案的创建、查询、取消与确认执行生命周期管理。
#   3. 全链路安全审计 (AuditLog) 记录，追踪“谁在何时以何种权限触发/批准了何种外部变更”。
# ==============================================================================

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import ActionDraft, AuditLog, Issue, PullRequest, Repository, User
from app.db.session import get_db
from app.schemas.action_drafts import ActionDraftConfirmResponse, ActionDraftCreateRequest, ActionDraftResponse, AuditLogResponse
from app.services.github.client import GitHubClient
from app.services.permissions import require_permission, write_audit_log

# 初始化 FastAPI 路由子模块
router = APIRouter()


def _target_number(db: Session, draft: ActionDraft) -> int | None:
    """
    【辅助方法】提取动作草案所绑定的目标 Issue 或 Pull Request 的序号 (Number)。

    设计原理与 Python 语法技巧:
      - 多源回退策略 (Fallback Strategy):
        1. 优先从草案元数据 `metadata` 字典中提取显式传递的数字 (适配前端直接指定场景)。
        2. 若元数据未携带，则根据 `target_type` 动态从数据库实体中回查对应的 `number`。
      - `or` 运算符的短路取值: `a or b or c` 依次求值直到找到第一个 Truthy 值。
      - 强类型防崩处理: 通过 `try...except (TypeError, ValueError)` 确保即使传入非法格式也不会抛出未捕获异常。

    Args:
        db (Session): SQLAlchemy 数据库会话。
        draft (ActionDraft): 动作草案 ORM 实体。

    Returns:
        int | None: 解析出的目标 GitHub 实体序号（如 #123），若解析失败或未关联则返回 None。
    """
    # 步骤 1: 优先从元数据字典中提取
    meta = draft.meta or {}
    raw = meta.get("target_number") or meta.get("issue_number") or meta.get("pr_number")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    # 步骤 2: 若元数据未直接提供，通过 target_id 关联实体查询
    if draft.target_id and draft.target_type == "issue":
        issue = db.get(Issue, draft.target_id)
        return issue.number if issue else None
    if draft.target_id and draft.target_type in {"pull_request", "pr"}:
        pr = db.get(PullRequest, draft.target_id)
        return pr.number if pr else None
    return None


async def _execute_github_draft(db: Session, draft: ActionDraft, repo: Repository) -> dict:
    """
    【内部核心执行器】调用外部 GitHub API 执行已获人工批准的动作草案。

    安全设计与工程规范:
      - 凭证按需即时解密: `decrypt_token` 仅在准备发起网络请求的瞬间在内存中解密，绝不在持久层明文存储。
      - 客户端生命周期隔离: 基于解密后的 Token 和仓库专属 API Base URL 实例化异步客户端。
      - 多态动作分发 (Polymorphic Dispatch): 根据 `draft_type` 路由至不同的 GitHubClient 异步请求。

    Args:
        db (Session): SQLAlchemy 会话。
        draft (ActionDraft): 待执行的草案对象。
        repo (Repository): 关联的仓库对象（包含加密凭证与 GitHub 属主信息）。

    Returns:
        dict: GitHub API 返回的 JSON 响应数据。

    Raises:
        RuntimeError: 参数缺失或不支持的草案类型。
    """
    # 1. 内存中按需即时解密 GitHub 访问令牌 (AES-GCM)
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    client = GitHubClient(token=token, base_url=repo.api_base_url)
    
    draft_type = draft.draft_type.lower()
    issue_number = _target_number(db, draft)

    # 2. 评论发布 (Issue Comment / PR Comment)
    if draft_type in {"issue_comment", "pr_comment", "comment"}:
        if issue_number is None:
            raise RuntimeError("评论草稿缺少 issue/pr number")
        return await client.create_issue_comment(repo.owner, repo.name, issue_number, draft.content)

    # 3. 创建全新 Issue
    if draft_type == "create_issue":
        # 若未指定标题，智能提取正文第一行前 120 个字符作为 Issue 标题
        title = draft.title or (draft.content.splitlines()[0][:120] if draft.content else "DevFlow AI Issue")
        return await client.create_issue(repo.owner, repo.name, title, draft.content)

    # 4. 关闭 Issue
    if draft_type == "close_issue":
        if issue_number is None:
            raise RuntimeError("关闭 Issue 草稿缺少 issue number")
        return await client.close_issue(repo.owner, repo.name, issue_number)

    # 5. 打标签 (Labels)
    if draft_type in {"label", "add_label", "add_labels"}:
        if issue_number is None:
            raise RuntimeError("Label 草稿缺少 issue/pr number")
        labels = draft.meta.get("labels") if draft.meta else None
        if not isinstance(labels, list) or not labels:
            labels = [draft.title] if draft.title else []
        if not labels:
            raise RuntimeError("Label 草稿缺少 labels")
        return await client.add_issue_labels(repo.owner, repo.name, issue_number, [str(item) for item in labels])

    # 6. 本地/内部系统报告（无需调用外部 GitHub）
    if draft_type == "send_report":
        return {"status": "recorded", "message": "报告草稿已在 DevFlow AI 中标记为已发送，未接入外部消息渠道。"}

    raise RuntimeError(f"暂不支持执行草稿类型：{draft.draft_type}")


@router.post("", response_model=ActionDraftResponse)
async def create_action_draft(
    payload: ActionDraftCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> ActionDraft:
    """
    【API 接口】创建新的动作草案并记录审计日志。

    权限要求: `draft:create` (普通开发者或系统 Agent 身份)
    业务逻辑:
      1. 验证目标仓库是否存在。
      2. 实例化 `ActionDraft`，初始状态由数据库字段默认指定为 `pending_confirmation`。
      3. 写入操作审计日志 (AuditLog)，记录是谁创建了该草案及其请求参数。
      4. 提交事务并刷新实体返回。
    """
    # 验证仓库存在性
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    # 创建动作草案实体
    draft = ActionDraft(
        repo_id=payload.repo_id,
        draft_type=payload.draft_type,
        target_type=payload.target_type,
        target_id=payload.target_id,
        title=payload.title,
        content=payload.content,
        risk_level=payload.risk_level,
        created_by=user.id,
        meta=payload.metadata,
    )
    db.add(draft)

    # 写入审计日志（不可逆追踪）
    write_audit_log(
        db,
        user=user,
        repo_id=payload.repo_id,
        action="action_draft.create",
        target_type=payload.target_type,
        target_id=str(payload.target_id) if payload.target_id else None,
        request_json=payload.model_dump(mode="json"),
    )
    db.commit()
    db.refresh(draft)
    return draft


@router.get("", response_model=list[ActionDraftResponse])
async def list_action_drafts(
    repo_id: UUID | None = Query(default=None),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[ActionDraft]:
    """
    【API 接口】多条件筛选查询动作草案列表。

    查询参数:
      - `repo_id`: 可选，按所属仓库过滤。
      - `status`: 可选，按草案状态过滤 (如 `pending_confirmation`, `executed`, `cancelled`, `failed`)。
    """
    query = db.query(ActionDraft)
    if repo_id:
        query = query.filter(ActionDraft.repo_id == repo_id)
    if status:
        query = query.filter(ActionDraft.status == status)
    # 按创建时间倒序排列，限制返回最近 100 条
    return query.order_by(ActionDraft.created_at.desc()).limit(100).all()


@router.get("/audit-logs", response_model=list[AuditLogResponse])
async def list_audit_logs(
    repo_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("settings:manage")),
) -> list[AuditLog]:
    """
    【API 接口】查询系统安全审计日志列表。

    权限要求: `settings:manage` (仅管理员或高级权限用户可查看)
    """
    query = db.query(AuditLog)
    if repo_id:
        query = query.filter(AuditLog.repo_id == repo_id)
    return query.order_by(AuditLog.created_at.desc()).limit(100).all()


@router.get("/{draft_id}", response_model=ActionDraftResponse)
async def get_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> ActionDraft:
    """
    【API 接口】获取单个动作草案详情。
    """
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    return draft


@router.post("/{draft_id}/cancel", response_model=ActionDraftResponse)
async def cancel_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> ActionDraft:
    """
    【API 接口】取消待确认的动作草案。

    状态机流转约束:
      - 仅允许处于 `pending_confirmation` 状态的草案被取消。
      - 取消后状态变更为 `cancelled`，并写入操作审计日志。
    """
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以取消")

    # 变更状态为已取消
    draft.status = "cancelled"
    write_audit_log(db, user=user, repo_id=draft.repo_id, action="action_draft.cancel", target_type=draft.target_type, target_id=str(draft.id))
    db.commit()
    db.refresh(draft)
    return draft


@router.post("/{draft_id}/confirm", response_model=ActionDraftConfirmResponse)
async def confirm_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:approve")),
) -> dict:
    """
    【API 核心接口】人工审批确认并正式执行动作草案 (Human-in-the-Loop)。

    权限要求: `draft:approve` (需要拥有审批权限的用户)
    核心执行流程:
      1. 严格校验草案状态是否为 `pending_confirmation`。
      2. 提取所属仓库配置并尝试执行 `_execute_github_draft`。
      3. 成功流:
         - 更新草案状态为 `executed`，记录批准人 `confirmed_by` 与完成时间 `executed_at`。
         - 保存执行响应 `execution_result`，写入状态为 `success` 的审计日志。
      4. 失败流 (异常捕获防御):
         - 捕获任何第三方 API 异常，更新草案状态为 `failed`，记录错误详情 `error_message`。
         - 写入状态为 `failed` 的审计日志，防止异常吞没或事务挂起。
      5. 统一提交事务并返回执行结果与生成的审计日志 ID。
    """
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以执行")

    repo = db.get(Repository, draft.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    try:
        # 执行外部调用（如向 GitHub 发起真实请求）
        result = await _execute_github_draft(db, draft, repo)
        draft.status = "executed"
        draft.confirmed_by = user.id
        draft.executed_at = datetime.now(timezone.utc)
        draft.execution_result = result
        audit = write_audit_log(
            db,
            user=user,
            repo_id=draft.repo_id,
            action=f"action_draft.confirm.{draft.draft_type}",
            target_type=draft.target_type,
            target_id=str(draft.target_id or draft.id),
            status="success",
            request_json={"draft_id": str(draft.id), "draft_type": draft.draft_type},
            result_json=result if isinstance(result, dict) else {"result": result},
        )
    except Exception as exc:
        # 异常兜底保护：记录失败原因，确保审计日志完备
        draft.status = "failed"
        draft.confirmed_by = user.id
        draft.executed_at = datetime.now(timezone.utc)
        draft.error_message = str(exc)
        audit = write_audit_log(
            db,
            user=user,
            repo_id=draft.repo_id,
            action=f"action_draft.confirm.{draft.draft_type}",
            target_type=draft.target_type,
            target_id=str(draft.target_id or draft.id),
            status="failed",
            request_json={"draft_id": str(draft.id), "draft_type": draft.draft_type},
            result_json={"error": str(exc)},
        )

    # 提交事务并刷新返回
    db.commit()
    db.refresh(draft)
    db.refresh(audit)
    return {"draft": draft, "audit_id": audit.id}

