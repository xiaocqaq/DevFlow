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

router = APIRouter()


def _target_number(db: Session, draft: ActionDraft) -> int | None:
    meta = draft.meta or {}
    raw = meta.get("target_number") or meta.get("issue_number") or meta.get("pr_number")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    if draft.target_id and draft.target_type == "issue":
        issue = db.get(Issue, draft.target_id)
        return issue.number if issue else None
    if draft.target_id and draft.target_type in {"pull_request", "pr"}:
        pr = db.get(PullRequest, draft.target_id)
        return pr.number if pr else None
    return None


async def _execute_github_draft(db: Session, draft: ActionDraft, repo: Repository) -> dict:
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    client = GitHubClient(token=token, base_url=repo.api_base_url)
    draft_type = draft.draft_type.lower()
    issue_number = _target_number(db, draft)
    if draft_type in {"issue_comment", "pr_comment", "comment"}:
        if issue_number is None:
            raise RuntimeError("评论草稿缺少 issue/pr number")
        return await client.create_issue_comment(repo.owner, repo.name, issue_number, draft.content)
    if draft_type == "create_issue":
        title = draft.title or (draft.content.splitlines()[0][:120] if draft.content else "DevFlow AI Issue")
        return await client.create_issue(repo.owner, repo.name, title, draft.content)
    if draft_type == "close_issue":
        if issue_number is None:
            raise RuntimeError("关闭 Issue 草稿缺少 issue number")
        return await client.close_issue(repo.owner, repo.name, issue_number)
    if draft_type in {"label", "add_label", "add_labels"}:
        if issue_number is None:
            raise RuntimeError("Label 草稿缺少 issue/pr number")
        labels = draft.meta.get("labels") if draft.meta else None
        if not isinstance(labels, list) or not labels:
            labels = [draft.title] if draft.title else []
        if not labels:
            raise RuntimeError("Label 草稿缺少 labels")
        return await client.add_issue_labels(repo.owner, repo.name, issue_number, [str(item) for item in labels])
    if draft_type == "send_report":
        return {"status": "recorded", "message": "报告草稿已在 DevFlow AI 中标记为已发送，未接入外部消息渠道。"}
    raise RuntimeError(f"暂不支持执行草稿类型：{draft.draft_type}")


@router.post("", response_model=ActionDraftResponse)
async def create_action_draft(
    payload: ActionDraftCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> ActionDraft:
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
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
    query = db.query(ActionDraft)
    if repo_id:
        query = query.filter(ActionDraft.repo_id == repo_id)
    if status:
        query = query.filter(ActionDraft.status == status)
    return query.order_by(ActionDraft.created_at.desc()).limit(100).all()


@router.get("/audit-logs", response_model=list[AuditLogResponse])
async def list_audit_logs(
    repo_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("settings:manage")),
) -> list[AuditLog]:
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
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以取消")
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
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以执行")
    repo = db.get(Repository, draft.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    try:
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
    db.commit()
    db.refresh(draft)
    db.refresh(audit)
    return {"draft": draft, "audit_id": audit.id}
