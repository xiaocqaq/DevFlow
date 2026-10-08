from collections.abc import Callable

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import AuditLog, User
from app.db.session import get_db

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "owner": {"repo:read", "repo:sync", "agent:run", "draft:create", "draft:approve", "eval:read", "settings:manage"},
    "maintainer": {"repo:read", "repo:sync", "agent:run", "draft:create", "draft:approve", "eval:read"},
    "developer": {"repo:read", "agent:run", "draft:create", "eval:read"},
    "viewer": {"repo:read", "eval:read"},
}


def ensure_demo_user(db: Session) -> User:
    user = db.query(User).order_by(User.created_at.asc()).first()
    if user is None:
        user = User(name="demo-owner", role="owner")
        db.add(user)
        db.commit()
        db.refresh(user)
    if not getattr(user, "role", None):
        user.role = "owner"
        db.commit()
        db.refresh(user)
    return user


def has_permission(user: User, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get((user.role or "viewer").lower(), set())


def require_permission(permission: str) -> Callable:
    def dependency(db: Session = Depends(get_db)) -> User:
        user = ensure_demo_user(db)
        if not has_permission(user, permission):
            raise HTTPException(status_code=403, detail=f"需要权限：{permission}")
        return user

    return dependency


def write_audit_log(
    db: Session,
    *,
    user: User | None,
    repo_id=None,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    status: str = "success",
    request_json: dict | None = None,
    result_json: dict | None = None,
) -> AuditLog:
    row = AuditLog(
        user_id=user.id if user else None,
        repo_id=repo_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        status=status,
        request_json=request_json or {},
        result_json=result_json or {},
    )
    db.add(row)
    db.flush()
    return row
