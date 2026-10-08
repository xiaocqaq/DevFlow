from datetime import datetime, time, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Issue, PullRequest, Repository, User, WorkflowRun, Workspace
from app.db.session import get_db
from app.schemas.workspaces import MultiRepoReportRequest, MultiRepoReportResponse, WorkspaceCreateRequest, WorkspaceResponse
from app.services.permissions import require_permission, write_audit_log
from app.services.rag.vector_store import add_document

router = APIRouter()


def _bounds(payload: MultiRepoReportRequest) -> tuple[datetime, datetime]:
    return (
        datetime.combine(payload.start_date, time.min, tzinfo=timezone.utc),
        datetime.combine(payload.end_date, time.max, tzinfo=timezone.utc),
    )


def _normalize(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _in_range(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    normalized = _normalize(value)
    return normalized is not None and start_at <= normalized <= end_at


def _repo_ids_from_payload(db: Session, payload: MultiRepoReportRequest) -> list[UUID]:
    repo_ids = list(payload.repo_ids)
    if payload.workspace_id:
        workspace = db.get(Workspace, payload.workspace_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail="未找到 workspace")
        repo_ids.extend(UUID(str(item)) for item in (workspace.repo_ids or []))
    seen: set[UUID] = set()
    output = []
    for repo_id in repo_ids:
        if repo_id not in seen:
            seen.add(repo_id)
            output.append(repo_id)
    return output


@router.post("", response_model=WorkspaceResponse)
async def create_workspace(
    payload: WorkspaceCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("settings:manage")),
) -> Workspace:
    known_repo_ids = {repo.id for repo in db.query(Repository).filter(Repository.id.in_(payload.repo_ids)).all()} if payload.repo_ids else set()
    missing = [str(repo_id) for repo_id in payload.repo_ids if repo_id not in known_repo_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"仓库不存在：{', '.join(missing)}")
    workspace = Workspace(
        name=payload.name,
        description=payload.description,
        repo_ids=[str(repo_id) for repo_id in payload.repo_ids],
        created_by=user.id,
    )
    db.add(workspace)
    write_audit_log(db, user=user, action="workspace.create", request_json=payload.model_dump(mode="json"))
    db.commit()
    db.refresh(workspace)
    return workspace


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[Workspace]:
    return db.query(Workspace).order_by(Workspace.updated_at.desc()).all()


@router.post("/multi-repo-report", response_model=MultiRepoReportResponse)
async def multi_repo_report(
    payload: MultiRepoReportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("agent:run")),
) -> dict:
    repo_ids = _repo_ids_from_payload(db, payload)
    if not repo_ids:
        raise HTTPException(status_code=400, detail="至少选择一个仓库")
    start_at, end_at = _bounds(payload)
    repos = db.query(Repository).filter(Repository.id.in_(repo_ids)).all()
    summaries = []
    totals = {"open_issues": 0, "open_prs": 0, "merged_prs": 0, "failed_ci": 0, "repos": len(repos)}
    for repo in repos:
        issues = db.query(Issue).filter(Issue.repo_id == repo.id).all()
        prs = db.query(PullRequest).filter(PullRequest.repo_id == repo.id).all()
        runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).all()
        ranged_issues = [item for item in issues if any(_in_range(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.closed_at])]
        ranged_prs = [item for item in prs if any(_in_range(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.merged_at])]
        ranged_runs = [item for item in runs if any(_in_range(value, start_at, end_at) for value in [item.created_at, item.updated_at])]
        open_issues = len([item for item in issues if item.state == "open"])
        open_prs = len([item for item in prs if item.state == "open"])
        merged_prs = len([item for item in ranged_prs if item.merged_at])
        failed_ci = len([item for item in ranged_runs if item.conclusion == "failure"])
        totals["open_issues"] += open_issues
        totals["open_prs"] += open_prs
        totals["merged_prs"] += merged_prs
        totals["failed_ci"] += failed_ci
        summaries.append(
            {
                "repo_id": str(repo.id),
                "repo_name": repo.full_name,
                "range_issues": len(ranged_issues),
                "range_prs": len(ranged_prs),
                "open_issues": open_issues,
                "open_prs": open_prs,
                "merged_prs": merged_prs,
                "failed_ci": failed_ci,
                "risk_level": "high" if failed_ci or open_prs > 5 else "medium" if open_issues > 5 else "low",
            }
        )
    risk_repos = [item for item in summaries if item["risk_level"] == "high"]
    lines = [
        f"# 多仓库研发周报 {payload.start_date} - {payload.end_date}",
        "",
        f"- 覆盖仓库：{len(repos)} 个",
        f"- Open Issues：{totals['open_issues']}",
        f"- Open PRs：{totals['open_prs']}",
        f"- 本周期 merged PRs：{totals['merged_prs']}",
        f"- 失败 CI：{totals['failed_ci']}",
        "",
        "## 仓库明细",
    ]
    for item in summaries:
        lines.append(
            f"- {item['repo_name']}：open issue {item['open_issues']}，open PR {item['open_prs']}，merged {item['merged_prs']}，失败 CI {item['failed_ci']}，风险 {item['risk_level']}"
        )
    lines.extend(["", "## 重点关注", *(f"- {item['repo_name']} 需要优先处理失败 CI 或 PR backlog。" for item in risk_repos)])
    if not risk_repos:
        lines.append("- 暂无高风险仓库，建议继续关注 PR 停留时间和评审积压。")
    report_markdown = "\n".join(lines) + "\n"
    for repo in repos:
        add_document(
            db,
            repo.id,
            "weekly_report",
            None,
            f"多仓库研发周报 {payload.start_date} - {payload.end_date}",
            report_markdown,
            {
                "display_title": f"多仓库研发周报 {payload.start_date} - {payload.end_date}",
                "start_date": str(payload.start_date),
                "end_date": str(payload.end_date),
                "workspace_id": str(payload.workspace_id) if payload.workspace_id else None,
                "repo_count": len(repos),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            commit=False,
        )
    write_audit_log(db, user=user, action="workspace.multi_repo_report", request_json=payload.model_dump(mode="json"), result_json=totals)
    db.commit()
    return {"report_markdown": report_markdown, "metrics": totals, "repo_summaries": summaries}
