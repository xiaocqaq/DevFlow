from datetime import date, datetime, time, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Issue, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.reports import WeeklyReportRequest, WeeklyReportResponse
from app.services.agents.report_agent import ReportAgent
from app.services.rag.vector_store import add_document

router = APIRouter()


def _range_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        datetime.combine(end_date, time.max, tzinfo=timezone.utc),
    )


def _normalize_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _in_range(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    normalized = _normalize_dt(value)
    return normalized is not None and start_at <= normalized <= end_at


def _issue_in_range(issue: Issue, start_at: datetime, end_at: datetime) -> bool:
    return any(_in_range(value, start_at, end_at) for value in [issue.created_at, issue.updated_at, issue.closed_at])


def _pr_in_range(pr: PullRequest, start_at: datetime, end_at: datetime) -> bool:
    return any(_in_range(value, start_at, end_at) for value in [pr.created_at, pr.updated_at, pr.merged_at])


@router.post("/weekly", response_model=WeeklyReportResponse)
async def weekly_report(payload: WeeklyReportRequest, db: Session = Depends(get_db)) -> dict:
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    start_at, end_at = _range_bounds(payload.start_date, payload.end_date)
    issues = [
        item
        for item in db.query(Issue).filter(Issue.repo_id == payload.repo_id).all()
        if _issue_in_range(item, start_at, end_at)
    ]
    prs = [
        item
        for item in db.query(PullRequest).filter(PullRequest.repo_id == payload.repo_id).all()
        if _pr_in_range(item, start_at, end_at)
    ]
    runs = [
        item
        for item in db.query(WorkflowRun).filter(WorkflowRun.repo_id == payload.repo_id).all()
        if _in_range(item.created_at, start_at, end_at) or _in_range(item.updated_at, start_at, end_at)
    ]
    merged_prs = [item for item in prs if item.merged_at]
    open_prs = [item for item in prs if item.state == "open"]
    failed_ci = [item for item in runs if item.conclusion == "failure"]
    result = await ReportAgent().run(
        {
            "issues": [{"number": item.number, "title": item.title, "state": item.state, "closed_at": item.closed_at} for item in issues],
            "pull_requests": [
                {"number": item.number, "title": item.title, "state": item.state, "merged_at": item.merged_at}
                for item in prs
            ],
            "workflow_runs": [{"name": item.name, "conclusion": item.conclusion} for item in runs],
            "metrics": {
                "issues": len(issues),
                "pull_requests": len(prs),
                "merged_prs": len(merged_prs),
                "open_prs": len(open_prs),
                "failed_ci": len(failed_ci),
            },
        },
        {
            "repo_id": str(payload.repo_id),
            "repo_name": repo.full_name,
            "start_date": str(payload.start_date),
            "end_date": str(payload.end_date),
        },
    )
    add_document(
        db,
        payload.repo_id,
        "weekly_report",
        None,
        f"研发周报 {payload.start_date} - {payload.end_date}",
        result["report_markdown"],
        {
            "display_title": f"研发周报 {payload.start_date} - {payload.end_date}",
            "start_date": str(payload.start_date),
            "end_date": str(payload.end_date),
            "repo_name": repo.full_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


@router.post("/repo-health")
async def repo_health(payload: WeeklyReportRequest, db: Session = Depends(get_db)) -> dict:
    open_issues = db.query(Issue).filter(Issue.repo_id == payload.repo_id, Issue.state == "open").count()
    open_prs = db.query(PullRequest).filter(PullRequest.repo_id == payload.repo_id, PullRequest.state == "open").count()
    failed_ci = db.query(WorkflowRun).filter(WorkflowRun.repo_id == payload.repo_id, WorkflowRun.conclusion == "failure").count()
    return {
        "summary": "仓库健康度以 demo 指标估算，建议结合近期失败 CI 和高风险 PR 判断。",
        "metrics": {"open_issues": open_issues, "open_prs": open_prs, "failed_ci_runs": failed_ci, "merged_prs_7d": 0},
        "risks": ["存在失败 CI，需要排查"] if failed_ci else [],
    }
