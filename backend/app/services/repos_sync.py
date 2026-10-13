"""GitHub 远端仓库数据增量同步服务模块.

本模块负责通过 GitHub REST API 批量拉取 Issue、Pull Request、变更文件、
代码审查行内评论（Review Comments）及 GitHub Actions 工作流构建记录与失败日志，
并利用 SQLAlchemy ORM 执行数据库 Upsert（更新或插入）操作。
"""

from datetime import datetime, timezone
from uuid import UUID

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import Issue, PRFile, PRReviewComment, PullRequest, Repository, WorkflowRun
from app.schemas.repos import RepoSyncRequest, RepoSyncResponse
from app.services.github.actions import get_workflow_run_logs, list_workflow_run_jobs, list_workflow_runs
from app.services.github.issues import list_issues
from app.services.github.pull_requests import list_pull_request_files, list_pull_request_review_comments, list_pull_requests


def parse_github_datetime(value: str | None) -> datetime | None:
    """解析 GitHub API 返回的 ISO 8601 格式 UTC 时间字符串.

    【Python 语法要点 - 字符串替换与时间解析】:
    - GitHub 返回时间格式如 '2026-10-13T08:00:00Z'；
    - Python `datetime.fromisoformat` 在 3.11 之前对后缀 'Z' 支持有限，
      将 'Z' 替换为 '+00:00' 可确保跨版本稳定解析为带 UTC 时区（Timezone-aware）的对象。

    Args:
        value: 原始时间字符串或 None。

    Returns:
        datetime | None: 解析后的 UTC datetime 对象。
    """
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def login_of(user: dict | None) -> str | None:
    """从 GitHub API 返回的用户字典中提取登录用户名（login 字段）.

    【Python 语法要点 - 字典安全访问】:
    利用 `user.get("login") if user else None` 防御式编程，避免 NoneType 访问 KeyError。
    """
    return user.get("login") if user else None


async def sync_repository(db: Session, repo: Repository, payload: RepoSyncRequest) -> RepoSyncResponse:
    """执行单个代码仓库与 GitHub 远端数据的同步主逻辑.

    【Python / SQLAlchemy 业务与语法要点】:
    1. 动态凭证解密：调用 `decrypt_token` 将加密存储的 GitHub Token 还原为明文请求头；
    2. Upsert（存在则更新，不存在则插入）模式：
       - `db.query(Model).filter(...).one_or_none()` 查询数据库现有记录；
       - 若为 None 则 `Model(...)` 实例化并 `db.add()`；
       - 若已存在则更新其 title/body/state/updated_at 等动态字段；
    3. 子表级联重建模式：
       - 针对 PRFile 和 PRReviewComment，由于远端可能被 force push 修改，
         先执行 `db.query(SubModel).filter(SubModel.pr_id == pr.id).delete()` 清理旧记录，
         再批量插入最新全量列表，避免产生脏数据；
    4. 事务生命周期：
       - `db.flush()`：将挂起的 SQL 推送到数据库获取主键或校验外键约束；
       - `db.commit()`：最终提交整个事务，原子化保证数据一致性。

    Args:
        db: 活跃的 SQLAlchemy 数据库会话。
        repo: 要同步的 Repository ORM 实例。
        payload: 同步请求参数（包含是否同步 Issue/PR/CI 及拉取条数上限）。

    Returns:
        RepoSyncResponse: 同步结果统计（包含各模块同步的记录数）。
    """
    if repo.provider not in {"github", "github_compatible"}:
        raise HTTPException(status_code=400, detail=f"{repo.provider} sync is not implemented yet")

    # 统计各维度已同步条数的计数器字典
    synced = {"issues": 0, "pull_requests": 0, "workflow_runs": 0}
    
    # 动态解密保存在数据库中的访问凭证
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    base_url = repo.api_base_url

    # ----------------- 1. 同步 GitHub Issues -----------------
    if payload.sync_issues:
        try:
            remote_issues = await list_issues(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=400, detail=f"GitHub issues sync failed: {exc.response.status_code} {exc.response.text}") from exc
        
        for item in remote_issues:
            # 根据仓库 ID 与 GitHub 远端 Issue 唯一 ID 查重
            issue = db.query(Issue).filter(Issue.repo_id == repo.id, Issue.github_issue_id == item.get("id")).one_or_none()
            if issue is None:
                issue = Issue(repo_id=repo.id, github_issue_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(issue)
            
            # 更新字段属性
            issue.number = item.get("number", issue.number)
            issue.title = item.get("title") or issue.title
            issue.body = item.get("body")
            issue.state = item.get("state") or "open"
            # 列表推导式提取标签名称并过滤空值
            issue.labels = [label.get("name") for label in item.get("labels", []) if label.get("name")]
            issue.author = login_of(item.get("user"))
            issue.assignees = [login_of(user) for user in item.get("assignees", []) if login_of(user)]
            issue.created_at = parse_github_datetime(item.get("created_at"))
            issue.updated_at = parse_github_datetime(item.get("updated_at"))
            issue.closed_at = parse_github_datetime(item.get("closed_at"))
            synced["issues"] += 1

    # ----------------- 2. 同步 Pull Requests 与评论 -----------------
    if payload.sync_pull_requests:
        try:
            remote_prs = await list_pull_requests(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=400, detail=f"GitHub pull requests sync failed: {exc.response.status_code} {exc.response.text}") from exc
        
        for item in remote_prs:
            pr = db.query(PullRequest).filter(PullRequest.repo_id == repo.id, PullRequest.github_pr_id == item.get("id")).one_or_none()
            if pr is None:
                pr = PullRequest(repo_id=repo.id, github_pr_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(pr)
                # flush 提前生成 pr.id 以供下方关联外键使用
                db.flush()
            
            pr.number = item.get("number", pr.number)
            pr.title = item.get("title") or pr.title
            pr.body = item.get("body")
            pr.state = item.get("state") or "open"
            pr.author = login_of(item.get("user"))
            pr.base_branch = (item.get("base") or {}).get("ref")
            pr.head_branch = (item.get("head") or {}).get("ref")
            pr.merged_at = parse_github_datetime(item.get("merged_at"))
            pr.created_at = parse_github_datetime(item.get("created_at"))
            pr.updated_at = parse_github_datetime(item.get("updated_at"))

            # 清理旧 PRFile 关联并拉取最新变动文件
            db.query(PRFile).filter(PRFile.pr_id == pr.id).delete()
            try:
                files = await list_pull_request_files(repo.owner, repo.name, pr.number, token=token, base_url=base_url)
            except httpx.HTTPStatusError:
                files = []
            for file_item in files:
                db.add(
                    PRFile(
                        pr_id=pr.id,
                        filename=file_item.get("filename") or "",
                        status=file_item.get("status") or "modified",
                        additions=file_item.get("additions") or 0,
                        deletions=file_item.get("deletions") or 0,
                        patch=file_item.get("patch"),
                    )
                )

            # 清理旧行内审查评论并重新拉取
            db.query(PRReviewComment).filter(PRReviewComment.pr_id == pr.id).delete()
            try:
                review_comments = await list_pull_request_review_comments(repo.owner, repo.name, pr.number, token=token, base_url=base_url)
            except httpx.HTTPStatusError:
                review_comments = []
            for comment_item in review_comments:
                db.add(
                    PRReviewComment(
                        pr_id=pr.id,
                        github_comment_id=comment_item.get("id"),
                        body=comment_item.get("body"),
                        path=comment_item.get("path"),
                        line=comment_item.get("line") or comment_item.get("original_line"),
                        author=login_of(comment_item.get("user")),
                        created_at=parse_github_datetime(comment_item.get("created_at")),
                        updated_at=parse_github_datetime(comment_item.get("updated_at")),
                    )
                )
            synced["pull_requests"] += 1

    # ----------------- 3. 同步 GitHub Actions CI 运行记录 -----------------
    if payload.sync_workflow_runs:
        try:
            remote_runs = await list_workflow_runs(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError:
            remote_runs = []
        
        for item in remote_runs:
            run = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id, WorkflowRun.github_run_id == item.get("id")).one_or_none()
            if run is None:
                run = WorkflowRun(repo_id=repo.id, github_run_id=item.get("id"), name=item.get("name") or "workflow")
                db.add(run)
            
            run.name = item.get("name") or run.name
            run.status = item.get("status") or "unknown"
            run.conclusion = item.get("conclusion")
            run.html_url = item.get("html_url")
            run.created_at = parse_github_datetime(item.get("created_at"))
            run.updated_at = parse_github_datetime(item.get("updated_at"))
            
            # 若包含 GitHub Run ID，进一步同步每个流水线内的各 Job 与 Step 运行明细
            if run.github_run_id:
                try:
                    jobs = await list_workflow_run_jobs(repo.owner, repo.name, run.github_run_id, token=token, base_url=base_url)
                    run.jobs = [
                        {
                            "id": job.get("id"),
                            "name": job.get("name"),
                            "status": job.get("status"),
                            "conclusion": job.get("conclusion"),
                            "html_url": job.get("html_url"),
                            "started_at": job.get("started_at"),
                            "completed_at": job.get("completed_at"),
                            "steps": [
                                {
                                    "name": step.get("name"),
                                    "status": step.get("status"),
                                    "conclusion": step.get("conclusion"),
                                    "number": step.get("number"),
                                }
                                for step in job.get("steps", [])
                            ],
                        }
                        for job in jobs
                    ]
                except httpx.HTTPStatusError:
                    run.jobs = run.jobs or []
            
            # 失败的流水线专门抓取构建错误日志（用于后续 CI 智能诊断 Agent）
            if run.conclusion == "failure" and run.github_run_id:
                try:
                    run.logs_text = await get_workflow_run_logs(repo.owner, repo.name, run.github_run_id, token=token, base_url=base_url)
                except httpx.HTTPStatusError:
                    run.logs_text = run.logs_text
            synced["workflow_runs"] += 1

    # 记录最后同步时间并原子化提交事务
    repo.last_sync_at = datetime.now(timezone.utc)
    db.flush()
    db.commit()
    return RepoSyncResponse(repo_id=repo.id, status="completed", synced=synced)


async def sync_repository_by_id(repo_id: UUID, payload: RepoSyncRequest) -> RepoSyncResponse | None:
    """根据仓库主键 UUID 独立创建会话并触发同步（常用于后台异步任务调用）.

    【Python 语法要点 - 局部依赖导入与会话保证】:
    - 局部导入 `from app.db.session import SessionLocal` 避免循环导入（Circular Import）；
    - `try ... finally: db.close()` 保证即使外部异常，数据库会话连接也必然归还连接池。
    """
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        # SQLAlchemy 2.0 推荐的按主键高效获取单个实例的方式: db.get(Model, pk)
        repo = db.get(Repository, repo_id)
        if repo is None:
            return None
        return await sync_repository(db, repo, payload)
    finally:
        db.close()
