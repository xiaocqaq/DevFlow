"""后台异步定时任务调度器模块.

本模块负责在系统后台以独立协程运行周期性代码库数据同步任务（Periodic Repo Sync Loop），
自动拉取远端 GitHub 仓库最新的 Issue、PR 和 Actions 构建数据。
"""

import asyncio
import logging

from app.core.config import settings
from app.db.models import Repository
from app.db.session import SessionLocal
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository_by_id

logger = logging.getLogger(__name__)


async def periodic_repo_sync_loop() -> None:
    """后台仓库定时同步常驻循环协程.

    【Python 语法要点 - 后台常驻协程与手动会话生命周期】:
    1. 预热延迟：启动初期先 `await asyncio.sleep(5)` 释放控制权，确保数据库表结构补丁已完成初始化；
    2. 手动 ORM 会话管理：
       在非 HTTP 请求生命周期的后台协程中，无法使用 FastAPI 的 `Depends(get_db)` 自动注入，
       因此必须显式调用 `db = SessionLocal()`，并在 `finally` 块中调用 `db.close()` 确保连接归还池中；
    3. 异常捕获与隔离：遍历仓库同步时使用 `try ... except` 包裹单个仓库，避免单一仓库的网络失败中断整个轮询调度器；
    4. 防御性最小间隔：`max(settings.auto_sync_interval_seconds, 30)` 设定 30 秒下限，防止配置失误触发死循环或被 GitHub API 限流封禁。
    """
    await asyncio.sleep(5)
    while settings.auto_sync_enabled:
        db = SessionLocal()
        try:
            # SQLAlchemy in_() 对应 SQL 中的 `WHERE provider IN (...)`
            repos = db.query(Repository).filter(Repository.provider.in_(["github", "github_compatible"])).all()
            repo_ids = [repo.id for repo in repos]
        finally:
            db.close()

        for repo_id in repo_ids:
            try:
                await sync_repository_by_id(repo_id, RepoSyncRequest(limit=settings.auto_sync_limit))
            except Exception as exc:
                logger.warning("Periodic repo sync failed for %s: %s", repo_id, exc)

        await asyncio.sleep(max(settings.auto_sync_interval_seconds, 30))
