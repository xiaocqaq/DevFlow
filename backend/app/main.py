"""FastAPI Web 应用程序入口与主工厂模块.

本模块负责构建 FastAPI 应用实例、配置跨域资源共享（CORS）中间件、
挂载各业务路由蓝图（Routers）、注册全局异常处理器，
并在生命周期钩子中执行数据库表结构自动初始化与后台常驻任务管理。
"""

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import inspect, text

from app.api.routes import (
    action_drafts,
    chat,
    ci,
    code_graph,
    conversations,
    evaluations,
    feedback,
    issues,
    knowledge,
    project_index,
    pull_requests,
    rag,
    repos,
    reports,
    search,
    skills,
    webhooks,
    workspaces,
)
from app.core.config import settings
from app.db.models import Base
from app.db.session import engine
from app.services.scheduler import periodic_repo_sync_loop
from app.services.rag.vector_store import MilvusUnavailableError


def _cors_origins() -> list[str]:
    """解析并返回合法的 CORS 跨域源白名单列表.

    【Python 语法要点 - 列表推导式 (List Comprehension)】:
    - `[origin.strip() for origin in ... if origin.strip()]`:
      1. 迭代逗号分隔的字符串配置；
      2. 去除两端空格；
      3. 利用条件表达式过滤掉空白字符串（避免白名单中出现空字符串引发安全漏洞）。

    Returns:
        list[str]: 允许进行跨域访问的 Origin 字符串列表。
    """
    return [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]


def _cors_origin_regex() -> str | None:
    """获取本地开发环境匹配的 CORS 跨域正则表达式.

    【Python 语法要点 - 正则表达式与环境分支】:
    - 若当前处于生产环境（production），返回 None 强制仅使用白名单匹配。
    - 开发模式下返回原始字符串（raw string，`r"..."`），利用正则表达式匹配
      `localhost` 或 `127.0.0.1` 任意端口号的前端本地调试服务。

    Returns:
        str | None: 正则表达式模式字符串，非开发环境返回 None。
    """
    if settings.app_env.lower() == "production":
        return None
    return r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


def create_app() -> FastAPI:
    """FastAPI 应用工厂函数.

    【Python 设计模式 - 应用工厂模式 (Application Factory Pattern)】:
    将应用的初始化、路由注册与中间件挂载封装在工厂函数中，便于测试用例中按需创建独立实例。

    Returns:
        FastAPI: 配置完毕的 FastAPI 核心应用实例。
    """
    app = FastAPI(title=settings.app_name, version="0.1.0")

    @app.exception_handler(MilvusUnavailableError)
    async def milvus_unavailable_handler(_request, exc: MilvusUnavailableError) -> JSONResponse:
        """全局异常处理器：拦截向量数据库不可用异常并优雅降级返回 503 状态码."""
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "service": "milvus", "status": "degraded"},
        )

    # 挂载 CORS 跨域资源共享中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_origin_regex=_cors_origin_regex(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册各模块路由
    app.include_router(repos.router, prefix="/api/repos", tags=["repos"])
    app.include_router(conversations.router, prefix="/api/repos", tags=["conversations"])
    app.include_router(workspaces.router, prefix="/api/workspaces", tags=["workspaces"])
    app.include_router(issues.router, prefix="/api", tags=["issues"])
    app.include_router(pull_requests.router, prefix="/api", tags=["pull_requests"])
    app.include_router(ci.router, prefix="/api", tags=["ci"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(feedback.router, prefix="/api/chat/feedback", tags=["chat_feedback"])
    app.include_router(action_drafts.router, prefix="/api/action-drafts", tags=["action_drafts"])
    app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    app.include_router(skills.router, prefix="/api/skills", tags=["skills"])
    app.include_router(knowledge.router, prefix="/api/repos", tags=["knowledge"])
    app.include_router(project_index.router, prefix="/api/repos", tags=["project_index"])
    app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
    app.include_router(code_graph.router, prefix="/api", tags=["code_graph"])
    app.include_router(evaluations.router, prefix="/api/evals", tags=["evaluations"])
    app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        """健康检查接口，供云平台探针或负载均衡器检测服务可用性."""
        return {"status": "ok", "app": settings.app_name}

    @app.on_event("startup")
    async def create_demo_tables() -> None:
        """应用启动钩子函数：执行数据表初态补丁并启动后台常驻异步协程任务.

        【Python 语法要点 - 事务上下文与异步任务】:
        - `with engine.begin() as connection:`:
          SQLAlchemy 连接事务上下文管理器，块内执行成功自动 commit，抛出异常自动 rollback。
        - `inspector = inspect(engine)`: SQLAlchemy 动态反射机制，检测数据表和列定义是否存在。
        - `asyncio.create_task(...)`: 将异步协程封装为后台 Task 丢给事件循环运行，不阻塞主服务启动流程。
        - `app.state`: FastAPI 应用级状态存储空间，可在整个应用生命周期中跨请求共享对象。
        """
        # 课程项目默认自动建表；生产风格流程会在后续加入 Alembic 迁移。
        Base.metadata.create_all(bind=engine)
        inspector = inspect(engine)
        if "users" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("users")}
            if "role" not in columns:
                with engine.begin() as connection:
                    connection.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR(80) DEFAULT 'owner'"))
        if "repositories" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("repositories")}
            with engine.begin() as connection:
                if "provider" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN provider VARCHAR(80) DEFAULT 'github'"))
                if "api_base_url" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN api_base_url TEXT"))
                if "clone_url" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN clone_url TEXT"))
                if "local_path" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN local_path TEXT"))
                if "checkout_mode" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN checkout_mode VARCHAR(80) DEFAULT 'managed'"))
                if "github_token_encrypted" not in columns:
                    connection.execute(text("ALTER TABLE repositories ADD COLUMN github_token_encrypted TEXT"))
        if "workflow_runs" in inspector.get_table_names():
            columns = {column["name"] for column in inspector.get_columns("workflow_runs")}
            if "jobs" not in columns:
                with engine.begin() as connection:
                    connection.execute(text("ALTER TABLE workflow_runs ADD COLUMN jobs JSON"))
        for table_name in ["code_symbols", "code_relations"]:
            if table_name in inspector.get_table_names():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                with engine.begin() as connection:
                    if "pr_id" not in columns:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN pr_id UUID"))
                    if "pr_number" not in columns:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN pr_number INTEGER"))
        for table_name in ["chat_messages", "chat_sessions", "agent_runs", "evidence_items"]:
            if table_name in inspector.get_table_names():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                if "conversation_id" not in columns:
                    with engine.begin() as connection:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN conversation_id UUID"))
        for table_name in ["thread_memories", "conversation_memories"]:
            if table_name in inspector.get_table_names():
                columns = {column["name"] for column in inspector.get_columns(table_name)}
                for column_name in ["facts", "tasks", "user_preferences", "repo_context", "citations"]:
                    if column_name not in columns:
                        with engine.begin() as connection:
                            connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} JSON"))
        if "documents" in inspector.get_table_names():
            document_columns = {column["name"] for column in inspector.get_columns("documents")}
            if "embedding" in document_columns:
                with engine.begin() as connection:
                    connection.execute(text("ALTER TABLE documents DROP COLUMN embedding"))
        from app.db.session import SessionLocal
        from app.services.conversations import backfill_all_default_conversations
        from app.services.code_analysis import purge_legacy_code_documents
        from app.services.rag.vector_store import delete_milvus_documents, milvus_runtime_status

        db = SessionLocal()
        try:
            backfill_all_default_conversations(db)
            if purge_legacy_code_documents(db):
                db.commit()
        finally:
            db.close()
        app.state.rag_vector_status = milvus_runtime_status()
        delete_milvus_documents(source_types=["code_file"], strict=False)
        if settings.auto_sync_enabled:
            app.state.repo_sync_task = asyncio.create_task(periodic_repo_sync_loop())

    @app.on_event("shutdown")
    async def stop_background_tasks() -> None:
        """应用关闭钩子函数：优雅取消后台正在执行的异步轮询任务."""
        task = getattr(app.state, "repo_sync_task", None)
        if task:
            task.cancel()

    return app


app = create_app()
