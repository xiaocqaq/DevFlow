# ==============================================================================
# 模块名称: CI 持续集成故障智能排查与根因分析路由 (CI Debug & Workflow Diagnostics API)
# 架构定位: Web 接口层 (API Routes Layer) / 自动化质量保障端点
# 核心架构与技术亮点:
#   1. CI 失败日志启发式挖掘与清洗 (Log Mining & Heuristics):
#      - `_first_error`: 基于滑动窗口与异常特征词 (Traceback, AssertionError, fatal, npm ERR)
#        准确定位导致崩溃的“首个关键错误块”，过滤成千上万行的无关编译日志噪音。
#      - `_related_files_from_logs`: 利用精准文件扩展名正则提取日志中提及的源码/测试文件。
#   2. CI 运行与 PR 分支快照自动关联 (PR Association & Snapshot):
#      自动反查触发该 CI 运行的关联 PR (`_related_pr_for_run`)，并调起 Worktree 快照和代码语法图谱。
#   3. 标准 6 步故障排查工作流 (6-Step CI Debug Workflow):
#      定位失败 job/step -> 提取关键错误块 -> 判断失败类型 -> 检索代码与 workflow 配置 ->
#      关联最近变更 -> 输出根因、修复建议与合并阻塞性判定 (is_merge_blocking)。
#   4. Server-Sent Events (SSE) 实时透明排查轨迹推送。
# ==============================================================================

import asyncio
import json
import re
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.db.models import AnalysisResult, CodeRelation, CodeSymbol, Conversation, Document, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.ci import CIAnalysisResponse, WorkflowRunResponse
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.chat_memory import ContextAssembler
from app.services.code_search import search_repository_code
from app.services.rag.indexing import sanitize_ci_log
from app.services.rag.retrieval import search_similar_documents
from app.services.worktree_manager import prepare_pr_worktree

# 路由子模块
router = APIRouter()

# 知识库文档类型过滤
PROJECT_DOC_TYPES = {"knowledge_file", "memory_note", "project_doc"}

# 提示词：引导 Agent 输出面向用户的结构化可解释性排查过程
PUBLIC_REASONING_PROMPT = """
你是 DevFlow AI 的 CI 根因分析 Agent。请用中文流式输出一段“公开排查过程”，帮助用户理解你如何按计划定位失败 CI。
要求：
- 只输出面向用户的可审计分析摘要，不输出隐藏链路思维或私有推理链。
- 按计划说明你检查了失败 job、失败 step、首个错误块、失败类型、相关代码/配置、最近 PR，并如何判断是否阻塞合并。
- 不要编造输入中没有的 job、step、文件、PR 或错误日志。
- 不要输出最终 JSON。
"""


def _encode_sse(event: str, data: dict) -> str:
    """
    【协议编码】构造符合规范的 SSE 消息帧。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _trace_item(kind: str, title: str, content: str, status: str = "done") -> dict:
    """
    【追踪卡片】封装排查链路前端时间线节点。
    """
    return {"kind": kind, "title": title, "content": content, "status": status}


def _document_evidence(items: list[dict], default_source_type: str) -> list[dict]:
    """
    【数据清洗】将多源检索项规整为标准证据字典。
    """
    evidence = []
    for item in items:
        snippet = str(item.get("snippet") or "").strip()
        title = str(item.get("title") or "").strip()
        if not snippet or not title:
            continue
        evidence.append(
            {
                "source_type": item.get("source_type") or default_source_type,
                "title": title,
                "snippet": snippet,
                "source_id": str(item.get("source_id") or item.get("id") or ""),
                "score": item.get("score"),
                "metadata": item.get("metadata") or {},
            }
        )
    return evidence


def _dedupe_evidence(items: list[dict], limit: int) -> list[dict]:
    """
    【证据去重】根据 (source_type, source_id, title) 唯一标识符剪除重复条目。
    """
    output = []
    seen = set()
    for item in items:
        key = (item.get("source_type"), item.get("source_id"), item.get("title"))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _fallback_project_documents(db: Session, repo_id: UUID) -> list[dict]:
    """
    【冷启动兜底】向量库无命中时加载最新的通用项目规范文档。
    """
    rows = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type.in_(PROJECT_DOC_TYPES))
        .order_by(Document.created_at.desc())
        .limit(4)
        .all()
    )
    return [
        {
            "source_type": doc.source_type,
            "title": doc.title,
            "snippet": doc.content[:260],
            "source_id": str(doc.source_id),
            "score": None,
            "metadata": doc.meta or {},
        }
        for doc in rows
    ]


def _conversation_evidence(db: Session, run: WorkflowRun, conversation_id: UUID | None, query: str) -> list[dict]:
    """
    【会话上下文抽取】提取对话中关于 CI 构建相关的讨论记录。
    """
    if not conversation_id:
        return []
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.repo_id != run.repo_id:
        return []
    repo = db.get(Repository, run.repo_id)
    assembled = ContextAssembler(db).assemble(repo, conversation, query, limit=8)
    evidence = _document_evidence(assembled.evidence, "conversation")
    if assembled.system_context.strip():
        evidence.append(
            {
                "source_type": "conversation",
                "title": f"会话上下文：{conversation.title}",
                "snippet": assembled.system_context[:320],
                "source_id": str(conversation.id),
                "score": None,
            }
        )
    return evidence[:5]


def _ci_plan() -> list[str]:
    """
    【规范化大纲】返回 CI 故障 6 步标准排查流程。
    """
    return [
        "找出失败 job 和失败 step。",
        "提取首个关键错误块。",
        "判断失败类型：依赖、测试、构建、权限、环境变量或未知。",
        "搜索相关代码、测试文件和配置文件。",
        "关联最近 PR 或变更模块。",
        "输出根因、修复步骤和阻塞判断。",
    ]


def _workflow_payload(run: WorkflowRun) -> dict:
    """
    【载荷转换】将 WorkflowRun ORM 实体解构为 Agent 可读的干净字典。
    """
    return {
        "id": str(run.id),
        "name": run.name,
        "status": run.status,
        "conclusion": run.conclusion,
        "jobs": run.jobs or [],
        "logs_text": run.logs_text,
        "html_url": run.html_url,
    }


def _failed_jobs(run: WorkflowRun) -> list[dict]:
    """
    【故障筛选】从多 Job 执行树中过滤出状态为 failure 的 Job 并提取其失败 Step。
    """
    return [
        {
            **job,
            "failed_steps": [step for step in job.get("steps", []) if step.get("conclusion") == "failure"],
        }
        for job in (run.jobs or [])
        if job.get("conclusion") == "failure"
    ]


def _first_error(logs: str | None) -> str | None:
    """
    【日志核心算法】基于特征词和滑动窗口，从冗长构建日志中提取首个致命错误上下文。

    算法原理:
      1. 逐行扫描日志，寻找特征标识词（error, failed, exception, traceback, assertionerror 等）。
      2. 命中后以当前行为中心，向前扩展 2 行、向后抓取 5 行，组成高价值错误上下文块。
      3. 最多截取 900 字符，杜绝超出 Token 限制。
    """
    if not logs or not logs.strip():
        return None
    lines = [line.strip() for line in logs.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(token in lowered for token in ["error", "failed", "failure", "exception", "traceback", "assertionerror", "npm err", "fatal"]):
            start = max(0, index - 2)
            end = min(len(lines), index + 5)
            return "\n".join(lines[start:end])[:900]
    return "\n".join(lines[:6])[:900]


def _related_files_from_logs(logs: str | None) -> list[str]:
    """
    【正则文件挖掘】从控制台堆栈输出中提取提及的代码文件和配置路径。

    匹配规则: 捕获常见代码与配置扩展名 (`.py`, `.ts`, `.json`, `.yml` 等) 并去除首尾标点。
    """
    if not logs:
        return []
    candidates = re.findall(r"[\w./\\-]+\.(?:py|ts|tsx|js|jsx|json|ya?ml|toml|ini|cfg|md)", logs)
    normalized = []
    for item in candidates:
        cleaned = item.strip("`'\"()[]{}:,;").replace("\\", "/")
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized[:8]


def _related_pr_for_run(db: Session, run: WorkflowRun) -> PullRequest | None:
    """
    【智能 PR 溯源】通过工作流日志和分支名称自动关联对应的 Pull Request。

    匹配策略:
      1. 加载仓库最近活跃的 8 个 PR。
      2. 检查 PR 的 `head_branch` 是否出现在 CI 任务名称或日志文本中。
      3. 若未完全匹配，默认关联最近更新的一个 PR 作为分析上下文。
    """
    candidates = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files))
        .filter(PullRequest.repo_id == run.repo_id)
        .order_by(PullRequest.updated_at.desc(), PullRequest.created_at.desc())
        .limit(8)
        .all()
    )
    if not candidates:
        return None
    logs = f"{run.name}\n{run.logs_text or ''}".lower()
    for pr in candidates:
        if pr.head_branch and pr.head_branch.lower() in logs:
            return pr
    return candidates[0]


def _snapshot_summary(snapshot: dict) -> dict:
    """
    【快照精炼】封装关联 PR 的 Worktree 状态信息。
    """
    return {
        "status": snapshot.get("status"),
        "message": snapshot.get("message"),
        "name": snapshot.get("snapshot_name") or "PR 分支快照",
        "branch": snapshot.get("branch"),
        "commit_sha": snapshot.get("commit_sha"),
        "pr_id": snapshot.get("pr_id"),
        "pr_number": snapshot.get("pr_number"),
        "indexed_symbols": snapshot.get("indexed_symbols"),
        "pull_request": snapshot.get("pull_request"),
    }


def _prepare_ci_pr_snapshot(db: Session, run: WorkflowRun, enabled: bool) -> dict:
    """
    【快照桥接】反查 PR 并准备专属 Worktree 快照。
    """
    if not enabled:
        return {"status": "skipped", "message": "未请求 PR 分支快照。"}
    repo = db.get(Repository, run.repo_id)
    if repo is None:
        return {"status": "missing_repo", "message": "未找到仓库。"}
    pr = _related_pr_for_run(db, run)
    if pr is None:
        return {"status": "missing_pr", "message": "未找到可关联的 PR。"}
    snapshot = prepare_pr_worktree(db, repo, pr)
    snapshot["pull_request"] = {
        "id": str(pr.id),
        "number": pr.number,
        "title": pr.title,
        "head_branch": pr.head_branch,
        "state": pr.state,
    }
    return snapshot


def _ci_code_graph_impact(db: Session, pr_id: UUID, related_files: list[str]) -> dict:
    """
    【图谱碰撞】查询日志中提及的文件在关联 PR 中涉及的符号与函数调用拓扑。
    """
    paths = list(dict.fromkeys(related_files))
    symbol_query = db.query(CodeSymbol).filter(CodeSymbol.pr_id == pr_id)
    if paths:
        symbol_query = symbol_query.filter(CodeSymbol.path.in_(paths))
    symbols = symbol_query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(30).all()
    symbol_names = [symbol.name for symbol in symbols]

    relation_query = db.query(CodeRelation).filter(CodeRelation.pr_id == pr_id)
    relation_filters = []
    if paths:
        relation_filters.append(CodeRelation.path.in_(paths))
    if symbol_names:
        relation_filters.extend([CodeRelation.source_name.in_(symbol_names), CodeRelation.target_name.in_(symbol_names)])
    if relation_filters:
        relation_query = relation_query.filter(or_(*relation_filters))
    relations = relation_query.limit(60).all()

    return {
        "related_files": paths[:30],
        "symbols": [
            {
                "name": symbol.name,
                "kind": symbol.kind,
                "path": symbol.path,
                "start_line": symbol.start_line,
                "branch": symbol.branch,
                "commit_sha": symbol.commit_sha,
            }
            for symbol in symbols
        ],
        "relations": [
            {
                "source": relation.source_name,
                "target": relation.target_name,
                "type": relation.relation_type,
                "path": relation.path,
                "line": (relation.meta or {}).get("line"),
            }
            for relation in relations
        ],
    }


def _augment_ci_context_with_snapshot(db: Session, run: WorkflowRun, query: str, context: dict, counts: dict, snapshot: dict) -> None:
    """
    【快照证据合流】将 PR 快照中的相关代码段与受影响符号融入 CI 分析上下文。
    """
    context["pr_snapshot"] = _snapshot_summary(snapshot)
    counts["snapshot_code"] = 0
    counts["impacted_symbols"] = 0
    counts["impacted_relations"] = 0
    if snapshot.get("status") != "ready" or not snapshot.get("pr_id"):
        context["code_graph_impact"] = {"related_files": context.get("related_files_from_logs") or [], "symbols": [], "relations": []}
        return

    repo = db.get(Repository, run.repo_id)
    snapshot_docs = _document_evidence(
        search_repository_code(
            repo,
            query,
            limit=8,
            checkout_path=snapshot.get("snapshot_path"),
            metadata={
                "checkout_kind": "pr_snapshot",
                "pr_id": str(snapshot["pr_id"]),
                "pr_number": snapshot.get("pr_number"),
                "branch": snapshot.get("branch"),
                "commit_sha": snapshot.get("commit_sha"),
            },
        )
        if repo and snapshot.get("snapshot_path")
        else [],
        "code",
    )
    context["code_references"] = _dedupe_evidence([*snapshot_docs, *(context.get("code_references") or [])], 10)
    pr_uuid = UUID(str(snapshot["pr_id"]))
    impact = _ci_code_graph_impact(db, pr_uuid, context.get("related_files_from_logs") or [])
    context["code_graph_impact"] = impact
    counts["snapshot_code"] = len(snapshot_docs)
    counts["impacted_symbols"] = len(impact["symbols"])
    counts["impacted_relations"] = len(impact["relations"])
    counts["code"] = len(context["code_references"])


def _build_ci_analysis_context(db: Session, run: WorkflowRun, conversation_id: UUID | None = None) -> tuple[str, dict, dict]:
    """
    【排查上下文构建器】聚合失败 Job、首个错误块、日志涉事文件、源码引用与 Workflow 配置文件。
    """
    repo = db.get(Repository, run.repo_id)
    first_error = _first_error(run.logs_text)
    failed_jobs = _failed_jobs(run)
    failed_steps = [
        step
        for job in failed_jobs
        for step in job.get("failed_steps", [])
        if isinstance(step, dict)
    ]
    related_files = _related_files_from_logs(run.logs_text)

    # 构造检索代码与语义向量的查询串
    code_query = "\n".join(
        filter(None, [run.name, run.conclusion or "", first_error or "", run.logs_text[:5000] if run.logs_text else ""])
    )
    # 脱敏清洗日志（移除 IP、密钥格式字符串）后用于语义向量检索
    semantic_query = sanitize_ci_log(
        "\n".join(filter(None, [run.name, first_error or "", run.logs_text[:3000] if run.logs_text else ""]))
    )

    # 1. 源码检索
    code_docs = _document_evidence(
        search_repository_code(repo, code_query, limit=6) if repo else [],
        "code",
    )

    # 2. 知识库与项目运维文档
    project_docs = _document_evidence(
        search_similar_documents(
            db,
            str(run.repo_id),
            semantic_query,
            limit=4,
            source_types=sorted(PROJECT_DOC_TYPES),
        ),
        "document",
    )
    if not project_docs:
        project_docs = _fallback_project_documents(db, run.repo_id)

    # 3. GitHub Actions 配置文件检索 (.github/workflows)
    workflow_docs = _document_evidence(
        search_repository_code(repo, run.name, limit=4, relative_path=".github/workflows") if repo else [],
        "workflow",
    )

    # 4. 最近相关的 PR 列表
    recent_prs = [
        {
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "author": pr.author,
            "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
        }
        for pr in (
            db.query(PullRequest)
            .filter(PullRequest.repo_id == run.repo_id)
            .order_by(PullRequest.updated_at.desc())
            .limit(5)
            .all()
        )
    ]

    # 5. 会话上下文
    chat_evidence = _conversation_evidence(db, run, conversation_id, semantic_query)

    context = {
        "repo_id": str(run.repo_id),
        "plan": _ci_plan(),
        "failed_jobs": failed_jobs,
        "failed_steps": failed_steps,
        "first_error": first_error,
        "related_files_from_logs": related_files,
        "code_references": code_docs,
        "workflow_references": workflow_docs,
        "project_documents": project_docs,
        "recent_prs": recent_prs,
        "conversation_evidence": chat_evidence,
    }
    counts = {
        "failed_jobs": len(failed_jobs),
        "failed_steps": len(failed_steps),
        "code": len(code_docs),
        "workflow": len(workflow_docs),
        "documents": len(project_docs),
        "recent_prs": len(recent_prs),
        "conversation": len(chat_evidence),
        "related_files": len(related_files),
    }
    return code_query, context, counts


def _persist_ci_analysis(
    db: Session,
    run: WorkflowRun,
    result: dict,
    model_name: str,
    conversation_id: UUID | None,
    evidence_counts: dict,
) -> AnalysisResult:
    """
    【持久化】将 CI 根因、修复建议与阻塞判断入库保存。
    """
    analysis = AnalysisResult(
        target_type="workflow_run",
        target_id=run.id,
        analysis_type="ci_debug",
        input_snapshot={
            "run_id": str(run.id),
            "conversation_id": str(conversation_id) if conversation_id else None,
            "evidence_counts": evidence_counts,
        },
        result_json=result,
        model_name=model_name,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _public_reasoning_fallback(run: WorkflowRun, context: dict, counts: dict) -> list[str]:
    """
    【公开推理备用话术】在离线状态下汇报排查步骤。
    """
    first_error = context.get("first_error") or "日志中没有明确错误块"
    return [
        f"- 我先制定 CI 排查计划，再读取 workflow「{run.name}」的状态、结论、job 和日志。\n",
        f"- 当前找到失败 job {counts['failed_jobs']} 个、失败 step {counts['failed_steps']} 个，首个关键错误块是：{str(first_error)[:180]}。\n",
        f"- 我从日志中提取到 {counts['related_files']} 个相关文件，并检索代码/配置线索 {counts['code'] + counts['workflow']} 条，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条。\n",
        f"- 同时参考最近 PR {counts['recent_prs']} 条、项目文档/记忆 {counts['documents']} 条、会话证据 {counts['conversation']} 条。\n",
        "- 最后输出失败类型、根因、修复步骤，以及该 CI 是否阻塞合并。\n",
    ]


async def _stream_public_reasoning(agent: CIDebugAgent, run: WorkflowRun, payload: dict, context: dict, counts: dict) -> AsyncIterator[str]:
    """
    【流式执行过程】流式输出 CI 排查思考链路。
    """
    streamed = False
    user_payload = {
        "workflow_run": payload,
        "plan": context.get("plan") or [],
        "evidence_counts": counts,
        "context_preview": {
            "failed_jobs": context.get("failed_jobs", [])[:3],
            "first_error": context.get("first_error"),
            "related_files_from_logs": context.get("related_files_from_logs", [])[:5],
            "recent_prs": context.get("recent_prs", [])[:3],
            "pr_snapshot": context.get("pr_snapshot"),
            "code_graph_impact": context.get("code_graph_impact"),
            "code_references": context.get("code_references", [])[:3],
        },
    }
    async for chunk in agent.llm.chat_text_stream(
        PUBLIC_REASONING_PROMPT,
        [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
    ):
        streamed = True
        yield chunk
    if streamed:
        return
    for line in _public_reasoning_fallback(run, context, counts):
        await asyncio.sleep(0.06)
        yield line


async def _ci_analysis_sse(run_id: UUID, conversation_id: UUID | None, db: Session) -> AsyncIterator[str]:
    """
    【SSE 核心生成器】执行 CI 6 步故障诊断全过程并推送流式事件。
    """
    run = db.get(WorkflowRun, run_id)
    if run is None:
        yield _encode_sse("error", {"message": "未找到 workflow 运行"})
        return

    agent = CIDebugAgent()
    payload = _workflow_payload(run)
    plan = _ci_plan()

    try:
        # 发送排查大纲
        yield _encode_sse("trace", _trace_item("step", "制定排查计划", "\n".join(f"{index + 1}. {item}" for index, item in enumerate(plan)), "done"))
        yield _encode_sse("trace", _trace_item("step", "执行 1/6：定位失败 job/step", f"读取 workflow「{run.name}」的 jobs、steps、状态和结论。", "running"))
        
        # 关联 PR 快照
        snapshot = _prepare_ci_pr_snapshot(db, run, enabled=True)
        yield _encode_sse(
            "trace",
            _trace_item(
                "tool",
                "关联 PR 分支快照",
                str(snapshot.get("message") or f"状态：{snapshot.get('status')}；代码图符号：{snapshot.get('indexed_symbols') or 0}。"),
            ),
        )

        _query, context, counts = _build_ci_analysis_context(db, run, conversation_id)
        _augment_ci_context_with_snapshot(db, run, _query, context, counts, snapshot)
        context["plan"] = plan

        failed_job_names = [str(job.get("name")) for job in context.get("failed_jobs", []) if job.get("name")]
        yield _encode_sse("trace", _trace_item("tool", "执行 1/6：定位失败 job/step", f"失败 job：{', '.join(failed_job_names[:3]) or '未提供'}；失败 step 数：{counts['failed_steps']}。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 2/6：提取首个错误块", str(context.get("first_error") or "日志中没有提取到明确 error 块。")))
        yield _encode_sse("trace", _trace_item("tool", "执行 3/6：判断失败类型", "根据日志关键词、失败命令和 job/step 名称进行分类。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 4/6：搜索代码和配置", f"找到 {counts['code']} 条代码线索，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条；workflow 配置线索 {counts['workflow']} 条、日志文件路径 {counts['related_files']} 个。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 5/6：关联最近 PR", f"读取最近 PR {counts['recent_prs']} 条，会话证据 {counts['conversation']} 条。"))

        # 流式透明推导演示
        yield _encode_sse("trace", _trace_item("thinking", "LLM 公开排查过程", "开始综合失败日志、代码配置、最近 PR 和项目文档。", "running"))
        async for chunk in _stream_public_reasoning(agent, run, payload, context, counts):
            yield _encode_sse("thinking_delta", {"content": chunk})

        # 执行 Agent 诊断
        result = await agent.run(payload, context)
        analysis = _persist_ci_analysis(db, run, result, agent.llm.model, conversation_id, counts)

        # 汇报根因判定
        yield _encode_sse("trace", _trace_item("result", "执行 6/6：输出根因和阻塞判断", f"失败类型：{result.get('failure_type')}；阻塞合并：{'是' if result.get('is_merge_blocking') else '否'}。"))
        yield _encode_sse("final", {"analysis_id": str(analysis.id), "result": result})
    except Exception as exc:
        db.rollback()
        yield _encode_sse("error", {"message": f"分析失败：{exc}"})


@router.get("/repos/{repo_id}/workflow-runs", response_model=list[WorkflowRunResponse])
async def list_workflow_runs(repo_id: UUID, db: Session = Depends(get_db)) -> list[WorkflowRun]:
    """
    【API 接口】查询指定仓库的所有 CI 工作流运行记录（按创建时间倒序）。
    """
    return db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).order_by(WorkflowRun.created_at.desc()).all()


@router.get("/workflow-runs/{run_id}", response_model=WorkflowRunResponse)
async def get_workflow_run(run_id: UUID, db: Session = Depends(get_db)) -> WorkflowRun:
    """
    【API 接口】获取单个 CI 工作流运行记录详情。
    """
    run = db.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到 workflow 运行")
    return run


@router.get("/workflow-runs/{run_id}/failed-jobs")
async def get_failed_jobs(run_id: UUID, db: Session = Depends(get_db)) -> dict:
    """
    【API 接口】快速筛选当前工作流中所有状态为 failure 的 Job 及具体失败 Steps。
    """
    run = db.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到 workflow 运行")
    failed_jobs = [
        {
            **job,
            "failed_steps": [step for step in job.get("steps", []) if step.get("conclusion") == "failure"],
        }
        for job in (run.jobs or [])
        if job.get("conclusion") == "failure"
    ]
    return {"run_id": str(run.id), "workflow": run.name, "failed_jobs": failed_jobs}


@router.post("/workflow-runs/{run_id}/analyze", response_model=CIAnalysisResponse)
async def analyze_workflow_run(
    run_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    use_pr_snapshot: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> CIAnalysisResponse:
    """
    【API 接口 - 同步阻塞】对 CI 失败运行进行全量诊断，返回错误根因、修复计划与阻塞性判断。
    """
    run = db.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到 workflow 运行")
    agent = CIDebugAgent()
    snapshot = _prepare_ci_pr_snapshot(db, run, enabled=use_pr_snapshot)
    _query, context, counts = _build_ci_analysis_context(db, run, conversation_id)
    _augment_ci_context_with_snapshot(db, run, _query, context, counts, snapshot)
    result = await agent.run(_workflow_payload(run), context)
    analysis = _persist_ci_analysis(db, run, result, agent.llm.model, conversation_id, counts)
    return CIAnalysisResponse(analysis_id=analysis.id, result=result)


@router.post("/workflow-runs/{run_id}/analyze/stream")
async def analyze_workflow_run_stream(
    run_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    【API 接口 - SSE 流式传输】通过 Server-Sent Events 实时推送 CI 诊断推导过程。
    """
    return StreamingResponse(_ci_analysis_sse(run_id, conversation_id, db), media_type="text/event-stream")

