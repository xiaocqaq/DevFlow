# ==============================================================================
# 模块名称: Pull Request 智能代码审查与就绪度评估路由 (PR Review & Analysis API)
# 架构定位: Web 接口层 (API Routes Layer) / 核心工程提效模块
# 核心架构与技术亮点:
#   1. 规避 N+1 查询 (selectinload):
#      利用 SQLAlchemy 2.0 的 `selectinload(PullRequest.files)` 与 `selectinload(PullRequest.review_comments)`，
#      一次性通过高效的子查询预加载所有关联变更文件和历史评审评论，杜绝性能雪崩。
#   2. 6 步结构化审查规范 (Structured 6-Step Review Plan):
#      理解元数据 -> 扫描 Diff 敏感风险 -> 快照源码与知识检索 -> CI 失败信号检测 -> 历史评审与上下文碰撞 -> 综合生成决策。
#   3. 动态快照与代码知识图谱影响面分析 (Code Graph Impact Analysis):
#      协同 `worktree_manager` 准备 PR 分支快照，提取变更文件直接影响的 `CodeSymbol` 符号与 `CodeRelation` 拓扑依赖。
#   4. 实时双流 SSE (Server-Sent Events):
#      实时将审查计划步骤推进、工具执行结果、透明思考流及最终审查 Checklist 推送至前端。
# ==============================================================================

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.db.models import AnalysisResult, CodeRelation, CodeSymbol, Conversation, Document, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.pull_requests import PRAnalysisResponse, PullRequestResponse
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.chat_memory import ContextAssembler
from app.services.code_search import search_repository_code
from app.services.rag.retrieval import search_similar_documents
from app.services.worktree_manager import cleanup_pr_snapshots, prepare_pr_worktree

# 路由子模块
router = APIRouter()

# 项目知识文档数据源类型
PROJECT_DOC_TYPES = {"knowledge_file", "memory_note", "project_doc"}

# 引导 Agent 输出用户可审计的审查执行摘要
PUBLIC_REASONING_PROMPT = """
你是 DevFlow AI 的 PR 合并就绪度 Agent。请用中文流式输出一段“公开执行过程”，帮助用户理解你如何按计划审查当前 PR。
要求：
- 只输出面向用户的可审计分析摘要，不输出隐藏链路思维或私有推理链。
- 按计划说明你检查了 PR 目标、diff 风险、相关代码/文档、CI、Review 评论，并如何形成合入建议。
- 最终建议只能围绕：建议合入 / 修改后合入 / 暂缓 / 拒绝。
- 不要编造输入中没有的文件、CI、Issue、人员或评论。
- 不要输出最终 JSON。
"""


def _encode_sse(event: str, data: dict) -> str:
    """
    【协议封装】将事件流打包为标准 SSE (Server-Sent Events) 数据帧。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _trace_item(kind: str, title: str, content: str, status: str = "done") -> dict:
    """
    【追踪节点构造器】封装前端审查时间轴步骤卡片。
    """
    return {"kind": kind, "title": title, "content": content, "status": status}


def _document_evidence(items: list[dict], default_source_type: str) -> list[dict]:
    """
    【数据规范化】将检索项清洗转换为包含 source_type、title、snippet、score 的结构化字典。
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
    【证据去重器】根据 (source_type, source_id, title) 三元组唯一性去重，避免上下文冗余。
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
    【冷启动兜底】向量库未命中时直接读取仓库最新 4 篇规范文档。
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


def _conversation_evidence(db: Session, pr: PullRequest, conversation_id: UUID | None, query: str) -> list[dict]:
    """
    【会话上下文抽取】组装当前会话与该 PR 相关的讨论背景。
    """
    if not conversation_id:
        return []
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.repo_id != pr.repo_id:
        return []
    repo = db.get(Repository, pr.repo_id)
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


def _pr_payload(pr: PullRequest) -> dict:
    """
    【数据实体解包】将已通过 selectinload 预加载关联的 PullRequest 转换为干净的数据字典。
    """
    return {
        "id": str(pr.id),
        "number": pr.number,
        "title": pr.title,
        "body": pr.body,
        "state": pr.state,
        "author": pr.author,
        "base_branch": pr.base_branch,
        "head_branch": pr.head_branch,
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
        "files": [
            {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "patch": file.patch,
            }
            for file in pr.files
        ],
        "comments": [
            {"body": comment.body, "path": comment.path, "line": comment.line, "author": comment.author}
            for comment in pr.review_comments
        ],
    }


def _pr_review_plan() -> list[str]:
    """
    【规划驱动】返回标准 6 步代码审查执行大纲。
    """
    return [
        "理解 PR 标题、描述、变更文件和关联上下文。",
        "按模块拆分 diff 风险。",
        "读取相关源代码、测试文件和项目文档。",
        "检查 CI 状态和失败信号。",
        "检索团队约定、历史对话和 review comments。",
        "形成合入建议和 review comments。",
    ]


def _build_pr_analysis_context(db: Session, pr: PullRequest, conversation_id: UUID | None = None) -> tuple[str, dict, dict]:
    """
    【多维度上下文构建】聚合 Diff 特征、敏感文件、代码索引、CI 日志与历史 Issue。

    安全设计:
      - 敏感文件探测: 扫描文件名中是否包含 auth, security, migration, schema, payment 等高危关键词。
      - CI 失败关联: 抓取仓库最近失败的构建日志（前 260 字符），定位是否引入构建中断。
    """
    repo = db.get(Repository, pr.repo_id)
    file_names = [file.filename for file in pr.files]
    diff_text = "\n".join(f"{file.filename}\n{file.patch or ''}" for file in pr.files[:12])
    code_query = "\n".join(filter(None, [pr.title, pr.body or "", " ".join(file_names), diff_text[:5000]]))
    review_text = "\n".join(comment.body or "" for comment in pr.review_comments)
    semantic_query = "\n".join(filter(None, [pr.title, pr.body or "", review_text[:3000]]))

    # 1. 源码检索
    code_docs = _document_evidence(
        search_repository_code(repo, code_query, limit=6) if repo else [],
        "code",
    )

    # 2. 项目与架构规范文档
    project_docs = _document_evidence(
        search_similar_documents(
            db,
            str(pr.repo_id),
            semantic_query,
            limit=4,
            source_types=sorted(PROJECT_DOC_TYPES),
        ),
        "document",
    )
    if not project_docs:
        project_docs = _fallback_project_documents(db, pr.repo_id)

    # 3. 关联的 Issue
    related_issues = _document_evidence(
        search_similar_documents(db, str(pr.repo_id), semantic_query, "issue", 4),
        "issue",
    )

    # 4. 会话上下文
    chat_evidence = _conversation_evidence(db, pr, conversation_id, semantic_query)

    # 5. CI 运行状态与失败分析
    recent_runs = (
        db.query(WorkflowRun)
        .filter(WorkflowRun.repo_id == pr.repo_id)
        .order_by(WorkflowRun.created_at.desc())
        .limit(8)
        .all()
    )
    failed_runs = [run for run in recent_runs if run.conclusion == "failure"]

    # 6. 团队画像
    team_members = [
        {
            "name": (doc.meta or {}).get("name") or doc.title.replace("团队成员：", "").replace("Team member: ", ""),
            "role": (doc.meta or {}).get("role") or "",
            "strengths": (doc.meta or {}).get("strengths") or "",
            "techStack": (doc.meta or {}).get("tech_stack") or "",
        }
        for doc in db.query(Document).filter(Document.repo_id == pr.repo_id, Document.source_type == "team_member").all()
    ]

    total_additions = sum(file.additions or 0 for file in pr.files)
    total_deletions = sum(file.deletions or 0 for file in pr.files)
    # 敏感高危文件识别
    sensitive_files = [
        name
        for name in file_names
        if any(token in name.lower() for token in ["auth", "security", "migration", "schema", "payment", "permission", "config", "ci", "workflow"])
    ]

    context = {
        "repo_id": str(pr.repo_id),
        "plan": _pr_review_plan(),
        "diff_stats": {
            "files": len(pr.files),
            "additions": total_additions,
            "deletions": total_deletions,
            "sensitive_files": sensitive_files,
        },
        "code_references": code_docs,
        "project_documents": project_docs,
        "related_issues": related_issues,
        "conversation_evidence": chat_evidence,
        "review_comments": _pr_payload(pr)["comments"],
        "ci_summary": {
            "recent_runs": len(recent_runs),
            "failed_runs": len(failed_runs),
            "failed": [
                {"name": run.name, "conclusion": run.conclusion, "logs_text": (run.logs_text or "")[:260]}
                for run in failed_runs[:4]
            ],
        },
        "team_members": team_members,
    }
    counts = {
        "files": len(pr.files),
        "comments": len(pr.review_comments),
        "code": len(code_docs),
        "documents": len(project_docs),
        "issues": len(related_issues),
        "conversation": len(chat_evidence),
        "failed_ci": len(failed_runs),
        "team_members": len(team_members),
    }
    return code_query, context, counts


def _snapshot_summary(snapshot: dict) -> dict:
    """
    【快照精简】提炼快照核心状态字段。
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
    }


def _prepare_pr_snapshot_context(db: Session, pr: PullRequest, enabled: bool) -> dict:
    """
    【Worktree 快照调度】根据开关触发 git worktree 独立环境准备。
    """
    if not enabled:
        return {"status": "skipped", "message": "未请求 PR 分支快照。"}
    repo = db.get(Repository, pr.repo_id)
    if repo is None:
        return {"status": "missing_repo", "message": "未找到仓库。"}
    return prepare_pr_worktree(db, repo, pr)


def _pr_code_graph_impact(db: Session, pr: PullRequest) -> dict:
    """
    【代码知识图谱碰撞】查询该 PR 变更文件涉及的符号与函数依赖调用链。

    SQL 技巧与优化:
      - 仅查询与 `changed_paths` 相关的符号。
      - 使用 `or_(*relation_filters)` 动态匹配 `source_name` 或 `target_name` 位于受影响符号名集合中的依赖关系。
    """
    changed_paths = list(dict.fromkeys(file.filename for file in pr.files if file.filename))
    symbol_query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == pr.repo_id, CodeSymbol.pr_id == pr.id)
    if changed_paths:
        symbol_query = symbol_query.filter(CodeSymbol.path.in_(changed_paths))
    symbols = symbol_query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(30).all()
    symbol_names = [symbol.name for symbol in symbols]

    relation_query = db.query(CodeRelation).filter(CodeRelation.repo_id == pr.repo_id, CodeRelation.pr_id == pr.id)
    relation_filters = []
    if changed_paths:
        relation_filters.append(CodeRelation.path.in_(changed_paths))
    if symbol_names:
        relation_filters.extend([CodeRelation.source_name.in_(symbol_names), CodeRelation.target_name.in_(symbol_names)])
    if relation_filters:
        relation_query = relation_query.filter(or_(*relation_filters))
    relations = relation_query.limit(60).all()

    return {
        "changed_paths": changed_paths[:30],
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


def _augment_pr_context_with_snapshot(db: Session, pr: PullRequest, query: str, context: dict, counts: dict, snapshot: dict) -> None:
    """
    【快照与图谱证据注入】将 PR 专属 Worktree 快照检索结果与依赖图谱合流至主分析上下文中。
    """
    context["pr_snapshot"] = _snapshot_summary(snapshot)
    counts["snapshot_code"] = 0
    counts["impacted_symbols"] = 0
    counts["impacted_relations"] = 0
    if snapshot.get("status") != "ready":
        context["code_graph_impact"] = {"changed_paths": [file.filename for file in pr.files], "symbols": [], "relations": []}
        return

    repo = db.get(Repository, pr.repo_id)
    # 在 PR 独立快照路径下检索受影响代码段
    snapshot_docs = _document_evidence(
        search_repository_code(
            repo,
            query,
            limit=8,
            checkout_path=snapshot.get("snapshot_path"),
            metadata={
                "checkout_kind": "pr_snapshot",
                "pr_id": str(pr.id),
                "pr_number": pr.number,
                "branch": snapshot.get("branch"),
                "commit_sha": snapshot.get("commit_sha"),
            },
        )
        if repo and snapshot.get("snapshot_path")
        else [],
        "code",
    )
    # 合并并去重代码引用
    context["code_references"] = _dedupe_evidence([*snapshot_docs, *(context.get("code_references") or [])], 10)
    impact = _pr_code_graph_impact(db, pr)
    context["code_graph_impact"] = impact
    counts["snapshot_code"] = len(snapshot_docs)
    counts["impacted_symbols"] = len(impact["symbols"])
    counts["impacted_relations"] = len(impact["relations"])
    counts["code"] = len(context["code_references"])


def _persist_pr_analysis(
    db: Session,
    pr: PullRequest,
    result: dict,
    model_name: str,
    conversation_id: UUID | None,
    evidence_counts: dict,
) -> AnalysisResult:
    """
    【落库持久化】将 PR Review 最终结论、风险清单与合入建议存入 AnalysisResult 表。
    """
    analysis = AnalysisResult(
        target_type="pull_request",
        target_id=pr.id,
        analysis_type="pr_review",
        input_snapshot={
            "pr_id": str(pr.id),
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


def _public_reasoning_fallback(pr: PullRequest, context: dict, counts: dict) -> list[str]:
    """
    【公开推理备用话术】在无流式输出时平滑输出审查进度说明。
    """
    stats = context.get("diff_stats") or {}
    return [
        f"- 我先制定 Review 计划，再读取 PR #{pr.number} 的标题、描述、分支和 {counts['files']} 个变更文件。\n",
        f"- Diff 规模为 +{stats.get('additions', 0)} / -{stats.get('deletions', 0)}，其中敏感文件 {len(stats.get('sensitive_files') or [])} 个。\n",
        f"- 我检索了相关代码 {counts['code']} 条，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条；项目文档/记忆 {counts['documents']} 条、历史 Issue {counts['issues']} 条和会话证据 {counts['conversation']} 条。\n",
        f"- CI 侧读取到 {counts['failed_ci']} 个失败信号，并结合 {counts['comments']} 条 review comment 判断是否阻塞。\n",
        "- 最后将风险点、阻塞项、测试建议汇总为合入建议：建议合入、修改后合入、暂缓或拒绝。\n",
    ]


async def _stream_public_reasoning(agent: PRReviewAgent, pr: PullRequest, payload: dict, context: dict, counts: dict) -> AsyncIterator[str]:
    """
    【流式执行过程】通过异步生成器输出 Agent 审查推导文本。
    """
    streamed = False
    user_payload = {
        "pull_request": payload,
        "plan": context.get("plan") or [],
        "evidence_counts": counts,
        "context_preview": {
            "diff_stats": context.get("diff_stats"),
            "pr_snapshot": context.get("pr_snapshot"),
            "code_graph_impact": context.get("code_graph_impact"),
            "code_references": context.get("code_references", [])[:3],
            "project_documents": context.get("project_documents", [])[:3],
            "ci_summary": context.get("ci_summary"),
            "review_comments": context.get("review_comments", [])[:3],
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
    for line in _public_reasoning_fallback(pr, context, counts):
        await asyncio.sleep(0.06)
        yield line


async def _pr_analysis_sse(pr_id: UUID, conversation_id: UUID | None, db: Session) -> AsyncIterator[str]:
    """
    【SSE 核心生成器】端到端驱动 6 步审查计划，向客户端持续推送进度事件帧。
    """
    # 使用 selectinload 预加载关联文件和评论，杜绝 N+1
    pr = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        yield _encode_sse("error", {"message": "未找到 PR"})
        return

    agent = PRReviewAgent()
    payload = _pr_payload(pr)
    plan = _pr_review_plan()

    try:
        # 发送整体计划规划
        yield _encode_sse("trace", _trace_item("step", "制定 Review 计划", "\n".join(f"{index + 1}. {item}" for index, item in enumerate(plan)), "done"))
        yield _encode_sse("trace", _trace_item("step", "执行 1/6：理解 PR", f"读取 PR #{pr.number} 的标题、描述、分支、作者和变更文件列表。", "running"))
        
        # 准备专属快照
        snapshot = _prepare_pr_snapshot_context(db, pr, enabled=True)
        yield _encode_sse(
            "trace",
            _trace_item(
                "tool",
                "准备 PR 分支快照",
                str(snapshot.get("message") or f"状态：{snapshot.get('status')}；代码图符号：{snapshot.get('indexed_symbols') or 0}。"),
            ),
        )

        _query, context, counts = _build_pr_analysis_context(db, pr, conversation_id)
        _augment_pr_context_with_snapshot(db, pr, _query, context, counts, snapshot)
        context["plan"] = plan

        # 逐步上报审查进展
        yield _encode_sse("trace", _trace_item("tool", "执行 2/6：拆分 diff 风险", f"扫描 {counts['files']} 个文件，识别敏感文件 {len((context.get('diff_stats') or {}).get('sensitive_files') or [])} 个。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 3/6：读取代码和测试", f"找到 {counts['code']} 条代码线索，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条；项目文档/记忆线索 {counts['documents']} 条。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 4/6：检查 CI", f"读取最近 CI 信号，失败 run 数量：{counts['failed_ci']}。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 5/6：检索上下文", f"找到 {counts['issues']} 条历史 Issue、{counts['conversation']} 条会话证据、{counts['comments']} 条 review comment。"))

        # 开始流式透明审查推导
        yield _encode_sse("trace", _trace_item("thinking", "LLM 公开执行过程", "开始综合 PR diff、代码、CI、文档和 review comment。", "running"))
        async for chunk in _stream_public_reasoning(agent, pr, payload, context, counts):
            yield _encode_sse("thinking_delta", {"content": chunk})

        # 执行 Agent 终审并保存结果
        result = await agent.run(payload, context)
        analysis = _persist_pr_analysis(db, pr, result, agent.llm.model, conversation_id, counts)

        # 汇报合入裁决
        yield _encode_sse("trace", _trace_item("result", "执行 6/6：形成合入建议", f"结论：{result.get('merge_recommendation')}；风险点：{len(result.get('risk_points') or [])}；阻塞项：{len(result.get('blocking_issues') or [])}。"))
        yield _encode_sse("final", {"analysis_id": str(analysis.id), "result": result})
    except Exception as exc:
        db.rollback()
        yield _encode_sse("error", {"message": f"分析失败：{exc}"})


@router.get("/repos/{repo_id}/pull-requests", response_model=list[PullRequestResponse])
async def list_pull_requests(repo_id: UUID, db: Session = Depends(get_db)) -> list[PullRequest]:
    """
    【API 接口】查询指定仓库的所有 PR 列表（使用 selectinload 预加载关联文件和评论）。
    """
    return (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .order_by(PullRequest.number.desc())
        .all()
    )


@router.get("/pull-requests/{pr_id}", response_model=PullRequestResponse)
async def get_pull_request(pr_id: UUID, db: Session = Depends(get_db)) -> PullRequest:
    """
    【API 接口】查询单个 PR 的详情及其完整变更文件和评审评论。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    return pr


@router.post("/pull-requests/{pr_id}/analyze", response_model=PRAnalysisResponse)
async def analyze_pull_request(
    pr_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    use_pr_snapshot: bool = Query(default=True),
    use_worktree: bool | None = Query(default=None),
    db: Session = Depends(get_db),
) -> PRAnalysisResponse:
    """
    【API 接口 - 同步阻塞】对 PR 进行全量智能审查，返回合入建议、风险点、阻塞项与评审 Checklist。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    agent = PRReviewAgent()
    snapshot_enabled = use_pr_snapshot if use_worktree is None else use_worktree
    snapshot = _prepare_pr_snapshot_context(db, pr, snapshot_enabled)
    _query, context, counts = _build_pr_analysis_context(db, pr, conversation_id)
    _augment_pr_context_with_snapshot(db, pr, _query, context, counts, snapshot)
    result = await agent.run(_pr_payload(pr), context)
    analysis = _persist_pr_analysis(db, pr, result, agent.llm.model, conversation_id, counts)
    return PRAnalysisResponse(analysis_id=analysis.id, result=result)


@router.post("/pull-requests/{pr_id}/analyze/stream")
async def analyze_pull_request_stream(
    pr_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    【API 接口 - SSE 流式传输】实时推送 PR 6 步审查进展与思考轨迹。
    """
    return StreamingResponse(_pr_analysis_sse(pr_id, conversation_id, db), media_type="text/event-stream")


@router.post("/pull-requests/{pr_id}/snapshot")
async def prepare_pull_request_snapshot(pr_id: UUID, db: Session = Depends(get_db)) -> dict:
    """
    【API 接口】手动触发预检出 PR 专属 Worktree 快照。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    return _prepare_pr_snapshot_context(db, pr, enabled=True)


@router.post("/pull-requests/{pr_id}/worktree")
async def prepare_pull_request_worktree(pr_id: UUID, db: Session = Depends(get_db)) -> dict:
    """
    【API 接口兼容别名】触发 PR Worktree 准备。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    return _prepare_pr_snapshot_context(db, pr, enabled=True)


@router.post("/repos/{repo_id}/pr-snapshots/cleanup")
async def cleanup_pull_request_snapshots(
    repo_id: UUID,
    max_age_hours: int = Query(default=168, ge=1, le=24 * 90),
    db: Session = Depends(get_db),
) -> dict:
    """
    【运维接口】批量清理仓库超期的 PR 快照磁盘目录与数据库符号。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return cleanup_pr_snapshots(db, repo, max_age_hours=max_age_hours)


@router.post("/pull-requests/{pr_id}/review-checklist")
async def review_checklist(pr_id: UUID, db: Session = Depends(get_db)) -> dict:
    """
    【API 接口】快速获取该 PR 的推荐审查清单 (Checklist)。
    """
    response = await analyze_pull_request(pr_id, db=db)
    return {"pr_id": str(pr_id), "checklist": response.result.review_checklist}

