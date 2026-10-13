# ==============================================================================
# 模块名称: Issue 智能分析与排查路由 (Issue Agent API & Streaming Analysis)
# 架构定位: Web 接口层 (API Routes Layer) / 智能体业务交互端点
# 核心业务与技术架构:
#   1. 多源证据融合 (Multi-Source Evidence Aggregation):
#      分析 Issue 时自动并行/串行聚合：
#        - 向量库中相似的历史 Issue（判重、关联历史方案）。
#        - 仓库源码上下文（基于关键词/ripgrep检索受影响代码段）。
#        - 团队画像与技能矩阵（精准建议 Issue 负责人）。
#        - 当前对话上下文记忆。
#   2. 渐进式实时推理流 (Streaming Public Reasoning via SSE):
#      通过 Server-Sent Events (text/event-stream) 分布式推送 Agent 分析轨迹 (`trace`)、
#      透明推理增量 (`thinking_delta`) 以及最终结构化分析报告 (`final`)。
#   3. 分析结果不可变持久化:
#      所有分析结论存档至 `AnalysisResult` 表，记录输入快照与所用模型，供后续溯源与评测。
# ==============================================================================

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.models import AnalysisResult, Conversation, Document, Issue, Repository
from app.db.session import get_db
from app.schemas.issues import IssueAnalysisResponse, IssueResponse
from app.services.agents.issue_agent import IssueAgent
from app.services.chat_memory import ContextAssembler
from app.services.code_search import search_repository_code
from app.services.rag.retrieval import search_similar_documents

# 路由子模块
router = APIRouter()

# 允许作为参考项目文档的数据源类型
PROJECT_DOC_TYPES = {"knowledge_file", "memory_note", "project_doc"}

# 引导 LLM 输出面向用户的可解释性分析摘要（非内部链式黑盒）
PUBLIC_REASONING_PROMPT = """
你是 DevFlow AI 的 Issue Agent。请用中文流式输出一段“公开分析过程”，帮助用户理解你如何分析当前 Issue。
要求：
- 只输出面向用户的可审计分析摘要，不输出隐藏链路思维或私有推理链。
- 简洁说明你检查了哪些证据、如何判断结论/优先级/复杂度/负责人。
- 不要编造输入中没有的文件、人员、Issue 或 PR。
- 不要输出最终 JSON。
"""


def _encode_sse(event: str, data: dict) -> str:
    """
    【协议编码器】将事件名与数据字典打包为标准 SSE (Server-Sent Events) 数据帧。

    格式规范:
      event: <event_name>\n
      data: <json_string>\n\n
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _document_evidence(items: list[dict], default_source_type: str) -> list[dict]:
    """
    【数据清洗与归一化】将不同检索源返回的异构数据统一格式化为标准证据字典列表。
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


def _fallback_project_documents(db: Session, repo_id: UUID) -> list[dict]:
    """
    【冷启动兜底】当向量检索未命中任何项目文档时，直接从数据库加载最新的 4 篇文档作为通用背景。
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


def _conversation_evidence(db: Session, issue: Issue, conversation_id: UUID | None, query: str) -> list[dict]:
    """
    【上下文集成】提取当前对话历史及记忆沉淀作为分析线索。
    """
    if not conversation_id:
        return []
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.repo_id != issue.repo_id:
        return []
    repo = db.get(Repository, issue.repo_id)
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


def _issue_payload(issue: Issue) -> dict:
    """
    【视图转换】提取 Issue 实体的核心业务属性并转化为标准 JSON 兼容字典。
    """
    return {
        "id": str(issue.id),
        "title": issue.title,
        "body": issue.body,
        "labels": issue.labels,
        "number": issue.number,
        "state": issue.state,
        "assignees": issue.assignees,
        "author": issue.author,
    }


def _build_issue_analysis_context(db: Session, issue: Issue, conversation_id: UUID | None = None) -> tuple[str, dict, dict]:
    """
    【核心上下文编排器】全方位搜集当前 Issue 的所有关联线索（向量判重、源码引用、项目背景、团队画像）。

    返回值说明:
      - `query`: 综合标题和正文的搜索串。
      - `context`: 包含 duplicate_candidates, code_references, project_documents, team_members 的大字典。
      - `counts`: 各类型证据命中条数统计（用于快照与监控打点）。
    """
    query = f"{issue.title}\n{issue.body or ''}"
    repo = db.get(Repository, issue.repo_id)

    # 1. 向量相似度检索：寻找相似历史 Issue（判重检测）
    similar_docs = search_similar_documents(db, str(issue.repo_id), query, "issue", 4)
    duplicates = [
        {
            "issue_id": item["source_id"],
            "title": item["title"],
            "score": max(0.0, min(1.0, float(item["score"]))),
        }
        for item in similar_docs
        if item["source_id"] != str(issue.id)
    ]

    # 2. 源码检索：根据 Issue 报错关键词搜寻本地工作区代码
    code_docs = _document_evidence(
        search_repository_code(repo, query, limit=4) if repo else [],
        "code",
    )

    # 3. 知识库与项目规范检索
    project_docs = _document_evidence(
        search_similar_documents(
            db,
            str(issue.repo_id),
            query,
            limit=4,
            source_types=sorted(PROJECT_DOC_TYPES),
        ),
        "document",
    )
    if not project_docs:
        project_docs = _fallback_project_documents(db, issue.repo_id)

    # 4. 当前会话上下文证据
    chat_evidence = _conversation_evidence(db, issue, conversation_id, query)

    # 5. 团队技能画像：根据文档中维护的开发者专长信息辅助建议负责人
    team_members = [
        {
            "name": (doc.meta or {}).get("name") or doc.title.replace("团队成员：", "").replace("Team member: ", ""),
            "role": (doc.meta or {}).get("role") or "",
            "strengths": (doc.meta or {}).get("strengths") or "",
            "techStack": (doc.meta or {}).get("tech_stack") or "",
        }
        for doc in db.query(Document).filter(Document.repo_id == issue.repo_id, Document.source_type == "team_member").all()
    ]

    context = {
        "repo_id": str(issue.repo_id),
        "duplicate_candidates": duplicates,
        "code_references": code_docs,
        "project_documents": project_docs,
        "conversation_evidence": chat_evidence,
        "team_members": team_members,
    }
    counts = {
        "duplicates": len(duplicates),
        "code": len(code_docs),
        "documents": len(project_docs),
        "conversation": len(chat_evidence),
        "team_members": len(team_members),
    }
    return query, context, counts


def _persist_issue_analysis(
    db: Session,
    issue: Issue,
    result: dict,
    model_name: str,
    conversation_id: UUID | None,
    evidence_counts: dict,
) -> AnalysisResult:
    """
    【持久化】将 Agent 分析出的结论、证据清单及输入快照存入 AnalysisResult 表。
    """
    analysis = AnalysisResult(
        target_type="issue",
        target_id=issue.id,
        analysis_type="issue_analysis",
        input_snapshot={
            "issue_id": str(issue.id),
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


def _trace_item(kind: str, title: str, content: str, status: str = "done") -> dict:
    """
    【辅助方法】构造前端执行链路时间线节点（Step/Tool/Thinking/Result）。
    """
    return {"kind": kind, "title": title, "content": content, "status": status}


def _public_reasoning_fallback(issue: Issue, context: dict) -> list[str]:
    """
    【离线降级话术】当 LLM 流式文本接口未能成功输出时，生成基于事实统计的透明分析过程。
    """
    duplicates = context.get("duplicate_candidates") or []
    code_refs = context.get("code_references") or []
    docs = context.get("project_documents") or []
    members = context.get("team_members") or []
    return [
        f"- 我先读取 Issue #{issue.number} 的标题、正文、标签和当前状态，确认它描述的是「{issue.title}」。\n",
        f"- 然后检索相似历史 Issue，找到 {len(duplicates)} 个候选；如果相似度很高，会优先建议合并到已有 Issue。\n",
        f"- 接着查看代码和项目文档证据：代码线索 {len(code_refs)} 条，文档/记忆线索 {len(docs)} 条，用于判断影响范围和复杂度。\n",
        f"- 最后结合团队画像，当前可参考成员 {len(members)} 人；负责人只会从已配置成员或明确方向中选择。\n",
        "- 这些公开分析步骤会汇总成结构化结论：结论、优先级、复杂度、负责人、证据、checklist 和可选草稿。\n",
    ]


async def _stream_public_reasoning(agent: IssueAgent, issue: Issue, payload: dict, context: dict) -> AsyncIterator[str]:
    """
    【异步生成器】流式输出面向用户的公开推理过程（优先调用模型流式，失败自动回退）。
    """
    streamed = False
    user_payload = {
        "issue": payload,
        "evidence_counts": {
            "duplicates": len(context.get("duplicate_candidates") or []),
            "code_references": len(context.get("code_references") or []),
            "project_documents": len(context.get("project_documents") or []),
            "conversation_evidence": len(context.get("conversation_evidence") or []),
            "team_members": len(context.get("team_members") or []),
        },
        "context_preview": {
            "duplicates": context.get("duplicate_candidates", [])[:3],
            "code_references": context.get("code_references", [])[:3],
            "project_documents": context.get("project_documents", [])[:3],
            "team_members": context.get("team_members", [])[:3],
        },
    }
    # 尝试模型流式吐字
    async for chunk in agent.llm.chat_text_stream(
        PUBLIC_REASONING_PROMPT,
        [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
    ):
        streamed = True
        yield chunk

    if streamed:
        return

    # 若未吐字，启动平滑回退
    for line in _public_reasoning_fallback(issue, context):
        await asyncio.sleep(0.06)
        yield line


async def _issue_analysis_sse(issue_id: UUID, conversation_id: UUID | None, db: Session) -> AsyncIterator[str]:
    """
    【SSE 核心生成器】端到端驱动 Issue 智能分析生命周期并按需向前端推送 SSE 帧。

    事件流生命周期:
      1. `trace` (step: reading) -> 读取基础数据
      2. `trace` (tools) -> 检索相似 Issue、代码与团队画像
      3. `thinking_delta` -> 实时流式输出 LLM 的分析思路
      4. `agent.run()` -> 生成高置信度 JSON 分析结果并保存入库
      5. `final` -> 发送最终分析结果与生成的 analysis_id
    """
    issue = db.get(Issue, issue_id)
    if issue is None:
        yield _encode_sse("error", {"message": "未找到 Issue"})
        return

    agent = IssueAgent()
    payload = _issue_payload(issue)

    try:
        # 发送起始追踪节点
        yield _encode_sse("trace", _trace_item("step", "读取 Issue", f"读取 Issue #{issue.number} 的标题、正文、标签、状态和已分配人员。", "running"))
        query, context, counts = _build_issue_analysis_context(db, issue, conversation_id)
        
        # 汇报证据搜集进展
        yield _encode_sse("trace", _trace_item("tool", "检索相似历史", f"找到 {counts['duplicates']} 个相似 Issue 候选。"))
        yield _encode_sse("trace", _trace_item("tool", "检索代码与文档", f"找到 {counts['code']} 条代码线索、{counts['documents']} 条项目文档/记忆线索。"))
        yield _encode_sse("trace", _trace_item("tool", "读取团队画像", f"读取 {counts['team_members']} 位团队成员，用于建议负责人。"))
        if counts["conversation"]:
            yield _encode_sse("trace", _trace_item("tool", "读取会话上下文", f"找到 {counts['conversation']} 条与当前会话相关的证据。"))

        # 开始流式透明思考
        yield _encode_sse("trace", _trace_item("thinking", "LLM 公开分析过程", "开始综合 Issue、证据和团队画像。", "running"))
        async for chunk in _stream_public_reasoning(agent, issue, payload, context):
            yield _encode_sse("thinking_delta", {"content": chunk})

        # 执行 Agent 结构化推导与结果落库
        result = await agent.run(payload, context)
        analysis = _persist_issue_analysis(db, issue, result, agent.llm.model, conversation_id, counts)

        # 发送完成事件与最终报告
        yield _encode_sse("trace", _trace_item("result", "生成结构化结论", f"结论：{result.get('conclusion')}；优先级：{result.get('priority')}；复杂度：{result.get('complexity')}；负责人：{result.get('suggested_owner')}。"))
        yield _encode_sse("final", {"analysis_id": str(analysis.id), "result": result})
    except Exception as exc:
        db.rollback()
        yield _encode_sse("error", {"message": f"分析失败：{exc}"})


@router.get("/repos/{repo_id}/issues", response_model=list[IssueResponse])
async def list_issues(repo_id: UUID, db: Session = Depends(get_db)) -> list[Issue]:
    """
    【API 接口】获取仓库的所有 Issue 列表（按编号倒序）。
    """
    return db.query(Issue).filter(Issue.repo_id == repo_id).order_by(Issue.number.desc()).all()


@router.get("/issues/{issue_id}", response_model=IssueResponse)
async def get_issue(issue_id: UUID, db: Session = Depends(get_db)) -> Issue:
    """
    【API 接口】获取单个 Issue 的详情。
    """
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="未找到 Issue")
    return issue


@router.post("/issues/{issue_id}/analyze", response_model=IssueAnalysisResponse)
async def analyze_issue(
    issue_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> IssueAnalysisResponse:
    """
    【API 接口 - 同步阻塞模式】对指定 Issue 进行全量分析并直接返回分析报告。
    """
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="未找到 Issue")
    agent = IssueAgent()
    _query, context, counts = _build_issue_analysis_context(db, issue, conversation_id)
    result = await agent.run(_issue_payload(issue), context)
    analysis = _persist_issue_analysis(db, issue, result, agent.llm.model, conversation_id, counts)
    return IssueAnalysisResponse(analysis_id=analysis.id, result=result)


@router.post("/issues/{issue_id}/analyze/stream")
async def analyze_issue_stream(
    issue_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    【API 接口 - SSE 流式模式】通过 Server-Sent Events 流式返回思考过程与分析阶段结论。
    """
    return StreamingResponse(_issue_analysis_sse(issue_id, conversation_id, db), media_type="text/event-stream")


@router.get("/issues/{issue_id}/similar")
async def similar_issues(issue_id: UUID, db: Session = Depends(get_db)) -> list[dict]:
    """
    【API 接口】基于向量特征检索与当前 Issue 语义相似的 Top-5 历史 Issue。
    """
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="未找到 Issue")
    return search_similar_documents(db, str(issue.repo_id), f"{issue.title}\n{issue.body or ''}", "issue", 5)

