# ==============================================================================
# 模块名称: 知识库、长期记忆中心与知识图谱路由 (Knowledge, Memory Hub & Graph API)
# 架构定位: Web 接口层 (API Routes Layer) / 项目知识与长期记忆管理
# 核心架构与设计模式:
#   1. 多态切片逻辑聚合算法 (`_group_documents`):
#      底层的分块存储将一篇长文档切为多个 `Document` 记录，本模块利用 `(source_type, source_id)` 二元组
#      字典将物理切片动态折叠聚合为完整的逻辑知识卡片 (`KnowledgeItemResponse`)。
#   2. 记忆萃取与人在回路审批 (Memory Candidate Review Workflow):
#      Agent 对话中萃取的关键事实先沉淀为 `MemoryCandidate` (pending)，人工审批通过 (`approve_candidate`) 后，
#      正式转入持久层记忆，并自动触发证据库与知识图谱的联动重建。
#   3. 确定性主键生成与 Upsert 模式 (`stable_source_id`):
#      团队成员画像等结构化数据基于 `(repo_id, source_type, identifier)` 派生确定性 UUID，
#      创建时先清理旧切片再写入，优雅实现无锁幂等更新。
#   4. 知识图谱 (Knowledge Graph) 子图下发与全量重建端点。
# ==============================================================================

import asyncio
import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document, Repository
from app.db.session import get_db
from app.schemas.knowledge import (
    KnowledgeGraphRebuildResponse,
    KnowledgeGraphResponse,
    KnowledgeItemResponse,
    KnowledgeUploadResponse,
    MemoryCandidateActionResponse,
    MemoryCandidateResponse,
    MemorySearchResponse,
    MemoryStatusResponse,
    MemoryNoteRequest,
    RecallEventResponse,
    TeamMemberRequest,
    TeamMemberResponse,
)
from app.services.knowledge_graph import knowledge_graph_payload, rebuild_knowledge_graph
from app.services.chat_memory import EvidenceStore
from app.services.memory_hub import (
    approve_memory_candidate,
    list_memory_candidates,
    list_recall_events,
    memory_status,
    record_recall_event,
    reject_memory_candidate,
)
from app.services.rag.chunking import chunk_document
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES, extract_document_text
from app.services.rag.document_processing import SUPPORTED_KNOWLEDGE_SUFFIXES
from app.services.rag.embeddings import embed_texts, embedding_provider_name
from app.services.rag.vector_store import DocumentPayload, add_document, add_documents, delete_document_group, stable_source_id

# 路由子模块
router = APIRouter()

# 用户可直接查看与操作的知识源类型白名单
USER_KNOWLEDGE_SOURCE_TYPES = [
    "memory_note",
    "knowledge_file",
    "team_member",
    "weekly_report",
    "project_overview",
    "project_doc",
    "project_manifest",
]
# 向量批处理切片批量大小
EMBEDDING_BATCH_SIZE = 64


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """
    【前置拦截器】校验指定仓库实体是否存在。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


def _clip(text: str, limit: int = 360) -> str:
    """
    【文本摘要截断】规整换行并截取前 limit 字符，超出补省略号。
    """
    text = (text or "").strip().replace("\r\n", "\n")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _safe_filename(filename: str) -> str:
    """
    【文件名安全过滤】去除文件名中危险路径字符，仅保留常规英文字符、数字与中文。
    """
    value = Path(filename or "upload.txt").name
    return re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", value).strip("._") or "upload.txt"


def _upload_root(repo_id: uuid.UUID) -> Path:
    """
    【路径管理】获取指定仓库的物理上传工作目录。
    """
    root = Path(settings.upload_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    path = root / str(repo_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_upload_text(filename: str, data: bytes) -> str:
    """
    【文本提取安全包装】调用底层解析器，发生损坏时平滑回退为空串而不崩溃进程。
    """
    try:
        return extract_document_text(filename, data)
    except Exception:
        return ""


def _embed_default_in_batches(texts: list[str]) -> list[list[float]]:
    """
    【分批向量化处理】按固定步长 `EMBEDDING_BATCH_SIZE` 切片调用嵌入服务，防单次请求 HTTP Payload 超限。
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(
            embed_texts(
                texts[start : start + EMBEDDING_BATCH_SIZE],
                dimensions=settings.embedding_dimensions,
            )
        )
    return vectors


def _group_documents(documents: list[Document]) -> list[KnowledgeItemResponse]:
    """
    【切片聚合算法】将打散的多条 Document 切片重新折叠为用户可见的逻辑知识项。

    Python 数据结构与算法解析:
      1. 使用字典 `groups: dict[tuple[str, uuid.UUID], list[Document]]` 按 `(source_type, source_id)` 归类。
      2. 针对每个分组内的切片列表，按元数据 `chunk_index` 和创建时间升序排列。
      3. 提取首个切片的元数据作为逻辑项的代表属性，合并前 2 个切片的内容生成预览。
      4. 最终按最新创建时间倒序返回列表。
    """
    groups: dict[tuple[str, uuid.UUID], list[Document]] = {}
    for doc in documents:
        groups.setdefault((doc.source_type, doc.source_id), []).append(doc)

    items: list[KnowledgeItemResponse] = []
    for (source_type, source_id), docs in groups.items():
        docs.sort(key=lambda item: ((item.meta or {}).get("chunk_index", 0), item.created_at))
        first = docs[0]
        content = "\n".join(doc.content for doc in docs[:2])
        metadata = dict(first.meta or {})
        metadata.setdefault("source_type", source_type)
        items.append(
            KnowledgeItemResponse(
                id=source_id,
                document_ids=[doc.id for doc in docs],
                source_type=source_type,
                title=metadata.get("display_title") or first.title,
                content_preview=_clip(content),
                metadata=metadata,
                chunk_count=len(docs),
                created_at=first.created_at,
            )
        )
    items.sort(key=lambda item: item.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items


def _team_from_document(doc: Document) -> TeamMemberResponse:
    """
    【实体反序列化】从存储团队画像的 Document 实体中提取元数据，还原为 TeamMemberResponse。
    """
    meta = doc.meta or {}
    return TeamMemberResponse(
        id=doc.source_id,
        name=str(meta.get("name") or doc.title.replace("团队成员：", "").replace("Team member: ", "")),
        role=str(meta.get("role") or ""),
        strengths=str(meta.get("strengths") or ""),
        techStack=str(meta.get("tech_stack") or meta.get("techStack") or ""),
        created_at=doc.created_at,
    )


@router.get("/{repo_id}/knowledge", response_model=list[KnowledgeItemResponse])
async def list_knowledge(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[KnowledgeItemResponse]:
    """
    【API 接口】查询指定仓库下所有类型的知识库条目列表（已按逻辑源聚合）。
    """
    _repo_or_404(db, repo_id)
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type.in_(USER_KNOWLEDGE_SOURCE_TYPES))
        .order_by(Document.created_at.desc())
        .all()
    )
    return _group_documents(docs)


@router.get("/{repo_id}/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    repo_id: uuid.UUID,
    q: str = Query(default=""),
    limit: int = Query(default=8, ge=1, le=30),
    db: Session = Depends(get_db),
) -> MemorySearchResponse:
    """
    【API 接口】工作区记忆检索并记录召回审计事件。

    特点:
      - 基于 `EvidenceStore.search` 进行混合搜索。
      - 记录不可篡改的 `record_recall_event` 召回日志，供系统评估检索命中率。
    """
    repo = _repo_or_404(db, repo_id)
    query = q.strip()
    results = EvidenceStore(db).search(repo.id, None, query, limit=limit) if query else []
    record_recall_event(
        db,
        repo_id=repo.id,
        conversation_id=None,
        tool_name="manual_memory_search",
        query=query,
        results=results,
        citations=[{"type": item["source_type"], "id": item["source_id"], "title": item["title"]} for item in results],
        scope="project",
        mode="hybrid",
        metadata={"limit": limit, "surface": "workspace_memory"},
    )
    db.commit()
    return MemorySearchResponse(query=query, results=results, result_count=len(results))


@router.get("/{repo_id}/memory/recall-events", response_model=list[RecallEventResponse])
async def get_recall_events(
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=30, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[RecallEventResponse]:
    """
    【API 接口】获取记忆召回审计事件列表。
    """
    _repo_or_404(db, repo_id)
    return [RecallEventResponse.model_validate(item) for item in list_recall_events(db, repo_id=repo_id, conversation_id=conversation_id, limit=limit)]


@router.get("/{repo_id}/memory/status", response_model=MemoryStatusResponse)
async def get_memory_status(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryStatusResponse:
    """
    【API 接口】查询当前仓库的记忆中心统计健康度（待审批候选数、已确认数等）。
    """
    repo = _repo_or_404(db, repo_id)
    return MemoryStatusResponse.model_validate(memory_status(db, repo))


@router.get("/{repo_id}/memory/candidates", response_model=list[MemoryCandidateResponse])
async def get_memory_candidates(
    repo_id: uuid.UUID,
    status: str | None = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[MemoryCandidateResponse]:
    """
    【API 接口】查询 Agent 对话中自动提炼的待审核记忆候选列表。
    """
    _repo_or_404(db, repo_id)
    return [MemoryCandidateResponse.model_validate(item) for item in list_memory_candidates(db, repo_id=repo_id, status=status, limit=limit)]


@router.post("/{repo_id}/memory/candidates/{candidate_id}/approve", response_model=MemoryCandidateActionResponse)
async def approve_candidate(repo_id: uuid.UUID, candidate_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryCandidateActionResponse:
    """
    【API 接口 - 人在回路】人工批准 Agent 沉淀的记忆候选。

    副作用:
      1. 将候选状态标记为 approved。
      2. 转录为正式的永久记忆 Document。
      3. 自动触发 `EvidenceStore.rebuild_repo` 与 `rebuild_knowledge_graph` 更新图谱实体。
    """
    repo = _repo_or_404(db, repo_id)
    try:
        candidate, doc = approve_memory_candidate(db, repo, candidate_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="candidate_not_found") from None
    EvidenceStore(db).rebuild_repo(repo)
    rebuild_knowledge_graph(db, repo)
    db.commit()
    return MemoryCandidateActionResponse(id=candidate.id, status=candidate.status, document_id=doc.id if doc else None)


@router.post("/{repo_id}/memory/candidates/{candidate_id}/reject", response_model=MemoryCandidateActionResponse)
async def reject_candidate(repo_id: uuid.UUID, candidate_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryCandidateActionResponse:
    """
    【API 接口 - 人在回路】拒绝不准确或无价值的记忆候选，状态标记为 rejected。
    """
    repo = _repo_or_404(db, repo_id)
    try:
        candidate = reject_memory_candidate(db, repo, candidate_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="candidate_not_found") from None
    db.commit()
    return MemoryCandidateActionResponse(id=candidate.id, status=candidate.status)


@router.post("/{repo_id}/knowledge/notes", response_model=KnowledgeItemResponse)
async def create_memory_note(
    repo_id: uuid.UUID,
    payload: MemoryNoteRequest,
    db: Session = Depends(get_db),
) -> KnowledgeItemResponse:
    """
    【API 接口】手动新建一条记忆便签（即时向量化并增量重塑图谱）。
    """
    _repo_or_404(db, repo_id)
    content = payload.content.strip()
    title = payload.title.strip() or "未命名记忆"
    if not content and not title:
        raise HTTPException(status_code=400, detail="记忆标题或内容至少需要填写一项")
    source_id = uuid.uuid4()
    doc = add_document(
        db,
        repo_id,
        "memory_note",
        source_id,
        title,
        content or title,
        {"display_title": title, "created_by": "user"},
    )
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return _group_documents([doc])[0]


@router.post("/{repo_id}/knowledge/upload", response_model=KnowledgeUploadResponse)
async def upload_knowledge_file(
    repo_id: uuid.UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> KnowledgeUploadResponse:
    """
    【API 接口 - 同步切片模式】上传知识库文件并同步进行分块、分批 Embedding 与图谱重构。
    """
    repo = _repo_or_404(db, repo_id)
    data = await file.read(MAX_DOCUMENT_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded file is too large; max size is 12 MB")

    filename = _safe_filename(file.filename or "upload.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_KNOWLEDGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_KNOWLEDGE_SUFFIXES))
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Supported: {supported}")

    # 哈希秒传校验
    checksum = hashlib.sha256(data).hexdigest()
    existing = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type == "knowledge_file")
        .order_by(Document.created_at.asc())
        .all()
    )
    duplicate_docs = [doc for doc in existing if (doc.meta or {}).get("checksum") == checksum]
    if duplicate_docs:
        return KnowledgeUploadResponse(
            id=duplicate_docs[0].source_id,
            filename=str((duplicate_docs[0].meta or {}).get("filename") or filename),
            chunks=len(duplicate_docs),
            indexed=True,
            duplicate=True,
        )

    # 异步线程池提取文本
    text = await asyncio.to_thread(_extract_upload_text, filename, data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from this file")

    source_id = uuid.uuid4()
    stored_path = _upload_root(repo_id) / f"{source_id}-{filename}"
    chunks = await asyncio.to_thread(chunk_document, text, filename, 1800, 180)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document did not produce indexable chunks")

    # 构建统一 Payload
    payloads = [
        DocumentPayload(
            source_type="knowledge_file",
            source_id=source_id,
            title=f"{filename}#{index + 1}",
            content=str(chunk["content"]),
            metadata={
                **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                "display_title": filename,
                "filename": filename,
                "stored_path": str(stored_path),
                "content_type": file.content_type,
                "checksum": checksum,
                "file_size": len(data),
                "chunk_index": index,
                "chunk_chars": len(str(chunk["content"])),
                "total_chunks": len(chunks),
                "chunk_strategy": "structure_aware_v1",
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "embedding_provider": embedding_provider_name(),
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
            },
        )
        for index, chunk in enumerate(chunks)
    ]
    embedding_inputs = [f"{payload.title}\n{payload.content}" for payload in payloads]
    vectors = await asyncio.to_thread(_embed_default_in_batches, embedding_inputs)

    try:
        await asyncio.to_thread(stored_path.write_bytes, data)
        docs = add_documents(db, repo_id, payloads, commit=False, vectors=vectors)
        rebuild_knowledge_graph(db, repo)
    except Exception:
        db.rollback()
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return KnowledgeUploadResponse(id=source_id, filename=filename, chunks=len(docs), indexed=bool(docs))


@router.get("/{repo_id}/knowledge/graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    repo_id: uuid.UUID,
    center_type: str | None = Query(default=None),
    center_id: str | None = Query(default=None),
    depth: int = Query(default=1, ge=1, le=2),
    db: Session = Depends(get_db),
) -> KnowledgeGraphResponse:
    """
    【API 接口】查询知识图谱结构化网络（节点与关联边，支持聚焦中心节点并按深度探索子图）。
    """
    repo = _repo_or_404(db, repo_id)
    return KnowledgeGraphResponse.model_validate(
        knowledge_graph_payload(db, repo, center_type=center_type, center_id=center_id, depth=depth)
    )


@router.post("/{repo_id}/knowledge/graph/rebuild", response_model=KnowledgeGraphRebuildResponse)
async def rebuild_knowledge_graph_route(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeGraphRebuildResponse:
    """
    【API 接口】手动触发仓库知识图谱的全量重构（基于源码符号、文档、PR、Issue 关联关系）。
    """
    repo = _repo_or_404(db, repo_id)
    result = rebuild_knowledge_graph(db, repo)
    return KnowledgeGraphRebuildResponse(repo_id=repo.id, edges=result["edges"], nodes=result["nodes"])


@router.delete("/{repo_id}/knowledge/{source_id}")
async def delete_knowledge_item(
    repo_id: uuid.UUID,
    source_id: uuid.UUID,
    source_type: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    【API 接口】删除某条逻辑知识项（自动清理其包含的全部物理分块切片并刷新图谱）。
    """
    _repo_or_404(db, repo_id)
    count = delete_document_group(db, repo_id, source_id, source_type=source_type)
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return {"repo_id": str(repo_id), "source_id": str(source_id), "deleted": count}


@router.get("/{repo_id}/team-members", response_model=list[TeamMemberResponse])
async def list_team_members(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[TeamMemberResponse]:
    """
    【API 接口】查询团队所有成员的技能画像列表。
    """
    _repo_or_404(db, repo_id)
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type == "team_member")
        .order_by(Document.created_at.asc())
        .all()
    )
    return [_team_from_document(doc) for doc in docs]


@router.post("/{repo_id}/team-members", response_model=TeamMemberResponse)
async def create_team_member(
    repo_id: uuid.UUID,
    payload: TeamMemberRequest,
    db: Session = Depends(get_db),
) -> TeamMemberResponse:
    """
    【API 接口】创建或更新团队成员技能画像（采用确定性 source_id 实现优雅 Upsert）。
    """
    _repo_or_404(db, repo_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="团队成员姓名为必填项")
    content = "\n".join(
        [
            f"name: {name}",
            f"role: {payload.role.strip()}",
            f"strengths: {payload.strengths.strip()}",
            f"tech_stack: {payload.tech_stack.strip()}",
        ]
    )
    # 基于 name 派生确定性主键
    source_id = stable_source_id(repo_id, "team_member", name.lower())
    # 清理旧同名记录
    delete_document_group(db, repo_id, source_id, source_type="team_member", commit=False)
    # 写入新记录
    doc = add_document(
        db,
        repo_id,
        "team_member",
        source_id,
        f"团队成员：{name}",
        content,
        {
            "display_title": name,
            "name": name,
            "role": payload.role.strip(),
            "strengths": payload.strengths.strip(),
            "tech_stack": payload.tech_stack.strip(),
        },
    )
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return _team_from_document(doc)


@router.delete("/{repo_id}/team-members/{member_id}")
async def delete_team_member(repo_id: uuid.UUID, member_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    【API 接口】删除团队成员技能画像。
    """
    _repo_or_404(db, repo_id)
    count = delete_document_group(db, repo_id, member_id, source_type="team_member")
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return {"repo_id": str(repo_id), "member_id": str(member_id), "deleted": count}

