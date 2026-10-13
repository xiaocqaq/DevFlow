# ==============================================================================
# 模块名称: RAG 向量检索与文档知识库路由 (RAG & Knowledge Documents API)
# 架构定位: Web 接口层 (API Routes Layer) / RAG 数据流中枢
# 核心业务与技术架构:
#   1. 文档上传流式安全控制 (File Upload & Safe Storage):
#      - 读取配额限制 (`MAX_DOCUMENT_BYTES` 12MB) 防止大文件 OOM 攻击。
#      - 文件名正规化 (`_safe_filename`) 过滤非法路径字符。
#      - SHA-256 校验和幂等去重: 相同内容秒传，复用现有切片，避免重复向量化开销。
#   2. FastAPI BackgroundTasks 异步解耦:
#      文件落盘后即刻响应 HTTP 202 Accepted，后台线程执行长耗时的 OCR/PDF/Word 解析、
#      Chunking 分块及 Embedding 向量批处理入库。
#   3. RAG 检索参数调优沙箱 (Retrieval Testing Playground):
#      提供可视化召回测试端点，动态调试 Top-K、混合检索、Rerank 排序模型与得分阈值。
# ==============================================================================

import asyncio
import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    Document,
    KnowledgeSourceDocument,
    Repository,
    RetrievalTestRun,
)
from app.db.session import get_db
from app.schemas.rag import (
    KnowledgeBaseConfigUpdate,
    KnowledgeBaseResponse,
    KnowledgeChunkResponse,
    KnowledgeDocumentResponse,
    ModelCatalogResponse,
    RAGAnswerResponse,
    RAGQueryRequest,
    RAGReindexResponse,
    RAGSearchResponse,
    RetrievalTestHistoryResponse,
    RetrievalTestRequest,
    RetrievalTestResponse,
)
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES
from app.services.rag.document_processing import SUPPORTED_KNOWLEDGE_SUFFIXES, process_knowledge_document
from app.services.rag.embeddings import validate_embedding_contract
from app.services.rag.knowledge_base import (
    config_dict,
    get_or_create_config,
    model_catalog,
    record_retrieval_test,
    retrieve_knowledge,
    runtime_status,
)
from app.services.rag.qa import answer_question, build_sources, rag_status, reindex_documents
from app.services.rag.vector_store import delete_document_group

# 创建 RAG 专用路由
router = APIRouter()


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """
    【前置检查】校验仓库存在性，且排除单纯的独立知识库类型。
    """
    repo = db.get(Repository, repo_id)
    if repo is None or repo.provider == "knowledge_base":
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


def _safe_filename(filename: str) -> str:
    """
    【文件名安全过滤】去除文件名中可能包含的跨目录符号与非常规特殊字符。

    正则表达式解析:
      - 保留英文字母、数字、点号、下划线、减号及中文字符 `\\u4e00-\\u9fff`。
      - 其余字符一律替换为 `_`，防止路径遍历或终端乱码。
    """
    value = Path(filename or "upload.txt").name
    return re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", value).strip("._") or "upload.txt"


def _upload_root(repo_id: uuid.UUID) -> Path:
    """
    【路径管理】根据 repo_id 生成仓库专属的文件物理上传目录。
    """
    root = Path(settings.upload_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    path = root / str(repo_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_chunking(chunk_size: int, overlap: int) -> None:
    """
    【分块参数校验】严格要求切片重叠区间小于单切片长度，防止递归切片死循环。
    """
    if overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="切片重叠必须小于切片长度")


def _document_response(item: KnowledgeSourceDocument) -> KnowledgeDocumentResponse:
    """
    【Schema 组装】将原始文档模型转换为标准 API 响应对象。
    """
    return KnowledgeDocumentResponse(
        id=item.id,
        repo_id=item.repo_id,
        name=item.name,
        source_type=item.source_type,
        status=item.status,
        content_type=item.content_type,
        file_size=item.file_size,
        character_count=item.character_count,
        chunk_count=item.chunk_count,
        error_message=item.error_message,
        created_at=item.created_at,
        updated_at=item.updated_at,
        completed_at=item.completed_at,
    )


def _knowledge_base(db: Session, repo: Repository) -> KnowledgeBaseResponse:
    """
    【知识库概览聚合】聚合查询仓库关联的文档总数、切片总数、字符总数以及底层向量库运行时状态。

    SQL 优化技巧:
      - 使用 `func.coalesce(func.sum(...), 0)` 确保无文档时统计值为 0 而非 NULL。
      - 一次查询同时统计 `count` 与 `sum`，减少数据库往返延迟。
    """
    config = get_or_create_config(db, repo.id)
    document_count, character_count = (
        db.query(func.count(KnowledgeSourceDocument.id), func.coalesce(func.sum(KnowledgeSourceDocument.character_count), 0))
        .filter(KnowledgeSourceDocument.repo_id == repo.id)
        .one()
    )
    chunk_count = db.query(func.count(Document.id)).filter(Document.repo_id == repo.id).scalar() or 0
    return KnowledgeBaseResponse(
        id=repo.id,
        name=repo.name,
        full_name=repo.full_name,
        description=repo.description or "",
        provider=repo.provider or "github",
        document_count=int(document_count or 0),
        chunk_count=int(chunk_count),
        character_count=int(character_count or 0),
        config=config_dict(config),
        runtime=runtime_status(config),
        created_at=repo.created_at,
        updated_at=repo.updated_at,
    )


@router.get("/models", response_model=ModelCatalogResponse)
async def get_model_catalog() -> ModelCatalogResponse:
    """
    【API 接口】获取系统支持的 Embedding 模型与 LLM 模型目录清单及默认推荐配置。
    """
    return ModelCatalogResponse.model_validate(model_catalog())


@router.get("/repositories", response_model=list[KnowledgeBaseResponse])
async def list_rag_repositories(db: Session = Depends(get_db)) -> list[KnowledgeBaseResponse]:
    """
    【API 接口】查询所有已配置 RAG 知识库的仓库列表。
    """
    repos = (
        db.query(Repository)
        .filter(Repository.provider != "knowledge_base")
        .order_by(Repository.updated_at.desc())
        .all()
    )
    return [_knowledge_base(db, repo) for repo in repos]


@router.get("/repositories/{repo_id}", response_model=KnowledgeBaseResponse)
async def get_rag_repository(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeBaseResponse:
    """
    【API 接口】获取单个知识库的详细统计信息与分块配置。
    """
    return _knowledge_base(db, _repo_or_404(db, repo_id))


@router.patch("/repositories/{repo_id}/config", response_model=KnowledgeBaseResponse)
async def update_rag_config(
    repo_id: uuid.UUID,
    payload: KnowledgeBaseConfigUpdate,
    db: Session = Depends(get_db),
) -> KnowledgeBaseResponse:
    """
    【API 接口】动态更新知识库分块策略与向量化嵌入模型配置。

    校验与一致性保护:
      1. 校验 `chunk_size` 与 `chunk_overlap` 合法性。
      2. 校验 `validate_embedding_contract`: 检查厂商/模型名称/向量维度是否匹配。
    """
    repo = _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id, commit=False)
    changes = payload.model_dump(exclude_unset=True)
    next_size = int(changes.get("chunk_size", config.chunk_size))
    next_overlap = int(changes.get("chunk_overlap", config.chunk_overlap))
    _validate_chunking(next_size, next_overlap)

    try:
        validate_embedding_contract(
            str(changes.get("embedding_provider", config.embedding_provider)),
            str(changes.get("embedding_model", config.embedding_model)),
            int(changes.get("embedding_dimensions", config.embedding_dimensions)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    for key, value in changes.items():
        setattr(config, key, value)
    db.commit()
    db.refresh(repo)
    return _knowledge_base(db, repo)


@router.get("/repositories/{repo_id}/documents", response_model=list[KnowledgeDocumentResponse])
async def list_documents(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[KnowledgeDocumentResponse]:
    """
    【API 接口】获取指定知识库下所有上传的原始文档列表。
    """
    _repo_or_404(db, repo_id)
    items = (
        db.query(KnowledgeSourceDocument)
        .filter(KnowledgeSourceDocument.repo_id == repo_id)
        .order_by(KnowledgeSourceDocument.created_at.desc())
        .all()
    )
    return [_document_response(item) for item in items]


@router.post("/repositories/{repo_id}/documents", response_model=KnowledgeDocumentResponse, status_code=202)
async def upload_document(
    repo_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> KnowledgeDocumentResponse:
    """
    【API 核心接口】上传知识库文档（支持秒级幂等去重与后台异步解耦）。

    关键流程:
      1. 读取文件流并检查是否超过 `MAX_DOCUMENT_BYTES`（12MB）。
      2. 检查扩展名是否在 `SUPPORTED_KNOWLEDGE_SUFFIXES` 支持列表中。
      3. 计算文件 SHA-256 哈希值，若已存在相同哈希的已处理文档，直接返回既有记录 (秒传)。
      4. 登记 `KnowledgeSourceDocument` 实体，状态初始化为 `queued`。
      5. 使用 `asyncio.to_thread` 异步持久化到专属上传目录。
      6. 利用 `background_tasks.add_task(process_knowledge_document, document.id)`
         将后续文档提取、AST/Markdown分块与嵌入入库推入后台执行，接口直接返回 202 Accepted。
    """
    _repo_or_404(db, repo_id)
    # 限制读取大小防止 OOM 攻击
    data = await file.read(MAX_DOCUMENT_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="文件不能超过 12 MB")

    filename = _safe_filename(file.filename or "upload.txt")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_KNOWLEDGE_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"暂不支持 {suffix or '无扩展名'} 文件")

    # SHA-256 校验和幂等秒传
    checksum = hashlib.sha256(data).hexdigest()
    duplicate = (
        db.query(KnowledgeSourceDocument)
        .filter(
            KnowledgeSourceDocument.repo_id == repo_id,
            KnowledgeSourceDocument.checksum == checksum,
            KnowledgeSourceDocument.status != "failed",
        )
        .order_by(KnowledgeSourceDocument.created_at.desc())
        .first()
    )
    if duplicate is not None:
        return _document_response(duplicate)

    # 登记排队状态
    document = KnowledgeSourceDocument(
        repo_id=repo_id,
        name=filename,
        content_type=file.content_type,
        checksum=checksum,
        file_size=len(data),
        stored_path="pending",
        status="queued",
    )
    db.add(document)
    db.flush()

    stored_path = _upload_root(repo_id) / f"{document.id}-{filename}"
    try:
        # 线程池隔离执行物理文件落盘
        await asyncio.to_thread(stored_path.write_bytes, data)
        document.stored_path = str(stored_path)
        db.commit()
    except Exception:
        db.rollback()
        try:
            await asyncio.to_thread(stored_path.unlink, missing_ok=True)
        except OSError:
            pass
        raise

    db.refresh(document)
    # 异步非阻塞执行后台分块与向量化
    background_tasks.add_task(process_knowledge_document, document.id)
    return _document_response(document)


@router.get("/repositories/{repo_id}/documents/{document_id}/chunks", response_model=list[KnowledgeChunkResponse])
async def list_document_chunks(
    repo_id: uuid.UUID,
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[KnowledgeChunkResponse]:
    """
    【API 接口】查询某篇文档切片后的所有知识分块 (Chunks) 内容与字符统计。
    """
    _repo_or_404(db, repo_id)
    rows = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_id == document_id)
        .order_by(Document.created_at.asc())
        .all()
    )
    return [
        KnowledgeChunkResponse(
            id=row.id,
            position=int((row.meta or {}).get("chunk_index", index)) + 1,
            title=row.title,
            content=row.content,
            character_count=len(row.content),
        )
        for index, row in enumerate(rows)
    ]


@router.post("/repositories/{repo_id}/documents/{document_id}/retry", response_model=KnowledgeDocumentResponse)
async def retry_document(
    repo_id: uuid.UUID,
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> KnowledgeDocumentResponse:
    """
    【API 接口】对处理失败的文档重新入队重试分块与嵌入。
    """
    _repo_or_404(db, repo_id)
    document = db.get(KnowledgeSourceDocument, document_id)
    if document is None or document.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="未找到文档")
    document.status = "queued"
    document.error_message = None
    db.commit()
    db.refresh(document)
    background_tasks.add_task(process_knowledge_document, document.id)
    return _document_response(document)


@router.delete("/repositories/{repo_id}/documents/{document_id}")
async def delete_document(repo_id: uuid.UUID, document_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    【API 接口】级联删除文档：清理 Milvus/PostgreSQL 向量切片、本地磁盘物理文件并删除元数据。
    """
    _repo_or_404(db, repo_id)
    source = db.get(KnowledgeSourceDocument, document_id)
    if source is None or source.repo_id != repo_id:
        raise HTTPException(status_code=404, detail="未找到文档")
    # 清理向量切片
    delete_document_group(db, repo_id, document_id, source_type="knowledge_file", commit=False)
    # 删除物理文件
    path = Path(source.stored_path)
    if path.exists() and path.is_file():
        path.unlink()
    db.delete(source)
    db.commit()
    return {"id": str(document_id), "deleted": True}


@router.post("/repositories/{repo_id}/retrieval-tests", response_model=RetrievalTestResponse)
async def run_retrieval_test(
    repo_id: uuid.UUID,
    payload: RetrievalTestRequest,
    db: Session = Depends(get_db),
) -> RetrievalTestResponse:
    """
    【API 接口】RAG 召回调试沙箱：在指定参数（Top-K、重排、阈值、元数据过滤）下执行一次检索测试并记录指标。
    """
    _repo_or_404(db, repo_id)
    items, trace, duration_ms = await retrieve_knowledge(
        db,
        repo_id,
        payload.query.strip(),
        top_k=payload.top_k,
        retrieval_method=payload.retrieval_method,
        rerank_enabled=payload.rerank_enabled,
        score_threshold_enabled=payload.score_threshold_enabled,
        score_threshold=payload.score_threshold,
        metadata_filters=payload.metadata_filters,
    )
    results = build_sources(items)
    run = record_retrieval_test(db, repo_id, payload.query.strip(), trace, results, duration_ms)
    return RetrievalTestResponse(
        id=run.id,
        query=run.query,
        results=results,
        result_count=len(results),
        took_ms=duration_ms,
        retrieval_config=trace,
        created_at=run.created_at,
    )


@router.get("/repositories/{repo_id}/retrieval-tests", response_model=list[RetrievalTestHistoryResponse])
async def list_retrieval_tests(
    repo_id: uuid.UUID,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[RetrievalTestHistoryResponse]:
    """
    【API 接口】查询 RAG 召回测试的历史记录列表。
    """
    _repo_or_404(db, repo_id)
    runs = (
        db.query(RetrievalTestRun)
        .filter(RetrievalTestRun.repo_id == repo_id)
        .order_by(RetrievalTestRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        RetrievalTestHistoryResponse(
            id=run.id,
            query=run.query,
            result_count=len(run.results_json or []),
            took_ms=run.duration_ms,
            created_at=run.created_at,
        )
        for run in runs
    ]


@router.get("/{repo_id}/status")
async def get_rag_status(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    【API 接口】查询知识库运行状态（包含向量数据库连通性与切片分布）。
    """
    _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id)
    return {**rag_status(db, repo_id), "config": config_dict(config), "runtime": runtime_status(config)}


@router.post("/{repo_id}/search", response_model=RAGSearchResponse)
async def search_rag(repo_id: uuid.UUID, payload: RAGQueryRequest, db: Session = Depends(get_db)) -> RAGSearchResponse:
    """
    【API 接口】纯检索模式：根据问题检索匹配的切片和佐证证据，不调用 LLM 生成回答。
    """
    _repo_or_404(db, repo_id)
    items, trace, duration_ms = await retrieve_knowledge(
        db,
        repo_id,
        payload.question.strip(),
        top_k=payload.top_k,
        source_types=payload.source_types,
        metadata_filters=payload.metadata_filters,
    )
    results = build_sources(items)
    return RAGSearchResponse(
        query=payload.question.strip(),
        results=results,
        result_count=len(results),
        took_ms=duration_ms,
        retrieval_config=trace,
    )


@router.post("/{repo_id}/ask", response_model=RAGAnswerResponse)
async def ask_rag(repo_id: uuid.UUID, payload: RAGQueryRequest, db: Session = Depends(get_db)) -> RAGAnswerResponse:
    """
    【API 接口】完整 RAG 问答：检索相关切片、通过 Answer Gate 判定充分性并驱动 LLM 生成结构化回答。
    """
    _repo_or_404(db, repo_id)
    result = await answer_question(
        db,
        repo_id,
        payload.question,
        payload.top_k,
        payload.source_types,
        payload.metadata_filters,
    )
    return RAGAnswerResponse.model_validate(result)


@router.post("/{repo_id}/reindex", response_model=RAGReindexResponse)
async def reindex_rag(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> RAGReindexResponse:
    """
    【API 运维接口】对仓库现存的所有文档重新执行切片与向量化重索引。
    """
    _repo_or_404(db, repo_id)
    config = get_or_create_config(db, repo_id)
    count = reindex_documents(db, repo_id)
    return RAGReindexResponse(
        repo_id=repo_id,
        indexed_documents=count,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
    )

