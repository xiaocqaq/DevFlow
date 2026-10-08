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

router = APIRouter()

USER_KNOWLEDGE_SOURCE_TYPES = [
    "memory_note",
    "knowledge_file",
    "team_member",
    "weekly_report",
    "project_overview",
    "project_doc",
    "project_manifest",
]
EMBEDDING_BATCH_SIZE = 64


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return repo


def _clip(text: str, limit: int = 360) -> str:
    text = (text or "").strip().replace("\r\n", "\n")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _safe_filename(filename: str) -> str:
    value = Path(filename or "upload.txt").name
    return re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]+", "_", value).strip("._") or "upload.txt"


def _upload_root(repo_id: uuid.UUID) -> Path:
    root = Path(settings.upload_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    path = root / str(repo_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_upload_text(filename: str, data: bytes) -> str:
    try:
        return extract_document_text(filename, data)
    except Exception:
        return ""


def _embed_default_in_batches(texts: list[str]) -> list[list[float]]:
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
    _repo_or_404(db, repo_id)
    return [RecallEventResponse.model_validate(item) for item in list_recall_events(db, repo_id=repo_id, conversation_id=conversation_id, limit=limit)]


@router.get("/{repo_id}/memory/status", response_model=MemoryStatusResponse)
async def get_memory_status(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryStatusResponse:
    repo = _repo_or_404(db, repo_id)
    return MemoryStatusResponse.model_validate(memory_status(db, repo))


@router.get("/{repo_id}/memory/candidates", response_model=list[MemoryCandidateResponse])
async def get_memory_candidates(
    repo_id: uuid.UUID,
    status: str | None = Query(default="pending"),
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[MemoryCandidateResponse]:
    _repo_or_404(db, repo_id)
    return [MemoryCandidateResponse.model_validate(item) for item in list_memory_candidates(db, repo_id=repo_id, status=status, limit=limit)]


@router.post("/{repo_id}/memory/candidates/{candidate_id}/approve", response_model=MemoryCandidateActionResponse)
async def approve_candidate(repo_id: uuid.UUID, candidate_id: uuid.UUID, db: Session = Depends(get_db)) -> MemoryCandidateActionResponse:
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
    text = await asyncio.to_thread(_extract_upload_text, filename, data)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from this file")

    source_id = uuid.uuid4()
    stored_path = _upload_root(repo_id) / f"{source_id}-{filename}"
    chunks = await asyncio.to_thread(chunk_document, text, filename, 1800, 180)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document did not produce indexable chunks")
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
    repo = _repo_or_404(db, repo_id)
    return KnowledgeGraphResponse.model_validate(
        knowledge_graph_payload(db, repo, center_type=center_type, center_id=center_id, depth=depth)
    )


@router.post("/{repo_id}/knowledge/graph/rebuild", response_model=KnowledgeGraphRebuildResponse)
async def rebuild_knowledge_graph_route(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> KnowledgeGraphRebuildResponse:
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
    _repo_or_404(db, repo_id)
    count = delete_document_group(db, repo_id, source_id, source_type=source_type)
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return {"repo_id": str(repo_id), "source_id": str(source_id), "deleted": count}


@router.get("/{repo_id}/team-members", response_model=list[TeamMemberResponse])
async def list_team_members(repo_id: uuid.UUID, db: Session = Depends(get_db)) -> list[TeamMemberResponse]:
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
    source_id = stable_source_id(repo_id, "team_member", name.lower())
    delete_document_group(db, repo_id, source_id, source_type="team_member", commit=False)
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
    _repo_or_404(db, repo_id)
    count = delete_document_group(db, repo_id, member_id, source_type="team_member")
    rebuild_knowledge_graph(db, _repo_or_404(db, repo_id))
    return {"repo_id": str(repo_id), "member_id": str(member_id), "deleted": count}
