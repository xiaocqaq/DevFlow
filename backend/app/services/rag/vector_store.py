import json
import logging
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Document
from app.services.rag.embeddings import embed_text, embed_texts, embedding_contract, embedding_provider_name
from app.services.rag.source_policy import policy_metadata, should_vectorize

logger = logging.getLogger(__name__)


class MilvusUnavailableError(RuntimeError):
    pass

try:  # pymilvus 在测试和本地演示模式中是可选依赖。
    from pymilvus import DataType, MilvusClient
except Exception:  # pragma: no cover - 依赖缺失时会走到这里。
    DataType = None  # type: ignore[assignment]
    MilvusClient = None  # type: ignore[assignment]


@dataclass
class DocumentPayload:
    source_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_id: uuid.UUID | str | None = None


def _as_uuid(value: uuid.UUID | str | None) -> uuid.UUID:
    if value is None:
        return uuid.uuid4()
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def stable_source_id(repo_id: uuid.UUID | str, source_type: str, key: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"devflow:{repo_id}:{source_type}:{key}")


def _escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _clip_utf8(value: str, max_bytes: int) -> str:
    data = value.encode("utf-8")
    if len(data) <= max_bytes:
        return value
    return data[:max_bytes].decode("utf-8", errors="ignore")


def _json_dumps(value: dict[str, Any], max_chars: int = 4096) -> str:
    text = json.dumps(value or {}, ensure_ascii=False, default=str)
    return _clip_utf8(text, max_chars)


class MilvusVectorStore:
    def __init__(self) -> None:
        if MilvusClient is None or DataType is None:
            raise RuntimeError("pymilvus is not installed")
        self.collection_name = settings.milvus_collection
        self.dimensions = settings.embedding_dimensions
        self.client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or "")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            description = self.client.describe_collection(self.collection_name)
            actual_dimensions = _collection_dimensions(description)
            if actual_dimensions != self.dimensions:
                raise RuntimeError(
                    f"Milvus collection '{self.collection_name}' has dimension {actual_dimensions}; "
                    f"the configured embedding contract requires {self.dimensions}. "
                    "Use a new collection name and rebuild the index."
                )
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="repo_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source_type", datatype=DataType.VARCHAR, max_length=80)
        schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name="metadata_json", datatype=DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimensions)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert_documents(self, documents: list[Document], vectors: list[list[float]]) -> None:
        if len(documents) != len(vectors):
            raise ValueError("document/vector count mismatch")
        rows = []
        for doc, vector in zip(documents, vectors):
            if not should_vectorize(doc.source_type):
                continue
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"document {doc.id} has {len(vector)} embedding dimensions; expected {self.dimensions}"
                )
            rows.append(
                {
                    "pk": str(doc.id),
                    "repo_id": str(doc.repo_id),
                    "document_id": str(doc.id),
                    "source_type": doc.source_type,
                    "source_id": str(doc.source_id),
                    "title": _clip_utf8(doc.title, 500),
                    "content": _clip_utf8(doc.content or "", 8192),
                    "metadata_json": _json_dumps(doc.meta or {}),
                    "embedding": vector,
                }
            )
        if rows:
            self.client.upsert(collection_name=self.collection_name, data=rows)

    def delete_documents(
        self,
        *,
        repo_id: uuid.UUID | str | None = None,
        source_types: list[str] | None = None,
        source_id: uuid.UUID | str | None = None,
        document_ids: list[uuid.UUID | str] | None = None,
    ) -> None:
        expressions: list[str] = []
        if document_ids:
            ids = ", ".join(f'"{_escape_filter_value(str(item))}"' for item in document_ids)
            expressions.append(f"document_id in [{ids}]")
        if repo_id is not None:
            expressions.append(f'repo_id == "{_escape_filter_value(str(repo_id))}"')
        if source_types:
            values = ", ".join(f'"{_escape_filter_value(item)}"' for item in source_types)
            expressions.append(f"source_type in [{values}]")
        if source_id is not None:
            expressions.append(f'source_id == "{_escape_filter_value(str(source_id))}"')
        if expressions:
            self.client.delete(collection_name=self.collection_name, filter=" and ".join(expressions))

    def search(
        self,
        *,
        repo_id: uuid.UUID | str,
        query_embedding: list[float],
        source_type: str | None = None,
        source_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        expressions = [f'repo_id == "{_escape_filter_value(str(repo_id))}"']
        if source_type:
            expressions.append(f'source_type == "{_escape_filter_value(source_type)}"')
        elif source_types:
            values = ", ".join(f'"{_escape_filter_value(item)}"' for item in source_types)
            expressions.append(f"source_type in [{values}]")
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            filter=" and ".join(expressions),
            limit=limit,
            output_fields=["document_id", "source_type", "source_id", "title", "content", "metadata_json"],
        )
        hits = results[0] if results else []
        output: list[dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity") or {}
            metadata_text = entity.get("metadata_json") or "{}"
            try:
                metadata = json.loads(metadata_text)
            except json.JSONDecodeError:
                metadata = {}
            output.append(
                {
                    "id": str(entity.get("document_id") or hit.get("id")),
                    "title": str(entity.get("title") or ""),
                    "source_type": str(entity.get("source_type") or ""),
                    "source_id": str(entity.get("source_id") or ""),
                    "score": round(float(hit.get("distance") or 0.0), 4),
                    "metadata": metadata,
                    "snippet": str(entity.get("content") or "")[:260],
                    "_raw_content": str(entity.get("content") or ""),
                }
            )
        return output


def _collection_dimensions(description: dict[str, Any]) -> int:
    for schema_field in description.get("fields") or []:
        if schema_field.get("name") != "embedding":
            continue
        params = schema_field.get("params") or schema_field.get("type_params") or {}
        value = params.get("dim") or schema_field.get("dim")
        if value is not None:
            return int(value)
    raise RuntimeError("Milvus collection does not expose an embedding dimension")


@lru_cache(maxsize=1)
def _milvus_store() -> MilvusVectorStore:
    if not settings.milvus_enabled:
        raise MilvusUnavailableError("Milvus is disabled; vector and hybrid RAG are unavailable")
    try:
        return MilvusVectorStore()
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError(f"Milvus is unavailable: {exc}") from exc


def reset_milvus_store_cache() -> None:
    _milvus_store.cache_clear()


def milvus_runtime_status() -> dict[str, Any]:
    try:
        store = _milvus_store()
    except MilvusUnavailableError as exc:
        return {
            "backend": "milvus",
            "status": "degraded",
            "available": False,
            "error": str(exc),
            "collection": settings.milvus_collection,
            "embedding": embedding_contract(),
        }
    return {
        "backend": "milvus",
        "status": "ready",
        "available": True,
        "error": None,
        "collection": store.collection_name,
        "embedding": embedding_contract(),
    }


def upsert_milvus_documents(documents: list[Document], vectors: list[list[float]]) -> bool:
    if len(documents) != len(vectors):
        raise ValueError("document/vector count mismatch")
    for vector in vectors:
        if len(vector) != settings.embedding_dimensions:
            raise ValueError(
                f"embedding has {len(vector)} dimensions; expected {settings.embedding_dimensions}. "
                "Refusing to truncate or pad vectors."
            )
    pairs = [(doc, vector) for doc, vector in zip(documents, vectors) if should_vectorize(doc.source_type)]
    eligible_documents = [item[0] for item in pairs]
    eligible_vectors = [item[1] for item in pairs]
    if not eligible_documents:
        return True
    store = _milvus_store()
    try:
        store.upsert_documents(eligible_documents, eligible_vectors)
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError("failed to write documents to Milvus") from exc
    return True


def delete_milvus_documents(
    *,
    repo_id: uuid.UUID | str | None = None,
    source_types: list[str] | None = None,
    source_id: uuid.UUID | str | None = None,
    document_ids: list[uuid.UUID | str] | None = None,
    strict: bool = True,
) -> bool:
    try:
        store = _milvus_store()
    except MilvusUnavailableError:
        if strict:
            raise
        return False
    try:
        store.delete_documents(repo_id=repo_id, source_types=source_types, source_id=source_id, document_ids=document_ids)
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        if not strict:
            logger.warning("从 Milvus 删除文档失败：%s", exc)
            return False
        raise MilvusUnavailableError("failed to delete documents from Milvus") from exc
    return True


def search_milvus_documents(
    repo_id: str | uuid.UUID,
    query: str,
    source_type: str | None = None,
    limit: int = 20,
    query_embedding: list[float] | None = None,
    source_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    store = _milvus_store()
    try:
        vector = query_embedding or embed_text(query, settings.embedding_dimensions)
        return store.search(
            repo_id=repo_id,
            query_embedding=vector,
            source_type=source_type,
            source_types=source_types,
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - 依赖本地基础设施。
        raise MilvusUnavailableError("failed to search documents in Milvus") from exc


def add_document(
    db: Session,
    repo_id: str | uuid.UUID,
    source_type: str,
    source_id: str | uuid.UUID | None,
    title: str,
    content: str,
    metadata: dict[str, Any] | None = None,
    *,
    commit: bool = True,
) -> Document:
    if source_type == "code_file":
        raise ValueError("源码必须通过工作区实时搜索，不能保存为 RAG Document")
    vectorized = should_vectorize(source_type)
    vectors = [embed_text(f"{title}\n{content}", settings.embedding_dimensions)] if vectorized else []
    embedding_metadata = (
        {
            "embedding_provider": embedding_provider_name(),
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
        }
        if vectorized
        else {}
    )
    doc = Document(
        repo_id=_as_uuid(repo_id),
        source_type=source_type,
        source_id=_as_uuid(source_id),
        title=title[:500],
        content=content,
        meta={
            **(metadata or {}),
            **policy_metadata(source_type),
            **embedding_metadata,
        },
    )
    db.add(doc)
    db.flush()
    if vectorized and settings.milvus_enabled:
        upsert_milvus_documents([doc], vectors)
    if commit:
        db.commit()
        db.refresh(doc)
    return doc


def add_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
    vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> list[Document]:
    clean_payloads = [payload for payload in payloads if payload.content.strip()]
    if any(payload.source_type == "code_file" for payload in clean_payloads):
        raise ValueError("源码必须通过工作区实时搜索，不能保存为 RAG Document")
    if vectors is not None and len(vectors) != len(clean_payloads):
        raise ValueError("embedding count does not match document payload count")

    embeddings: list[list[float] | None] = [None] * len(clean_payloads)
    vector_indexes = [
        index for index, payload in enumerate(clean_payloads) if should_vectorize(payload.source_type)
    ]
    if vectors is not None:
        for index in vector_indexes:
            embeddings[index] = vectors[index]
    elif vector_indexes and generate_embeddings:
        generated = embed_texts(
            [f"{clean_payloads[index].title}\n{clean_payloads[index].content}" for index in vector_indexes],
            settings.embedding_dimensions,
        )
        if len(generated) != len(vector_indexes):
            raise ValueError("embedding count does not match vectorized document payload count")
        for index, embedding in zip(vector_indexes, generated):
            embeddings[index] = embedding
    if sync_vector_store and any(embeddings[index] is None for index in vector_indexes):
        raise ValueError("RAG documents require embeddings before Milvus synchronization")

    docs: list[Document] = []
    for payload, embedding in zip(clean_payloads, embeddings):
        vectorized = embedding is not None
        embedding_metadata = (
            {
                "embedding_provider": embedding_provider_name(),
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
                **(embedding_info or {}),
            }
            if vectorized
            else {}
        )
        docs.append(
            Document(
                repo_id=_as_uuid(repo_id),
                source_type=payload.source_type,
                source_id=_as_uuid(payload.source_id),
                title=payload.title[:500],
                content=payload.content,
                meta={
                    **(payload.metadata or {}),
                    **policy_metadata(payload.source_type),
                    **embedding_metadata,
                },
            )
        )
    if not docs:
        return []
    db.add_all(docs)
    db.flush()
    if sync_vector_store and settings.milvus_enabled:
        vector_documents = [docs[index] for index in vector_indexes]
        vector_values = [embeddings[index] for index in vector_indexes]
        upsert_milvus_documents(
            vector_documents,
            [vector for vector in vector_values if vector is not None],
        )
    if commit:
        db.commit()
        for doc in docs:
            db.refresh(doc)
    return docs


def replace_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    source_types: list[str],
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
    vectors: list[list[float]] | None = None,
    embedding_info: dict[str, Any] | None = None,
    generate_embeddings: bool = True,
    sync_vector_store: bool = True,
) -> list[Document]:
    repo_uuid = _as_uuid(repo_id)
    db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_type.in_(source_types)).delete(
        synchronize_session=False
    )
    db.flush()
    if sync_vector_store and settings.milvus_enabled:
        delete_milvus_documents(repo_id=repo_uuid, source_types=source_types)
    docs = add_documents(
        db,
        repo_uuid,
        payloads,
        commit=False,
        vectors=vectors,
        embedding_info=embedding_info,
        generate_embeddings=generate_embeddings,
        sync_vector_store=sync_vector_store,
    )
    if commit:
        db.commit()
    return docs


def replace_documents_by_metadata(
    db: Session,
    repo_id: str | uuid.UUID,
    source_type: str,
    metadata_filters: dict[str, Any],
    payloads: list[DocumentPayload],
    *,
    commit: bool = True,
) -> list[Document]:
    repo_uuid = _as_uuid(repo_id)
    rows = db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_type == source_type).all()
    document_ids = [
        row.id
        for row in rows
        if all((row.meta or {}).get(key) == value for key, value in metadata_filters.items())
    ]
    if document_ids:
        db.query(Document).filter(Document.id.in_(document_ids)).delete(synchronize_session=False)
        if should_vectorize(source_type) and settings.milvus_enabled:
            delete_milvus_documents(repo_id=repo_uuid, document_ids=document_ids)
    db.flush()
    docs = add_documents(db, repo_uuid, payloads, commit=False)
    if commit:
        db.commit()
    return docs


def delete_document_group(
    db: Session,
    repo_id: str | uuid.UUID,
    source_id: str | uuid.UUID,
    source_type: str | None = None,
    *,
    commit: bool = True,
) -> int:
    repo_uuid = _as_uuid(repo_id)
    source_uuid = _as_uuid(source_id)
    query = db.query(Document).filter(Document.repo_id == repo_uuid, Document.source_id == source_uuid)
    if source_type:
        query = query.filter(Document.source_type == source_type)
    document_ids = [row.id for row in query.all()]
    count = query.delete(synchronize_session=False)
    if settings.milvus_enabled and (source_type is None or should_vectorize(source_type)):
        delete_milvus_documents(repo_id=repo_uuid, source_id=source_uuid, document_ids=document_ids)
    if commit:
        db.commit()
    return count
