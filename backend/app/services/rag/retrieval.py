import heapq
import hashlib
import math
import re
import uuid
from collections import Counter
from datetime import datetime, time, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Document
from app.services.context_compression import compress_retrieval_results
from app.services.rag.rerank import rerank_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES, is_searchable_source, normalize_source_types
from app.services.rag.vector_store import MilvusUnavailableError, search_milvus_documents


BM25_K1 = 1.5
BM25_B = 0.75
VECTOR_FUSION_WEIGHT = 0.62
KEYWORD_FUSION_WEIGHT = 0.30
DUAL_RECALL_BONUS = 0.08
MAX_CHUNKS_PER_PARENT = 2
NEAR_DUPLICATE_JACCARD = 0.9


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def _as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _parse_datetime(value: str | None, end_of_day: bool = False) -> datetime | None:
    if not value:
        return None
    try:
        if len(value) == 10:
            parsed_date = datetime.fromisoformat(value).date()
            return datetime.combine(parsed_date, time.max if end_of_day else time.min, tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _metadata_time(doc_meta: dict) -> datetime | None:
    for key in ["updated_at", "created_at", "merged_at", "closed_at"]:
        parsed = _parse_datetime(doc_meta.get(key))
        if parsed:
            return parsed
    return None


def _matches_metadata(doc_meta: dict, filters: dict | None) -> bool:
    if not filters:
        return True

    label = filters.get("label")
    if label and label not in (doc_meta.get("labels") or []):
        return False

    state = filters.get("state")
    if state and doc_meta.get("state") != state:
        return False

    path = (filters.get("path") or filters.get("path_prefix") or filters.get("module") or "").lower()
    if path:
        files = [str(item).lower() for item in doc_meta.get("files") or []]
        if files and not any(path in filename for filename in files):
            return False
        if not files and path not in " ".join(str(value).lower() for value in doc_meta.values()):
            return False

    start_at = _parse_datetime(filters.get("start_date") or filters.get("updated_after"))
    end_at = _parse_datetime(filters.get("end_date") or filters.get("updated_before"), end_of_day=True)
    doc_time = _metadata_time(doc_meta)
    if start_at and (doc_time is None or doc_time < start_at):
        return False
    if end_at and (doc_time is None or doc_time > end_at):
        return False

    for key, value in filters.items():
        if key in {
            "label",
            "path",
            "path_prefix",
            "module",
            "start_date",
            "end_date",
            "updated_after",
            "updated_before",
        } or value in (None, ""):
            continue
        if key == "branch":
            current = doc_meta.get("branch") or doc_meta.get("base_branch") or doc_meta.get("head_branch")
            if current != value:
                return False
            continue
        current = doc_meta.get(key)
        if isinstance(current, list):
            if value not in current:
                return False
        elif current != value:
            return False
    return True


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def _clip(text: str, limit: int = 260) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def _normalise_scores(items: list[dict[str, Any]], field: str) -> None:
    if not items:
        return
    values = [max(0.0, float(item.get(field) or 0.0)) for item in items]
    max_value = max(values) or 1.0
    for item, value in zip(items, values):
        item[field] = round(value / max_value, 4)


def _document_result(doc: Document, score: float, source: str, score_field: str) -> dict[str, Any]:
    metadata = doc.meta or {}
    result = {
        "id": str(doc.id),
        "title": doc.title,
        "source_type": doc.source_type,
        "source_id": str(doc.source_id),
        "score": round(max(0.0, score), 4),
        "metadata": metadata,
        "snippet": _clip(doc.content),
        "_raw_content": doc.content,
        score_field: round(max(0.0, score), 4),
        "retrieval": {"sources": [source], score_field: round(max(0.0, score), 4)},
    }
    return result


def _milvus_vector_candidates(
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    milvus_results = search_milvus_documents(
        repo_id,
        query,
        source_type=source_types[0] if len(source_types) == 1 else None,
        source_types=source_types if len(source_types) > 1 else None,
        limit=fetch_limit,
        query_embedding=query_embedding,
    )
    candidates = []
    for item in milvus_results:
        if not _matches_metadata(item.get("metadata") or {}, metadata_filters):
            continue
        score = max(0.0, float(item.get("score") or 0.0))
        candidates.append(
            {
                **item,
                "id": str(item.get("id") or item.get("source_id") or ""),
                "score": score,
                "vector_score": score,
                "vector_score_raw": score,
                "retrieval": {
                    "sources": ["vector"],
                    "vector_score_raw": score,
                    "vector_backend": "milvus",
                },
            }
        )
    _normalise_scores(candidates, "vector_score")
    for rank, item in enumerate(candidates, start=1):
        item["score"] = item["vector_score"]
        item["retrieval"]["vector_score"] = item["vector_score"]
        item["vector_rank"] = rank
        item["retrieval"]["vector_rank"] = rank
    return candidates


def _bm25_document_stats(doc: Document, query_terms: list[str]) -> dict[str, Any]:
    metadata = doc.meta or {}
    field_tokens = {
        "title": _tokens(doc.title or ""),
        "path": _tokens(str(metadata.get("path") or "")),
        "symbol": _tokens(str(metadata.get("symbol") or metadata.get("name") or "")),
        "labels": _tokens(" ".join(str(item) for item in metadata.get("labels") or [])),
        "content": _tokens(doc.content or ""),
    }
    field_weights = {"title": 4.0, "path": 3.0, "symbol": 4.5, "labels": 2.5, "content": 1.0}
    counters = {field: Counter(tokens) for field, tokens in field_tokens.items()}
    weighted_tf = {
        term: sum(float(counters[field].get(term, 0)) * field_weights[field] for field in field_tokens)
        for term in query_terms
    }
    matched_terms = [term for term in query_terms if weighted_tf.get(term, 0.0) > 0.0]
    document_length = max(1, sum(len(tokens) for tokens in field_tokens.values()))
    return {
        "doc": doc,
        "weighted_tf": weighted_tf,
        "matched_terms": matched_terms,
        "document_length": document_length,
    }


def _bm25_score(
    stats: dict[str, Any],
    *,
    document_count: int,
    document_frequency: Counter[str],
    average_document_length: float,
) -> float:
    score = 0.0
    document_length = float(stats["document_length"])
    length_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * document_length / max(average_document_length, 1.0))
    for term, weighted_tf in stats["weighted_tf"].items():
        if weighted_tf <= 0:
            continue
        df = float(document_frequency.get(term, 0))
        idf = math.log1p((document_count - df + 0.5) / (df + 0.5))
        score += idf * (weighted_tf * (BM25_K1 + 1.0)) / (weighted_tf + length_norm)
    return max(0.0, score)


def _keyword_candidates(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    source_types: list[str],
    fetch_limit: int,
    metadata_filters: dict | None,
) -> list[dict[str, Any]]:
    terms = list(dict.fromkeys(_tokens(query)))
    if not terms:
        return []
    q = db.query(Document).filter(
        Document.repo_id == _as_uuid(repo_id),
        Document.source_type.in_(source_types),
    )
    state = (metadata_filters or {}).get("state")
    if state:
        q = q.filter(Document.meta["state"].as_string() == str(state))

    corpus: list[dict[str, Any]] = []
    document_frequency: Counter[str] = Counter()
    total_document_length = 0
    for doc in q.yield_per(500):
        if not _matches_metadata(doc.meta or {}, metadata_filters):
            continue
        stats = _bm25_document_stats(doc, terms)
        corpus.append(stats)
        total_document_length += int(stats["document_length"])
        document_frequency.update(set(stats["matched_terms"]))

    if not corpus:
        return []

    average_document_length = total_document_length / len(corpus)
    top_candidates: list[tuple[float, int, dict[str, Any]]] = []
    for sequence, stats in enumerate(corpus):
        doc = stats["doc"]
        score = _bm25_score(
            stats,
            document_count=len(corpus),
            document_frequency=document_frequency,
            average_document_length=average_document_length,
        )
        if score <= 0:
            continue
        item = _document_result(doc, score, "keyword", "keyword_score")
        item["bm25_score"] = round(score, 4)
        item["bm25_matched_terms"] = list(stats["matched_terms"])
        item["retrieval"].update(
            {
                "keyword_backend": "bm25",
                "bm25_score": round(score, 4),
                "bm25_matched_terms": list(stats["matched_terms"]),
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
            }
        )
        entry = (score, sequence, item)
        if len(top_candidates) < fetch_limit:
            heapq.heappush(top_candidates, entry)
        elif score > top_candidates[0][0]:
            heapq.heapreplace(top_candidates, entry)
    candidates = [
        entry[2]
        for entry in sorted(top_candidates, key=lambda entry: (entry[0], entry[1]), reverse=True)
    ]
    _normalise_scores(candidates, "keyword_score")
    for rank, item in enumerate(candidates, start=1):
        item["score"] = item["keyword_score"]
        item["retrieval"]["keyword_score"] = item["keyword_score"]
        item["keyword_rank"] = rank
        item["retrieval"]["keyword_rank"] = rank
    return candidates


def _merge_candidates(vector_items: list[dict[str, Any]], keyword_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source_name, items in [("vector", vector_items), ("keyword", keyword_items)]:
        for item in items:
            metadata = item.get("metadata") or {}
            key = str(
                metadata.get("chunk_id")
                or item.get("id")
                or f"{item.get('source_type')}:{item.get('source_id')}:{item.get('title')}"
            )
            current = merged.get(key)
            if current is None:
                current = {**item, "retrieval": {**(item.get("retrieval") or {})}}
                current["retrieval"]["sources"] = list(dict.fromkeys(current["retrieval"].get("sources") or [source_name]))
                merged[key] = current
            else:
                current["vector_score"] = max(float(current.get("vector_score") or 0.0), float(item.get("vector_score") or 0.0))
                current["keyword_score"] = max(float(current.get("keyword_score") or 0.0), float(item.get("keyword_score") or 0.0))
                current["vector_score_raw"] = max(
                    float(current.get("vector_score_raw") or 0.0), float(item.get("vector_score_raw") or 0.0)
                )
                current["bm25_score"] = max(float(current.get("bm25_score") or 0.0), float(item.get("bm25_score") or 0.0))
                current["snippet"] = max([str(current.get("snippet") or ""), str(item.get("snippet") or "")], key=len)
                current["_raw_content"] = max(
                    [str(current.get("_raw_content") or ""), str(item.get("_raw_content") or "")], key=len
                )
                current["retrieval"]["sources"] = list(dict.fromkeys([*(current["retrieval"].get("sources") or []), source_name]))
                for trace_key, trace_value in (item.get("retrieval") or {}).items():
                    if trace_key == "sources":
                        continue
                    current["retrieval"][trace_key] = trace_value
    output = []
    for item in merged.values():
        vector_score = float(item.get("vector_score") or 0.0)
        keyword_score = float(item.get("keyword_score") or 0.0)
        both_bonus = DUAL_RECALL_BONUS if vector_score and keyword_score else 0.0
        combined = min(1.0, VECTOR_FUSION_WEIGHT * vector_score + KEYWORD_FUSION_WEIGHT * keyword_score + both_bonus)
        item["combined_score"] = round(combined, 4)
        item["fusion_score"] = round(combined, 4)
        item["score"] = round(combined, 4)
        item["retrieval"].update(
            {
                "fusion_method": "weighted",
                "fusion_weights": {
                    "vector": VECTOR_FUSION_WEIGHT,
                    "keyword": KEYWORD_FUSION_WEIGHT,
                    "dual_recall_bonus": DUAL_RECALL_BONUS,
                },
                "dual_recall_bonus": both_bonus,
                "combined_score": item["combined_score"],
                "fusion_score": item["fusion_score"],
            }
        )
        output.append(item)
    output.sort(key=lambda item: float(item.get("combined_score") or 0.0), reverse=True)
    for rank, item in enumerate(output, start=1):
        item["fusion_rank"] = rank
        item["retrieval"]["fusion_rank"] = rank
    return output


def _normalised_fingerprint(text: str) -> str:
    normalised = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", str(text or "").lower())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest() if normalised else ""


def _candidate_term_set(item: dict[str, Any]) -> set[str]:
    text = str(item.get("_raw_content") or item.get("snippet") or "")
    return set(_tokens(text))


def _same_duplicate_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_meta = left.get("metadata") or {}
    right_meta = right.get("metadata") or {}
    left_parent = str(left_meta.get("parent_id") or "")
    right_parent = str(right_meta.get("parent_id") or "")
    if left_parent and right_parent:
        return left_parent == right_parent
    return (
        str(left.get("source_type") or "") == str(right.get("source_type") or "")
        and str(left.get("source_id") or "") == str(right.get("source_id") or "")
    )


def _deduplicate_candidates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    fingerprints: dict[str, dict[str, Any]] = {}
    term_sets: dict[str, set[str]] = {}
    for item in items:
        item_id = str(item.get("id") or "")
        raw_text = str(item.get("_raw_content") or item.get("snippet") or "")
        fingerprint = _normalised_fingerprint(raw_text)
        duplicate_of: dict[str, Any] | None = fingerprints.get(fingerprint) if fingerprint else None
        if duplicate_of is not None and not _same_duplicate_scope(item, duplicate_of):
            duplicate_of = None
        if duplicate_of is None:
            current_terms = _candidate_term_set(item)
            for existing in kept:
                if not _same_duplicate_scope(item, existing):
                    continue
                existing_id = str(existing.get("id") or "")
                existing_terms = term_sets.get(existing_id) or _candidate_term_set(existing)
                if not current_terms or not existing_terms:
                    continue
                union = current_terms | existing_terms
                similarity = len(current_terms & existing_terms) / max(len(union), 1)
                if similarity >= NEAR_DUPLICATE_JACCARD:
                    duplicate_of = existing
                    break
            term_sets[item_id] = current_terms
        if duplicate_of is not None:
            duplicate_id = str(duplicate_of.get("id") or "")
            duplicate_trace = duplicate_of.setdefault("retrieval", {})
            duplicate_trace.setdefault("duplicates_removed", []).append(item_id)
            removed.append({"id": item_id, "duplicate_of": duplicate_id})
            continue
        kept.append(item)
        if fingerprint:
            fingerprints[fingerprint] = item
    return kept, removed


def _limit_parent_occupancy(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for item in items:
        metadata = item.get("metadata") or {}
        parent_id = str(metadata.get("parent_id") or "")
        if parent_id and counts[parent_id] >= MAX_CHUNKS_PER_PARENT:
            removed.append({"id": str(item.get("id") or ""), "parent_id": parent_id})
            continue
        kept.append(item)
        if parent_id:
            counts[parent_id] += 1
    return kept, removed


def _same_version(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("branch", "commit_sha", "version", "checksum", "knowledge_document_id"):
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is not None and right_value is not None and left_value != right_value:
            return False
    return True


def expand_parent_context(
    db: Session,
    repo_id: str | uuid.UUID,
    items: list[dict[str, Any]],
    *,
    neighbor_window: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    expanded_items = 0
    expanded_chunks = 0
    for item in items:
        metadata = dict(item.get("metadata") or {})
        parent_id = str(metadata.get("parent_id") or "")
        child_index = metadata.get("child_index")
        source_id = item.get("source_id")
        source_type = str(item.get("source_type") or "")
        retrieval = dict(item.get("retrieval") or {})
        if not parent_id or child_index is None or not source_id or not source_type:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "parent metadata unavailable"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        try:
            source_uuid = _as_uuid(source_id)
        except (TypeError, ValueError):
            retrieval["parent_expansion"] = {"expanded": False, "reason": "invalid source id"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        siblings = (
            db.query(Document)
            .filter(
                Document.repo_id == _as_uuid(repo_id),
                Document.source_type == source_type,
                Document.source_id == source_uuid,
            )
            .all()
        )
        selected: list[Document] = []
        current_index = int(child_index)
        for sibling in siblings:
            sibling_meta = sibling.meta or {}
            if str(sibling_meta.get("parent_id") or "") != parent_id:
                continue
            if not _same_version(metadata, sibling_meta):
                continue
            sibling_index = int(sibling_meta.get("child_index") or 0)
            if abs(sibling_index - current_index) <= max(0, neighbor_window):
                selected.append(sibling)
        selected.sort(key=lambda row: int((row.meta or {}).get("child_index") or 0))
        if not selected:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "no matching siblings"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        parts: list[str] = []
        seen_parts: set[str] = set()
        expanded_ids: list[str] = []
        for sibling in selected:
            content = str(sibling.content or "").strip()
            fingerprint = _normalised_fingerprint(content)
            if not content or fingerprint in seen_parts:
                continue
            seen_parts.add(fingerprint)
            parts.append(content)
            expanded_ids.append(str(sibling.id))
        if not parts:
            retrieval["parent_expansion"] = {"expanded": False, "reason": "empty sibling content"}
            expanded.append({**item, "retrieval": retrieval})
            continue
        parent_expanded = len(expanded_ids) > 1 or str(item.get("id") or "") not in expanded_ids
        retrieval["parent_expansion"] = {
            "expanded": parent_expanded,
            "parent_id": parent_id,
            "chunk_ids": expanded_ids,
            "neighbor_window": max(0, neighbor_window),
        }
        next_metadata = {**metadata, "expanded_chunk_ids": expanded_ids}
        expanded.append(
            {
                **item,
                "metadata": next_metadata,
                "_raw_content": "\n\n".join(parts),
                "parent_expanded": parent_expanded,
                "retrieval": retrieval,
            }
        )
        if parent_expanded:
            expanded_items += 1
            expanded_chunks += max(0, len(expanded_ids) - 1)
    return expanded, {
        "name": "parent_expansion",
        "input_count": len(items),
        "output_count": len(expanded),
        "expanded_items": expanded_items,
        "additional_chunks": expanded_chunks,
    }


def search_similar_documents(
    db: Session,
    repo_id: str | uuid.UUID,
    query: str,
    source_type: str | None = None,
    limit: int = 5,
    metadata_filters: dict | None = None,
    query_embedding: list[float] | None = None,
    retrieval_method: str = "hybrid",
    apply_rerank: bool = True,
    apply_compression: bool = True,
    source_types: list[str] | None = None,
    pipeline_trace: dict[str, Any] | None = None,
) -> list[dict]:
    requested_types = normalize_source_types([source_type, *(source_types or [])]) or sorted(RAG_SOURCE_TYPES)
    searchable_types = list(dict.fromkeys(item for item in requested_types if is_searchable_source(item)))
    if not searchable_types:
        return []

    fetch_limit = min(max(limit * 8, 40), 200)
    method = retrieval_method.strip().lower()
    vector_types = searchable_types
    vector_candidates: list[dict[str, Any]] = []
    keyword_candidates: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    vector_status = "skipped"
    if method in {"vector", "hybrid"} and vector_types:
        try:
            vector_candidates = _milvus_vector_candidates(
                repo_id,
                query,
                vector_types,
                fetch_limit,
                metadata_filters,
                query_embedding=query_embedding,
            )
            vector_status = "success"
        except MilvusUnavailableError:
            vector_status = "unavailable"
            if method == "vector":
                raise
    stages.append(
        {
            "name": "vector_recall",
            "status": vector_status,
            "candidate_count": len(vector_candidates),
            "backend": "milvus",
        }
    )
    keyword_types = searchable_types if method in {"fulltext", "hybrid"} else []
    if keyword_types:
        keyword_candidates = _keyword_candidates(
            db,
            repo_id,
            query,
            keyword_types,
            fetch_limit,
            metadata_filters,
        )
    stages.append(
        {
            "name": "keyword_recall",
            "status": "success" if keyword_types else "skipped",
            "candidate_count": len(keyword_candidates),
            "backend": "bm25",
        }
    )
    if method == "vector":
        merged = vector_candidates
    elif method == "fulltext":
        merged = keyword_candidates
    else:
        merged = _merge_candidates(vector_candidates, keyword_candidates)
    stages.append(
        {
            "name": "weighted_fusion" if method == "hybrid" else "single_retriever_selection",
            "method": "weighted" if method == "hybrid" else method,
            "candidate_count": len(merged),
            "weights": {
                "vector": VECTOR_FUSION_WEIGHT,
                "keyword": KEYWORD_FUSION_WEIGHT,
                "dual_recall_bonus": DUAL_RECALL_BONUS,
            },
        }
    )
    deduplicated, duplicates_removed = _deduplicate_candidates(merged)
    aggregated, parent_overflow_removed = _limit_parent_occupancy(deduplicated)
    stages.append(
        {
            "name": "deduplication_and_parent_aggregation",
            "input_count": len(merged),
            "output_count": len(aggregated),
            "duplicates_removed": duplicates_removed,
            "parent_overflow_removed": parent_overflow_removed,
        }
    )
    ranked = rerank_documents(query, aggregated, limit) if apply_rerank else aggregated[:limit]
    if apply_rerank:
        stages.append(
            {
                "name": "rerank",
                "input_count": len(aggregated),
                "output_count": len(ranked),
                "mode": ranked[0].get("retrieval", {}).get("rerank_mode") if ranked else None,
            }
        )
    if pipeline_trace is not None:
        pipeline_trace.update(
            {
                "retrieval_method": method,
                "fusion_method": "weighted" if method == "hybrid" else method,
                "stages": stages,
            }
        )
    if not apply_compression:
        return ranked
    compressed = compress_retrieval_results(query, ranked, max_tokens_per_item=260)
    if pipeline_trace is not None:
        pipeline_trace["stages"].append(
            {
                "name": "context_compression",
                "input_count": len(ranked),
                "output_count": len(compressed),
            }
        )
    return compressed
