import math
import re
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_\-]+", lowered)
    for segment in re.findall(r"[\u4e00-\u9fff]+", lowered):
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


def _normalise(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _candidate_text(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    path = str(metadata.get("path") or "")
    return "\n".join(
        [
            str(candidate.get("title") or ""),
            path,
            str(candidate.get("snippet") or ""),
        ]
    ).lower()


def _freshness_bonus(metadata: dict[str, Any]) -> float:
    for key in ("updated_at", "created_at", "merged_at", "closed_at"):
        value = metadata.get(key)
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        age_days = max(0, (datetime.now(timezone.utc) - parsed).days)
        if age_days <= 30:
            return 0.04
        if age_days <= 180:
            return 0.02
        return 0.0
    return 0.0


def _heuristic_score(query: str, candidate: dict[str, Any]) -> tuple[float, str]:
    terms = _tokens(query)
    if not terms:
        return _normalise(float(candidate.get("score") or 0.0)), "base score only"
    text = _candidate_text(candidate)
    title = str(candidate.get("title") or "").lower()
    metadata = candidate.get("metadata") or {}
    path = str(metadata.get("path") or "").lower()
    matched = [term for term in terms if term in text]
    coverage = len(set(matched)) / max(len(set(terms)), 1)
    title_hits = sum(1 for term in terms if term in title)
    path_hits = sum(1 for term in terms if term in path)
    phrase_bonus = 0.12 if query.lower().strip() and query.lower().strip() in text else 0.0
    source_bonus = 0.08 if len({term for term in terms if term in text}) >= 2 else 0.0
    base = _normalise(float(candidate.get("score") or 0.0))
    keyword = _normalise(float(candidate.get("keyword_score") or 0.0))
    vector = _normalise(float(candidate.get("vector_score") or 0.0))
    sources = set((candidate.get("retrieval") or {}).get("sources") or [])
    dual_recall_bonus = 0.05 if {"vector", "keyword"}.issubset(sources) else 0.0
    freshness_bonus = _freshness_bonus(metadata)
    source_type = str(candidate.get("source_type") or "").lower()
    source_type_bonus = 0.03 if source_type and any(term in source_type for term in terms) else 0.0
    score = (
        0.31 * base
        + 0.20 * vector
        + 0.16 * keyword
        + 0.17 * coverage
        + 0.04 * min(title_hits, 3) / 3
        + 0.02 * min(path_hits, 3) / 3
        + phrase_bonus
        + source_bonus
        + dual_recall_bonus
        + freshness_bonus
        + source_type_bonus
    )
    reasons = []
    if coverage:
        reasons.append(f"matched {len(set(matched))}/{len(set(terms))} query terms")
    if title_hits:
        reasons.append("title match")
    if path_hits:
        reasons.append("path match")
    if vector:
        reasons.append("vector candidate")
    if keyword:
        reasons.append("keyword candidate")
    if dual_recall_bonus:
        reasons.append("dual recall")
    if freshness_bonus:
        reasons.append("fresh evidence")
    if source_type_bonus:
        reasons.append("source type match")
    return _normalise(score), ", ".join(reasons) or "弱关键词匹配"


def rerank_documents(query: str, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    top_n = min(max(settings.rag_rerank_top_n, limit), len(candidates))
    head = candidates[:top_n]
    tail = candidates[top_n:]
    ranked: list[dict[str, Any]] = []
    for candidate in head:
        heuristic_score, heuristic_reason = _heuristic_score(query, candidate)
        ranked.append(
            {
                **candidate,
                "score": round(heuristic_score, 4),
                "rerank_score": round(heuristic_score, 4),
                "rank_reason": heuristic_reason,
                "retrieval": {
                    **(candidate.get("retrieval") or {}),
                    "rerank_mode": "heuristic",
                    "rerank_score": round(heuristic_score, 4),
                    "rank_reason": heuristic_reason,
                },
            }
        )
    ranked.sort(key=lambda item: (float(item.get("rerank_score") or item.get("score") or 0.0), float(item.get("combined_score") or 0.0)), reverse=True)
    return [*ranked, *tail][:limit]
