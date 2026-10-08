import re
from typing import Any


AMBIGUOUS_MARKERS = ("这个", "那个", "它", "这段", "这里", "怎么办", "怎么处理")
CONFLICT_POLARITIES = (
    ("permission", ("允许", "可以", "enabled", "enable"), ("禁止", "不允许", "不能", "disabled", "disable")),
    ("outcome", ("成功", "通过", "success", "passed"), ("失败", "未通过", "failure", "failed")),
    ("requirement", ("必须", "需要", "required"), ("可选", "无需", "optional")),
)


def _normalise(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _evidence_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return _normalise(
        "\n".join(
            [
                str(item.get("title") or ""),
                str(item.get("snippet") or item.get("_raw_content") or ""),
                str(metadata.get("path") or ""),
                str(metadata.get("symbol") or ""),
                str(metadata.get("branch") or metadata.get("base_branch") or ""),
            ]
        )
    )


def _strong_query_signals(query: str) -> list[str]:
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_./\-]*|#\d+|\b\d{3,}\b", query)
    signals: list[str] = []
    for candidate in candidates:
        if (
            any(marker in candidate for marker in ("_", ".", "/", "-"))
            or any(char.isdigit() for char in candidate)
            or candidate.endswith(("Error", "Exception"))
        ):
            signals.append(candidate.lower())
    return list(dict.fromkeys(signals))


def _detect_conflict(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    high_confidence = [item for item in items if float(item.get("score") or 0.0) >= 0.6]
    if len(high_confidence) < 2:
        return None
    texts = [(str(item.get("id") or item.get("source_id") or ""), _evidence_text(item)) for item in high_confidence]
    for conflict_type, positive_markers, negative_markers in CONFLICT_POLARITIES:
        negative_ids = [item_id for item_id, text in texts if any(marker in text for marker in negative_markers)]
        positive_ids = [
            item_id
            for item_id, text in texts
            if any(marker in text for marker in positive_markers)
            and not any(marker in text for marker in negative_markers)
        ]
        if positive_ids and negative_ids and set(positive_ids) != set(negative_ids):
            return {
                "type": conflict_type,
                "positive_evidence_ids": positive_ids,
                "negative_evidence_ids": negative_ids,
            }
    return None


def _is_ambiguous_query(query: str, items: list[dict[str, Any]]) -> bool:
    compact = re.sub(r"\s+", "", query)
    if not any(marker in compact for marker in AMBIGUOUS_MARKERS):
        return False
    source_ids = {str(item.get("source_id") or "") for item in items if item.get("source_id")}
    return len(source_ids) > 1 or len(compact) <= 8


def evaluate_answer_gate(
    query: str,
    items: list[dict[str, Any]],
    *,
    score_threshold: float = 0.0,
    threshold_enabled: bool = True,
) -> dict[str, Any]:
    top_score = max((float(item.get("score") or 0.0) for item in items), default=0.0)
    base = {
        "evidence_count": len(items),
        "top_score": round(top_score, 4),
        "score_threshold": round(float(score_threshold), 4) if threshold_enabled else None,
    }
    if not items:
        return {**base, "decision": "insufficient_evidence", "reason": "没有检索到满足条件的证据"}
    if threshold_enabled and top_score < score_threshold:
        return {**base, "decision": "insufficient_evidence", "reason": "最高证据分数未达到回答门槛"}

    signals = _strong_query_signals(query)
    evidence_text = "\n".join(_evidence_text(item) for item in items)
    missing_signals = [signal for signal in signals if signal not in evidence_text]
    if signals and len(missing_signals) == len(signals):
        return {
            **base,
            "decision": "insufficient_evidence",
            "reason": "查询中的关键名称没有出现在候选证据中",
            "required_signals": signals,
            "missing_signals": missing_signals,
        }

    conflict = _detect_conflict(items)
    if conflict:
        return {**base, "decision": "conflict", "reason": "高排名证据给出了相反结论", "conflict": conflict}

    if _is_ambiguous_query(query, items):
        return {
            **base,
            "decision": "ask_clarification",
            "reason": "问题缺少可定位的对象或范围",
            "suggested_fields": ["文件或模块", "错误名称", "分支或版本"],
        }

    return {
        **base,
        "decision": "answer",
        "reason": "证据达到分数门槛，关键名称可以在候选中核对",
        "matched_signals": [signal for signal in signals if signal in evidence_text],
    }


def gate_response_message(gate: dict[str, Any]) -> str:
    decision = str(gate.get("decision") or "insufficient_evidence")
    if decision == "ask_clarification":
        return "当前问题的对象或范围不够明确。请补充文件、模块、错误名称、分支或版本后再试。"
    if decision == "conflict":
        return "当前知识库中的高排名证据存在冲突，暂时无法给出确定结论。请先核对对应来源或版本。"
    return "当前知识库资料不足，无法确认。请补充相关资料、检查检索范围，或换一种更具体的问法。"
