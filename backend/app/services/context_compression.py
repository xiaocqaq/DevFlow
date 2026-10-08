import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import settings


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_\-]+|[^\s]")
WORD_PATTERN = re.compile(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?\n\u3002\uff01\uff1f])\s+")

MODEL_CONTEXT_WINDOWS = {
    "deepseek-v4-pro": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
}

MODEL_OUTPUT_WINDOWS = {
    "deepseek-v4-pro": 384_000,
    "gpt-4o": 16_384,
    "gpt-4o-mini": 16_384,
    "gpt-4.1": 32_768,
    "gpt-4.1-mini": 32_768,
    "gpt-4.1-nano": 32_768,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    "o1": 100_000,
    "o1-mini": 65_536,
    "o3": 100_000,
    "o3-mini": 100_000,
    "o4-mini": 100_000,
    "claude-3-5-sonnet": 8_192,
    "claude-3-7-sonnet": 64_000,
    "claude-sonnet-4": 64_000,
    "claude-opus-4": 64_000,
}

COMPACTABLE_TOOL_HINTS = (
    "workspace.read_file",
    "workspace.search_code",
    "workspace.list_files",
    "rag.search_similar_documents",
    "devflow_search_evidence",
    "repo_health.generate_summary",
    "ci_debug_agent.analyze_workflow_run",
    "read",
    "grep",
    "search",
    "shell",
    "web_fetch",
    "webfetch",
)

COMPACT_BOUNDARY_SUBTYPE = "compact_boundary"
MICROCOMPACT_MARKER = "[Old regenerable tool/context result cleared by microcompact]"


PressureStage = Literal["normal", "warning", "auto_compact", "aggressive", "manual_path"]


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    total = 0
    for token in TOKEN_PATTERN.findall(text):
        if re.fullmatch(r"[\u4e00-\u9fff]", token):
            total += 1
        elif re.fullmatch(r"[a-zA-Z0-9_\-]+", token):
            total += max(1, (len(token) + 3) // 4)
        else:
            total += 1
    return total


def is_context_overflow_error(exc: Exception) -> bool:
    """Recognize provider-specific prompt/context overflow failures.

    OpenAI-compatible gateways do not expose one stable exception type, so we
    deliberately use a narrow set of message markers and only allow callers to
    retry once.
    """
    message = str(exc or "").lower()
    markers = (
        "context_length_exceeded",
        "maximum context length",
        "prompt is too long",
        "prompt too long",
        "context window",
        "too many tokens",
        "request too large",
    )
    return any(marker in message for marker in markers)


def _normalize_model_name(model: str | None) -> str:
    value = str(model or settings.llm_model or "").strip().lower()
    if not value:
        return "unknown"
    value = value.split("/")[-1]
    value = re.sub(r"-(20\d{2}[-_]\d{2}[-_]\d{2}|\d{4}[-_]\d{2}[-_]\d{2})$", "", value)
    return value


def _model_lookup(model: str | None, table: dict[str, int]) -> int | None:
    normalized = _normalize_model_name(model)
    if normalized in table:
        return table[normalized]
    for key, value in table.items():
        if normalized.startswith(key):
            return value
    return None


def model_context_window_tokens(model: str | None = None) -> int:
    if settings.context_model_window_tokens:
        return max(4096, settings.context_model_window_tokens)
    known = _model_lookup(model, MODEL_CONTEXT_WINDOWS)
    if known:
        return known
    return max(4096, settings.context_max_input_tokens + settings.context_reserved_response_tokens)


def model_output_window_tokens(model: str | None = None) -> int:
    known = _model_lookup(model, MODEL_OUTPUT_WINDOWS)
    if known:
        return known
    return max(512, settings.context_reserved_response_tokens)


def effective_context_window_tokens(model: str | None = None, *, for_compaction: bool = False) -> int:
    model_window = model_context_window_tokens(model)
    if for_compaction:
        reserved = min(model_output_window_tokens(model), max(512, settings.context_compact_reserved_output_tokens))
    else:
        reserved = max(512, min(settings.context_reserved_response_tokens, model_window // 2))
    dynamic_window = max(4096, model_window - reserved)
    if settings.context_dynamic_model_window_enabled:
        return max(settings.context_max_input_tokens, dynamic_window)
    return max(4096, min(settings.context_max_input_tokens, dynamic_window))


@dataclass(frozen=True)
class ContextPressure:
    stage: PressureStage
    token_usage: int
    effective_window_tokens: int
    remaining_tokens: int
    warning_threshold: int
    auto_compact_threshold: int
    aggressive_threshold: int
    manual_threshold: int
    utilization: float

    @property
    def should_warn(self) -> bool:
        return self.stage != "normal"

    @property
    def should_auto_compact(self) -> bool:
        return self.stage in {"auto_compact", "aggressive", "manual_path"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "token_usage": self.token_usage,
            "effective_window_tokens": self.effective_window_tokens,
            "remaining_tokens": self.remaining_tokens,
            "warning_threshold": self.warning_threshold,
            "auto_compact_threshold": self.auto_compact_threshold,
            "aggressive_threshold": self.aggressive_threshold,
            "manual_threshold": self.manual_threshold,
            "utilization": self.utilization,
        }


def _thresholds_for_window(window: int) -> tuple[int, int, int, int]:
    warning_buffer = min(settings.context_warning_buffer_tokens, max(512, window // 4))
    auto_buffer = min(settings.context_auto_compact_buffer_tokens, max(512, window // 5))
    aggressive_buffer = min(settings.context_aggressive_buffer_tokens, max(512, window // 8))
    manual_buffer = min(settings.context_manual_buffer_tokens, max(256, window // 16))
    auto_threshold = max(1024, window - auto_buffer)
    warning_threshold = max(512, min(auto_threshold - 1, auto_threshold - warning_buffer))
    aggressive_threshold = max(auto_threshold + 1, window - aggressive_buffer)
    manual_threshold = max(aggressive_threshold + 1, window - manual_buffer)
    return warning_threshold, auto_threshold, aggressive_threshold, manual_threshold


def calculate_context_pressure(token_usage: int, model: str | None = None) -> ContextPressure:
    effective_window = effective_context_window_tokens(model, for_compaction=True)
    warning_threshold, auto_threshold, aggressive_threshold, manual_threshold = _thresholds_for_window(effective_window)
    if token_usage >= manual_threshold:
        stage: PressureStage = "manual_path"
    elif token_usage >= aggressive_threshold:
        stage = "aggressive"
    elif token_usage >= auto_threshold:
        stage = "auto_compact"
    elif token_usage >= warning_threshold:
        stage = "warning"
    else:
        stage = "normal"
    return ContextPressure(
        stage=stage,
        token_usage=token_usage,
        effective_window_tokens=effective_window,
        remaining_tokens=max(0, effective_window - token_usage),
        warning_threshold=warning_threshold,
        auto_compact_threshold=auto_threshold,
        aggressive_threshold=aggressive_threshold,
        manual_threshold=manual_threshold,
        utilization=round(token_usage / effective_window, 4) if effective_window else 0.0,
    )


def is_compact_boundary_message(message: dict[str, Any]) -> bool:
    meta = message.get("meta") or message.get("metadata") or {}
    return str(message.get("role") or "") == "system" and meta.get("subtype") == COMPACT_BOUNDARY_SUBTYPE


def find_last_compact_boundary_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if is_compact_boundary_message(messages[index]):
            return index
    return -1


def messages_after_compact_boundary(messages: list[dict[str, Any]], *, include_boundary: bool = True) -> list[dict[str, Any]]:
    boundary_index = find_last_compact_boundary_index(messages)
    if boundary_index < 0:
        return messages
    start = boundary_index if include_boundary else boundary_index + 1
    return messages[start:]


def _message_meta(message: dict[str, Any]) -> dict[str, Any]:
    meta = message.get("meta") or message.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _message_provider_id(message: dict[str, Any]) -> str | None:
    meta = _message_meta(message)
    value = (
        message.get("message_id")
        or message.get("provider_message_id")
        or meta.get("message_id")
        or meta.get("provider_message_id")
    )
    return str(value) if value else None


def _content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _tool_use_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            value = (
                call.get("id")
                or call.get("tool_call_id")
                or call.get("tool_use_id")
                or call.get("call_id")
            )
            if value:
                ids.add(str(value))
    for block in _content_blocks(message):
        if block.get("type") == "tool_use" and block.get("id"):
            ids.add(str(block["id"]))
    meta = _message_meta(message)
    for key in ("tool_call_id", "tool_use_id"):
        if meta.get(key) and str(message.get("role") or "") in {"assistant", "tool_use"}:
            ids.add(str(meta[key]))
    return ids


def _tool_result_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    role = str(message.get("role") or "")
    for key in ("tool_call_id", "tool_use_id"):
        value = message.get(key)
        if value:
            ids.add(str(value))
    for block in _content_blocks(message):
        if block.get("type") == "tool_result" and block.get("tool_use_id"):
            ids.add(str(block["tool_use_id"]))
    meta = _message_meta(message)
    for key in ("tool_call_id", "tool_use_id"):
        if meta.get(key) and role in {"tool", "tool_result", "user"}:
            ids.add(str(meta[key]))
    return ids


def adjust_keep_start_to_preserve_api_invariants(messages: list[dict[str, Any]], start_index: int) -> int:
    """Move a suffix boundary left when slicing there would orphan tool results."""
    if start_index <= 0 or start_index >= len(messages):
        return max(0, min(start_index, len(messages)))

    adjusted = start_index
    kept_results: set[str] = set()
    kept_uses: set[str] = set()
    for message in messages[adjusted:]:
        kept_results.update(_tool_result_ids(message))
        kept_uses.update(_tool_use_ids(message))

    missing_uses = kept_results - kept_uses
    if missing_uses:
        for index in range(adjusted - 1, -1, -1):
            uses = _tool_use_ids(messages[index])
            if uses & missing_uses:
                adjusted = index
                missing_uses -= uses
                if not missing_uses:
                    break

    kept_provider_ids = {
        provider_id
        for message in messages[adjusted:]
        if str(message.get("role") or "") == "assistant"
        for provider_id in [_message_provider_id(message)]
        if provider_id
    }
    while adjusted > 0:
        previous = messages[adjusted - 1]
        if str(previous.get("role") or "") != "assistant":
            break
        previous_provider_id = _message_provider_id(previous)
        if not previous_provider_id or previous_provider_id not in kept_provider_ids:
            break
        adjusted -= 1

    return adjusted


def _words(text: str) -> list[str]:
    return WORD_PATTERN.findall((text or "").lower())


def _unique_preserve_order(items: list[str], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        clean = " ".join(str(item or "").strip().split())
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
        if len(output) >= limit:
            break
    return output


def clip_to_token_budget(text: str, max_tokens: int, suffix: str = "...") -> str:
    clean = str(text or "").strip()
    if max_tokens <= 0 or not clean:
        return ""
    if estimate_tokens(clean) <= max_tokens:
        return clean
    approx_chars = max(32, max_tokens * 4)
    clipped = clean[:approx_chars].rstrip()
    while clipped and estimate_tokens(clipped + suffix) > max_tokens:
        clipped = clipped[: max(0, len(clipped) - 24)].rstrip()
    return f"{clipped}{suffix}" if clipped else ""


@dataclass(frozen=True)
class ContextBudget:
    max_input_tokens: int
    reserved_response_tokens: int
    system_tokens: int
    memory_tokens: int
    evidence_tokens: int
    recent_tokens: int
    tool_observation_tokens: int

    @property
    def available_prompt_tokens(self) -> int:
        return max(512, self.max_input_tokens - self.reserved_response_tokens)


def default_context_budget(model: str | None = None) -> ContextBudget:
    if settings.context_dynamic_model_window_enabled:
        max_input = max(settings.context_max_input_tokens, model_context_window_tokens(model))
    else:
        max_input = max(4096, settings.context_max_input_tokens)
    reserved = max(512, min(settings.context_reserved_response_tokens, max_input // 2))
    available = max_input - reserved
    return ContextBudget(
        max_input_tokens=max_input,
        reserved_response_tokens=reserved,
        system_tokens=max(512, int(available * settings.context_system_ratio)),
        memory_tokens=max(512, int(available * settings.context_memory_ratio)),
        evidence_tokens=max(512, int(available * settings.context_evidence_ratio)),
        recent_tokens=max(512, int(available * settings.context_recent_ratio)),
        tool_observation_tokens=max(512, int(available * settings.context_tool_ratio)),
    )


def compress_messages(messages: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    kept_reversed: list[dict[str, str]] = []
    used = 0
    truncated_messages = 0
    exact_suffix_messages = 0
    exact_suffix_open = True
    for message in reversed(messages):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role not in {"user", "assistant"} or not content.strip():
            continue
        per_message_budget = max(96, min(900, max_tokens // 3))
        clipped = clip_to_token_budget(content, per_message_budget)
        cost = estimate_tokens(role) + estimate_tokens(clipped)
        if kept_reversed and used + cost > max_tokens:
            break
        kept_reversed.append({"role": role, "content": clipped})
        if clipped == content.strip():
            if exact_suffix_open:
                exact_suffix_messages += 1
        else:
            truncated_messages += 1
            exact_suffix_open = False
        used += cost
        if used >= max_tokens:
            break
    kept = list(reversed(kept_reversed))
    total_eligible_messages = sum(
        1
        for message in messages
        if str(message.get("role") or "") in {"user", "assistant"}
        and str(message.get("content") or "").strip()
    )
    return kept, {
        "input_messages": len(messages),
        "eligible_messages": total_eligible_messages,
        "kept_messages": len(kept),
        "omitted_messages": max(0, total_eligible_messages - len(kept)),
        "truncated_messages": truncated_messages,
        "exact_suffix_messages": exact_suffix_messages,
        "estimated_tokens": used,
        "budget_tokens": max_tokens,
    }


def _tool_names(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    route = str(message.get("route") or "").strip()
    if route:
        names.append(route)
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            name = str(call.get("tool_name") or "").strip()
            if name:
                names.append(name)
    return names


def is_regenerable_context_message(message: dict[str, Any]) -> bool:
    role = str(message.get("role") or "")
    meta = _message_meta(message)
    if role not in {"tool", "tool_result"} and meta.get("subtype") != "tool_result":
        return False
    if meta.get("regenerable") is False:
        return False
    haystack = " ".join(_tool_names(message)).lower()
    if not haystack:
        return False
    return any(hint in haystack for hint in COMPACTABLE_TOOL_HINTS)


def microcompact_messages(
    messages: list[dict[str, Any]],
    query: str,
    *,
    keep_recent: int | None = None,
    max_excerpt_tokens: int = 90,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keep = keep_recent if keep_recent is not None else settings.context_microcompact_keep_recent_messages
    keep_start = adjust_keep_start_to_preserve_api_invariants(
        messages,
        max(0, len(messages) - max(0, keep)),
    )
    output: list[dict[str, Any]] = []
    compacted = 0
    tokens_before = 0
    tokens_after = 0
    compacted_tools: list[str] = []

    for index, message in enumerate(messages):
        content = str(message.get("content") or "")
        before = estimate_tokens(content)
        tokens_before += before
        next_message = dict(message)
        if (
            index < keep_start
            and is_regenerable_context_message(message)
            and before > max_excerpt_tokens * 2
        ):
            names = _tool_names(message)
            excerpt = extract_relevant_snippet(query, content, max_tokens=max_excerpt_tokens)
            tool_label = ", ".join(_unique_preserve_order(names, 4)) or "tool output"
            compacted_content = f"{MICROCOMPACT_MARKER}\nSource: {tool_label}\nRecoverable: rerun the corresponding workspace, RAG, or GitHub tool if exact details are needed."
            if excerpt:
                compacted_content += f"\nKey excerpt: {excerpt}"
            next_message["content"] = compacted_content
            compacted += 1
            compacted_tools.extend(names)
        tokens_after += estimate_tokens(str(next_message.get("content") or ""))
        output.append(next_message)

    return output, {
        "mode": "microcompact_regenerable_context",
        "input_messages": len(messages),
        "compacted_messages": compacted,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "estimated_tokens_saved": max(0, tokens_before - tokens_after),
        "kept_recent_messages": max(0, min(keep, len(messages))),
        "compacted_tools": _unique_preserve_order(compacted_tools, 12),
    }


def extract_relevant_snippet(query: str, text: str, max_tokens: int = 220) -> str:
    clean = " ".join(str(text or "").replace("\r", "\n").split())
    if not clean:
        return ""
    terms = set(_words(query))
    if not terms:
        return clip_to_token_budget(clean, max_tokens)

    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(clean) if part.strip()]
    if not sentences:
        sentences = [clean]

    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        matched = [term for term in terms if term and term in lower]
        score = len(set(matched)) * 3.0 + sum(lower.count(term) for term in matched)
        if query.lower().strip() and query.lower().strip() in lower:
            score += 4.0
        scored.append((score, index, sentence))

    positive = [item for item in scored if item[0] > 0]
    if not positive:
        return clip_to_token_budget(clean, max_tokens)

    positive.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(positive[:4], key=lambda item: item[1])
    snippet = " ".join(sentence for _score, _index, sentence in selected)
    return clip_to_token_budget(snippet, max_tokens)


def compress_retrieval_results(
    query: str,
    items: list[dict[str, Any]],
    max_tokens_per_item: int = 260,
    max_total_tokens: int | None = None,
) -> list[dict[str, Any]]:
    compressed: list[dict[str, Any]] = []
    total_budget = max(0, int(max_total_tokens)) if max_total_tokens is not None else None
    used_tokens = 0
    for item in items:
        if total_budget is not None and used_tokens >= total_budget:
            break
        if (
            total_budget is not None
            and compressed
            and total_budget - used_tokens < min(24, max(1, int(max_tokens_per_item)))
        ):
            break
        item_budget = max(1, int(max_tokens_per_item))
        if total_budget is not None:
            item_budget = min(item_budget, total_budget - used_tokens)
        raw_text = str(item.get("_raw_content") or item.get("content") or item.get("snippet") or "")
        snippet = extract_relevant_snippet(query, raw_text, max_tokens=item_budget)
        original_tokens = estimate_tokens(raw_text)
        compressed_tokens = estimate_tokens(snippet)
        next_item = {key: value for key, value in item.items() if key not in {"_raw_content", "content"}}
        next_item["snippet"] = snippet
        next_item["compression"] = {
            "mode": "query_aware_extractive",
            "original_estimated_tokens": original_tokens,
            "compressed_estimated_tokens": compressed_tokens,
            "ratio": round(compressed_tokens / original_tokens, 4) if original_tokens else 1.0,
            "item_budget_tokens": item_budget,
            "global_budget_tokens": total_budget,
            "global_tokens_before_item": used_tokens,
        }
        retrieval = dict(next_item.get("retrieval") or {})
        retrieval["context_compression"] = next_item["compression"]
        retrieval["selected_for_context"] = True
        next_item["retrieval"] = retrieval
        compressed.append(next_item)
        used_tokens += compressed_tokens
    return compressed


COMPACT_SUMMARY_SYSTEM_PROMPT = """请在上下文压缩后，为 DevFlow AI 创建一份可延续工作的摘要。
目标不是单纯减少 token，而是为软件工程 Agent 保留工作连续性。

请用纯文本返回以下部分：
1. 用户意图和明确要求。
2. 当前工作面：现在正在处理什么；如相关，请包含仓库、文件、函数、命令、测试和 UI 状态。
3. 持久决策和约束。
4. 重要事实、代码引用、错误和修复。
5. 待办任务和下一步，并以最近一次用户请求为依据。

当文件路径、标识符、日期、命令、错误消息和用户偏好很重要时，请保留原文。
不要编造细节。如果某个细节只存在于可以重新生成的工具输出中，请说明它可以通过相关工具恢复。"""

PATH_LIKE_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/])?(?:[\w.@()+\- ]+[\\/])+[\w.@()+\- ]+\.[A-Za-z0-9_]+(?::\d+)?"
)
COMMAND_HINT_PATTERN = re.compile(
    r"\b(pytest|npm|pnpm|yarn|git|docker|uvicorn|alembic|python|pip|ruff|mypy|pytest)\b",
    flags=re.IGNORECASE,
)
PRESERVE_LINE_HINTS = (
    "error",
    "failed",
    "failure",
    "exception",
    "traceback",
    "decision",
    "decide",
    "agreed",
    "todo",
    "next step",
    "prefer",
    "preference",
    "错误",
    "失败",
    "异常",
    "决定",
    "确定",
    "待办",
    "下一步",
    "偏好",
)


def _must_preserve_candidates(messages: list[dict[str, Any]], limit: int = 18) -> list[str]:
    candidates: list[str] = []
    for message in messages:
        content = str(message.get("content") or "")
        if not content:
            continue
        candidates.extend(match.group(0).strip() for match in PATH_LIKE_PATTERN.finditer(content))
        for raw_line in re.split(r"[\r\n]+", content):
            line = " ".join(raw_line.strip(" -\t").split())
            if not line:
                continue
            lower = line.lower()
            if COMMAND_HINT_PATTERN.search(line) or any(hint in lower for hint in PRESERVE_LINE_HINTS):
                candidates.append(clip_to_token_budget(line, 120))
    return _unique_preserve_order(candidates, limit)


def compaction_preservation_report(messages: list[dict[str, Any]], summary: str) -> dict[str, Any]:
    required = _must_preserve_candidates(messages)
    summary_lower = str(summary or "").lower()
    missing = [item for item in required if item.lower() not in summary_lower]
    return {
        "required_count": len(required),
        "missing_count": len(missing),
        "required": required,
        "missing": missing,
    }


def ensure_compaction_summary_preserves_key_facts(
    messages: list[dict[str, Any]],
    summary: str,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    report = compaction_preservation_report(messages, summary)
    missing = [str(item) for item in report.get("missing") or []]
    if not missing:
        return clip_to_token_budget(summary, max_tokens), report

    appendix = "保真补充：\n" + "\n".join(
        f"- {clip_to_token_budget(item, 96)}" for item in missing[:12]
    )
    body_budget = max(160, max_tokens - estimate_tokens(appendix) - 12)
    combined = "\n\n".join([clip_to_token_budget(summary, body_budget), appendix]).strip()
    combined = clip_to_token_budget(combined, max_tokens)
    final_report = compaction_preservation_report(messages, combined)
    final_report["appended_missing_count"] = len(missing)
    return combined, final_report


def build_compaction_summary_payload(messages: list[dict[str, Any]], max_tokens: int = 6000) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        if role == "system" and is_compact_boundary_message(message):
            role = "compact_boundary"
        content = clip_to_token_budget(str(message.get("content") or ""), max(160, max_tokens // max(len(messages), 1)))
        if not content:
            continue
        tool_names = _tool_names(message)
        suffix = f" tools={tool_names}" if tool_names else ""
        lines.append(f"{role}{suffix}: {content}")
    return clip_to_token_budget("\n\n".join(lines), max_tokens)


def format_compaction_summary(summary: str) -> str:
    formatted = str(summary or "").strip()
    formatted = re.sub(r"<analysis>[\s\S]*?</analysis>", "", formatted, flags=re.IGNORECASE).strip()
    match = re.search(r"<summary>([\s\S]*?)</summary>", formatted, flags=re.IGNORECASE)
    if match:
        formatted = "摘要：\n" + match.group(1).strip()
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted.strip()


def fallback_compaction_summary(messages: list[dict[str, Any]], max_tokens: int | None = None) -> str:
    user_messages = [str(item.get("content") or "") for item in messages if item.get("role") == "user"]
    assistant_messages = [str(item.get("content") or "") for item in messages if item.get("role") == "assistant"]
    decisions: list[str] = []
    tasks: list[str] = []
    facts: list[str] = []
    for content in [*user_messages, *assistant_messages]:
        for line in [part.strip(" -\t") for part in content.splitlines() if part.strip()]:
            lower = line.lower()
            if any(token in lower for token in ["decision", "decide", "agreed", "\u51b3\u5b9a", "\u786e\u5b9a"]):
                decisions.append(line)
            if any(token in lower for token in ["todo", "next step", "implement", "fix", "\u5f85\u529e", "\u4e0b\u4e00\u6b65"]):
                tasks.append(line)
            if len(line) > 20 and len(facts) < 10:
                facts.append(line)
    summary = "\n\n".join(
        [
            "This conversation was compacted. Continue from the preserved recent messages and this summary.",
            "User intent:\n" + "\n".join(f"- {clip_to_token_budget(item, 120)}" for item in user_messages[-8:]),
            "Assistant context:\n" + "\n".join(f"- {clip_to_token_budget(item, 120)}" for item in assistant_messages[-5:]),
            "Durable decisions:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(decisions, 12)),
            "Pending tasks:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(tasks, 12)),
            "Important facts:\n" + "\n".join(f"- {item}" for item in _unique_preserve_order(facts, 12)),
        ]
    )
    return clip_to_token_budget(summary, max_tokens or settings.context_compact_summary_tokens)


def structured_memory_from_model(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": str(parsed.get("summary") or "").strip(),
        "facts": _unique_preserve_order([str(item) for item in parsed.get("facts") or []], 16),
        "decisions": _unique_preserve_order([str(item) for item in parsed.get("decisions") or []], 16),
        "open_questions": _unique_preserve_order([str(item) for item in parsed.get("open_questions") or []], 12),
        "tasks": _unique_preserve_order([str(item) for item in parsed.get("tasks") or []], 16),
        "user_preferences": _unique_preserve_order([str(item) for item in parsed.get("user_preferences") or []], 12),
        "repo_context": _unique_preserve_order([str(item) for item in parsed.get("repo_context") or []], 16),
        "citations": [item for item in parsed.get("citations") or [] if isinstance(item, dict)][:16],
    }


def fallback_memory_update(user_message: str, assistant_message: str, route: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    joined = "\n".join([user_message or "", assistant_message or ""])
    lines = [line.strip(" -\t") for line in joined.splitlines() if line.strip()]
    decisions: list[str] = []
    open_questions: list[str] = []
    tasks: list[str] = []
    facts: list[str] = []
    preferences: list[str] = []

    for line in lines:
        lower = line.lower()
        if "?" in line or "\uff1f" in line or lower.startswith(("why ", "how ", "what ")):
            open_questions.append(line)
        if any(token in lower for token in ["decide", "decision", "agreed", "chosen", "\u786e\u5b9a", "\u51b3\u5b9a"]):
            decisions.append(line)
        if any(token in lower for token in ["todo", "next step", "action", "fix", "implement", "\u5f85\u529e", "\u4e0b\u4e00\u6b65"]):
            tasks.append(line)
        if any(token in lower for token in ["prefer", "preference", "\u559c\u6b22", "\u504f\u597d"]):
            preferences.append(line)
        if len(line) > 20 and len(facts) < 6:
            facts.append(line)

    tools = ", ".join(str(call.get("tool_name") or "未知") for call in tool_calls) or "无"
    summary = (
        f"路由：{route}\n"
        f"用户请求：{clip_to_token_budget(user_message, 90)}\n"
        f"助手回答：{clip_to_token_budget(assistant_message, 140)}\n"
        f"工具：{tools}"
    )
    return structured_memory_from_model(
        {
            "summary": summary,
            "facts": facts,
            "decisions": decisions,
            "open_questions": open_questions,
            "tasks": tasks,
            "user_preferences": preferences,
            "repo_context": [f"使用路由：{route}"] if route else [],
            "citations": [],
        }
    )


def merge_memory(existing: dict[str, Any], update: dict[str, Any], digest: str, max_tokens: int) -> dict[str, Any]:
    existing_summary = str(existing.get("summary") or "").strip()
    update_summary = str(update.get("summary") or digest or "").strip()
    summary_parts = [part for part in [existing_summary, update_summary] if part]
    summary = "\n\n".join(summary_parts)
    summary = clip_to_token_budget(summary, max_tokens)
    return {
        "summary": summary,
        "facts": _unique_preserve_order([*(existing.get("facts") or []), *(update.get("facts") or [])], 16),
        "decisions": _unique_preserve_order([*(existing.get("decisions") or []), *(update.get("decisions") or [])], 16),
        "open_questions": _unique_preserve_order([*(existing.get("open_questions") or []), *(update.get("open_questions") or [])], 12),
        "tasks": _unique_preserve_order([*(existing.get("tasks") or []), *(update.get("tasks") or [])], 16),
        "user_preferences": _unique_preserve_order([*(existing.get("user_preferences") or []), *(update.get("user_preferences") or [])], 12),
        "repo_context": _unique_preserve_order([*(existing.get("repo_context") or []), *(update.get("repo_context") or [])], 16),
        "citations": [*(existing.get("citations") or []), *(update.get("citations") or [])][-16:],
    }


def render_structured_memory(memory: Any, max_tokens: int | None = None) -> str:
    sections: list[str] = []
    summary = str(getattr(memory, "summary", "") or "").strip()
    if summary:
        sections.append("摘要：\n" + summary)
    for label, attr in [
        ("事实", "facts"),
        ("决策", "decisions"),
        ("未决问题", "open_questions"),
        ("任务", "tasks"),
        ("用户偏好", "user_preferences"),
        ("仓库上下文", "repo_context"),
    ]:
        values = getattr(memory, attr, None) or []
        if values:
            sections.append(label + ":\n" + "\n".join(f"- {item}" for item in values[:12]))
    text = "\n\n".join(sections)
    return clip_to_token_budget(text, max_tokens) if max_tokens else text


def memory_document_text(memory: Any) -> str:
    return render_structured_memory(memory, max_tokens=None)


def prompt_cache_model_kwargs() -> dict[str, Any]:
    if not settings.prompt_cache_enabled:
        return {}
    kwargs: dict[str, Any] = {}
    if settings.prompt_cache_key:
        kwargs["prompt_cache_key"] = settings.prompt_cache_key
    if settings.prompt_cache_retention:
        kwargs["prompt_cache_retention"] = settings.prompt_cache_retention
    return kwargs


def compression_eval_summary(before_text: str, after_text: str, retrieved: int, kept: int) -> dict[str, Any]:
    before = estimate_tokens(before_text)
    after = estimate_tokens(after_text)
    return {
        "estimated_tokens_before": before,
        "estimated_tokens_after": after,
        "compression_ratio": round(after / before, 4) if before else 1.0,
        "retrieved_items": retrieved,
        "kept_items": kept,
        "token_savings": max(0, before - after),
    }


def dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
