import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    AgentRun,
    ChatMessage,
    ChatSession,
    ChatSession as ChatSessionModel,
    Conversation,
    ConversationMemory,
    Document,
    EvidenceItem,
    Issue,
    PullRequest,
    Repository,
    ThreadMemory,
    WorkflowRun,
)
from app.services.context_compression import (
    COMPACT_BOUNDARY_SUBTYPE,
    COMPACT_SUMMARY_SYSTEM_PROMPT,
    adjust_keep_start_to_preserve_api_invariants,
    build_compaction_summary_payload,
    calculate_context_pressure,
    clip_to_token_budget,
    compress_messages,
    default_context_budget,
    dumps_compact,
    ensure_compaction_summary_preserves_key_facts,
    estimate_tokens,
    extract_relevant_snippet,
    fallback_memory_update,
    fallback_compaction_summary,
    format_compaction_summary,
    is_compact_boundary_message,
    messages_after_compact_boundary,
    memory_document_text,
    merge_memory,
    microcompact_messages,
    render_structured_memory,
    structured_memory_from_model,
)
from app.services.llm.client import LLMClient
from app.services.memory_hub import capture_memory_candidates, record_recall_event
from app.services.rag.indexing import CI_LOG_CHUNK_CHARS, sanitize_ci_log
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES
from app.services.rag.vector_store import MilvusUnavailableError


def _repo_filter(model: Any, repo_id: uuid.UUID | None) -> Any:
    return model.repo_id.is_(None) if repo_id is None else model.repo_id == repo_id


def _conversation_filter(model: Any, conversation_id: uuid.UUID | None) -> Any:
    return model.conversation_id.is_(None) if conversation_id is None else model.conversation_id == conversation_id


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+", text.lower())


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


class MessageStore:
    def __init__(self, db: Session) -> None:
        self.db = db

    def append(
        self,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        role: str,
        content: str,
        route: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            repo_id=repo_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            route=route,
            tool_calls=tool_calls or [],
            meta=meta or {},
        )
        self.db.add(message)
        self.db.flush()
        return message

    def recent(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int = 20) -> list[ChatMessage]:
        """Return the active context projection, not the complete transcript.

        New compact boundaries keep references to the original recent messages
        instead of copying them.  Older databases can still contain
        ``preserved_after_compact`` copies, so this method understands both
        layouts during the migration period.
        """
        boundary = self.last_compact_boundary(repo_id, conversation_id)
        if boundary is None:
            return self.timeline(repo_id, conversation_id, limit=limit, include_system=True)

        boundary_meta = boundary.meta or {}
        preserved_ids = [
            value
            for value in boundary_meta.get("preserved_message_ids") or []
            if value
        ]
        preserved_rows: list[ChatMessage] = []
        if preserved_ids:
            parsed_ids: list[uuid.UUID] = []
            for value in preserved_ids:
                try:
                    parsed_ids.append(uuid.UUID(str(value)))
                except ValueError:
                    continue
            if parsed_ids:
                by_id = {
                    row.id: row
                    for row in self.db.query(ChatMessage)
                    .filter(
                        _repo_filter(ChatMessage, repo_id),
                        _conversation_filter(ChatMessage, conversation_id),
                        ChatMessage.id.in_(parsed_ids),
                    )
                    .all()
                }
                preserved_rows = [by_id[item] for item in parsed_ids if item in by_id]

        rows_after_boundary = (
            self.db.query(ChatMessage)
            .filter(
                _repo_filter(ChatMessage, repo_id),
                _conversation_filter(ChatMessage, conversation_id),
                ChatMessage.created_at >= boundary.created_at,
            )
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            .all()
        )
        projected = [boundary, *preserved_rows]
        projected.extend(
            row
            for row in rows_after_boundary
            if row.id != boundary.id
            and str(row.id) not in preserved_ids
            and not (
                preserved_ids
                and (row.meta or {}).get("subtype") == "preserved_after_compact"
            )
        )
        return projected[-max(1, limit) :]

    def timeline(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        *,
        limit: int | None = 500,
        include_system: bool = False,
        before_message_id: uuid.UUID | None = None,
        after_message_id: uuid.UUID | None = None,
    ) -> list[ChatMessage]:
        """Return the durable transcript while hiding compatibility copies.

        Compact boundaries change the model's active projection; they do not
        erase the transcript.  This method is the authoritative read path for
        UI history and exact-memory tools.
        """
        query = self.db.query(ChatMessage).filter(
            _repo_filter(ChatMessage, repo_id),
            _conversation_filter(ChatMessage, conversation_id),
        )
        if not include_system:
            query = query.filter(ChatMessage.role != "system")

        if before_message_id:
            before = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == before_message_id,
                )
                .one_or_none()
            )
            if before and before.created_at:
                query = query.filter(
                    or_(
                        ChatMessage.created_at < before.created_at,
                        and_(
                            ChatMessage.created_at == before.created_at,
                            ChatMessage.id < before.id,
                        ),
                    )
                )
        if after_message_id:
            after = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == after_message_id,
                )
                .one_or_none()
            )
            if after and after.created_at:
                query = query.filter(
                    or_(
                        ChatMessage.created_at > after.created_at,
                        and_(
                            ChatMessage.created_at == after.created_at,
                            ChatMessage.id > after.id,
                        ),
                    )
                )

        rows = query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).all()
        durable = [
            row
            for row in rows
            if (row.meta or {}).get("subtype") != "preserved_after_compact"
        ]
        if limit is None:
            return durable
        if after_message_id and not before_message_id:
            return durable[: max(1, limit)]
        return durable[-max(1, limit) :]

    def active_for_context(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        limit: int = 1000,
    ) -> list[ChatMessage]:
        return self.recent(repo_id, conversation_id, limit=limit)

    def all_for_context(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int = 200) -> list[ChatMessage]:
        rows = (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id))
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))

    def last_compact_boundary(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ChatMessage | None:
        rows = (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id), ChatMessage.role == "system")
            .order_by(ChatMessage.created_at.desc())
            .limit(50)
            .all()
        )
        for row in rows:
            if is_compact_boundary_message({"role": row.role, "meta": row.meta or {}}):
                return row
        return None


class EvidenceStore:
    def __init__(self, db: Session) -> None:
        self.db = db

    def rebuild_repo(self, repo: Repository | None, conversation: Conversation | None = None) -> None:
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None
        if conversation_id:
            self.db.query(EvidenceItem).filter(
                _repo_filter(EvidenceItem, repo_id),
                or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)),
            ).delete(synchronize_session=False)
        else:
            self.db.query(EvidenceItem).filter(_repo_filter(EvidenceItem, repo_id)).delete(synchronize_session=False)

        if repo:
            self._add(
                repo_id,
                None,
                "repository",
                str(repo.id),
                repo.full_name,
                "\n".join(filter(None, [repo.description or "", f"默认分支：{repo.default_branch or '未知'}"])),
                {"anchor": f"repo:{repo.full_name}"},
            )
            for issue in self.db.query(Issue).filter(Issue.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "issue",
                    str(issue.id),
                    f"Issue #{issue.number}: {issue.title}",
                    "\n".join(filter(None, [issue.title, issue.body or "", f"state: {issue.state}", f"labels: {', '.join(issue.labels or [])}"])),
                    {"number": issue.number, "state": issue.state},
                )
            for pr in self.db.query(PullRequest).filter(PullRequest.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "pull_request",
                    str(pr.id),
                    f"PR #{pr.number}: {pr.title}",
                    "\n".join(filter(None, [pr.title, pr.body or "", f"state: {pr.state}", f"base: {pr.base_branch}", f"head: {pr.head_branch}"])),
                    {"number": pr.number, "state": pr.state},
                )
            for run in self.db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).all():
                if run.conclusion != "failure" or not (run.logs_text or "").strip():
                    continue
                sanitized_logs = sanitize_ci_log(run.logs_text or "")
                safe_logs = sanitized_logs[:CI_LOG_CHUNK_CHARS]
                self._add(
                    repo.id,
                    None,
                    "workflow_run",
                    str(run.id),
                    f"CI: {run.name}",
                    "\n".join(filter(None, [run.name, run.status, run.conclusion or "", safe_logs])),
                    {
                        "status": run.status,
                        "conclusion": run.conclusion,
                        "logs_truncated": len(sanitized_logs) > CI_LOG_CHUNK_CHARS,
                        "secrets_redacted": True,
                    },
                )
            for doc in (
                self.db.query(Document)
                .filter(Document.repo_id == repo.id, Document.source_type.in_(RAG_SOURCE_TYPES))
                .all()
            ):
                self._add(repo.id, None, doc.source_type, str(doc.source_id), doc.title, doc.content, doc.meta or {})

        memories = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).all()
            if conversation_id
            else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).all()
        )
        for memory in memories:
            self._add(
                repo_id,
                conversation_id,
                "memory",
                str(memory.id),
                "Thread memory",
                memory_document_text(memory),
                {"sessions_incorporated": memory.sessions_incorporated},
            )
        session_query = self.db.query(ChatSession).filter(_repo_filter(ChatSession, repo_id))
        run_query = self.db.query(AgentRun).filter(_repo_filter(AgentRun, repo_id))
        if conversation_id:
            session_query = session_query.filter(ChatSession.conversation_id == conversation_id)
            run_query = run_query.filter(AgentRun.conversation_id == conversation_id)
        for session in session_query.order_by(ChatSession.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "session", str(session.id), f"Session digest: {session.route}", session.digest, {"route": session.route})
        for run in run_query.order_by(AgentRun.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "agent_run", str(run.id), f"Agent run: {run.route}", f"{run.user_message}\n{run.final_answer}", {"route": run.route})
        self.db.flush()

    def search(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_terms = _tokens(query)
        if not query_terms:
            return []
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        if repo_id:
            try:
                rag_results = search_similar_documents(
                    self.db,
                    repo_id,
                    query,
                    source_type=None,
                    limit=max(limit, 8),
                )
            except MilvusUnavailableError:
                rag_results = []
            for item in rag_results:
                key = (str(item.get("source_type") or ""), str(item.get("source_id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "title": item.get("title") or "",
                        "source_type": item.get("source_type") or "",
                        "source_id": item.get("source_id") or "",
                        "snippet": item.get("snippet") or "",
                        "score": item.get("score") or 0,
                        "metadata": item.get("metadata") or {},
                        "retrieval": item.get("retrieval") or {},
                    }
                )
                if len(results) >= limit:
                    return results
        candidates = (
            self.db.query(EvidenceItem)
            .filter(
                _repo_filter(EvidenceItem, repo_id),
                or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)),
            )
            .order_by(EvidenceItem.created_at.desc())
            .limit(500)
            .all()
        )
        ranked: list[tuple[int, EvidenceItem]] = []
        for item in candidates:
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            haystack = f"{item.title}\n{item.content}".lower()
            score = sum(haystack.count(term) for term in query_terms)
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        for score, item in ranked:
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "title": item.title,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "snippet": extract_relevant_snippet(query, item.content, max_tokens=90),
                    "score": score,
                    "metadata": item.meta or {},
                    "retrieval": {"sources": ["evidence_item_keyword"]},
                }
            )
            if len(results) >= limit:
                break
        return results

    def _add(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, source_type: str, source_id: str, title: str, content: str, meta: dict[str, Any]) -> None:
        if not content.strip():
            return
        self.db.add(EvidenceItem(repo_id=repo_id, conversation_id=conversation_id, source_type=source_type, source_id=source_id, title=title, content=content, meta=meta))


@dataclass
class AssembledContext:
    system_context: str
    history_messages: list[dict[str, str]]
    evidence: list[dict[str, Any]]
    compression_stats: dict[str, Any] = field(default_factory=dict)


class ContextAssembler:
    def __init__(self, db: Session) -> None:
        self.db = db

    def assemble(self, repo: Repository | None, conversation: Conversation | None, current_message: str, limit: int = 16) -> AssembledContext:
        budget = default_context_budget()
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None
        message_store = MessageStore(self.db)
        boundary = message_store.last_compact_boundary(repo_id, conversation_id)
        memory = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
            if conversation_id
            else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()
        )
        latest_session = (
            self.db.query(ChatSession)
            .filter(
                _repo_filter(ChatSession, repo_id),
                _conversation_filter(ChatSession, conversation_id),
                ChatSession.status.in_(["sealed", "sealed_with_warnings"]),
            )
            .order_by(ChatSession.created_at.desc())
            .first()
        )
        evidence = EvidenceStore(self.db).search(repo_id, conversation_id, current_message, limit=8)
        history_fetch_limit = max(limit, 30)
        recent = message_store.recent(repo_id, conversation_id, limit=history_fetch_limit)
        raw_history = [
            {
                "role": item.role,
                "content": item.content,
                "route": item.route,
                "tool_calls": item.tool_calls or [],
                "meta": item.meta or {},
            }
            for item in recent
            if item.role in {"user", "assistant"}
        ]
        micro_stats: dict[str, Any] | None = None
        raw_history_tokens = sum(
            estimate_tokens(str(item.get("role") or ""))
            + estimate_tokens(str(item.get("content") or ""))
            + (
                estimate_tokens(dumps_compact(item.get("tool_calls")))
                if item.get("tool_calls")
                else 0
            )
            for item in raw_history
        )
        current_message_in_history = bool(
            raw_history
            and raw_history[-1].get("role") == "user"
            and " ".join(str(raw_history[-1].get("content") or "").split())
            == " ".join(str(current_message or "").split())
        )
        pending_user_tokens = 0 if current_message_in_history else estimate_tokens(current_message)
        raw_history_pressure = calculate_context_pressure(raw_history_tokens + pending_user_tokens)
        history_source = raw_history
        if raw_history_pressure.should_warn:
            history_source, micro_stats = microcompact_messages(raw_history, current_message)
        history, history_stats = compress_messages(history_source, budget.recent_tokens)

        def build_context(next_history: list[dict[str, str]], next_evidence: list[dict[str, Any]]) -> str:
            system_sections: list[str] = []
            if repo:
                system_sections.append(f"[Current Repository]\n{repo.full_name}\n[/Current Repository]")
            if conversation:
                system_sections.append(f"[Current Conversation]\n{conversation.title}\n[/Current Conversation]")

            sections: list[str] = []
            stable_system = clip_to_token_budget("\n\n".join(system_sections), budget.system_tokens)
            if stable_system:
                sections.append(stable_system)

            memory_sections: list[str] = []
            if boundary and boundary.content:
                transcript_ref = (boundary.meta or {}).get("transcript_ref") or {}
                rehydration = (boundary.meta or {}).get("rehydration") or {}
                recovery_hint = (
                    "\nExact transcript recovery: call devflow_get_thread_context with this conversation"
                    if transcript_ref
                    else ""
                )
                if rehydration:
                    memory_sections.append(
                        "[Rehydrated Working State]\n"
                        + dumps_compact(rehydration)
                        + "\n[/Rehydrated Working State]"
                    )
                memory_sections.append(
                    f"[Compaction Boundary]\n{boundary.content}{recovery_hint}\n[/Compaction Boundary]"
                )
            if memory and memory_document_text(memory).strip():
                memory_text = render_structured_memory(memory, max_tokens=max(256, budget.memory_tokens // 2))
                memory_sections.append(f"[Thread Memory: {memory.sessions_incorporated} sessions]\n{memory_text}\n[/Thread Memory]")
            if latest_session and latest_session.digest:
                memory_sections.append(f"[Previous Session Digest]\n{latest_session.digest}\n[/Previous Session Digest]")
            memory_context = clip_to_token_budget("\n\n".join(memory_sections), budget.memory_tokens)
            if memory_context:
                sections.append(memory_context)
            if next_evidence:
                evidence_budget = max(120, budget.evidence_tokens // max(len(next_evidence), 1))
                lines = ["[Related Evidence]"]
                for item in next_evidence:
                    snippet = clip_to_token_budget(item["snippet"], evidence_budget)
                    lines.append(f"- [{item['source_type']}] {item['title']}: {snippet}")
                lines.append("[/Related Evidence]")
                sections.append(clip_to_token_budget("\n".join(lines), budget.evidence_tokens))
            return "\n\n".join(sections)

        system_context = build_context(history, evidence)
        history_prompt_tokens = sum(
            estimate_tokens(item.get("role")) + estimate_tokens(item.get("content"))
            for item in history
        )
        initial_prompt_tokens = estimate_tokens(system_context) + history_prompt_tokens + pending_user_tokens
        initial_pressure = calculate_context_pressure(initial_prompt_tokens)
        if initial_pressure.should_warn and micro_stats is None:
            micro_raw_history, micro_stats = microcompact_messages(raw_history, current_message)
            history, history_stats = compress_messages(micro_raw_history, budget.recent_tokens)
            system_context = build_context(history, evidence)

        history_prompt_tokens = sum(
            estimate_tokens(item.get("role")) + estimate_tokens(item.get("content"))
            for item in history
        )
        assembled_prompt_tokens = estimate_tokens(system_context) + history_prompt_tokens + pending_user_tokens
        final_pressure = calculate_context_pressure(assembled_prompt_tokens)
        microcompacted_messages = int((micro_stats or {}).get("compacted_messages") or 0)
        history_coverage = {
            "source_messages": len(raw_history),
            "prompt_messages": len(history),
            "omitted_messages": int(history_stats.get("omitted_messages") or 0),
            "truncated_messages": int(history_stats.get("truncated_messages") or 0),
            "exact_suffix_messages": int(history_stats.get("exact_suffix_messages") or 0),
            "source_limit_reached": len(recent) >= history_fetch_limit,
            "has_compaction_boundary": boundary is not None,
            "microcompacted_messages": microcompacted_messages,
        }
        history_coverage["complete_and_exact"] = not any(
            [
                history_coverage["omitted_messages"],
                history_coverage["truncated_messages"],
                history_coverage["source_limit_reached"],
                history_coverage["has_compaction_boundary"],
                history_coverage["microcompacted_messages"],
            ]
        )

        return AssembledContext(
            system_context=system_context,
            history_messages=history,
            evidence=evidence,
            compression_stats={
                "budget": budget.__dict__,
                "history": history_stats,
                "history_coverage": history_coverage,
                "microcompact": micro_stats,
                "pressure": final_pressure.as_dict(),
                "initial_pressure": initial_pressure.as_dict(),
                "raw_history_pressure": raw_history_pressure.as_dict(),
                "compact_boundary": {
                    "id": str(boundary.id) if boundary else None,
                    "created_at": boundary.created_at.isoformat() if boundary and boundary.created_at else None,
                    "mode": (boundary.meta or {}).get("mode") if boundary else None,
                },
                "system_context_estimated_tokens": estimate_tokens(system_context),
                "history_estimated_tokens": history_prompt_tokens,
                "pending_user_estimated_tokens": pending_user_tokens,
                "assembled_prompt_estimated_tokens": assembled_prompt_tokens,
                "prompt_breakdown": {
                    "system_memory_evidence": estimate_tokens(system_context),
                    "recent_history": history_prompt_tokens,
                    "pending_user": pending_user_tokens,
                    "tool_observation_budget": budget.tool_observation_tokens,
                    "reserved_response": budget.reserved_response_tokens,
                },
                "evidence_items": len(evidence),
                "mode": "progressive_budgeted_context",
            },
        )


class ProgressiveContextManager:
    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()

    async def ensure_headroom(self, repo: Repository | None, conversation: Conversation | None, current_message: str) -> dict[str, Any]:
        if repo is None or conversation is None:
            return {"attempted": False, "reason": "no_repo_or_conversation"}
        assembled = ContextAssembler(self.db).assemble(repo, conversation, current_message)
        assembled_pressure = assembled.compression_stats.get("pressure") or {}
        micro_stats = assembled.compression_stats.get("microcompact") or {}
        raw_segment_tokens = self._raw_segment_tokens(repo.id, conversation.id) + estimate_tokens(current_message)
        assembled_tokens = int(
            assembled.compression_stats.get("assembled_prompt_estimated_tokens")
            or assembled.compression_stats.get("system_context_estimated_tokens")
            or 0
        )
        raw_pressure_state = calculate_context_pressure(raw_segment_tokens)
        assembled_pressure_state = calculate_context_pressure(assembled_tokens)
        pressure_state = calculate_context_pressure(max(raw_segment_tokens, assembled_tokens))
        pressure = {
            **pressure_state.as_dict(),
            "raw_segment_estimated_tokens": raw_segment_tokens,
            "assembled_context_estimated_tokens": assembled_tokens,
            "raw_stage": raw_pressure_state.stage,
            "assembled_stage": assembled_pressure.get("stage") or assembled_pressure_state.stage,
            "microcompact_estimated_tokens_saved": micro_stats.get("estimated_tokens_saved", 0),
            "microcompact_compacted_messages": micro_stats.get("compacted_messages", 0),
        }
        raw_stage = raw_pressure_state.stage
        prompt_stage = str(pressure.get("assembled_stage") or assembled_pressure_state.stage)
        stage = str(pressure.get("stage") or "normal")
        if not settings.context_auto_compact_enabled:
            return {"attempted": False, "reason": "auto_compact_disabled", "pressure": pressure}
        if (
            prompt_stage in {"normal", "warning"}
            and raw_stage in {"auto_compact", "aggressive", "manual_path"}
            and int(micro_stats.get("compacted_messages") or 0) > 0
        ):
            return {
                "attempted": False,
                "reason": "microcompact_sufficient",
                "pressure": pressure,
                "microcompact": micro_stats,
            }
        if stage in {"normal", "warning"}:
            return {"attempted": False, "reason": stage, "pressure": pressure}

        failures = self._consecutive_failures(repo.id, conversation.id)
        if failures >= settings.context_max_compaction_failures:
            return {
                "attempted": False,
                "reason": "circuit_breaker_open",
                "consecutive_failures": failures,
                "pressure": pressure,
            }

        try:
            result = await self._compact(repo, conversation, current_message, pressure, stage)
            return {"attempted": True, **result, "pressure": pressure, "consecutive_failures": 0}
        except Exception as exc:  # pragma: no cover - defensive circuit breaker path
            self.db.rollback()
            failure = self._record_failure(repo.id, conversation.id, current_message, stage, exc)
            self.db.flush()
            return {
                "attempted": True,
                "compacted": False,
                "reason": "compaction_failed",
                "failure_id": str(failure.id),
                "error": str(exc),
                "pressure": pressure,
                "consecutive_failures": failures + 1,
            }

    async def _compact(
        self,
        repo: Repository,
        conversation: Conversation,
        current_message: str,
        pressure: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        store = MessageStore(self.db)
        all_rows = store.active_for_context(repo.id, conversation.id, limit=500)
        all_messages = [self._row_to_message(row) for row in all_rows]
        segment_messages = messages_after_compact_boundary(all_messages, include_boundary=False)
        message_ids = {str(item.get("id")) for item in segment_messages if item.get("id")}
        segment_rows = [row for row in all_rows if str(row.id) in message_ids]
        candidate_rows = [
            row
            for row in segment_rows
            if row.role in {"user", "assistant", "tool", "tool_use", "tool_result"}
        ]
        plan_stages = [stage]
        if stage == "auto_compact":
            plan_stages.extend(["aggressive", "manual_path"])
        elif stage == "aggressive":
            plan_stages.append("manual_path")
        plans = [self._compaction_plan_for_stage(item) for item in plan_stages]
        if len(candidate_rows) <= min(int(item["keep_recent"]) for item in plans) + 1:
            return {"compacted": False, "reason": "not_enough_messages"}

        attempts: list[dict[str, Any]] = []
        selected: tuple[
            dict[str, Any],
            list[ChatMessage],
            list[ChatMessage],
            str,
            str,
            dict[str, Any],
            int,
        ] | None = None
        rehydration = self._rehydration_state(candidate_rows)
        for index, plan in enumerate(plans, start=1):
            keep_rows = self._messages_to_keep(candidate_rows, keep_count=int(plan["keep_recent"]))
            keep_ids = {row.id for row in keep_rows}
            summarize_rows = [row for row in candidate_rows if row.id not in keep_ids]
            if not summarize_rows:
                continue
            summary, mode, preservation_report = await self._summary_for(
                repo,
                conversation,
                summarize_rows,
                current_message,
                max_summary_tokens=int(plan["summary_tokens"]),
            )
            mode_prefix = str(plan.get("mode_prefix") or "")
            if mode_prefix:
                mode = f"{mode_prefix}_{mode}"
            post_compact_tokens = self._post_compact_tokens(
                summary,
                keep_rows,
                current_message=current_message,
                rehydration=rehydration,
            )
            post_pressure = calculate_context_pressure(post_compact_tokens)
            attempts.append(
                {
                    "attempt": index,
                    "plan": plan,
                    "post_compact_estimated_tokens": post_compact_tokens,
                    "post_compact_stage": post_pressure.stage,
                }
            )
            selected = (
                plan,
                keep_rows,
                summarize_rows,
                summary,
                mode,
                preservation_report,
                post_compact_tokens,
            )
            if post_pressure.stage in {"normal", "warning"}:
                break

        if selected is None:
            return {"compacted": False, "reason": "nothing_to_summarize"}
        plan, keep_rows, summarize_rows, summary, mode, preservation_report, post_compact_tokens = selected
        boundary = self._insert_boundary(
            repo,
            conversation,
            summary,
            mode,
            pressure,
            summarize_rows,
            keep_rows,
            preservation_report,
            rehydration,
        )
        return {
            "compacted": True,
            "mode": mode,
            "stage": stage,
            "boundary_id": str(boundary.id),
            "messages_summarized": len(summarize_rows),
            "messages_preserved": len(keep_rows),
            "post_compact_estimated_tokens": post_compact_tokens,
            "post_compact_pressure": calculate_context_pressure(post_compact_tokens).as_dict(),
            "compaction_plan": plan,
            "compaction_attempts": attempts,
            "rehydration": rehydration,
        }

    def _compaction_plan_for_stage(self, stage: str) -> dict[str, Any]:
        default_keep = max(2, settings.context_compact_keep_recent_messages)
        default_summary_tokens = max(256, settings.context_compact_summary_tokens)
        if stage == "manual_path":
            return {
                "keep_recent": max(2, min(default_keep, max(2, default_keep // 3))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.5))),
                "mode_prefix": "manual_path",
            }
        if stage == "aggressive":
            return {
                "keep_recent": max(2, min(default_keep, max(3, default_keep // 2))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.67))),
                "mode_prefix": "aggressive",
            }
        return {"keep_recent": default_keep, "summary_tokens": default_summary_tokens, "mode_prefix": ""}

    def _messages_to_keep(self, rows: list[ChatMessage], keep_count: int | None = None) -> list[ChatMessage]:
        keep = max(2, keep_count if keep_count is not None else settings.context_compact_keep_recent_messages)
        start_index = max(0, len(rows) - keep)
        start_index = adjust_keep_start_to_preserve_api_invariants(
            [self._row_to_message(row) for row in rows],
            start_index,
        )
        keep_rows = rows[start_index:]
        if (
            keep_rows
            and keep_rows[0].role == "assistant"
            and start_index > 0
            and rows[start_index - 1].role == "user"
        ):
            keep_rows = [rows[start_index - 1], *keep_rows]
        return keep_rows

    async def _summary_for(
        self,
        repo: Repository,
        conversation: Conversation,
        rows: list[ChatMessage],
        current_message: str,
        max_summary_tokens: int,
    ) -> tuple[str, str, dict[str, Any]]:
        messages = [self._row_to_message(row) for row in rows]
        memory = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).first()
        if memory and memory_document_text(memory).strip():
            summary = render_structured_memory(memory, max_tokens=max_summary_tokens)
            continuation = (
                "This conversation was compacted using structured session memory. "
                "Continue from the preserved recent messages and the current user request.\n\n"
                f"{summary}"
            )
            summary, report = ensure_compaction_summary_preserves_key_facts(messages, continuation, max_summary_tokens)
            return summary, "session_memory", report

        payload = build_compaction_summary_payload(messages)
        llm_summary = await self.llm.chat_text(
            COMPACT_SUMMARY_SYSTEM_PROMPT,
            f"Current user request that triggered pressure:\n{current_message}\n\nConversation segment to compact:\n{payload}",
        )
        summary = format_compaction_summary(llm_summary)
        if not summary:
            summary = fallback_compaction_summary(messages, max_tokens=max_summary_tokens)
            summary, report = ensure_compaction_summary_preserves_key_facts(messages, summary, max_summary_tokens)
            return summary, "fallback_extractive", report
        summary, report = ensure_compaction_summary_preserves_key_facts(messages, summary, max_summary_tokens)
        return summary, "full_compaction", report

    def _insert_boundary(
        self,
        repo: Repository,
        conversation: Conversation,
        summary: str,
        mode: str,
        pressure: dict[str, Any],
        summarized_rows: list[ChatMessage],
        keep_rows: list[ChatMessage],
        preservation_report: dict[str, Any],
        rehydration: dict[str, Any],
    ) -> ChatMessage:
        now = datetime.now(timezone.utc)
        latest_source_time = max(
            (row.created_at for row in [*summarized_rows, *keep_rows] if row.created_at),
            default=None,
        )
        if latest_source_time:
            comparable_now = now if latest_source_time.tzinfo else now.replace(tzinfo=None)
            if latest_source_time >= comparable_now:
                now = latest_source_time + timedelta(microseconds=1)
        boundary = ChatMessage(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="system",
            content=summary,
            route="context_compaction",
            tool_calls=[],
            meta={
                "subtype": COMPACT_BOUNDARY_SUBTYPE,
                "trigger": "auto",
                "mode": mode,
                "pressure": pressure,
                "pre_compact_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                "summarized_from_message_id": str(summarized_rows[0].id) if summarized_rows else None,
                "summarized_to_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                "preserved_start_message_id": str(keep_rows[0].id) if keep_rows else None,
                "preserved_message_ids": [str(row.id) for row in keep_rows],
                "transcript_ref": {
                    "repo_id": str(repo.id),
                    "conversation_id": str(conversation.id),
                    "tool": "devflow_get_thread_context",
                },
                "messages_summarized": len(summarized_rows),
                "messages_preserved": len(keep_rows),
                "summary_estimated_tokens": estimate_tokens(summary),
                "preservation": preservation_report,
                "rehydration": rehydration,
            },
            created_at=now,
        )
        self.db.add(boundary)
        self.db.flush()
        return boundary

    def _post_compact_tokens(
        self,
        summary: str,
        keep_rows: list[ChatMessage],
        *,
        current_message: str = "",
        rehydration: dict[str, Any] | None = None,
    ) -> int:
        return (
            estimate_tokens(summary)
            + sum(estimate_tokens(row.content) for row in keep_rows)
            + estimate_tokens(current_message)
            + estimate_tokens(dumps_compact(rehydration or {}))
        )

    def _rehydration_state(self, rows: list[ChatMessage]) -> dict[str, Any]:
        working_files: list[str] = []
        active_skills: list[str] = []
        recent_tools: list[str] = []

        def remember(target: list[str], value: Any, limit: int) -> None:
            clean = str(value or "").strip()
            if not clean or clean in target:
                return
            target.append(clean)
            if len(target) > limit:
                del target[0 : len(target) - limit]

        for row in rows:
            calls = [item for item in (row.tool_calls or []) if isinstance(item, dict)]
            events = [item for item in (row.meta or {}).get("agent_events") or [] if isinstance(item, dict)]
            for call in calls:
                remember(recent_tools, call.get("tool_name"), 8)
                arguments = call.get("arguments") or {}
                if isinstance(arguments, dict):
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)
            for event in events:
                event_type = str(event.get("type") or "")
                if event_type in {"tool_use", "tool_result"}:
                    remember(recent_tools, event.get("tool_name"), 8)
                arguments = event.get("arguments") or {}
                if isinstance(arguments, dict):
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)
                if event_type == "skill_activated":
                    remember(active_skills, event.get("skill_name") or event.get("skill_title"), 6)

        return {
            "working_files": working_files,
            "active_skills": active_skills,
            "recent_tools": recent_tools,
            "recovery": {
                "exact_transcript_tool": "devflow_get_thread_context",
                "sealed_session_tool": "devflow_read_session_events",
            },
        }

    def _raw_segment_tokens(self, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> int:
        rows = MessageStore(self.db).active_for_context(repo_id, conversation_id, limit=1000)
        messages = messages_after_compact_boundary([self._row_to_message(row) for row in rows], include_boundary=False)
        total = 0
        for message in messages:
            total += estimate_tokens(str(message.get("role") or ""))
            total += estimate_tokens(str(message.get("content") or ""))
            if message.get("tool_calls"):
                total += estimate_tokens(dumps_compact(message.get("tool_calls")))
        return total

    def _row_to_message(self, row: ChatMessage) -> dict[str, Any]:
        return {
            "id": str(row.id),
            "role": row.role,
            "content": row.content,
            "route": row.route,
            "tool_calls": row.tool_calls or [],
            "meta": row.meta or {},
        }

    def _consecutive_failures(self, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> int:
        boundary = MessageStore(self.db).last_compact_boundary(repo_id, conversation_id)
        query = self.db.query(ChatMessage).filter(
            ChatMessage.repo_id == repo_id,
            ChatMessage.conversation_id == conversation_id,
            ChatMessage.role == "system",
            ChatMessage.route == "context_compaction",
        )
        if boundary and boundary.created_at:
            query = query.filter(ChatMessage.created_at > boundary.created_at)
        count = 0
        for row in query.order_by(ChatMessage.created_at.desc()).limit(settings.context_max_compaction_failures).all():
            if (row.meta or {}).get("subtype") == "compact_failure":
                count += 1
            else:
                break
        return count

    def _record_failure(self, repo_id: uuid.UUID, conversation_id: uuid.UUID, current_message: str, stage: str, exc: Exception) -> ChatMessage:
        failure = ChatMessage(
            repo_id=repo_id,
            conversation_id=conversation_id,
            role="system",
            content=clip_to_token_budget(f"Context compaction failed during {stage}: {exc}", 240),
            route="context_compaction",
            tool_calls=[],
            meta={
                "subtype": "compact_failure",
                "stage": stage,
                "current_message": clip_to_token_budget(current_message, 120),
            },
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(failure)
        return failure


class SessionSealer:
    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()

    async def seal(
        self,
        *,
        repo: Repository | None,
        conversation: Conversation | None,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        route: str,
        tool_calls: list[dict[str, Any]],
        agent_events: list[dict[str, Any]] | None = None,
        model_usage: list[dict[str, Any]] | None = None,
        context_diagnostics: dict[str, Any] | None = None,
    ) -> ChatSessionModel:
        agent_events = agent_events or []
        model_usage = model_usage or []
        context_diagnostics = context_diagnostics or {}
        fallback_update = fallback_memory_update(
            user_message.content,
            assistant_message.content,
            route,
            tool_calls,
        )
        fallback_digest = str(fallback_update.get("summary") or "")
        session = ChatSession(
            repo_id=repo.id if repo else None,
            conversation_id=conversation.id if conversation else None,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            route=route,
            transcript_json={
                "messages": [
                    {"role": "user", "content": user_message.content, "created_at": self._iso(user_message.created_at)},
                    {"role": "assistant", "content": assistant_message.content, "created_at": self._iso(assistant_message.created_at)},
                ],
                "tool_calls": tool_calls,
                "agent_events": agent_events,
                "tool_use_results": [
                    event for event in agent_events if event.get("type") in {"tool_use", "tool_result"}
                ],
                "model_usage": model_usage,
                "context_diagnostics": context_diagnostics,
                "memory_update": fallback_update,
                "sealing": {"phase": "raw_persisted", "errors": []},
            },
            digest=fallback_digest,
            status="sealing",
            sealed_at=None,
        )
        self.db.add(session)
        self.db.flush()

        errors: list[str] = []
        memory_update = fallback_update
        digest = fallback_digest
        try:
            memory_update = await self._build_memory_update(
                user_message.content,
                assistant_message.content,
                route,
                tool_calls,
                fallback_digest,
                repo_id=repo.id if repo else None,
                conversation_id=conversation.id if conversation else None,
            )
            digest = str(memory_update.pop("_digest", "") or memory_update.get("summary") or fallback_digest)
        except Exception as exc:  # model/network failures must not invalidate the completed chat turn
            errors.append(f"memory_extraction_failed: {exc}")
            memory_update = fallback_update
            digest = fallback_digest

        try:
            with self.db.begin_nested():
                self._update_thread_memory(
                    repo.id if repo else None,
                    conversation.id if conversation else None,
                    digest,
                    memory_update,
                )
                capture_memory_candidates(
                    self.db,
                    repo_id=repo.id if repo else None,
                    conversation_id=conversation.id if conversation else None,
                    session_id=session.id,
                    memory_update=memory_update,
                )
                self.db.flush()
        except Exception as exc:  # keep raw ChatSession even when enrichment fails
            errors.append(f"memory_persistence_failed: {exc}")

        transcript = dict(session.transcript_json or {})
        transcript["memory_update"] = {
            key: value for key, value in memory_update.items() if not key.startswith("_")
        }
        transcript["sealing"] = {
            "phase": "completed",
            "errors": errors,
            "evidence_refresh": "deferred_to_next_request",
        }
        session.transcript_json = transcript
        session.digest = digest
        session.status = "sealed_with_warnings" if errors else "sealed"
        session.sealed_at = datetime.now(timezone.utc)
        self.db.flush()
        return session

    async def _build_memory_update(
        self,
        user_message: str,
        assistant_message: str,
        route: str,
        tool_calls: list[dict[str, Any]],
        digest: str,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
    ) -> dict[str, Any]:
        fallback = fallback_memory_update(user_message, assistant_message, route, tool_calls)
        if repo_id and conversation_id:
            current_memory: ConversationMemory | ThreadMemory | None = (
                self.db.query(ConversationMemory)
                .filter(ConversationMemory.conversation_id == conversation_id)
                .first()
            )
        else:
            current_memory = (
                self.db.query(ThreadMemory)
                .filter(_repo_filter(ThreadMemory, repo_id))
                .first()
            )
        existing_memory = self._memory_state(current_memory) if current_memory else {}
        prompt = (
            "请增量更新软件工程 Agent 的数据库会话记忆。"
            "输入包含已有记忆和本轮原始记录；返回更新后的完整快照，而不是只返回本轮增量。"
            "只返回 JSON，包含 digest、summary、facts、decisions、open_questions、tasks、"
            "user_preferences、repo_context、citations。summary 必须描述当前工作状态和下一步；"
            "保留重要文件、函数、命令、错误修复和用户纠正，不要编造。"
        )
        parsed = await self.llm.chat_json(
            prompt,
            {
                "route": route,
                "digest": digest,
                "existing_memory": existing_memory,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "tool_calls": tool_calls,
            },
        )
        update = structured_memory_from_model(parsed) if parsed else fallback
        if not update.get("summary"):
            update["summary"] = fallback["summary"]
        if parsed:
            update["_update_mode"] = "snapshot"
            update["_digest"] = str(parsed.get("digest") or parsed.get("summary") or digest)
        return update

    def _memory_state(self, memory: ConversationMemory | ThreadMemory) -> dict[str, Any]:
        return {
            "summary": memory.summary or "",
            "facts": getattr(memory, "facts", None) or [],
            "decisions": memory.decisions or [],
            "open_questions": memory.open_questions or [],
            "tasks": getattr(memory, "tasks", None) or [],
            "user_preferences": getattr(memory, "user_preferences", None) or [],
            "repo_context": getattr(memory, "repo_context", None) or [],
            "citations": getattr(memory, "citations", None) or [],
        }

    def _new_memory(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ConversationMemory | ThreadMemory:
        if repo_id and conversation_id:
            memory = ConversationMemory(
                repo_id=repo_id,
                conversation_id=conversation_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        else:
            memory = ThreadMemory(
                repo_id=repo_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        self.db.add(memory)
        self.db.flush()
        return memory

    def _update_thread_memory(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        digest: str,
        memory_update: dict[str, Any],
    ) -> ConversationMemory | ThreadMemory | None:
        if repo_id and conversation_id:
            memory: ConversationMemory | ThreadMemory | None = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
        else:
            memory = self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()
        if not memory:
            memory = self._new_memory(repo_id, conversation_id)
        if memory_update.get("_update_mode") == "snapshot":
            merged = {
                **structured_memory_from_model(memory_update),
                "summary": clip_to_token_budget(str(memory_update.get("summary") or digest), 1100),
            }
        else:
            merged = merge_memory(self._memory_state(memory), memory_update, digest, max_tokens=1100)
        memory.summary = merged["summary"]
        memory.facts = merged["facts"]
        memory.decisions = merged["decisions"]
        memory.open_questions = merged["open_questions"]
        memory.tasks = merged["tasks"]
        memory.user_preferences = merged["user_preferences"]
        memory.repo_context = merged["repo_context"]
        memory.citations = merged["citations"]
        memory.sessions_incorporated = (memory.sessions_incorporated or 0) + 1
        return memory

    def _iso(self, value: datetime | None) -> str | None:
        return value.isoformat() if value else None


class MemoryTools:
    def __init__(self, db: Session, repo: Repository | None, conversation: Conversation | None = None) -> None:
        self.db = db
        self.repo = repo
        self.repo_id = repo.id if repo else None
        self.conversation = conversation
        self.conversation_id = conversation.id if conversation else None

    async def search_evidence(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        query = str(input_data.get("query") or input_data.get("message") or "")
        results = EvidenceStore(self.db).search(self.repo_id, self.conversation_id, query, limit=int(input_data.get("limit") or 5))
        answer = "No evidence matched this query." if not results else "\n".join(
            f"- [{item['source_type']}] {item['title']}: {item['snippet']}" for item in results
        )
        citations = [{"type": item["source_type"], "id": item["source_id"], "title": item["title"]} for item in results]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_search_evidence",
            query=query,
            results=results,
            citations=citations,
            scope="conversation" if self.conversation_id else "project",
            mode="hybrid",
            metadata={"limit": int(input_data.get("limit") or 5)},
        )
        return {
            "route": "memory_search",
            "tool_calls": [{"tool_name": "search_evidence", "arguments": {"query": query}, "status": "success"}],
            "answer": answer,
            "citations": citations,
            "results": results,
        }

    async def get_thread_context(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        keyword = str(input_data.get("keyword") or input_data.get("query") or "").strip().lower()
        limit = min(max(int(input_data.get("limit") or 30), 1), 100)
        message_id = self._uuid_or_none(input_data.get("message_id"))
        before_message_id = self._uuid_or_none(input_data.get("before_message_id"))
        after_message_id = self._uuid_or_none(input_data.get("after_message_id"))
        offset = max(int(input_data.get("offset") or 0), 0)
        max_chars = min(max(int(input_data.get("max_chars") or 6000), 200), 16000)

        store = MessageStore(self.db)
        if message_id:
            message = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, self.repo_id),
                    _conversation_filter(ChatMessage, self.conversation_id),
                    ChatMessage.id == message_id,
                    ChatMessage.role.in_(["user", "assistant", "tool", "tool_result"]),
                )
                .one_or_none()
            )
            messages = [message] if message and (message.meta or {}).get("subtype") != "preserved_after_compact" else []
        else:
            messages = store.timeline(
                self.repo_id,
                self.conversation_id,
                limit=None,
                before_message_id=before_message_id,
                after_message_id=after_message_id,
            )
            if keyword:
                terms = _tokens(keyword)
                messages = [msg for msg in messages if any(term in msg.content.lower() for term in terms)]
            messages = messages[-limit:]

        lines: list[str] = []
        results: list[dict[str, Any]] = []
        remaining_chars = max_chars
        for message in messages:
            content = message.content or ""
            chunk_offset = offset if message_id else 0
            chunk = content[chunk_offset : chunk_offset + remaining_chars]
            if not chunk and content:
                continue
            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(content)
            created_at = message.created_at.isoformat() if message.created_at else ""
            lines.append(
                f"[message_id={message.id} role={message.role} created_at={created_at} chars={chunk_offset}:{chunk_end}/{len(content)}]"
            )
            lines.append(chunk)
            if truncated:
                lines.append(
                    f"[内容未读完；继续调用 message_id={message.id}, offset={chunk_end}]"
                )
            results.append(
                {
                    "title": f"{message.role} message",
                    "source_type": "chat_message",
                    "source_id": str(message.id),
                    "snippet": chunk,
                    "metadata": {
                        "role": message.role,
                        "created_at": created_at,
                        "offset": chunk_offset,
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(content),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break

        citations = [
            {"type": "chat_message", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_get_thread_context",
            query=keyword or str(message_id or ""),
            results=results,
            citations=citations,
            scope="conversation",
            mode="exact_transcript",
            metadata={
                "limit": limit,
                "message_id": str(message_id) if message_id else None,
                "before_message_id": str(before_message_id) if before_message_id else None,
                "after_message_id": str(after_message_id) if after_message_id else None,
                "offset": offset,
                "max_chars": max_chars,
            },
        )
        return {
            "route": "thread_context",
            "tool_calls": [{"tool_name": "get_thread_context", "arguments": {"keyword": keyword, "message_id": str(message_id or "")}, "status": "success"}],
            "answer": "\n".join(lines) if lines else "当前会话还没有匹配的消息。",
            "citations": citations,
            "results": results,
        }

    async def read_session_events(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        session_id = input_data.get("session_id")
        include_transcript = bool(input_data.get("include_transcript", True))
        offset = max(int(input_data.get("offset") or 0), 0)
        max_chars = min(max(int(input_data.get("max_chars") or 8000), 400), 20000)
        query = self.db.query(ChatSession).filter(
            _repo_filter(ChatSession, self.repo_id),
            _conversation_filter(ChatSession, self.conversation_id),
        ).order_by(ChatSession.created_at.desc())
        if session_id:
            parsed_session_id = self._uuid_or_none(session_id)
            if parsed_session_id is None:
                sessions = []
            else:
                sessions = query.filter(ChatSession.id == parsed_session_id).limit(1).all()
        else:
            sessions = query.limit(min(max(int(input_data.get("limit") or 5), 1), 20)).all()
        lines: list[str] = []
        results: list[dict[str, Any]] = []
        remaining_chars = max_chars
        for session in sessions:
            header = f"Session {session.id} route={session.route} sealed_at={session.sealed_at}"
            payload = session.digest
            if include_transcript:
                payload += "\ntranscript=" + dumps_compact(session.transcript_json or {})
            chunk_offset = offset if session_id else 0
            chunk = payload[chunk_offset : chunk_offset + remaining_chars]
            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(payload)
            lines.append(header)
            lines.append(chunk)
            if truncated:
                lines.append(
                    f"[会话记录未读完；继续调用 session_id={session.id}, offset={chunk_end}。]"
                )
            results.append(
                {
                    "title": f"Session {session.route}",
                    "source_type": "session",
                    "source_id": str(session.id),
                    "snippet": chunk,
                    "metadata": {
                        "sealed_at": session.sealed_at.isoformat() if session.sealed_at else None,
                        "include_transcript": include_transcript,
                        "offset": chunk_offset,
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(payload),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break
        citations = [
            {"type": "session", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]
        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_read_session_events",
            query=str(session_id or "latest sessions"),
            results=results,
            citations=citations,
            scope="conversation",
            mode="sealed_transcript",
            metadata={"include_transcript": include_transcript, "offset": offset, "max_chars": max_chars},
        )
        return {
            "route": "session_events",
            "tool_calls": [{"tool_name": "read_session_events", "arguments": {"session_id": str(session_id or "")}, "status": "success"}],
            "answer": "\n\n".join(lines) if lines else "No sealed sessions found yet.",
            "citations": citations,
            "results": results,
        }

    @staticmethod
    def _uuid_or_none(value: Any) -> uuid.UUID | None:
        if not value:
            return None
        try:
            return uuid.UUID(str(value))
        except (TypeError, ValueError):
            return None
