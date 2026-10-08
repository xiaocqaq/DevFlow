import asyncio
import copy
import importlib.metadata
import inspect
import math
import re
import sys
import time
import types
import uuid
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Conversation, Document, Issue, PullRequest, Repository, WorkflowRun
from app.services.agents.chat_agent import ChatAgent
from app.services.chat_memory import ContextAssembler, EvidenceStore
from app.services.rag.embeddings import embed_text
from app.services.rag.source_policy import RAG_SOURCE_TYPES


CORE_RAGAS_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)
REQUIRED_RAGAS_METRICS = (*CORE_RAGAS_METRICS, "agent_goal_accuracy")
AGENT_RAGAS_METRICS = (
    "agent_goal_accuracy",
    "tool_call_accuracy",
    "tool_call_f1",
)

_FAITHFULNESS_CHUNK_CHARS = 700


def _split_faithfulness_response(text: str, max_chars: int = _FAITHFULNESS_CHUNK_CHARS) -> list[str]:
    """Split a long answer without dropping content before native Ragas scoring.

    Faithfulness first asks the judge to extract all claims. Some OpenAI-compatible
    reasoning models can exhaust their structured-output budget even for a moderate
    answer. Scoring smaller semantic chunks keeps that native metric reliable; the
    final score is weighted by the amount of answer text in each chunk.
    """

    normalized = text.strip()
    if not normalized:
        return [""]

    units = [unit.strip() for unit in re.split(r"(?<=[。！？.!?])\s+|\n+", normalized) if unit.strip()]
    chunks: list[str] = []
    current = ""
    for unit in units:
        while len(unit) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit[:max_chars])
            unit = unit[max_chars:]
        candidate = f"{current}\n{unit}".strip() if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [normalized]


class RagasUnavailableError(RuntimeError):
    """Ragas, its judge model, or its evaluation embedding is unavailable."""


@dataclass
class RagEvalCase:
    question: str
    reference: str
    expected_source_ids: list[str]
    source_type: str | None = None
    required_tools: list[str] = field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "positive"
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    answer_regex: str | None = None
    source_file: str | None = None
    id: str = ""


class SemanticScorer(Protocol):
    version: str
    judge_model: str

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
        tool_calls: list[dict[str, Any]],
        tool_observations: list[dict[str, Any]],
        expected_tool_calls: list[dict[str, Any]],
        metric_names: set[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]: ...


ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


async def _emit_progress(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is None:
        return
    pending = callback(payload)
    if inspect.isawaitable(pending):
        await pending


def _ensure_ragas_compatibility() -> None:
    """Bridge a Ragas 0.4.3 import removed by langchain-community 0.4.

    Ragas imports the legacy ChatVertexAI module only for an isinstance check. The
    project does not use VertexAI, so a scoped placeholder keeps the supported
    Collections API usable without downgrading LangChain/LangGraph.
    """

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`langchain-community` is being sunset.*",
                category=DeprecationWarning,
            )
            __import__(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        shim = types.ModuleType(module_name)
        shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = shim


class RagasCollectionsScorer:
    """Adapter from DevFlow's model clients to the Ragas 0.4 Collections API."""

    def __init__(self) -> None:
        _ensure_ragas_compatibility()
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import BaseRagasEmbedding
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AgentGoalAccuracyWithReference,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
                ToolCallAccuracy,
                ToolCallF1,
            )
        except Exception as exc:  # pragma: no cover - depends on optional packages
            raise RagasUnavailableError(f"Unable to import Ragas 0.4 Collections API: {exc}") from exc

        class ProjectEmbedding(BaseRagasEmbedding):
            def embed_text(self, text: str, **_: Any) -> list[float]:
                return embed_text(text)

            async def aembed_text(self, text: str, **_: Any) -> list[float]:
                return await asyncio.to_thread(embed_text, text)

        api_key = settings.ragas_judge_api_key or settings.llm_api_key
        base_url = (settings.ragas_judge_base_url or settings.llm_base_url).rstrip("/")
        model = settings.ragas_judge_model or settings.llm_model
        if not api_key:
            raise RagasUnavailableError("RAGAS judge API key is not configured")
        if not base_url or not model:
            raise RagasUnavailableError("RAGAS judge base URL or model is not configured")

        try:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=settings.ragas_timeout_seconds)
            llm = llm_factory(
                model,
                client=client,
                temperature=0,
                max_tokens=settings.ragas_judge_max_tokens,
            )
            embedding = ProjectEmbedding()
            self.metrics = {
                "context_precision": ContextPrecision(llm=llm),
                "context_recall": ContextRecall(llm=llm),
                "faithfulness": Faithfulness(llm=llm),
                "answer_relevancy": AnswerRelevancy(
                    llm=llm,
                    embeddings=embedding,
                    strictness=max(1, settings.ragas_answer_relevancy_strictness),
                ),
                "agent_goal_accuracy": AgentGoalAccuracyWithReference(llm=llm),
                "tool_call_accuracy": ToolCallAccuracy(strict_order=False),
                "tool_call_f1": ToolCallF1(),
            }
            self.message_types = self._message_types()
            self.version = importlib.metadata.version("ragas")
            self.judge_model = model
        except Exception as exc:
            raise RagasUnavailableError(f"Unable to initialize Ragas judge: {exc}") from exc

    @staticmethod
    def _message_types() -> tuple[Any, Any, Any, Any]:
        from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

        return HumanMessage, AIMessage, ToolMessage, ToolCall

    def _messages(
        self,
        user_input: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        tool_observations: list[dict[str, Any]],
    ) -> list[Any]:
        HumanMessage, AIMessage, ToolMessage, ToolCall = self.message_types
        messages: list[Any] = [HumanMessage(content=user_input)]
        ragas_calls = [
            ToolCall(
                name=str(item.get("tool_name") or item.get("name") or "unknown"),
                args=dict(item.get("arguments") or item.get("args") or {}),
            )
            for item in tool_calls
            if isinstance(item, dict)
        ]
        if ragas_calls:
            messages.append(AIMessage(content="", tool_calls=ragas_calls))
        for item in tool_observations:
            if isinstance(item, dict):
                messages.append(ToolMessage(content=str(item.get("content") or "")))
        messages.append(AIMessage(content=response))
        return messages

    async def _score_faithfulness(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> float:
        chunks = _split_faithfulness_response(response)
        semaphore = asyncio.Semaphore(2)

        async def score_chunk(chunk: str) -> Any:
            async with semaphore:
                return await asyncio.wait_for(
                    self.metrics["faithfulness"].ascore(
                        user_input=user_input,
                        response=chunk,
                        retrieved_contexts=retrieved_contexts,
                    ),
                    timeout=settings.ragas_timeout_seconds,
                )

        results = await asyncio.gather(*(score_chunk(chunk) for chunk in chunks), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            first = failures[0]
            raise RagasUnavailableError(
                f"Faithfulness chunk scoring failed ({len(failures)}/{len(chunks)}): "
                f"{type(first).__name__}: {first}"
            ) from first

        weights = [max(1, len(chunk)) for chunk in chunks]
        return sum(float(result.value) * weight for result, weight in zip(results, weights, strict=True)) / sum(weights)

    async def score(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
        tool_calls: list[dict[str, Any]],
        tool_observations: list[dict[str, Any]],
        expected_tool_calls: list[dict[str, Any]],
        metric_names: set[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        messages = self._messages(user_input, response, tool_calls, tool_observations)
        wants = lambda name: metric_names is None or name in metric_names
        calls: dict[str, Any] = {}
        if wants("context_precision"):
            calls["context_precision"] = self.metrics["context_precision"].ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
        if wants("context_recall"):
            calls["context_recall"] = self.metrics["context_recall"].ascore(
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
        if wants("faithfulness"):
            calls["faithfulness"] = self._score_faithfulness(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
        if wants("answer_relevancy"):
            calls["answer_relevancy"] = self.metrics["answer_relevancy"].ascore(
                user_input=user_input,
                response=response,
            )
        if wants("agent_goal_accuracy"):
            calls["agent_goal_accuracy"] = self.metrics["agent_goal_accuracy"].ascore(
                user_input=messages,
                reference=reference,
            )
        if expected_tool_calls and (wants("tool_call_accuracy") or wants("tool_call_f1")):
            _, _, _, ToolCall = self.message_types
            reference_calls = [
                ToolCall(
                    name=str(item.get("tool_name") or item.get("name") or "unknown"),
                    args=dict(item.get("arguments") or item.get("args") or {}),
                )
                for item in expected_tool_calls
            ]
            if wants("tool_call_accuracy"):
                calls["tool_call_accuracy"] = self.metrics["tool_call_accuracy"].ascore(
                    user_input=messages,
                    reference_tool_calls=reference_calls,
                )
            if wants("tool_call_f1"):
                calls["tool_call_f1"] = self.metrics["tool_call_f1"].ascore(
                    user_input=messages,
                    reference_tool_calls=reference_calls,
                )

        names = list(calls)
        results = await asyncio.gather(
            *(
                asyncio.wait_for(calls[name], timeout=settings.ragas_timeout_seconds)
                for name in names
            ),
            return_exceptions=True,
        )
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}
        for name, result in zip(names, results, strict=True):
            if isinstance(result, BaseException):
                errors[name] = f"{type(result).__name__}: {result}"
                continue
            value = result if isinstance(result, (int, float)) else result.value
            scores[name] = round(float(value), 4)
        return scores, errors


def _default_cases(db: Session, repo_id: str, limit: int | None = None) -> list[RagEvalCase]:
    repo_uuid = uuid.UUID(str(repo_id))
    case_limit = limit or settings.ragas_max_cases
    docs = (
        db.query(Document)
        .filter(Document.repo_id == repo_uuid, Document.source_type.in_(RAG_SOURCE_TYPES))
        .order_by(Document.created_at.desc())
        .limit(case_limit)
        .all()
    )
    cases: list[RagEvalCase] = []
    for index, doc in enumerate(docs, start=1):
        if doc.source_type == "issue":
            question = f"What context exists for issue {doc.title}?"
        elif doc.source_type == "pull_request":
            question = f"What changed in pull request {doc.title}?"
        elif doc.source_type == "workflow_run":
            question = f"What does the CI run {doc.title} show?"
        else:
            question = f"What project knowledge is available about {doc.title}?"
        cases.append(
            RagEvalCase(
                id=f"document-{index}",
                question=question,
                reference=doc.content[:2000],
                expected_source_ids=[str(doc.id), str(doc.source_id)],
                source_type=doc.source_type,
            )
        )
    if cases:
        return cases

    # A freshly connected repository can have structured Issue/PR/CI data before
    # the reusable Document index is populated. Production ChatAgent prefetches
    # those records through EvidenceStore, so the default suite must cover them.
    issues = (
        db.query(Issue)
        .filter(Issue.repo_id == repo_uuid)
        .order_by(Issue.updated_at.desc(), Issue.created_at.desc())
        .limit(case_limit)
        .all()
    )
    for issue in issues:
        reference = "\n".join(
            value
            for value in [
                issue.title,
                issue.body or "",
                f"state: {issue.state}",
                f"labels: {', '.join(issue.labels or [])}",
            ]
            if value
        )
        cases.append(
            RagEvalCase(
                id=f"issue-{issue.number}",
                question=f"What project context is recorded for issue #{issue.number}: {issue.title}?",
                reference=reference[:2000],
                expected_source_ids=[str(issue.id)],
                source_type="issue",
            )
        )
        if len(cases) >= case_limit:
            return cases

    pull_requests = (
        db.query(PullRequest)
        .filter(PullRequest.repo_id == repo_uuid)
        .order_by(PullRequest.updated_at.desc(), PullRequest.created_at.desc())
        .limit(case_limit - len(cases))
        .all()
    )
    for pull_request in pull_requests:
        reference = "\n".join(
            value
            for value in [
                pull_request.title,
                pull_request.body or "",
                f"state: {pull_request.state}",
                f"base: {pull_request.base_branch}",
                f"head: {pull_request.head_branch}",
            ]
            if value
        )
        cases.append(
            RagEvalCase(
                id=f"pull-request-{pull_request.number}",
                question=f"What changed in pull request #{pull_request.number}: {pull_request.title}?",
                reference=reference[:2000],
                expected_source_ids=[str(pull_request.id)],
                source_type="pull_request",
            )
        )
        if len(cases) >= case_limit:
            return cases

    workflow_runs = (
        db.query(WorkflowRun)
        .filter(WorkflowRun.repo_id == repo_uuid, WorkflowRun.conclusion == "failure")
        .order_by(WorkflowRun.created_at.desc())
        .limit(case_limit - len(cases))
        .all()
    )
    for workflow_run in workflow_runs:
        reference = "\n".join(
            value
            for value in [
                workflow_run.name,
                workflow_run.status,
                workflow_run.conclusion or "",
                (workflow_run.logs_text or "")[:1600],
            ]
            if value
        )
        cases.append(
            RagEvalCase(
                id=f"workflow-run-{workflow_run.github_run_id}",
                question=f"What does the failed CI run {workflow_run.name} show?",
                reference=reference[:2000],
                expected_source_ids=[str(workflow_run.id)],
                source_type="workflow_run",
            )
        )
    return cases


def _source_hit(result: dict[str, Any], expected_source_ids: set[str]) -> bool:
    return str(result.get("id")) in expected_source_ids or str(result.get("source_id")) in expected_source_ids


def _dcg(hits: list[int]) -> float:
    return sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))


def _retrieval_metrics(cases: list[RagEvalCase], rows: list[dict[str, Any]], k: int) -> dict[str, float]:
    if not cases:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}
    recall_values: list[float] = []
    precision_values: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    for case, row in zip(cases, rows, strict=True):
        expected = set(case.expected_source_ids)
        results = row["results"][:k]
        hits = [1 if _source_hit(result, expected) else 0 for result in results]
        recall_values.append(1.0 if not expected or any(hits) else 0.0)
        precision_values.append(sum(hits) / max(len(results), 1))
        first_hit = next((index + 1 for index, hit in enumerate(hits) if hit), None)
        reciprocal_ranks.append(1.0 / first_hit if first_hit else (1.0 if not expected else 0.0))
        ideal = [1] + [0] * max(len(hits) - 1, 0)
        ndcg_values.append(_dcg(hits) / (_dcg(ideal) or 1.0) if expected else 1.0)
    return {
        "recall_at_k": round(sum(recall_values) / len(recall_values), 4),
        "precision_at_k": round(sum(precision_values) / len(precision_values), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
        "ndcg": round(sum(ndcg_values) / len(ndcg_values), 4),
    }


def _trace_rows(trace: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    contexts: list[str] = []
    sources: list[dict[str, Any]] = []
    seen_contexts: set[str] = set()
    seen_sources: set[tuple[str, str, str]] = set()
    for retrieval in trace.get("retrievals") or []:
        if not isinstance(retrieval, dict):
            continue
        for value in retrieval.get("contexts") or []:
            text = str(value).strip()
            if text and text not in seen_contexts:
                seen_contexts.add(text)
                contexts.append(text)
        for raw_source in retrieval.get("sources") or []:
            if not isinstance(raw_source, dict):
                continue
            source = dict(raw_source)
            key = (
                str(source.get("id") or ""),
                str(source.get("source_id") or ""),
                str(source.get("title") or ""),
            )
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append(source)
    return contexts, sources


def _hard_rules(case: RagEvalCase, result: dict[str, Any], sources: list[dict[str, Any]], contexts: list[str]) -> dict[str, Any]:
    expected = set(case.expected_source_ids)
    source_hit = not expected or any(_source_hit(item, expected) for item in sources)
    actual_tools = [
        str(item.get("tool_name") or "")
        for item in result.get("tool_calls") or []
        if isinstance(item, dict)
    ]
    missing_tools = [name for name in case.required_tools if name not in actual_tools]
    tool_errors = [
        {
            "tool_name": item.get("tool_name"),
            "error": item.get("error"),
        }
        for item in result.get("tool_calls") or []
        if isinstance(item, dict) and item.get("status") == "error"
    ]
    answer = str(result.get("answer") or "").strip()
    answer_present = bool(answer)
    contexts_present = bool(contexts)
    missing_phrases = [value for value in case.must_include if value.casefold() not in answer.casefold()]
    forbidden_phrases = [value for value in case.must_not_include if value.casefold() in answer.casefold()]
    answer_pattern_matched = case.answer_regex is None or re.fullmatch(case.answer_regex, answer) is not None
    passed = (
        source_hit
        and not missing_tools
        and not tool_errors
        and answer_present
        and contexts_present
        and not missing_phrases
        and not forbidden_phrases
        and answer_pattern_matched
    )
    return {
        "passed": passed,
        "source_hit": source_hit,
        "answer_present": answer_present,
        "contexts_present": contexts_present,
        "missing_tools": missing_tools,
        "tool_errors": tool_errors,
        "missing_phrases": missing_phrases,
        "forbidden_phrases": forbidden_phrases,
        "answer_pattern_matched": answer_pattern_matched,
        "citation_count": len(result.get("citations") or []),
    }


def _evaluation_project_context(db: Session, repo_id: uuid.UUID) -> dict[str, Any]:
    """Load the same project-owned context that the frontend sends to ChatAgent."""

    documents = (
        db.query(Document)
        .filter(
            Document.repo_id == repo_id,
            Document.source_type.in_(("team_member", "memory_note")),
        )
        .order_by(Document.created_at.asc())
        .all()
    )
    team_members: list[dict[str, str]] = []
    project_memory: list[dict[str, str]] = []
    for document in documents:
        metadata = document.meta or {}
        if document.source_type == "team_member":
            team_members.append(
                {
                    "name": str(metadata.get("name") or document.title.replace("团队成员：", "")),
                    "role": str(metadata.get("role") or ""),
                    "strengths": str(metadata.get("strengths") or ""),
                    "techStack": str(metadata.get("tech_stack") or metadata.get("techStack") or ""),
                }
            )
        elif document.source_type == "memory_note":
            project_memory.append({"title": document.title, "content": document.content})
    return {"team_members": team_members, "project_memory": project_memory}


async def _run_chatagent_case(
    db: Session,
    repo: Repository,
    case: RagEvalCase,
    *,
    limit: int,
    agent_factory: Callable[[], ChatAgent],
) -> dict[str, Any]:
    # Each trial gets a clean conversation and rolls back every runtime write.
    savepoint = db.begin_nested()
    try:
        conversation = Conversation(
            repo_id=repo.id,
            title=f"[RAGAS Eval] {case.id or case.question[:40]}",
            status="evaluation",
        )
        db.add(conversation)
        db.flush()
        EvidenceStore(db).rebuild_repo(repo, conversation)
        assembled = ContextAssembler(db).assemble(repo, conversation, case.question)
        from app.api.routes.chat import _build_conversation_tools, _skill_runtime_context

        context: dict[str, Any] = {
            **_evaluation_project_context(db, repo.id),
            **_skill_runtime_context(),
            "repo_id": str(repo.id),
            "conversation_id": str(conversation.id),
            "server_context": assembled.system_context,
            "server_messages": assembled.history_messages,
            "evidence": assembled.evidence,
            "compression_stats": assembled.compression_stats,
            "source_type": case.source_type,
            "limit": limit,
            "evaluation_mode": True,
        }
        context["tools"] = _build_conversation_tools(repo, conversation, db)
        return await agent_factory().run({"message": case.question}, context)
    finally:
        if savepoint.is_active:
            savepoint.rollback()


def _thresholds() -> dict[str, float]:
    return {
        "context_precision": settings.ragas_context_precision_min,
        "context_recall": settings.ragas_context_recall_min,
        "faithfulness": settings.ragas_faithfulness_min,
        "answer_relevancy": settings.ragas_answer_relevancy_min,
        "agent_goal_accuracy": settings.ragas_agent_goal_accuracy_min,
        "tool_call_accuracy": settings.ragas_tool_call_accuracy_min,
        "tool_call_f1": settings.ragas_tool_call_f1_min,
    }


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return round(sum(finite) / len(finite), 4) if finite else None


def _metric_details(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    thresholds = _thresholds()
    values: dict[str, list[float]] = {
        name: [] for name in (*CORE_RAGAS_METRICS, *AGENT_RAGAS_METRICS)
    }
    for row in rows:
        for name, value in (row.get("ragas") or {}).items():
            if name in values and isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[name].append(float(value))
    details: dict[str, dict[str, Any]] = {}
    for name, metric_values in values.items():
        score = _mean(metric_values)
        details[name] = {
            "score": score,
            "threshold": thresholds[name],
            "passed": None if score is None else score >= thresholds[name],
            "scored_cases": len(metric_values),
        }
    return details


def _case_diagnoses(row: dict[str, Any]) -> list[dict[str, str]]:
    scores = row.get("ragas") or {}
    errors = row.get("ragas_errors") or {}
    hard = row.get("hard_rules") or {}
    thresholds = _thresholds()
    diagnoses: list[dict[str, str]] = []

    if errors:
        names = "、".join(errors)
        diagnoses.append(
            {
                "category": "judge_error",
                "severity": "error",
                "title": "Judge 指标计算异常",
                "summary": f"{names} 未得到可信分数；先仅重试评分，不要据此调整 RAG。",
                "action": "重试异常指标；若仍失败，再检查超时、模型结构化输出和答案长度。",
            }
        )
    if not hard.get("source_hit", True) or not hard.get("contexts_present", True):
        diagnoses.append(
            {
                "category": "retrieval",
                "severity": "failed",
                "title": "目标证据没有进入回答上下文",
                "summary": "预期来源未命中，或生产检索链没有返回可评分上下文。",
                "action": "检查切片、Query 改写、来源过滤、Top-K 和 Rerank。",
            }
        )
    rule_failures = (
        list(hard.get("missing_phrases") or [])
        + list(hard.get("forbidden_phrases") or [])
    )
    if rule_failures or hard.get("answer_pattern_matched") is False:
        diagnoses.append(
            {
                "category": "generation",
                "severity": "failed",
                "title": "回答违反确定性验收规则",
                "summary": "回答遗漏关键事实、包含禁用事实，或没有遵守要求的输出格式。",
                "action": "收紧回答指令，并审查 must_include、must_not_include 与格式规则。",
            }
        )

    precision = scores.get("context_precision")
    recall = scores.get("context_recall")
    faithfulness = scores.get("faithfulness")
    relevancy = scores.get("answer_relevancy")
    goal = scores.get("agent_goal_accuracy")
    precision_low = isinstance(precision, (int, float)) and precision < thresholds["context_precision"]
    recall_low = isinstance(recall, (int, float)) and recall < thresholds["context_recall"]
    if precision_low and not recall_low:
        diagnoses.append(
            {
                "category": "retrieval_noise",
                "severity": "failed",
                "title": "证据找全了，但上下文噪声过多",
                "summary": "Context Recall 达标而 Context Precision 未达标。",
                "action": "比较 Top-K、Rerank 和来源状态特征，减少旧资料或同主题噪声。",
            }
        )
    elif recall_low:
        diagnoses.append(
            {
                "category": "retrieval_recall",
                "severity": "failed",
                "title": "关键证据召回不完整",
                "summary": "Context Recall 未达到当前评测门槛。",
                "action": "检查切片、Query 改写、多文档召回和来源过滤。",
            }
        )
    if (
        isinstance(faithfulness, (int, float))
        and faithfulness < thresholds["faithfulness"]
    ):
        diagnoses.append(
            {
                "category": "generation_faithfulness",
                "severity": "failed",
                "title": "回答没有忠于检索证据",
                "summary": "回答包含证据无法支持的扩写、推断或错误选择。",
                "action": "缩短答案，要求逐项依据证据回答，并加强冲突来源处理。",
            }
        )
    if isinstance(relevancy, (int, float)) and relevancy < thresholds["answer_relevancy"]:
        diagnoses.append(
            {
                "category": "generation_relevancy",
                "severity": "failed",
                "title": "回答没有直接回应问题",
                "summary": "答案相关性低于当前门槛，可能存在过度展开或遗漏用户追问。",
                "action": "改进问题拆解和回答结构，删除无关背景与下一步建议。",
            }
        )
    if isinstance(goal, (int, float)) and goal < thresholds["agent_goal_accuracy"]:
        diagnoses.append(
            {
                "category": "agent_goal",
                "severity": "failed",
                "title": "ChatAgent 没有完成题目要求",
                "summary": "回答可能事实相关，但没有遵守句数、格式或任务约束。",
                "action": "让生成链优先执行用户明确约束，再补充必要解释。",
            }
        )
    if not diagnoses:
        diagnoses.append(
            {
                "category": "passed",
                "severity": "passed",
                "title": "该用例未发现门禁问题",
                "summary": "硬规则和已得到的语义指标均符合当前门槛。",
                "action": "保留为回归基线。",
            }
        )
    return diagnoses


def _diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    failed_cases = 0
    error_cases = 0
    for row in rows:
        row["diagnoses"] = _case_diagnoses(row)
        categories = {item["category"] for item in row["diagnoses"] if item["category"] != "passed"}
        if categories:
            failed_cases += 1
        if row.get("ragas_errors"):
            error_cases += 1
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
    return {
        "failed_cases": failed_cases,
        "error_cases": error_cases,
        "category_counts": counts,
    }


def _finalize_result(
    *,
    rows: list[dict[str, Any]],
    retrieval: dict[str, float],
    performance: dict[str, Any],
    judge_requested: bool,
    scorer: SemanticScorer | None,
    judge_error: str | None,
    repo_id: str,
    limit: int,
    evaluation_set: dict[str, Any] | None,
    rescore_of: str | None = None,
) -> dict[str, Any]:
    metric_rows = _metric_details(rows)
    hard_passed_cases = sum(1 for row in rows if (row.get("hard_rules") or {}).get("passed"))
    hard_pass_rate = round(hard_passed_cases / len(rows), 4) if rows else 0.0
    case_metric_errors = sum(bool(row.get("ragas_errors")) for row in rows)
    required_scored = all(metric_rows[name]["score"] is not None for name in REQUIRED_RAGAS_METRICS)
    ragas_available = bool(judge_requested and scorer is not None and required_scored)
    evaluation_complete = bool(ragas_available and case_metric_errors == 0)
    gated_names = list(REQUIRED_RAGAS_METRICS)
    gated_names.extend(
        name
        for name in ("tool_call_accuracy", "tool_call_f1")
        if metric_rows[name]["scored_cases"] > 0
    )
    semantic_failed = any(metric_rows[name]["passed"] is False for name in gated_names)
    ragas_passed = evaluation_complete and not semantic_failed
    hard_passed = bool(rows) and hard_passed_cases == len(rows)
    if not rows:
        quality_status = "not_evaluated"
        status = "no_cases"
    elif not hard_passed or semantic_failed:
        quality_status = "failed"
        status = "error" if judge_requested and not evaluation_complete else "failed"
    elif not judge_requested:
        quality_status = "partial"
        status = "partial"
    elif not evaluation_complete:
        quality_status = "incomplete"
        status = "error"
    else:
        quality_status = "passed"
        status = "passed"

    summary_metrics = {
        name: value["score"]
        for name, value in metric_rows.items()
        if value["score"] is not None
    }
    summary_metrics.update(
        {
            "hard_gate_pass_rate": hard_pass_rate,
            "retrieval_recall_at_k": retrieval.get("recall_at_k", 0.0),
            "retrieval_precision_at_k": retrieval.get("precision_at_k", 0.0),
        }
    )
    diagnostics = _diagnostic_summary(rows)
    return {
        "kind": "chatagent_ragas",
        "status": status,
        "execution_status": "completed",
        "quality_status": quality_status,
        "evaluation_complete": evaluation_complete,
        "passed": status == "passed",
        "run_mode": "rescore" if rescore_of else "full",
        "rescore_of": rescore_of,
        "target": {
            "name": "ChatAgent",
            "repo_id": str(repo_id),
            "execution": "stored_evidence_rescore" if rescore_of else "production_agent_with_isolated_conversation",
            "evaluation_mode": True,
        },
        "evaluation_set": evaluation_set or {
            "id": "auto-document-smoke",
            "name": "auto-document-smoke",
            "label": "自动文档冒烟测试",
            "source": "auto_smoke",
            "sha256": None,
            "case_count": len(rows),
        },
        "pipeline": {"top_k": limit},
        **summary_metrics,
        "summary_metrics": summary_metrics,
        "retrieval": retrieval,
        "performance": performance,
        "hard_gate": {
            "passed": hard_passed,
            "pass_rate": hard_pass_rate,
            "passed_cases": hard_passed_cases,
            "total_cases": len(rows),
        },
        "ragas": {
            "requested": judge_requested,
            "available": ragas_available,
            "complete": evaluation_complete,
            "passed": ragas_passed if evaluation_complete else None,
            "version": scorer.version if scorer is not None else None,
            "judge_model": scorer.judge_model if scorer is not None else None,
            "error": judge_error or (
                f"{case_metric_errors} evaluation case(s) contain Ragas metric errors"
                if case_metric_errors
                else None
            ),
            "case_errors": case_metric_errors,
            "metrics": metric_rows,
        },
        "diagnostics": diagnostics,
        "cases": rows,
    }


async def run_rag_quality_eval(
    db: Session,
    repo_id: str,
    *,
    cases: list[RagEvalCase] | None = None,
    limit: int = 5,
    run_judge: bool | None = None,
    scorer: SemanticScorer | None = None,
    agent_factory: Callable[[], ChatAgent] | None = None,
    progress_callback: ProgressCallback | None = None,
    evaluation_set: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the production ChatAgent path with Ragas and deterministic gates."""

    repo = db.get(Repository, uuid.UUID(str(repo_id)))
    if repo is None:
        raise ValueError(f"Repository {repo_id} was not found")
    evaluation_cases = cases if cases is not None else _default_cases(db, repo_id)
    judge_requested = settings.ragas_enabled if run_judge is None else run_judge
    total_cases = len(evaluation_cases)
    await _emit_progress(
        progress_callback,
        stage="preparing",
        message=f"已准备 {total_cases} 条评测用例，正在初始化 Judge。",
        completed_cases=0,
        total_cases=total_cases,
    )
    judge_error: str | None = None
    if judge_requested and scorer is None:
        try:
            scorer = RagasCollectionsScorer()
        except RagasUnavailableError as exc:
            judge_error = str(exc)

    rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, case in enumerate(evaluation_cases, start=1):
        await _emit_progress(
            progress_callback,
            stage="chatagent",
            message=f"正在运行第 {index}/{total_cases} 条真实 ChatAgent 用例。",
            current_case=index,
            case_id=case.id or f"case-{index}",
            completed_cases=index - 1,
            total_cases=total_cases,
        )
        item_started = time.perf_counter()
        result = await _run_chatagent_case(
            db,
            repo,
            case,
            limit=limit,
            agent_factory=agent_factory or ChatAgent,
        )
        trace = result.get("evaluation_trace") if isinstance(result.get("evaluation_trace"), dict) else {}
        contexts, sources = _trace_rows(trace)
        hard_rules = _hard_rules(case, result, sources, contexts)
        semantic_scores: dict[str, float] | None = None
        semantic_errors: dict[str, str] = {}
        skipped_reason: str | None = None
        if not judge_requested:
            skipped_reason = "judge_disabled"
        elif scorer is None:
            skipped_reason = "judge_unavailable"
        elif not contexts:
            skipped_reason = "no_retrieved_contexts"
        else:
            await _emit_progress(
                progress_callback,
                stage="ragas",
                message=f"ChatAgent 已完成第 {index}/{total_cases} 条，正在计算 RAGAS 指标。",
                current_case=index,
                case_id=case.id or f"case-{index}",
                completed_cases=index - 1,
                total_cases=total_cases,
            )
            semantic_scores, semantic_errors = await scorer.score(
                user_input=case.question,
                response=str(result.get("answer") or ""),
                retrieved_contexts=contexts,
                reference=case.reference,
                tool_calls=[dict(item) for item in result.get("tool_calls") or [] if isinstance(item, dict)],
                tool_observations=[
                    dict(item)
                    for item in trace.get("tool_observations") or []
                    if isinstance(item, dict)
                ],
                expected_tool_calls=case.expected_tool_calls,
            )

        latency_ms = round((time.perf_counter() - item_started) * 1000, 2)
        retrieval_rows.append({"results": sources})
        rows.append(
            {
                "id": case.id or f"case-{index}",
                "question": case.question,
                "reference": case.reference,
                "expected_source_ids": case.expected_source_ids,
                "source_file": case.source_file,
                "kind": case.kind,
                "must_include": case.must_include,
                "must_not_include": case.must_not_include,
                "answer_regex": case.answer_regex,
                "required_tools": case.required_tools,
                "expected_tool_calls": case.expected_tool_calls,
                "response": result.get("answer"),
                "route": result.get("route"),
                "citations": result.get("citations") or [],
                "tool_calls": result.get("tool_calls") or [],
                "tool_observations": trace.get("tool_observations") or [],
                "completion": result.get("completion") or {},
                "retrieved_contexts": contexts,
                "retrieved_sources": sources,
                "retrieval_trace": trace.get("retrievals") or [],
                "hard_rules": hard_rules,
                "ragas": semantic_scores,
                "ragas_errors": semantic_errors,
                "ragas_skipped_reason": skipped_reason,
                "latency_ms": latency_ms,
            }
        )
        await _emit_progress(
            progress_callback,
            stage="case_complete",
            message=f"第 {index}/{total_cases} 条评测已完成。",
            current_case=index,
            case_id=case.id or f"case-{index}",
            completed_cases=index,
            total_cases=total_cases,
        )

    await _emit_progress(
        progress_callback,
        stage="finalizing",
        message="正在汇总指标并生成最终判定。",
        completed_cases=total_cases,
        total_cases=total_cases,
    )
    retrieval = _retrieval_metrics(evaluation_cases, retrieval_rows, limit)
    latencies = [float(row["latency_ms"]) for row in rows]
    performance = {
        "cases": len(rows),
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }
    return _finalize_result(
        rows=rows,
        retrieval=retrieval,
        performance=performance,
        judge_requested=judge_requested,
        scorer=scorer,
        judge_error=judge_error,
        repo_id=str(repo.id),
        limit=limit,
        evaluation_set=evaluation_set,
    )


async def rescore_rag_result(
    result: dict[str, Any],
    *,
    eval_id: str,
    scorer: SemanticScorer | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Retry missing or failed Judge metrics without running ChatAgent again."""

    rows = copy.deepcopy(result.get("cases") or [])
    if not rows:
        raise ValueError("原评测没有可用于重评分的逐题现场")
    retry_by_case: list[set[str]] = []
    for row in rows:
        scores = row.get("ragas") or {}
        errors = row.get("ragas_errors") or {}
        retry_names = {name for name in errors if name in (*CORE_RAGAS_METRICS, *AGENT_RAGAS_METRICS)}
        retry_names.update(name for name in REQUIRED_RAGAS_METRICS if name not in scores)
        expected_calls = row.get("expected_tool_calls") or []
        if expected_calls:
            retry_names.update(
                name for name in ("tool_call_accuracy", "tool_call_f1") if name not in scores
            )
        retry_by_case.append(retry_names)
    retry_count = sum(bool(names) for names in retry_by_case)
    if retry_count == 0:
        raise ValueError("原评测没有失败或缺失的 Judge 指标")

    judge_error: str | None = None
    if scorer is None:
        try:
            scorer = RagasCollectionsScorer()
        except RagasUnavailableError as exc:
            judge_error = str(exc)
    started = time.perf_counter()
    completed = 0
    for index, (row, retry_names) in enumerate(zip(rows, retry_by_case, strict=True), start=1):
        if not retry_names:
            continue
        await _emit_progress(
            progress_callback,
            stage="rescore",
            message=f"正在重试第 {completed + 1}/{retry_count} 条异常评分。",
            current_case=index,
            case_id=str(row.get("id") or f"case-{index}"),
            completed_cases=completed,
            total_cases=retry_count,
        )
        existing_errors = dict(row.get("ragas_errors") or {})
        if scorer is None:
            for name in retry_names:
                existing_errors[name] = f"RagasUnavailableError: {judge_error or 'Judge unavailable'}"
            row["ragas_errors"] = existing_errors
            completed += 1
            continue
        contexts = [str(value) for value in row.get("retrieved_contexts") or [] if str(value).strip()]
        if not contexts:
            for name in retry_names:
                existing_errors[name] = "ValueError: no retrieved contexts"
            row["ragas_errors"] = existing_errors
            completed += 1
            continue
        scores, errors = await scorer.score(
            user_input=str(row.get("question") or ""),
            response=str(row.get("response") or ""),
            retrieved_contexts=contexts,
            reference=str(row.get("reference") or ""),
            tool_calls=[dict(item) for item in row.get("tool_calls") or [] if isinstance(item, dict)],
            tool_observations=[
                dict(item) for item in row.get("tool_observations") or [] if isinstance(item, dict)
            ],
            expected_tool_calls=[
                dict(item) for item in row.get("expected_tool_calls") or [] if isinstance(item, dict)
            ],
            metric_names=retry_names,
        )
        merged_scores = dict(row.get("ragas") or {})
        merged_scores.update(scores)
        for name in retry_names:
            existing_errors.pop(name, None)
        existing_errors.update(errors)
        row["ragas"] = merged_scores
        row["ragas_errors"] = existing_errors
        row["ragas_skipped_reason"] = None
        completed += 1
        await _emit_progress(
            progress_callback,
            stage="case_complete",
            message=f"第 {completed}/{retry_count} 条异常评分已重试。",
            current_case=index,
            case_id=str(row.get("id") or f"case-{index}"),
            completed_cases=completed,
            total_cases=retry_count,
        )

    performance = dict(result.get("performance") or {})
    performance.update(
        {
            "agent_reused": True,
            "rescore_cases": retry_count,
            "rescore_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    )
    top_k = int((result.get("pipeline") or {}).get("top_k") or 5)
    return _finalize_result(
        rows=rows,
        retrieval=dict(result.get("retrieval") or {}),
        performance=performance,
        judge_requested=True,
        scorer=scorer,
        judge_error=judge_error,
        repo_id=str((result.get("target") or {}).get("repo_id") or ""),
        limit=top_k,
        evaluation_set=dict(result.get("evaluation_set") or {}),
        rescore_of=eval_id,
    )
