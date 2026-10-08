import re
import json
from pathlib import Path

import pytest
from fastapi import BackgroundTasks
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Base, Conversation, Document, EvalRun, Repository
from app.api.routes.evaluations import _eval_comparison, _issue_markdown
from app.schemas.evals import EvalRunRequest
from app.services.agents.chat_agent import ChatAgent
from app.services.evals.ragas_runner import (
    CORE_RAGAS_METRICS,
    RagEvalCase,
    _default_cases,
    _hard_rules,
    _evaluation_project_context,
    _ensure_ragas_compatibility,
    _split_faithfulness_response,
    rescore_rag_result,
    run_rag_quality_eval,
)
from app.services.evals.suites import list_eval_suites, load_eval_suite
from app.services.evals.runner import run_basic_eval
from app.services.rag.vector_store import reset_milvus_store_cache


@pytest.mark.asyncio
async def test_basic_eval_runner_returns_real_metrics() -> None:
    result = await run_basic_eval()
    assert result["issue_classification_accuracy"] > 0
    assert result["tool_call_accuracy"] > 0
    assert result["rag_hit_rate_at_3"] == 1.0
    assert result["ci_pytest_failure_detection"] == 1.0
    assert "Eval Report" in result["report_markdown"]
    assert "CI pytest failure detection" in result["report_markdown"]


def test_ragas_collections_api_imports_with_current_langchain() -> None:
    _ensure_ragas_compatibility()
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    assert all([AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness])


def test_faithfulness_chunking_keeps_all_answer_content() -> None:
    response = "第一条事实。\n" + ("第二条很长的事实内容" * 90) + "！\n最后一条事实。"

    chunks = _split_faithfulness_response(response, max_chars=120)

    assert len(chunks) > 2
    assert all(0 < len(chunk) <= 120 for chunk in chunks)
    assert re.sub(r"\s+", "", "".join(chunks)) == re.sub(r"\s+", "", response)


def test_default_ragas_cases_fall_back_to_structured_repository_data() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="fallback", full_name="local/fallback")
    db.add(repo)
    db.flush()
    from app.db.models import Issue

    issue = Issue(repo_id=repo.id, number=7, title="Refresh token fails", body="Retry once with an idempotency key", state="open", labels=["bug"])
    db.add(issue)
    db.flush()

    cases = _default_cases(db, str(repo.id))

    assert len(cases) == 1
    assert cases[0].id == "issue-7"
    assert cases[0].source_type == "issue"
    assert cases[0].expected_source_ids == [str(issue.id)]


def test_fixed_eval_suite_resolves_repository_sources(tmp_path: Path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="DemoOrg", name="DevFlow-AI", full_name="DemoOrg/DevFlow-AI")
    db.add(repo)
    db.flush()
    source_id = __import__("uuid").uuid4()
    document = Document(
        repo_id=repo.id,
        source_type="knowledge_file",
        source_id=source_id,
        title="Facts",
        content="默认分支是 main。",
        meta={"filename": "01_repository_facts.md"},
    )
    db.add(document)
    db.flush()
    suite_dir = tmp_path / "evals"
    suite_dir.mkdir()
    (suite_dir / "baseline.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": "fixed-baseline",
                "label": "固定基线",
                "repository_names": ["DevFlow-AI"],
                "recommended": True,
                "cases": [
                    {
                        "id": "default-branch",
                        "question": "默认分支是什么？",
                        "reference": "main",
                        "source_file": "01_repository_facts.md",
                        "must_include": ["main"],
                        "answer_regex": "^main$",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    suites = list_eval_suites(db, repo.id, suite_dir=suite_dir)
    loaded = load_eval_suite(db, repo.id, "fixed-baseline", suite_dir=suite_dir)

    assert suites[0]["ready"] is True
    assert suites[0]["recommended"] is True
    assert suites[0]["sha256"]
    assert loaded.cases[0].expected_source_ids == [str(document.id), str(source_id)]
    assert loaded.cases[0].must_include == ["main"]
    assert loaded.cases[0].answer_regex == "^main$"


def test_hard_rules_enforce_required_forbidden_and_output_format() -> None:
    case = RagEvalCase(
        id="authority",
        question="权威来源？",
        reference="当前 Git checkout。",
        expected_source_ids=["source-1"],
        must_include=["当前 Git checkout"],
        must_not_include=["GitHub 仓库"],
        answer_regex=r"^当前 Git checkout[。.]?$",
    )
    result = {"answer": "GitHub 仓库", "tool_calls": [], "citations": []}
    hard = _hard_rules(case, result, [{"source_id": "source-1"}], ["evidence"])

    assert hard["passed"] is False
    assert hard["missing_phrases"] == ["当前 Git checkout"]
    assert hard["forbidden_phrases"] == ["GitHub 仓库"]
    assert hard["answer_pattern_matched"] is False


def test_eval_project_context_matches_frontend_team_and_memory_shape() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="context", full_name="local/context")
    db.add(repo)
    db.flush()
    db.add_all(
        [
            Document(
                repo_id=repo.id,
                source_type="team_member",
                source_id=__import__("uuid").uuid4(),
                title="团队成员：星哥",
                content="name: 星哥",
                meta={
                    "name": "星哥",
                    "role": "前端开发工程师",
                    "strengths": "擅长 Electron",
                    "tech_stack": "JavaScript、Electron、TypeScript",
                },
            ),
            Document(
                repo_id=repo.id,
                source_type="memory_note",
                source_id=__import__("uuid").uuid4(),
                title="桌面化上下文",
                content="Issue #1 是桌面化需求。",
                meta={},
            ),
        ]
    )
    db.flush()

    context = _evaluation_project_context(db, repo.id)

    assert context["team_members"] == [
        {
            "name": "星哥",
            "role": "前端开发工程师",
            "strengths": "擅长 Electron",
            "techStack": "JavaScript、Electron、TypeScript",
        }
    ]
    assert context["project_memory"] == [
        {"title": "桌面化上下文", "content": "Issue #1 是桌面化需求。"}
    ]


class EvalAnswerModel:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="Retries use an idempotency key and only retry once.")


class RecordingScorer:
    version = "test-ragas-0.4"
    judge_model = "recording-judge"

    def __init__(self) -> None:
        self.calls = []

    async def score(self, **kwargs):
        self.calls.append(kwargs)
        return (
            {
                "context_precision": 0.95,
                "context_recall": 0.96,
                "faithfulness": 0.97,
                "answer_relevancy": 0.94,
                "agent_goal_accuracy": 1.0,
            },
            {},
        )


class RetryScorer:
    version = "test-ragas-0.4"
    judge_model = "retry-judge"

    def __init__(self) -> None:
        self.metric_names = None

    async def score(self, **kwargs):
        self.metric_names = kwargs["metric_names"]
        return ({"faithfulness": 0.99}, {})


@pytest.mark.asyncio
async def test_rescore_reuses_saved_agent_evidence_and_only_retries_missing_metrics() -> None:
    scorer = RetryScorer()
    source_result = {
        "target": {"repo_id": "repo-1"},
        "evaluation_set": {"name": "fixed", "sha256": "abc"},
        "pipeline": {"top_k": 5},
        "retrieval": {"recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0, "ndcg": 1.0},
        "performance": {"cases": 1, "avg_latency_ms": 100},
        "cases": [
            {
                "id": "case-1",
                "question": "问题",
                "reference": "答案",
                "response": "答案",
                "retrieved_contexts": ["答案"],
                "retrieved_sources": [{"source_id": "source-1"}],
                "tool_calls": [],
                "tool_observations": [],
                "expected_tool_calls": [],
                "hard_rules": {"passed": True, "source_hit": True, "contexts_present": True, "answer_present": True},
                "ragas": {
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                    "answer_relevancy": 1.0,
                    "agent_goal_accuracy": 1.0,
                },
                "ragas_errors": {"faithfulness": "TimeoutError: "},
            }
        ],
    }

    rescored = await rescore_rag_result(source_result, eval_id="eval-1", scorer=scorer)

    assert scorer.metric_names == {"faithfulness"}
    assert rescored["run_mode"] == "rescore"
    assert rescored["rescore_of"] == "eval-1"
    assert rescored["performance"]["agent_reused"] is True
    assert rescored["cases"][0]["ragas_errors"] == {}
    assert rescored["cases"][0]["ragas"]["faithfulness"] == 0.99
    assert rescored["evaluation_complete"] is True
    assert rescored["status"] == "passed"


def test_eval_comparison_requires_same_baseline_and_reports_regressions() -> None:
    shared = {
        "evaluation_set": {"name": "fixed", "sha256": "abc"},
        "ragas": {"version": "0.4.3", "judge_model": "judge"},
        "pipeline": {"top_k": 5},
        "evaluation_complete": True,
    }
    base = EvalRun(
        name="base",
        result_json={
            **shared,
            "passed": True,
            "quality_status": "passed",
            "summary_metrics": {"faithfulness": 0.95},
            "cases": [{"id": "case-1", "hard_rules": {"passed": True}, "ragas_errors": {}}],
        },
    )
    candidate = EvalRun(
        name="candidate",
        result_json={
            **shared,
            "passed": False,
            "quality_status": "failed",
            "summary_metrics": {"faithfulness": 0.80},
            "cases": [
                {"id": "case-1", "hard_rules": {"passed": False}, "ragas_errors": {}},
            ],
        },
    )

    comparison = _eval_comparison(base, candidate)

    assert comparison["comparable"] is True
    assert comparison["metric_deltas"] == [
        {"name": "faithfulness", "base": 0.95, "candidate": 0.8, "delta": -0.15}
    ]
    assert comparison["regressed_cases"] == 1
    assert comparison["release_gate_passed"] is False

    candidate.result_json = {
        **candidate.result_json,
        "evaluation_set": {"name": "fixed", "sha256": "different"},
    }
    incompatible = _eval_comparison(base, candidate)
    assert incompatible["comparable"] is False
    assert "评测集 Hash 不一致" in incompatible["incompatibility_reasons"]


def test_issue_draft_markdown_contains_reproduction_and_actions() -> None:
    eval_run = EvalRun(
        name="failed-run",
        result_json={
            "status": "error",
            "quality_status": "failed",
            "evaluation_set": {"name": "fixed", "sha256": "abc"},
            "ragas": {
                "version": "0.4.3",
                "judge_model": "judge",
                "metrics": {
                    "faithfulness": {"score": None, "threshold": 0.9},
                    "tool_call_accuracy": {"score": None, "threshold": 0.9},
                },
            },
            "cases": [
                {
                    "id": "case-1",
                    "question": "问题",
                    "reference": "标准答案",
                    "response": "错误答案",
                    "ragas_errors": {"faithfulness": "TimeoutError"},
                    "diagnoses": [
                        {"category": "judge", "title": "Judge 异常", "action": "仅重试异常评分"}
                    ],
                }
            ],
        },
    )

    markdown = _issue_markdown(eval_run)

    assert "## 复现环境" in markdown
    assert "TimeoutError" in markdown
    assert "仅重试异常评分" in markdown
    assert "tool_call_accuracy" not in markdown
    assert "## 验收标准" in markdown


@pytest.mark.asyncio
async def test_rag_quality_eval_runs_real_chatagent_and_rolls_back_eval_conversation(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="ragas", full_name="local/ragas")
    db.add(repo)
    db.flush()
    source_id = __import__("uuid").uuid4()
    document = Document(
        repo_id=repo.id,
        source_type="project_doc",
        source_id=source_id,
        title="Retry policy",
        content="Authentication retries use an idempotency key and retry only once.",
        meta={},
    )
    db.add(document)
    db.commit()
    scorer = RecordingScorer()
    progress_events = []

    result = await run_rag_quality_eval(
        db,
        str(repo.id),
        cases=[
            RagEvalCase(
                id="retry-policy",
                question="How should authentication retries work?",
                reference="Use an idempotency key and retry only once.",
                expected_source_ids=[str(document.id), str(source_id)],
                source_type="project_doc",
            )
        ],
        scorer=scorer,
        agent_factory=lambda: ChatAgent(model=EvalAnswerModel()),  # type: ignore[arg-type]
        progress_callback=lambda event: progress_events.append(event),
    )

    assert result["kind"] == "chatagent_ragas"
    assert result["target"]["name"] == "ChatAgent"
    assert result["status"] == "passed"
    assert result["hard_gate"]["passed"] is True
    assert result["retrieval"]["recall_at_k"] == 1.0
    assert result["ragas"]["version"] == "test-ragas-0.4"
    assert all(result["ragas"]["metrics"][name]["score"] is not None for name in CORE_RAGAS_METRICS)
    assert scorer.calls[0]["response"].startswith("Retries use an idempotency key")
    assert scorer.calls[0]["retrieved_contexts"]
    assert [event["stage"] for event in progress_events] == [
        "preparing",
        "chatagent",
        "ragas",
        "case_complete",
        "finalizing",
    ]
    assert db.query(Conversation).filter(Conversation.status == "evaluation").count() == 0


@pytest.mark.asyncio
async def test_ragas_api_persists_summary_metrics(monkeypatch) -> None:
    from app.api.routes import evaluations as evaluation_routes

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    repo = Repository(owner="local", name="api", full_name="local/api")
    db.add(repo)
    db.commit()

    async def fake_run(db_arg, repo_id, **kwargs):
        assert db_arg is db
        assert repo_id == str(repo.id)
        assert kwargs["run_judge"] is True
        await kwargs["progress_callback"](
            {
                "stage": "ragas",
                "message": "scoring",
                "completed_cases": 0,
                "total_cases": 1,
            }
        )
        return {
            "kind": "chatagent_ragas",
            "status": "failed",
            "passed": False,
            "summary_metrics": {"faithfulness": 0.62, "context_recall": 0.91},
            "ragas": {"available": True, "metrics": {}},
            "cases": [],
        }

    monkeypatch.setattr(evaluation_routes, "run_rag_quality_eval", fake_run)
    monkeypatch.setattr(evaluation_routes, "SessionLocal", lambda: db)
    background_tasks = BackgroundTasks()
    response = await evaluation_routes.run_rag_eval(
        EvalRunRequest(name="api-ragas", repo_id=repo.id, run_judge=True),
        background_tasks=background_tasks,
        db=db,
        _user=object(),
    )
    queued_history = await evaluation_routes.list_evals(db=db, _user=object())

    assert response.status == "queued"
    assert queued_history[0]["status"] == "queued"

    await background_tasks()
    history = await evaluation_routes.list_evals(db=db, _user=object())
    detail = await evaluation_routes.get_eval(response.eval_id, db=db, _user=object())

    assert db.query(EvalRun).count() == 1
    assert history[0]["status"] == "failed"
    assert history[0]["metrics"] == {"faithfulness": 0.62, "context_recall": 0.91}
    assert detail["status"] == "failed"
