import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.api.routes.action_drafts import confirm_action_draft, create_action_draft
from app.api.routes.ci import _augment_ci_context_with_snapshot, _build_ci_analysis_context
from app.api.routes.code_graph import search_code_graph
from app.api.routes.pull_requests import _augment_pr_context_with_snapshot, _build_pr_analysis_context
from app.api.routes.workspaces import create_workspace, multi_repo_report
from app.core.config import settings
from app.db.models import Base, CodeRelation, CodeSymbol, Issue, PRFile, PRReviewComment, PullRequest, Repository, User, WorkflowRun
from app.schemas.action_drafts import ActionDraftCreateRequest
from app.schemas.workspaces import MultiRepoReportRequest, WorkspaceCreateRequest
from app.services.code_analysis import _sync_code_graph
from app.services.code_search import search_repository_code
from app.services.rag.vector_store import reset_milvus_store_cache
from app.services.worktree_manager import _sync_pr_snapshot_analysis, cleanup_pr_snapshots


def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _user(db):
    user = User(name="owner", role="owner")
    db.add(user)
    db.flush()
    return user


def _repo(db, name: str = "demo") -> Repository:
    repo = Repository(owner="local", name=name, full_name=f"local/{name}", checkout_mode="managed", clone_url=f"https://github.com/local/{name}.git")
    db.add(repo)
    db.flush()
    return repo


@pytest.mark.asyncio
async def test_send_report_action_draft_confirm_records_execution() -> None:
    db = _db_session()
    user = _user(db)
    repo = _repo(db)

    draft = await create_action_draft(
        ActionDraftCreateRequest(repo_id=repo.id, draft_type="send_report", target_type="repository", title="Weekly report", content="Report body"),
        db=db,
        user=user,
    )

    result = await confirm_action_draft(draft.id, db=db, user=user)

    assert result["draft"].status == "executed"
    assert result["draft"].execution_result["status"] == "recorded"
    assert result["audit_id"]


@pytest.mark.asyncio
async def test_multi_repo_report_aggregates_workspace_repositories(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    user = _user(db)
    repo_a = _repo(db, "api")
    repo_b = _repo(db, "web")
    db.add(Issue(repo_id=repo_a.id, number=1, title="Fix auth", state="open", created_at=datetime.now(timezone.utc)))
    db.add(PullRequest(repo_id=repo_b.id, number=2, title="UI update", state="open", created_at=datetime.now(timezone.utc)))
    db.add(WorkflowRun(repo_id=repo_b.id, github_run_id=3, name="test", status="completed", conclusion="failure", created_at=datetime.now(timezone.utc)))
    db.flush()

    workspace = await create_workspace(WorkspaceCreateRequest(name="team", repo_ids=[repo_a.id, repo_b.id]), db=db, user=user)
    result = await multi_repo_report(
        MultiRepoReportRequest(
            workspace_id=workspace.id,
            start_date=datetime.now(timezone.utc).date(),
            end_date=datetime.now(timezone.utc).date(),
        ),
        db=db,
        user=user,
    )

    assert result["metrics"]["repos"] == 2
    assert result["metrics"]["open_issues"] == 1
    assert result["metrics"]["open_prs"] == 1
    assert result["metrics"]["failed_ci"] == 1
    assert "多仓库研发周报" in result["report_markdown"]


@pytest.mark.asyncio
async def test_code_graph_indexes_symbols_and_relations(tmp_path) -> None:
    db = _db_session()
    user = _user(db)
    repo = _repo(db)
    project = tmp_path / "repo"
    project.mkdir()
    (project / "service.py").write_text(
        "from lib import helper\n\n"
        "def handler():\n"
        "    return helper()\n\n"
        "class Worker:\n"
        "    pass\n",
        encoding="utf-8",
    )

    count = _sync_code_graph(db, repo, project, branch="main", commit_sha="abc123")
    db.commit()
    result = await search_code_graph(repo.id, q="handler", limit=20, db=db, _user=user)

    assert count == 2
    assert result["symbols"][0].name == "handler"
    assert any(item["relation_type"] == "calls" for item in result["relations"])


def test_pr_snapshot_live_search_is_pr_scoped(tmp_path) -> None:
    db = _db_session()
    repo = _repo(db)
    project = tmp_path / "repo"
    project.mkdir()
    (project / "feature.py").write_text("def feature_flag():\n    return True\n", encoding="utf-8")

    snapshot_results = search_repository_code(
        repo,
        "feature_flag",
        checkout_path=project,
        metadata={
            "branch": "feature/pr-snapshot",
            "commit_sha": "abc123",
            "pr_id": "pr-123",
            "pr_number": 12,
            "checkout_kind": "pr_snapshot",
        },
    )

    assert snapshot_results
    metadata = snapshot_results[0]["metadata"]
    assert metadata["checkout_kind"] == "pr_snapshot"
    assert metadata["branch"] == "feature/pr-snapshot"
    assert metadata["commit_sha"] == "abc123"
    assert metadata["pr_id"] == "pr-123"
    assert metadata["pr_number"] == 12
    assert snapshot_results[0]["source_type"] == "workspace_file"


def test_pr_snapshot_analysis_writes_code_graph_without_documents(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    pr = PullRequest(repo_id=repo.id, number=12, title="Feature flag", state="open", head_branch="feature/pr-snapshot")
    db.add(pr)
    db.flush()
    project = tmp_path / "repo"
    project.mkdir()
    (project / "feature.py").write_text("def handler():\n    return helper()\n\ndef helper():\n    return True\n", encoding="utf-8")

    indexed = _sync_pr_snapshot_analysis(db, repo, pr, project, "feature/pr-snapshot", "abc123")
    db.commit()

    assert indexed == 2
    assert db.query(CodeSymbol).filter(CodeSymbol.pr_id == pr.id, CodeSymbol.pr_number == 12).count() == 2


def test_pr_analysis_context_prefers_snapshot_docs_and_graph(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    pr = PullRequest(repo_id=repo.id, number=4, title="Use handler", state="open", head_branch="feature/handler")
    db.add(pr)
    db.flush()
    db.add(PRFile(pr_id=pr.id, filename="service.py", status="modified", additions=4, deletions=1, patch="handler()"))
    db.flush()
    project = tmp_path / "repo"
    project.mkdir()
    (project / "service.py").write_text("def handler():\n    return helper()\n\ndef helper():\n    return True\n", encoding="utf-8")
    _sync_pr_snapshot_analysis(db, repo, pr, project, "feature/handler", "abc123")
    db.commit()
    db.refresh(pr)

    context = {"code_references": []}
    counts = {"code": 0}
    snapshot = {"status": "ready", "snapshot_name": "PR 分支快照", "pr_id": str(pr.id), "pr_number": pr.number, "branch": pr.head_branch, "commit_sha": "abc123", "indexed_symbols": 2, "snapshot_path": str(project)}
    _augment_pr_context_with_snapshot(db, pr, "handler service.py", context, counts, snapshot)

    assert counts["snapshot_code"] > 0
    assert context["code_references"][0]["metadata"]["pr_id"] == str(pr.id)
    assert context["code_graph_impact"]["symbols"][0]["name"] == "handler"


def test_pr_analysis_keeps_patch_out_of_semantic_queries(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    pr = PullRequest(
        repo_id=repo.id,
        number=8,
        title="Retry token refresh",
        body="Retry once after a refresh failure.",
        state="open",
    )
    db.add(pr)
    db.flush()
    db.add_all(
        [
            PRFile(
                pr_id=pr.id,
                filename="auth.py",
                status="modified",
                additions=2,
                deletions=1,
                patch="SENSITIVE_PATCH_ONLY_FOR_CODE_LOOKUP",
            ),
            PRReviewComment(
                pr_id=pr.id,
                body="Keep the retry idempotent.",
                path="auth.py",
                line=42,
                author="reviewer",
            ),
        ]
    )
    db.commit()
    db.refresh(pr)
    semantic_calls: list[str] = []
    code_calls: list[str] = []

    def fake_search(_db, _repo_id, query, source_type=None, _limit=5, **kwargs):
        semantic_calls.append(query)
        return []

    def fake_code_search(_repo, query, **_kwargs):
        code_calls.append(query)
        return []

    monkeypatch.setattr("app.api.routes.pull_requests.search_similar_documents", fake_search)
    monkeypatch.setattr("app.api.routes.pull_requests.search_repository_code", fake_code_search)

    code_query, _context, _counts = _build_pr_analysis_context(db, pr)

    assert "SENSITIVE_PATCH_ONLY_FOR_CODE_LOOKUP" in code_query
    assert any("SENSITIVE_PATCH_ONLY_FOR_CODE_LOOKUP" in query for query in code_calls)
    assert semantic_calls
    assert all("SENSITIVE_PATCH_ONLY_FOR_CODE_LOOKUP" not in query for query in semantic_calls)
    assert any("Keep the retry idempotent." in query for query in semantic_calls)


def test_ci_context_uses_related_pr_snapshot_docs_and_graph(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    pr = PullRequest(repo_id=repo.id, number=7, title="Fix service", state="open", head_branch="feature/ci")
    db.add(pr)
    db.flush()
    project = tmp_path / "repo"
    project.mkdir()
    (project / "service.py").write_text("def test_target():\n    return build()\n\ndef build():\n    return True\n", encoding="utf-8")
    _sync_pr_snapshot_analysis(db, repo, pr, project, "feature/ci", "abc123")
    run = WorkflowRun(repo_id=repo.id, github_run_id=99, name="tests", status="completed", conclusion="failure", logs_text="ERROR service.py test_target failed")
    db.add(run)
    db.commit()

    context = {"code_references": [], "related_files_from_logs": ["service.py"]}
    counts = {"code": 0}
    snapshot = {"status": "ready", "snapshot_name": "PR 分支快照", "pr_id": str(pr.id), "pr_number": pr.number, "branch": pr.head_branch, "commit_sha": "abc123", "indexed_symbols": 2, "snapshot_path": str(project)}
    _augment_ci_context_with_snapshot(db, run, "service.py test_target failed", context, counts, snapshot)

    assert counts["snapshot_code"] > 0
    assert context["code_references"][0]["metadata"]["checkout_kind"] == "pr_snapshot"
    assert context["code_graph_impact"]["symbols"][0]["name"] == "test_target"


def test_ci_analysis_redacts_semantic_queries_but_keeps_local_code_context(monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    db = _db_session()
    repo = _repo(db)
    secret = "ghp_" + "a" * 32
    run = WorkflowRun(
        repo_id=repo.id,
        github_run_id=101,
        name="tests",
        status="completed",
        conclusion="failure",
        logs_text=f"TOKEN={secret}\nERROR auth.py test_refresh failed",
    )
    db.add(run)
    db.commit()
    semantic_calls: list[str] = []
    code_calls: list[str] = []

    def fake_search(_db, _repo_id, query, source_type=None, _limit=5, **kwargs):
        semantic_calls.append(query)
        return []

    def fake_code_search(_repo, query, **_kwargs):
        code_calls.append(query)
        return []

    monkeypatch.setattr("app.api.routes.ci.search_similar_documents", fake_search)
    monkeypatch.setattr("app.api.routes.ci.search_repository_code", fake_code_search)

    code_query, _context, _counts = _build_ci_analysis_context(db, run)

    assert secret in code_query
    assert code_calls and secret in code_calls[0]
    assert semantic_calls
    assert all(secret not in query for query in semantic_calls)
    assert all("[REDACTED]" in query for query in semantic_calls)


def test_cleanup_pr_snapshots_removes_old_dirs_and_code_graph(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "milvus_enabled", False)
    reset_milvus_store_cache()
    checkout_root = tmp_path / "checkouts"
    monkeypatch.setattr(settings, "repo_checkout_dir", str(checkout_root))
    db = _db_session()
    repo = _repo(db)
    snapshot_dir = checkout_root / "worktrees" / str(repo.id) / "pr-1-local__demo"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / ".devflow-pr-snapshot.json").write_text(
        '{"updated_at": "'
        + (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        + '", "pr_id": "00000000-0000-0000-0000-000000000123"}',
        encoding="utf-8",
    )
    pr_id = UUID("00000000-0000-0000-0000-000000000123")
    symbol = CodeSymbol(
        repo_id=repo.id,
        path="service.py",
        name="handler",
        kind="function",
        language="py",
        start_line=1,
        pr_id=pr_id,
    )
    db.add(symbol)
    db.flush()
    db.add(CodeRelation(repo_id=repo.id, source_symbol_id=symbol.id, source_name="handler", target_name="helper", relation_type="calls", path="service.py", pr_id=pr_id))
    db.commit()

    result = cleanup_pr_snapshots(db, repo, max_age_hours=1)

    assert result["removed"] == 1
    assert not snapshot_dir.exists()
    assert db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id).count() == 0
    assert db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id).count() == 0
