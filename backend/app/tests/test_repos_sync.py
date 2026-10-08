import os

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Document, Issue, Repository
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository


def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.mark.asyncio
async def test_sync_repository_persists_facts_without_triggering_rag(monkeypatch) -> None:
    async def fake_list_issues(*_args, **_kwargs):
        return [
            {
                "id": 7800,
                "number": 78,
                "title": "Error: Could not import 'app'.",
                "body": "Realistic issue body",
                "state": "open",
                "labels": [{"name": "bug"}],
                "user": {"login": "contributor"},
                "assignees": [],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "closed_at": None,
            }
        ]

    def fail_if_vector_runtime_is_used(*_args, **_kwargs):
        raise AssertionError("GitHub fact sync must not depend on embeddings or Milvus")

    monkeypatch.setattr("app.services.repos_sync.list_issues", fake_list_issues)
    monkeypatch.setattr("app.services.rag.vector_store.embed_texts", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.delete_milvus_documents", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.upsert_milvus_documents", fail_if_vector_runtime_is_used)

    db = _db_session()
    repo = Repository(
        owner="openai",
        name="openai-quickstart-python",
        full_name="openai/openai-quickstart-python",
        provider="github",
        api_base_url="https://api.github.com",
    )
    db.add(repo)
    db.commit()

    result = await sync_repository(
        db,
        repo,
        RepoSyncRequest(sync_pull_requests=False, sync_workflow_runs=False, limit=1),
    )

    issue = db.query(Issue).filter(Issue.repo_id == repo.id).one()
    assert result.status == "completed"
    assert result.synced == {"issues": 1, "pull_requests": 0, "workflow_runs": 0}
    assert issue.number == 78
    assert issue.title == "Error: Could not import 'app'."
    assert db.query(Document).filter(Document.repo_id == repo.id).count() == 0
