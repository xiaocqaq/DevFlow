import os
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.api.routes import repos as repo_routes
from app.db.models import (
    AgentRun,
    AgentTaskRun,
    AgentWorkflowRun,
    Base,
    ChatMessage,
    ChatSession,
    Conversation,
    EvidenceItem,
    KnowledgeGraphEdge,
    MemoryCandidate,
    ProjectIndexState,
    RecallEvent,
    Repository,
    Workspace,
)


def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.mark.asyncio
async def test_delete_repo_removes_all_conversation_and_workflow_dependencies(monkeypatch) -> None:
    db = _db_session()
    repo = Repository(owner="devflow", name="delete-me", full_name=f"devflow/delete-{uuid.uuid4().hex[:8]}")
    db.add(repo)
    db.flush()

    workspace = Workspace(name="deletion-test", repo_ids=[str(repo.id)])
    conversation = Conversation(repo_id=repo.id, title="delete regression")
    db.add_all([workspace, conversation])
    db.flush()

    session = ChatSession(
        repo_id=None,
        conversation_id=conversation.id,
        route="test",
        transcript_json={},
        digest="delete regression",
    )
    workflow = AgentWorkflowRun(
        repo_id=None,
        conversation_id=conversation.id,
        goal="delete regression",
    )
    db.add_all([session, workflow])
    db.flush()

    db.add_all(
        [
            MemoryCandidate(
                repo_id=None,
                conversation_id=conversation.id,
                session_id=session.id,
                kind="fact",
                title="candidate",
                content="must be deleted before the session",
            ),
            RecallEvent(
                repo_id=None,
                conversation_id=conversation.id,
                tool_name="memory.search",
            ),
            AgentTaskRun(
                workflow_id=workflow.id,
                task_id="task-1",
                agent_name="test-agent",
                task_type="test",
            ),
            AgentRun(
                repo_id=None,
                conversation_id=conversation.id,
                user_message="delete",
                route="test",
                tool_calls=[],
                final_answer="deleted",
            ),
            ChatMessage(
                repo_id=None,
                conversation_id=conversation.id,
                role="user",
                content="delete this project",
            ),
            EvidenceItem(
                repo_id=None,
                conversation_id=conversation.id,
                source_type="test",
                source_id="delete-test",
                title="evidence",
                content="delete regression",
            ),
            KnowledgeGraphEdge(
                repo_id=repo.id,
                from_type="issue",
                from_id="1",
                to_type="pull_request",
                to_id="2",
                relation="related",
            ),
            ProjectIndexState(repo_id=repo.id),
        ]
    )
    db.commit()

    vector_calls = []

    def fake_delete_milvus_documents(**kwargs):
        vector_calls.append(kwargs)
        return True

    monkeypatch.setattr(repo_routes, "delete_milvus_documents", fake_delete_milvus_documents)

    result = await repo_routes.delete_repo(repo.id, db)

    assert result["deleted"] is True
    assert result["cleanup"]["milvus_deleted"] is True
    assert vector_calls == [{"repo_id": repo.id, "strict": False}]
    assert db.get(Repository, repo.id) is None
    assert db.query(Conversation).count() == 0
    assert db.query(ChatSession).count() == 0
    assert db.query(MemoryCandidate).count() == 0
    assert db.query(RecallEvent).count() == 0
    assert db.query(AgentWorkflowRun).count() == 0
    assert db.query(AgentTaskRun).count() == 0
    assert db.query(AgentRun).count() == 0
    assert db.query(ChatMessage).count() == 0
    assert db.query(EvidenceItem).count() == 0
    assert db.query(KnowledgeGraphEdge).count() == 0
    assert db.query(ProjectIndexState).count() == 0
    assert db.get(Workspace, workspace.id).repo_ids == []
