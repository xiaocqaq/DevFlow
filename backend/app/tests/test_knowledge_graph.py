import os
import uuid

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Document, Issue, PRFile, PullRequest, Repository
from app.services.knowledge_graph import knowledge_graph_payload, rebuild_knowledge_graph


def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_rebuild_knowledge_graph_creates_explicit_project_edges() -> None:
    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo")
    db.add(repo)
    db.flush()

    member_id = uuid.uuid4()
    note_id = uuid.uuid4()
    issue = Issue(repo_id=repo.id, number=12, title="Auth refresh fails", state="open", assignees=["alice"])
    pr = PullRequest(repo_id=repo.id, number=4, title="Refresh tokens", state="closed", body="Fixes #12", author="alice")
    db.add_all(
        [
            Document(
                repo_id=repo.id,
                source_type="team_member",
                source_id=member_id,
                title="Alice",
                content="name: alice",
                meta={"name": "alice", "role": "Backend"},
            ),
            Document(
                repo_id=repo.id,
                source_type="memory_note",
                source_id=note_id,
                title="Auth convention",
                content="Keep token refresh in backend/app/auth.py. See #12.",
                meta={"display_title": "Auth convention"},
            ),
            issue,
            pr,
        ]
    )
    db.flush()
    db.add(PRFile(pr_id=pr.id, filename="backend/app/auth.py", status="modified"))
    db.commit()

    result = rebuild_knowledge_graph(db, repo)
    graph = knowledge_graph_payload(db, repo, center_type="pull_request", center_id=str(pr.id), depth=1)
    overview = knowledge_graph_payload(db, repo)
    edge_keys = {(edge["from_type"], edge["to_type"], edge["relation"]) for edge in graph["edges"]}

    assert result["edges"] >= 4
    assert ("pull_request", "issue", "resolves") in edge_keys
    assert ("pull_request", "code_file", "touches_file") in edge_keys
    assert ("issue", "team_member", "assigned_to") in edge_keys
    assert any(edge["from_type"] == "memory_note" and edge["relation"] == "cites" for edge in overview["edges"])
    assert any(node["type"] == "pull_request" and node["id"] == str(pr.id) for node in graph["nodes"])
