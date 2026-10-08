import os

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, Document, EvidenceItem, Repository
from app.services.knowledge_graph import _collect_nodes
from app.services.project_indexing import discover_project_docs, get_project_index_state, scan_project_index, snooze_project_index
from app.services.rag.retrieval import search_similar_documents


def _db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _write_allowlist_config(project) -> None:
    config_dir = project / ".devflow"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "index.yml").write_text(
        """version: 1
project_docs:
  mode: allowlist
  include:
    - README.md
    - docs/architecture/**
  exclude:
    - docs/course/**
    - "docs/demo code/**"
include_manifests: true
""",
        encoding="utf-8",
    )


def test_scan_project_index_writes_state_and_project_documents(tmp_path, monkeypatch) -> None:
    def fail_if_vector_runtime_is_used(*_args, **_kwargs):
        raise AssertionError("project preflight must not depend on embeddings or Milvus")

    monkeypatch.setattr("app.services.rag.vector_store.embed_texts", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.delete_milvus_documents", fail_if_vector_runtime_is_used)
    monkeypatch.setattr("app.services.rag.vector_store.upsert_milvus_documents", fail_if_vector_runtime_is_used)
    project = tmp_path / "demo"
    (project / "docs").mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n\nProject overview.", encoding="utf-8")
    (project / "docs" / "ADR-001.md").write_text("# ADR 001\n\nUse FastAPI.", encoding="utf-8")
    (project / "package.json").write_text('{"name":"demo","version":"1.0.0"}', encoding="utf-8")

    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(project), checkout_mode="local")
    db.add(repo)
    db.flush()

    state = scan_project_index(db, repo)

    assert state["status"] == "ready"
    assert state["docs_total"] == 3
    assert state["docs_indexed"] >= 4
    assert state["summary_json"]["tierCoverage"]["authoritative"] == 2
    assert db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_doc").count() >= 2
    assert db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_manifest").count() == 1
    project_doc = db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_doc").first()
    manifest = db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_manifest").one()
    overview = db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_overview").one()
    assert project_doc is not None and "embedding_dimensions" not in project_doc.meta
    assert not hasattr(project_doc, "embedding")
    assert manifest.meta["retrieval_mode"] == "direct"
    assert not hasattr(overview, "embedding")


def test_project_index_state_reports_stale_when_files_change(tmp_path) -> None:
    project = tmp_path / "demo"
    project.mkdir()
    (project / "README.md").write_text("# Demo\n\nProject overview.", encoding="utf-8")

    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(project), checkout_mode="local")
    db.add(repo)
    db.flush()

    assert scan_project_index(db, repo)["status"] == "ready"
    (project / "CHANGELOG.md").write_text("# Changelog\n\nInitial release.", encoding="utf-8")

    assert get_project_index_state(db, repo)["status"] == "stale"


def test_allowlist_rescan_removes_course_docs_from_rag_evidence_and_graph(tmp_path) -> None:
    project = tmp_path / "demo"
    (project / "docs" / "architecture").mkdir(parents=True)
    (project / "docs" / "course").mkdir(parents=True)
    (project / "docs" / "demo code").mkdir(parents=True)
    (project / ".data" / "runtime-checkout").mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n\nProject overview.", encoding="utf-8")
    (project / "docs" / "architecture" / "system.md").write_text(
        "# System architecture\n\nFastAPI service boundary.", encoding="utf-8"
    )
    (project / "docs" / "course" / "lesson.md").write_text(
        "# Course lesson\n\nCOURSE_ONLY_MARKER should never reach project retrieval.", encoding="utf-8"
    )
    (project / "docs" / "demo code" / "README.md").write_text(
        "# Course demo\n\nDEMO_ONLY_MARKER is teaching material.", encoding="utf-8"
    )
    (project / "package.json").write_text('{"name":"demo","version":"1.0.0"}', encoding="utf-8")
    (project / ".data" / "runtime-checkout" / "package.json").write_text(
        '{"name":"runtime-copy","version":"1.0.0"}', encoding="utf-8"
    )

    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(project), checkout_mode="local")
    db.add(repo)
    db.flush()

    assert scan_project_index(db, repo)["status"] == "ready"
    assert any(
        str((doc.meta or {}).get("path") or "").startswith("docs/course/")
        for doc in db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_doc").all()
    )

    _write_allowlist_config(project)
    state = scan_project_index(db, repo)

    indexed_paths = {
        str((doc.meta or {}).get("path") or "")
        for doc in db.query(Document).filter(Document.repo_id == repo.id, Document.source_type == "project_doc").all()
    }
    assert state["status"] == "ready"
    assert state["docs_total"] == 3
    assert state["summary_json"]["indexPolicy"]["mode"] == "allowlist"
    assert state["summary_json"]["excludedFiles"] == 2
    assert "README.md" in indexed_paths
    assert "docs/architecture/system.md" in indexed_paths
    assert not any(path.startswith("docs/course/") or path.startswith("docs/demo code/") for path in indexed_paths)
    assert not any(path.startswith(".data/") for path in indexed_paths)
    assert not any(
        "COURSE_ONLY_MARKER" in item.content or "DEMO_ONLY_MARKER" in item.content
        for item in db.query(EvidenceItem).filter(EvidenceItem.repo_id == repo.id).all()
    )
    assert not search_similar_documents(
        db,
        repo.id,
        "COURSE_ONLY_MARKER",
        retrieval_method="fulltext",
        apply_rerank=False,
    )
    assert any(
        item["source_type"] == "project_doc"
        for item in search_similar_documents(
            db,
            repo.id,
            "FastAPI service boundary",
            retrieval_method="fulltext",
            apply_rerank=False,
        )
    )
    graph_nodes = _collect_nodes(db, repo.id).values()
    assert not any(
        str((node.get("metadata") or {}).get("path") or "").startswith(("docs/course/", "docs/demo code/"))
        for node in graph_nodes
    )


def test_excluded_files_do_not_make_project_index_stale(tmp_path) -> None:
    project = tmp_path / "demo"
    (project / "docs" / "architecture").mkdir(parents=True)
    (project / "docs" / "course").mkdir(parents=True)
    (project / "README.md").write_text("# Demo\n\nProject overview.", encoding="utf-8")
    architecture = project / "docs" / "architecture" / "system.md"
    architecture.write_text("# Architecture\n\nVersion one.", encoding="utf-8")
    course = project / "docs" / "course" / "lesson.md"
    course.write_text("# Lesson\n\nVersion one.", encoding="utf-8")
    _write_allowlist_config(project)

    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(project), checkout_mode="local")
    db.add(repo)
    db.flush()

    assert scan_project_index(db, repo)["status"] == "ready"
    course.write_text("# Lesson\n\nA much longer excluded course revision.", encoding="utf-8")
    assert get_project_index_state(db, repo)["status"] == "ready"

    architecture.write_text("# Architecture\n\nVersion two changes project behavior.", encoding="utf-8")
    assert get_project_index_state(db, repo)["status"] == "stale"


def test_discover_project_docs_defaults_to_automatic_mode_without_config(tmp_path) -> None:
    project = tmp_path / "demo"
    (project / "docs" / "course").mkdir(parents=True)
    lesson = project / "docs" / "course" / "lesson.md"
    lesson.write_text("# Lesson\n\nLegacy automatic discovery remains compatible.", encoding="utf-8")

    assert [item.relative_path for item in discover_project_docs(project)] == ["docs/course/lesson.md"]


def test_snooze_project_index_sets_missing_state_with_cooldown(tmp_path) -> None:
    db = _db_session()
    repo = Repository(owner="local", name="demo", full_name="local/demo", local_path=str(tmp_path), checkout_mode="local")
    db.add(repo)
    db.flush()

    row = snooze_project_index(db, repo, days=3)

    assert row.status == "missing"
    assert row.snoozed_until is not None
