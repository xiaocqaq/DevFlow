import uuid
from types import SimpleNamespace

from app.services.code_search import query_terms, search_checkout_code, search_repository_code


def test_query_terms_prioritizes_code_identifiers() -> None:
    terms = query_terms("为什么 refreshAccessToken 在 auth/service.py 里失败？")

    assert "refreshaccesstoken" in terms
    assert "auth/service.py" in terms


def test_search_checkout_code_reads_current_files_without_database_index(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "auth.py"
    source.write_text(
        "def refresh_access_token():\n"
        "    return rotate_credentials()\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("REFRESH_ACCESS_TOKEN=secret\n", encoding="utf-8")

    results = search_checkout_code(tmp_path, "refresh_access_token", limit=5)

    assert results
    assert results[0]["source_type"] == "workspace_file"
    assert results[0]["source_id"] == "src/auth.py"
    assert results[0]["metadata"]["retrieval_mode"] == "workspace"
    assert all(item["source_id"] != ".env" for item in results)


def test_search_checkout_code_respects_worktree_subdirectory(tmp_path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / ".github" / "workflows" / "test.yml").write_text("name: backend-tests\n", encoding="utf-8")
    (tmp_path / "src" / "test.py").write_text("backend_tests = True\n", encoding="utf-8")

    results = search_checkout_code(
        tmp_path,
        "backend-tests",
        relative_path=".github/workflows",
    )

    assert results and {item["source_id"] for item in results} == {".github/workflows/test.yml"}


def test_search_repository_code_uses_configured_local_checkout(tmp_path) -> None:
    (tmp_path / "service.py").write_text("def handle_payment_webhook():\n    pass\n", encoding="utf-8")
    repo = SimpleNamespace(id=uuid.uuid4(), local_path=str(tmp_path), full_name="local/demo")

    results = search_repository_code(repo, "handle_payment_webhook")

    assert results and results[0]["metadata"]["repo_id"] == str(repo.id)


def test_search_checkout_code_observes_file_changes_without_reindex(tmp_path) -> None:
    source = tmp_path / "service.py"
    source.write_text("def old_handler():\n    pass\n", encoding="utf-8")
    assert search_checkout_code(tmp_path, "old_handler")

    source.write_text("def current_handler():\n    pass\n", encoding="utf-8")

    assert search_checkout_code(tmp_path, "current_handler")
    assert search_checkout_code(tmp_path, "old_handler") == []
