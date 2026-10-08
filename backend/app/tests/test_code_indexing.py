import subprocess
import uuid
from types import SimpleNamespace

from app.core.config import settings
from app.services import code_analysis


def test_sync_checkout_uses_cached_checkout_when_update_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "repo_checkout_dir", str(tmp_path))
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
    )
    checkout = code_analysis._repo_checkout_path(repo)
    (checkout / ".git").mkdir(parents=True)

    def fail_run(args, cwd=None):
        raise subprocess.CalledProcessError(128, ["git", *args], stderr="network failed")

    monkeypatch.setattr(code_analysis, "_run_git", fail_run)

    assert code_analysis._sync_checkout(repo, None) == checkout


def test_sync_checkout_uses_configured_local_repo_without_updating(tmp_path, monkeypatch) -> None:
    local_repo = tmp_path / "local-repo"
    local_repo.mkdir()
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
        local_path=str(local_repo),
        checkout_mode="local",
    )

    monkeypatch.setattr(code_analysis, "_is_git_work_tree", lambda path: path == local_repo.resolve())

    def fail_run(args, cwd=None):
        raise AssertionError("local repository indexing should not fetch, checkout, or pull")

    monkeypatch.setattr(code_analysis, "_run_git", fail_run)

    assert code_analysis._sync_checkout(repo, None) == local_repo.resolve()


def test_sync_checkout_rejects_non_empty_configured_download_target(tmp_path, monkeypatch) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "README.md").write_text("# occupied\n", encoding="utf-8")
    repo = SimpleNamespace(
        id=uuid.uuid4(),
        full_name="owner/repo",
        clone_url="https://github.com/owner/repo.git",
        default_branch="main",
        local_path=str(target),
        checkout_mode="managed",
    )

    monkeypatch.setattr(code_analysis.shutil, "which", lambda name: "git")
    monkeypatch.setattr(code_analysis, "_is_git_work_tree", lambda path: False)

    try:
        code_analysis._sync_checkout(repo, None)
    except RuntimeError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected configured non-empty download target to be rejected")
