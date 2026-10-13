# ==============================================================================
# 模块名称: Git Worktree 隔离快照管理服务 (Git Worktree & PR Snapshot Manager)
# 架构定位: 底层基础设施层 (Infrastructure) / 代码分析支撑服务
# 核心技术与设计原理:
#   1. Git Worktree 核心优势:
#      传统的 `git checkout` 会切换整个仓库的当前目录，导致正在并发执行的主分支任务或编译中断。
#      通过 `git worktree add <path> <branch>`，多个独立的工作树可以共享同一个底层的 `.git/objects` 对象库，
#      却能在不同的文件路径下同时存在不同分支的代码，极速创建隔离目录，几乎零磁盘重复占用。
#   2. 细粒度线程互斥锁 (Fine-grained Thread Locking):
#      使用字典结合保护锁 `_SNAPSHOT_LOCKS_GUARD` 动态创建针对 `{repo_id}:{pr_id}` 的专属锁，
#      防止同一个 PR 在多并发请求下同时执行 fetch/checkout 引发 Git 锁竞争或 index.lock 报错。
#   3. TTL 自动过期与清理机制:
#      快照附带 `.devflow-pr-snapshot.json` 元数据，定时清理超期 worktree 及数据库符号索引，防止磁盘泄漏。
# ==============================================================================

import json
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import CodeRelation, CodeSymbol, PullRequest, Repository
from app.services.code_analysis import _git_text, _sync_checkout, _sync_code_graph

# 线程锁池：为每一个 (repo_id, pr_id) 分配独立的互斥锁
_SNAPSHOT_LOCKS: dict[str, threading.Lock] = {}
# 锁池守卫锁：保护 _SNAPSHOT_LOCKS 字典自身的并发读写安全
_SNAPSHOT_LOCKS_GUARD = threading.Lock()


def _checkout_root() -> Path:
    """
    【路径管理】获取并确保检出代码的全局根目录存在。

    Returns:
        Path: 解析为绝对路径后的工作根目录。
    """
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _snapshot_lock(key: str) -> threading.Lock:
    """
    【并发控制】按 Key 动态获取或创建专属互斥锁 (Double-Checked / Guarded Pattern)。

    设计说明:
      - 为什么不直接用全局大锁？
        若使用单一全局大锁，所有仓库、所有 PR 的检出分析将完全串行化，极大降低系统吞吐量。
      - 细粒度锁池按业务维度 (`repo.id:pr.id`) 分离锁粒度，不同 PR 互不干扰，完全并行。

    Args:
        key (str): 锁标识符 (如 "repo_id:pr_id")。

    Returns:
        threading.Lock: 对应的专属线程锁。
    """
    with _SNAPSHOT_LOCKS_GUARD:
        lock = _SNAPSHOT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SNAPSHOT_LOCKS[key] = lock
        return lock


def _run_git(args: list[str], cwd: Path) -> str:
    """
    【外部进程执行器】在指定目录下执行 git 命令并捕获控制台输出。

    参数配置技巧:
      - `check=True`: 若 git 返回非 0 退出码，自动抛出 `subprocess.CalledProcessError`。
      - `text=True`: 自动按系统编码将字节流解码为字符串。
      - `timeout=180`: 设定 3 分钟超时门禁，防止大仓库拉取网络假死挂起整个工作线程。

    Args:
        args (list[str]): 传递给 git 的子命令参数列表。
        cwd (Path): 执行命令的工作目录。

    Returns:
        str: 清理首尾空格后的标准输出文本。
    """
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    return result.stdout.strip()


def pr_worktree_path(repo: Repository, pr: PullRequest) -> Path:
    """
    【路径规范】生成指定仓库指定 PR 的 worktree 独立磁盘路径。

    命名规范:
      - 替换路径非法字符 (如 `/` 与 `\\`) 确保跨平台（Windows/Linux）兼容性。
      - 结构: `<checkout_root>/worktrees/<repo_id>/pr-<pr_number>-<safe_name>`
    """
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    return _checkout_root() / "worktrees" / str(repo.id) / f"pr-{pr.number}-{safe_name}"


def _snapshot_meta_path(path: Path) -> Path:
    """
    【内部辅助方法】获取存放 PR 快照元数据的隐藏文件路径。
    """
    return path / ".devflow-pr-snapshot.json"


def _write_snapshot_meta(path: Path, meta: dict[str, Any]) -> None:
    """
    【元数据持久化】将快照构建信息（包含分支名、Commit SHA、索引符号数、UTC 时间戳）写入本地 JSON 文件。
    """
    payload = {**meta, "updated_at": datetime.now(timezone.utc).isoformat()}
    _snapshot_meta_path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sync_pr_snapshot_analysis(
    db: Session,
    repo: Repository,
    pr: PullRequest,
    snapshot_path: Path,
    branch: str,
    commit_sha: str | None,
) -> int:
    """
    【语法与图谱分析触发】对 PR 快照工作目录触发 AST 解析与代码知识图谱提取。

    将提取出的符号（类、函数）打上 pr_id 与 pr_number 标签，与默认分支符号隔离。
    """
    return _sync_code_graph(
        db,
        repo,
        snapshot_path,
        branch=branch,
        commit_sha=commit_sha,
        pr_id=pr.id,
        pr_number=pr.number,
    )


def _snapshot_response(
    *,
    status: str,
    message: str | None = None,
    branch: str | None = None,
    commit_sha: str | None = None,
    pr: PullRequest,
    snapshot_path: Path | None = None,
    indexed_symbols: int | None = None,
    fallback: str | None = None,
) -> dict[str, Any]:
    """
    【响应统一封装】构建 PR 快照状态的标准结构化字典。
    """
    return {
        "status": status,
        "message": message,
        "snapshot_name": "PR 分支快照",
        "branch": branch,
        "commit_sha": commit_sha,
        "pr_id": str(pr.id),
        "pr_number": pr.number,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "indexed_symbols": indexed_symbols,
        "fallback": fallback,
        "checkout_kind": "pr_snapshot" if status == "ready" else "default_checkout",
    }


def prepare_pr_worktree(db: Session, repo: Repository, pr: PullRequest) -> dict[str, Any]:
    """
    【核心方法】准备 PR 分支的专属 Worktree 独立工作区，并同步解析更新其代码图谱。

    执行流程与并发保护:
      1. 获取当前仓库+PR的专属互斥锁 `with lock:`，避免多并发竞争。
      2. 前置防御校验:
         - 本地仓库模式 (`local`) 不额外创建快照。
         - 校验 `pr.head_branch` 必须存在。
         - 校验本地环境是否已安装 `git` 命令 (`shutil.which('git')`)。
      3. 检出与 Worktree 构建:
         - 获取主检出目录 `base_checkout`。
         - 在主检出目录中执行 `git fetch origin <head_branch>` 拉取远端最新提交。
         - 若目标 worktree 目录已存在: 直接在 worktree 中 fetch 并重置检出。
         - 若不存在: 在主检出执行 `git worktree add --force -B <local_branch> <target> FETCH_HEAD`。
      4. 代码图谱解析与持久化:
         - 解析 HEAD commit sha。
         - 触发 AST 符号提取 `_sync_pr_snapshot_analysis`。
         - 写入本地 `.devflow-pr-snapshot.json` 快照元数据。
         - 提交数据库事务并返回就绪状态。
      5. 异常容错回退机制:
         - 若网络断开或 Git 操作异常，回滚事务并返回 `fallback` 状态，平滑降级使用主分支。

    Args:
        db (Session): 数据库会话。
        repo (Repository): 仓库实体。
        pr (PullRequest): PR 实体。

    Returns:
        dict[str, Any]: 包含快照就绪状态、分支、Commit SHA 及本地路径的字典。
    """
    lock_key = f"{repo.id}:{pr.id}"
    lock = _snapshot_lock(lock_key)
    with lock:
        # 本地托管模式防御
        if (repo.checkout_mode or "managed") == "local":
            return _snapshot_response(status="unsupported", message="本地仓库模式不自动创建或删除 PR 分支快照。", pr=pr, fallback=repo.local_path)
        if not pr.head_branch:
            return _snapshot_response(status="missing_head", message="PR 缺少 head branch，继续使用默认分支代码。", pr=pr)
        if not shutil.which("git"):
            return _snapshot_response(status="failed", message="未找到 git，可继续使用默认分支代码。", pr=pr)

        token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
        target = pr_worktree_path(repo, pr)
        try:
            # 1. 确保主检出目录就绪
            base_checkout = _sync_checkout(repo, token)
            target.parent.mkdir(parents=True, exist_ok=True)
            ref = pr.head_branch
            local_branch = f"devflow/pr-{pr.number}"

            # 2. 从远端拉取 PR 分支数据
            _run_git(["fetch", "origin", ref], cwd=base_checkout)

            # 3. 增量更新或新建 Git Worktree 独立目录
            if target.exists():
                _run_git(["fetch", "origin", ref], cwd=target)
                _run_git(["checkout", "-B", local_branch, "FETCH_HEAD"], cwd=target)
            else:
                _run_git(["worktree", "add", "--force", "-B", local_branch, str(target), "FETCH_HEAD"], cwd=base_checkout)

            # 4. 获取实际检出的提交哈希
            commit_sha = _git_text(["rev-parse", "HEAD"], target)

            # 5. 提取 AST 代码知识图谱
            indexed_symbols = _sync_pr_snapshot_analysis(db, repo, pr, target, ref, commit_sha)

            # 6. 保存快照文件元数据
            _write_snapshot_meta(
                target,
                {
                    "repo_id": str(repo.id),
                    "pr_id": str(pr.id),
                    "pr_number": pr.number,
                    "branch": ref,
                    "commit_sha": commit_sha,
                    "indexed_symbols": indexed_symbols,
                    "snapshot_kind": "pr_snapshot",
                },
            )
            db.commit()
            return _snapshot_response(
                status="ready",
                branch=ref,
                commit_sha=commit_sha,
                pr=pr,
                snapshot_path=target,
                indexed_symbols=indexed_symbols,
            )
        except Exception as exc:
            db.rollback()
            # 降级容错：不中断整体 Review 流程，回退使用主分支代码
            return _snapshot_response(
                status="fallback",
                message=f"PR 分支快照准备失败，继续使用默认分支代码：{exc}",
                branch=repo.default_branch or "main",
                pr=pr,
            )


def cleanup_pr_snapshots(db: Session, repo: Repository, *, max_age_hours: int = 168) -> dict[str, Any]:
    """
    【运维与垃圾回收】定期清理过期的 PR Worktree 快照与数据库专属符号。

    设计思想:
      - 磁盘配额防护: PR 合并或关闭后，本地可能积攒大量过期的 worktree 目录。
      - 默认超时时间: 168 小时（7 天）。
      - 清理步骤:
        1. 检查快照 `.devflow-pr-snapshot.json` 的更新时间，若超时则标记清理。
        2. 优先调用 `git worktree remove --force <path>`，通知 Git 释放内部引用指针。
        3. 若 Git 命令异常，兜底调用 `shutil.rmtree` 物理清除。
        4. 从 `CodeRelation` 与 `CodeSymbol` 批量物理删除该 PR 的关联代码符号索引。

    Args:
        db (Session): 数据库会话。
        repo (Repository): 仓库实体。
        max_age_hours (int): 快照保留时长阈值（小时），默认 168 小时。

    Returns:
        dict[str, Any]: 包含清理数量和被清理目录列表的统计字典。
    """
    root = _checkout_root() / "worktrees" / str(repo.id)
    if not root.exists():
        return {"removed": 0, "paths": []}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    removed_paths: list[str] = []
    removed_pr_ids: set[uuid.UUID] = set()

    for path in root.iterdir():
        if not path.is_dir():
            continue

        meta_path = _snapshot_meta_path(path)
        updated_at = None
        snapshot_pr_id: uuid.UUID | None = None

        # 优先读取快照元数据时间戳
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                updated_raw = metadata.get("updated_at")
                updated_at = datetime.fromisoformat(updated_raw) if updated_raw else None
                if metadata.get("pr_id"):
                    snapshot_pr_id = uuid.UUID(str(metadata["pr_id"]))
            except (OSError, ValueError, json.JSONDecodeError):
                updated_at = None

        # 若无元数据文件，以操作系统文件夹最后修改时间为准
        if updated_at is None:
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

        # 尚未超时则保留
        if updated_at >= cutoff:
            continue

        # 触发清理
        try:
            # 优先从主检出库通知 Git 移除 worktree 引用
            main_checkout = next((item for item in _checkout_root().glob(f"{repo.id}-*") if item.is_dir()), None)
            if main_checkout:
                _run_git(["worktree", "remove", "--force", str(path)], cwd=main_checkout)
            elif path.exists():
                shutil.rmtree(path)
        except Exception:
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)

        removed_paths.append(str(path))
        if snapshot_pr_id:
            removed_pr_ids.add(snapshot_pr_id)

    # 级联物理清理数据库中该 PR 专属的代码符号及关系
    if removed_pr_ids:
        db.query(CodeRelation).filter(
            CodeRelation.repo_id == repo.id,
            CodeRelation.pr_id.in_(removed_pr_ids),
        ).delete(synchronize_session=False)
        db.query(CodeSymbol).filter(
            CodeSymbol.repo_id == repo.id,
            CodeSymbol.pr_id.in_(removed_pr_ids),
        ).delete(synchronize_session=False)

    db.commit()
    return {"removed": len(removed_paths), "paths": removed_paths}

