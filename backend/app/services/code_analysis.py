"""代码仓库本地检出与代码语义图谱（Code Graph）分析服务模块.

本模块负责：
1. 本地代码仓库检出与增量同步（Git Clone / Fetch / Pull）；
2. 源代码符号提取（Code Symbols，识别 Python/TypeScript 等函数、类、接口）；
3. 符号关联分析（Code Relations，识别模块导入 imports 与函数调用 calls）；
4. 基于 `asyncio.to_thread` 在事件循环中非阻塞执行耗时本地 I/O 与 Git 子进程任务。
"""

import asyncio
import logging
import os
import shutil
import subprocess
import uuid
import re
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import CodeRelation, CodeSymbol, Document, Repository
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

# 代码遍历时需要跳过的依赖目录、构建产物与版本控制缓存
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# 支持进行文本解析的源代码与配置文件后缀集合
TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".java",
    ".go",
    ".rs",
    ".cs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".md",
    ".mdx",
    ".txt",
    ".dockerfile",
}

TEXT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "README",
    "LICENSE",
    "package.json",
    "requirements.txt",
    "pyproject.toml",
}

# 【Python 语法要点 - 正则表达式多行匹配】:
# `re.MULTILINE` 使得 `^` 和 `$` 能够匹配每行的开头和结尾，而非仅仅匹配整个字符串的最开头和最结尾
PY_SYMBOL_RE = re.compile(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
TS_SYMBOL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const|let)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)
CALL_RE = re.compile(r"\b([A-Za-z_][$A-Za-z0-9_]*)\s*\(")
IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.\/-]+)\s+import|import\s+([\w.\/-]+)|import\s+.*?\s+from\s+['\"]([^'\"]+)['\"])", re.MULTILINE)


def _checkout_root() -> Path:
    """获取本地代码托管根目录并确保目录存在."""
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo_checkout_path(repo: Repository) -> Path:
    """计算单个代码库在本地磁盘上的绝对检出路径."""
    # 替换路径中的斜杠，防止形成非法文件名
    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    return _checkout_root() / f"{repo.id}-{safe_name}"


def _stored_checkout_path(repo: Repository) -> Path | None:
    """解析数据库中保存的用户自定义本地代码路径（展开环境变量与波浪号用户主目录）."""
    raw_path = getattr(repo, "local_path", None)
    if not raw_path:
        return None
    # 展开环境变量（如 $HOME 或 %USERPROFILE%）和用户目录（~）
    path = Path(os.path.expandvars(os.path.expanduser(str(raw_path))))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _clone_url(repo: Repository) -> str:
    """获取代码库克隆 URL."""
    if repo.clone_url:
        return repo.clone_url
    return f"https://github.com/{repo.owner}/{repo.name}.git"


def _clone_url_with_token(url: str, token: str | None) -> str:
    """在 Git 克隆 URL 中安全注入访问 Token.

    【Python 语法要点 - URL 解析与重构】:
    - 使用 `urllib.parse.urlparse` 将 URL 拆解为各组成部分；
    - `quote(token, safe="")` 将 Token 中的特殊符号进行百分号转义（URL Encoding）；
    - `parsed._replace(netloc=...)` 利用 NamedTuple 的 `_replace` 方法构造新的 netloc（包含用户名密码凭据）；
    - `urlunparse` 将各部分重新组装为合法 URL。
    """
    if not token:
        return url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or "@" in parsed.netloc:
        return url
    safe_token = quote(token, safe="")
    return urlunparse(parsed._replace(netloc=f"x-access-token:{safe_token}@{parsed.netloc}"))


def _run_git(args: list[str], cwd: Path | None = None) -> None:
    """执行 Git 命令子进程并验证退出码.

    【Python 语法要点 - subprocess.run】:
    - `check=True`: 当子进程退出码非 0 时自动抛出 `subprocess.CalledProcessError`；
    - `stdout=subprocess.DEVNULL`: 丢弃标准输出，只保留错误输出 `stderr`，减少内存占用；
    - `timeout=180`: 超时保护，防止长连接挂死阻塞工作流。
    """
    subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )


def _is_git_work_tree(path: Path) -> bool:
    """检测指定路径是否为一个合法的 Git 工作树目录."""
    if not path.exists() or not path.is_dir() or not shutil.which("git"):
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(path),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=30,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _sync_checkout(repo: Repository, token: str | None) -> Path:
    """执行仓库代码的克隆（Clone）或快速前向拉取（Pull ff-only）.

    【Git 与工程优化要点】:
    - 浅克隆（Shallow Clone，`--depth 1`）：仅下载最新单一提交历史，大幅降低网络流量与耗时；
    - 本地模式与托管模式分支：若用户接入的是本地既有目录则直接复用；若为远程仓库则统一放入缓存路径。
    """
    stored_path = _stored_checkout_path(repo)
    checkout_mode = getattr(repo, "checkout_mode", None) or "managed"

    # 1. 本地既有仓库模式
    if stored_path and checkout_mode == "local":
        if not _is_git_work_tree(stored_path):
            raise RuntimeError(f"已配置的本地仓库路径不是 git 工作树：{stored_path}")
        return stored_path

    # 2. 系统托管检出目录模式
    target = stored_path or _repo_checkout_path(repo)
    clone_url = _clone_url_with_token(_clone_url(repo), token)
    branch = repo.default_branch or "main"

    if not shutil.which("git"):
        raise RuntimeError("未找到 git 可执行文件，无法克隆仓库代码")

    # 若目标已存在 .git，则尝试 fetch 最新提交
    if (target / ".git").exists() or _is_git_work_tree(target):
        try:
            _run_git(["fetch", "--depth", "1", "origin", branch], cwd=target)
            _run_git(["checkout", branch], cwd=target)
            _run_git(["pull", "--ff-only", "origin", branch], cwd=target)
        except subprocess.CalledProcessError as exc:
            logger.warning("更新现有 checkout %s 失败，将使用缓存文件：%s", target, (exc.stderr or exc).strip())
        return target

    # 若目标不存在，清理后执行全新浅克隆
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if stored_path and any(target.iterdir()):
            raise RuntimeError(f"Configured download target already exists and is not an empty git work tree: {target}")
        shutil.rmtree(target)
    clone_args = ["clone", "--depth", "1"]
    if branch:
        clone_args.extend(["--branch", branch])
    clone_args.extend([clone_url, str(target)])
    _run_git(clone_args)
    return target


def _is_text_candidate(path: Path) -> bool:
    """根据文件名与后缀名判定文件是否可能为纯文本源代码."""
    if path.name in TEXT_FILENAMES:
        return True
    suffix = path.suffix.lower()
    return suffix in TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _read_text(path: Path, max_bytes: int = 300_000) -> str | None:
    """安全读取文件文本并智能尝试多种字符集编码.

    【Python 语法与文件读取要点】:
    1. 限制最大读取字节数 `max_bytes`，防止大文件吃爆内存；
    2. 二进制探测：检测前 4096 字节是否包含空字节 `b"\x00"`，包含则判定为编译后二进制文件并跳过；
    3. 编码回退列表：依次尝试 `utf-8-sig`（带 BOM）、`utf-8`、`gb18030`（中文国标码）与 `latin-1`。
    """
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        return None
    if not data or len(data) > max_bytes or b"\x00" in data[:4096]:
        return None
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _language_for(path: Path) -> str:
    """根据文件后缀推导编程语言名称."""
    suffix = path.suffix.lower().lstrip(".")
    return suffix or path.name


def _git_text(args: list[str], cwd: Path) -> str | None:
    """执行 Git 查询并返回单行输出字符串."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _line_for_offset(text: str, offset: int) -> int:
    """根据字符偏移量计算其在文本中的 1-indexed 行号.

    【Python 语法要点】:
    `text.count("\n", 0, offset) + 1`：统计从头到当前偏移量中换行符的个数，直接转换为行号。
    """
    return text.count("\n", 0, offset) + 1


def _symbol_matches(path: Path, text: str) -> list[dict[str, object]]:
    """使用多行正则表达式抽取文件中的核心符号（函数、类、接口等）."""
    language = _language_for(path)
    matches = []
    if path.suffix.lower() == ".py":
        for match in PY_SYMBOL_RE.finditer(text):
            matches.append({"name": match.group(2), "kind": match.group(1), "language": language, "start_line": _line_for_offset(text, match.start())})
    elif path.suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}:
        for match in TS_SYMBOL_RE.finditer(text):
            declaration = match.group(0)
            kind = "class" if "class " in declaration else "interface" if "interface " in declaration else "type" if "type " in declaration else "function"
            matches.append({"name": match.group(1), "kind": kind, "language": language, "start_line": _line_for_offset(text, match.start())})
    return matches


def _sync_code_graph(
    db: Session,
    repo: Repository,
    checkout_path: Path,
    *,
    branch: str | None,
    commit_sha: str | None,
    pr_id: uuid.UUID | None = None,
    pr_number: int | None = None,
    max_files: int = 500,
) -> int:
    """扫描检出目录，批量提取并构建代码符号与拓扑调用图（Code Graph）.

    【Python / SQLAlchemy 性能优化要点】:
    1. `delete(synchronize_session=False)`:
       执行批量清理时跳过 ORM Session 内存对象同步，直接下发 SQL DELETE，提升海量数据清理速度；
    2. `Path.rglob("*")`: 递归遍历所有下级文件与子目录；
    3. `path.relative_to(checkout_path).as_posix()`: 统一转为跨平台规范的 Unix 风格正斜杠相对路径。
    """
    # 1. 清理当前分支下的旧符号与旧关系数据
    db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id, CodeRelation.branch == branch, CodeRelation.pr_id == pr_id, CodeRelation.pr_number == pr_number).delete(synchronize_session=False)
    db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id, CodeSymbol.branch == branch, CodeSymbol.pr_id == pr_id, CodeSymbol.pr_number == pr_number).delete(synchronize_session=False)
    
    symbol_rows: list[CodeSymbol] = []
    files_seen = 0

    # 2. 遍历文件抽取符号 (Symbols)
    for path in checkout_path.rglob("*"):
        if files_seen >= max_files:
            break
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(checkout_path).parts):
            continue
        if not _is_text_candidate(path):
            continue
        text = _read_text(path, max_bytes=180_000)
        if not text:
            continue
        symbols = _symbol_matches(path, text)
        if not symbols:
            continue
        relative_path = path.relative_to(checkout_path).as_posix()
        files_seen += 1
        for item in symbols:
            row = CodeSymbol(
                repo_id=repo.id,
                path=relative_path,
                name=str(item["name"]),
                kind=str(item["kind"]),
                language=str(item["language"]),
                start_line=int(item["start_line"]),
                end_line=None,
                branch=branch,
                commit_sha=commit_sha,
                pr_id=pr_id,
                pr_number=pr_number,
                meta={},
            )
            db.add(row)
            symbol_rows.append(row)
    
    db.flush()
    # 建立符号名 -> ORM 实体快速索引字典
    by_name = {row.name: row for row in symbol_rows}

    # 3. 抽取依赖关系 (Relations: imports & calls)
    for path in checkout_path.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(checkout_path).parts) or not _is_text_candidate(path):
            continue
        text = _read_text(path, max_bytes=180_000)
        if not text:
            continue
        relative_path = path.relative_to(checkout_path).as_posix()
        
        # 抽取 imports 引用
        import_targets = [next((part for part in match.groups() if part), "") for match in IMPORT_RE.finditer(text)]
        for target in [item for item in import_targets if item]:
            db.add(
                CodeRelation(
                    repo_id=repo.id,
                    source_name=relative_path,
                    target_name=target,
                    relation_type="imports",
                    path=relative_path,
                    branch=branch,
                    pr_id=pr_id,
                    pr_number=pr_number,
                    meta={},
                )
            )
        
        # 抽取 calls 函数调用
        local_symbols = [row for row in symbol_rows if row.path == relative_path]
        source = local_symbols[0] if local_symbols else None
        seen_calls: set[str] = set()
        for match in CALL_RE.finditer(text):
            target_name = match.group(1)
            # 过滤关键字和内置函数
            if target_name in seen_calls or target_name in {"if", "for", "while", "return", "print", "len", "str", "int"}:
                continue
            seen_calls.add(target_name)
            target = by_name.get(target_name)
            db.add(
                CodeRelation(
                    repo_id=repo.id,
                    source_symbol_id=source.id if source else None,
                    target_symbol_id=target.id if target else None,
                    source_name=source.name if source else relative_path,
                    target_name=target_name,
                    relation_type="calls",
                    path=relative_path,
                    branch=branch,
                    pr_id=pr_id,
                    pr_number=pr_number,
                    meta={"line": _line_for_offset(text, match.start())},
                )
            )
    return len(symbol_rows)


def purge_legacy_code_documents(db: Session, repo_id: uuid.UUID | None = None) -> int:
    """清理遗留的基于文本直接索引的旧文档切片."""
    query = db.query(Document).filter(Document.source_type == "code_file")
    if repo_id is not None:
        query = query.filter(Document.repo_id == repo_id)
    return query.delete(synchronize_session=False)


async def sync_repository_code_analysis(db: Session, repo: Repository) -> int:
    """代码库分析主调度流程.

    【Python 语法要点 - asyncio.to_thread】:
    将耗时的本地文件读写与 Git 进程操作通过 `await asyncio.to_thread(...)` 派发到线程池执行，
    避免阻塞 FastAPI 主事件循环。
    """
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    checkout_path = await asyncio.to_thread(_sync_checkout, repo, token)
    branch = repo.default_branch or _git_text(["branch", "--show-current"], checkout_path) or "main"
    commit_sha = _git_text(["rev-parse", "HEAD"], checkout_path)
    symbol_count = await asyncio.to_thread(
        _sync_code_graph,
        db,
        repo,
        checkout_path,
        branch=branch,
        commit_sha=commit_sha,
    )
    purge_legacy_code_documents(db, repo.id)
    repo.clone_url = repo.clone_url or _clone_url(repo)
    db.commit()
    return symbol_count


async def sync_repository_code_analysis_by_id(repo_id: uuid.UUID) -> int | None:
    """根据仓库 UUID 独立运行代码分析（专供后台异步任务 background_tasks 使用）."""
    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return None
        try:
            return await sync_repository_code_analysis(db, repo)
        except Exception as exc:
            logger.warning("Failed to sync repository code analysis for %s: %s", repo_id, exc)
            return None
    finally:
        db.close()
