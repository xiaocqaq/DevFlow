import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.db.models import Repository


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

SECRET_FILENAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc"}

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

_QUERY_TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_.$:/\\-]*|[0-9]{3,}|[\u4e00-\u9fff]{2,}")
_STOP_TERMS = {
    "about",
    "after",
    "before",
    "code",
    "error",
    "failed",
    "failure",
    "file",
    "from",
    "into",
    "issue",
    "project",
    "pull",
    "request",
    "test",
    "tests",
    "that",
    "the",
    "this",
    "with",
    "代码",
    "问题",
    "失败",
    "项目",
    "相关",
    "这个",
}


def repository_checkout_path(repo: Repository) -> Path:
    raw_path = str(getattr(repo, "local_path", None) or "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    from app.services.code_analysis import _repo_checkout_path

    return _repo_checkout_path(repo).resolve()


def resolve_search_root(checkout_path: str | Path, relative_path: str | None = None) -> tuple[Path, Path]:
    checkout = Path(checkout_path).expanduser().resolve()
    if not checkout.exists() or not checkout.is_dir():
        raise ValueError("仓库代码还没有同步到本地。")

    requested = str(relative_path or ".").strip()
    requested = "." if requested in {"", ".", "./", "/", "\\"} else requested.lstrip("/\\")
    root = (checkout / requested).resolve()
    try:
        root.relative_to(checkout)
    except ValueError as exc:
        raise ValueError("搜索路径位于仓库工作区之外。") from exc
    if any(part in SKIP_DIRS for part in root.relative_to(checkout).parts):
        raise ValueError("搜索路径位于被跳过的目录中。")
    if root.name in SECRET_FILENAMES:
        raise ValueError("不会搜索敏感配置文件。")
    if not root.exists():
        raise ValueError("搜索路径不存在。")
    return checkout, root


def query_terms(query: str, limit: int = 16) -> list[str]:
    candidates: list[tuple[int, int, str]] = []
    fallback_candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for position, match in enumerate(_QUERY_TOKEN_RE.finditer(query or "")):
        value = match.group(0).strip()
        normalized = value.lower()
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        code_like = bool(
            re.search(r"[_.$:/\\\-0-9]", value)
            or (any(char.islower() for char in value) and any(char.isupper() for char in value))
        )
        candidate = (1 if code_like else 0, len(value), normalized)
        fallback_candidates.append(candidate)
        if normalized not in _STOP_TERMS:
            candidates.append(candidate)
    candidates = candidates or fallback_candidates
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[2] for item in candidates[: max(1, limit)]]


def _is_text_file(path: Path) -> bool:
    if path.name in SECRET_FILENAMES or path.suffix.lower() in {".log", ".db", ".sqlite", ".sqlite3"}:
        return False
    return path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _read_text(path: Path, max_bytes: int = 180_000) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data or len(data) > max_bytes or b"\x00" in data[:4096]:
        return None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def read_code_excerpt(checkout_path: str | Path, relative_path: str, start_line: int, line_count: int = 7) -> str:
    checkout, path = resolve_search_root(checkout_path, relative_path)
    if not path.is_file() or not _is_text_file(path):
        return ""
    text = _read_text(path)
    if text is None:
        return ""
    lines = text.splitlines()
    start = max(int(start_line), 1)
    end = min(start + max(int(line_count), 1) - 1, len(lines))
    if start > end:
        return ""
    relative = path.relative_to(checkout).as_posix()
    return f"{relative}:{start}-{end}\n" + "\n".join(
        f"{line_number}: {lines[line_number - 1]}" for line_number in range(start, end + 1)
    )


def _score_match(relative_path: str, line: str, terms: list[str]) -> float:
    path_text = relative_path.lower()
    line_text = line.lower()
    matched = sum(1 for term in terms if term in line_text or term in path_text)
    occurrences = sum(min(line_text.count(term), 4) for term in terms)
    path_hits = sum(path_text.count(term) for term in terms)
    return float(matched * 4 + occurrences + path_hits * 2)


def _ripgrep_matches(checkout: Path, root: Path, terms: list[str], max_matches: int) -> list[tuple[float, str, int, str]]:
    pattern = "|".join(re.escape(term) for term in terms)
    command = [
        "rg",
        "--json",
        "--ignore-case",
        "--line-number",
        "--max-columns",
        "500",
        "--glob",
        "!**/.env*",
        "--glob",
        "!**/.npmrc",
        "--glob",
        "!**/.pypirc",
        "--glob",
        "!**/*.log",
        "--glob",
        "!**/*.db",
        "--glob",
        "!**/*.sqlite*",
        pattern,
        str(root),
    ]
    result = subprocess.run(
        command,
        cwd=str(checkout),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "ripgrep 搜索失败")

    best_by_file: dict[str, tuple[float, str, int, str]] = {}
    for raw_line in result.stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data") or {}
        raw_path = str((data.get("path") or {}).get("text") or "")
        path = Path(raw_path)
        if not path.is_absolute():
            path = checkout / path
        try:
            relative = path.resolve().relative_to(checkout).as_posix()
        except ValueError:
            continue
        if any(part in SKIP_DIRS for part in Path(relative).parts) or Path(relative).name in SECRET_FILENAMES:
            continue
        line_number = int(data.get("line_number") or 1)
        line = str((data.get("lines") or {}).get("text") or "").strip()[:320]
        score = _score_match(relative, line, terms)
        candidate = (score, relative, line_number, line)
        current = best_by_file.get(relative)
        if current is None or candidate[0] > current[0]:
            best_by_file[relative] = candidate
        if len(best_by_file) >= max_matches:
            break
    return list(best_by_file.values())


def _python_matches(checkout: Path, root: Path, terms: list[str], max_files: int = 1000) -> list[tuple[float, str, int, str]]:
    paths = [root] if root.is_file() else root.rglob("*")
    matches: list[tuple[float, str, int, str]] = []
    files_seen = 0
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(checkout)
        if any(part in SKIP_DIRS for part in relative.parts) or not _is_text_file(path):
            continue
        files_seen += 1
        if files_seen > max_files:
            break
        text = _read_text(path)
        if text is None:
            continue
        relative_text = relative.as_posix()
        best: tuple[float, str, int, str] | None = None
        for line_number, line in enumerate(text.splitlines(), start=1):
            score = _score_match(relative_text, line, terms)
            if score <= 0:
                continue
            candidate = (score, relative_text, line_number, line.strip()[:320])
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best:
            matches.append(best)
    return matches


def search_checkout_code(
    checkout_path: str | Path,
    query: str,
    *,
    limit: int = 12,
    relative_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    terms = query_terms(query)
    if not terms:
        return []
    checkout, root = resolve_search_root(checkout_path, relative_path)
    capped_limit = min(max(int(limit), 1), 50)
    search_tool = "python_scan"
    if shutil.which("rg"):
        try:
            matches = _ripgrep_matches(checkout, root, terms, max_matches=max(capped_limit * 8, 80))
            search_tool = "ripgrep"
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            matches = _python_matches(checkout, root, terms)
    else:
        matches = _python_matches(checkout, root, terms)
    matches.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not matches:
        return []
    max_score = max(item[0] for item in matches) or 1.0
    output = []
    for score, path, line_number, line in matches[:capped_limit]:
        item_metadata = {
            "path": path,
            "line": line_number,
            "start_line": line_number,
            "end_line": line_number,
            "checkout_path": str(checkout),
            "retrieval_mode": "workspace",
            "search_tool": search_tool,
            **(metadata or {}),
        }
        output.append(
            {
                "id": f"workspace:{path}:{line_number}",
                "title": f"{path}:{line_number}",
                "source_type": "workspace_file",
                "source_id": path,
                "snippet": f"{path}:{line_number}: {line}",
                "score": score / max_score,
                "metadata": item_metadata,
            }
        )
    return output


def search_repository_code(
    repo: Repository,
    query: str,
    *,
    limit: int = 12,
    relative_path: str | None = None,
    checkout_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    root = Path(checkout_path).resolve() if checkout_path else repository_checkout_path(repo)
    if not root.exists():
        return []
    try:
        return search_checkout_code(
            root,
            query,
            limit=limit,
            relative_path=relative_path,
            metadata={"repo_id": str(repo.id), **(metadata or {})},
        )
    except ValueError:
        return []
