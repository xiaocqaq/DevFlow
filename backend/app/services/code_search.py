# ==============================================================================
# 模块名称: 代码检索与工作区搜索服务 (Code Search & Workspace Scanner)
# 架构定位: 检索层 (Retrieval Layer) / 基础设施支持
# 核心技术与设计原理:
#   1. 路径穿越安全防护 (Path Traversal Protection):
#      利用 `Path.resolve().relative_to(checkout)` 强制锁定工作区边界，
#      严格过滤系统敏感文件 (`.env`, `.npmrc`, `.pypirc`) 与大型二进制/缓存目录 (`node_modules`, `.venv`)。
#   2. 混合检索与平滑降级 (Hybrid Engine with Fallback):
#      - 优先引擎: 调用极速命令行工具 `ripgrep` (`rg --json`)，支持毫秒级全文正则匹配。
#      - 降级引擎: 若未安装 `rg` 或执行超时，自动回退到纯 Python 多编码回退扫描流。
#   3. 专为代码优化的特征分词与相关度打分:
#      - 正则精准识别驼峰命名、作用域标识符 (`ClassName.method_name`)。
#      - 过滤无效通用停用词，综合行内命中率、路径加权与词频进行归一化打分。
# ==============================================================================

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.db.models import Repository

# 默认跳过的非源码目录（包含版本控制、虚拟环境、前端产物、编译缓存）
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

# 敏感配置文件白名单拦截（防止 AI 检索泄露密钥与私有凭据）
SECRET_FILENAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc"}

# 支持文本全文检索的代码扩展名集合 (Set 具备 O(1) 查找性能)
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

# 常见无后缀文本及配置文件
TEXT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "README",
    "LICENSE",
    "package.json",
    "requirements.txt",
    "pyproject.toml",
}

# 正则表达式：提取代码标识符、数字常量或中文字词
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_.$:/\\-]*|[0-9]{3,}|[\u4e00-\u9fff]{2,}")

# 常见泛化停用词（检索时过滤以防止引入过多低质噪音）
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
    """
    【路径解析】获取仓库在本地磁盘的物理检出根目录。

    优先级策略:
      1. 若实体指定了 `local_path`（本地直连模式），展开用户目录 (`~`) 并解析为绝对路径。
      2. 否则回退到系统托管模式的专用下载目录 `_repo_checkout_path(repo)`。

    Args:
        repo (Repository): 仓库数据模型对象。

    Returns:
        Path: 解析完成的本地绝对路径。
    """
    raw_path = str(getattr(repo, "local_path", None) or "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    from app.services.code_analysis import _repo_checkout_path

    return _repo_checkout_path(repo).resolve()


def resolve_search_root(checkout_path: str | Path, relative_path: str | None = None) -> tuple[Path, Path]:
    """
    【安全沙箱校验】严格校验并限制代码搜索在合法的仓库工作目录内部。

    安全设计要点:
      - 防止路径穿越攻击 (Path Traversal): 如通过 `../../` 试图读取服务器 `/etc/passwd` 或系统文件。
      - `root.relative_to(checkout)`: 若计算相对路径抛出 `ValueError`，说明搜索目标越界脱离工作区。
      - 拦截被跳过的忽略目录（如 `node_modules`）和敏感机密文件（如 `.env`）。

    Args:
        checkout_path (str | Path): 仓库物理工作区根目录。
        relative_path (str | None): 用户请求指定的子目录或相对文件路径。

    Returns:
        tuple[Path, Path]: (checkout_root, target_search_root)

    Raises:
        ValueError: 路径非法、越界或文件不存在。
    """
    checkout = Path(checkout_path).expanduser().resolve()
    if not checkout.exists() or not checkout.is_dir():
        raise ValueError("仓库代码还没有同步到本地。")

    # 规范化相对路径
    requested = str(relative_path or ".").strip()
    requested = "." if requested in {"", ".", "./", "/", "\\"} else requested.lstrip("/\\")
    root = (checkout / requested).resolve()

    # 校验安全边界
    try:
        root.relative_to(checkout)
    except ValueError as exc:
        raise ValueError("搜索路径位于仓库工作区之外。") from exc

    # 校验是否处于排除目录或敏感文件
    if any(part in SKIP_DIRS for part in root.relative_to(checkout).parts):
        raise ValueError("搜索路径位于被跳过的目录中。")
    if root.name in SECRET_FILENAMES:
        raise ValueError("不会搜索敏感配置文件。")
    if not root.exists():
        raise ValueError("搜索路径不存在。")
    return checkout, root


def query_terms(query: str, limit: int = 16) -> list[str]:
    """
    【智能代码分词】从用户的自然语言或代码查询中提炼高价值的搜索关键词。

    分词与排序算法:
      1. 使用正则 `_QUERY_TOKEN_RE` 提取可能的 Token。
      2. 识别代码特征词: 包含特殊符号（如 `.`, `_`, `$`）或驼峰命名（同时包含大写和小写）的词优先级提升。
      3. 排除停用词 (`_STOP_TERMS`) 与超短词（字符数 < 2）。
      4. 降序排序依据: 是否具有代码特征 (-item[0])、词长倒序 (-item[1])，保证长关键词和标识符排在最前。

    Args:
        query (str): 用户输入的搜索字符串。
        limit (int): 返回的最大关键词数量，默认 16。

    Returns:
        list[str]: 过滤与排序后的关键词列表。
    """
    candidates: list[tuple[int, int, str]] = []
    fallback_candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()

    for position, match in enumerate(_QUERY_TOKEN_RE.finditer(query or "")):
        value = match.group(0).strip()
        normalized = value.lower()
        if len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)

        # 启发式判断是否像代码标识符（包含符号或包含驼峰）
        code_like = bool(
            re.search(r"[_.$:/\\\-0-9]", value)
            or (any(char.islower() for char in value) and any(char.isupper() for char in value))
        )
        candidate = (1 if code_like else 0, len(value), normalized)
        fallback_candidates.append(candidate)
        if normalized not in _STOP_TERMS:
            candidates.append(candidate)

    # 若过滤停用词后为空，降级采用备选列表
    candidates = candidates or fallback_candidates
    # 按优先级排序：代码特征(优先) -> 词长(优先) -> 字典序
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[2] for item in candidates[: max(1, limit)]]


def _is_text_file(path: Path) -> bool:
    """
    【文件类型嗅探】通过后缀与文件名快速判断是否为可分析的文本源码文件。
    """
    if path.name in SECRET_FILENAMES or path.suffix.lower() in {".log", ".db", ".sqlite", ".sqlite3"}:
        return False
    return path.name in TEXT_FILENAMES or path.suffix.lower() in TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _read_text(path: Path, max_bytes: int = 180_000) -> str | None:
    """
    【健壮文件读取】带大小保护与多编码自动回退的文本读取器。

    防护措施:
      - 大小限制: 超过 `max_bytes`（默认 180KB）直接跳过，防止大日志拖慢扫描。
      - 二进制探测: 前 4096 字节包含 `\\x00`（空字节）则判定为二进制文件直接拒绝。
      - 多编码解码顺序: `utf-8-sig` -> `utf-8` -> `gb18030` -> `latin-1`。
    """
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
    """
    【代码片段提取器】安全读取指定文件特定行号区间的代码上下文（附带行号前缀）。

    用于智能体在定位到匹配行后，抓取其上下文供大模型阅读。

    Args:
        checkout_path (str | Path): 检出目录。
        relative_path (str): 相对文件路径。
        start_line (int): 起始行号（1-indexed）。
        line_count (int): 提取行数，默认 7 行。

    Returns:
        str: 格式化后的代码切片文本，首行为 `filepath:start-end`。
    """
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
    """
    【相关度量化打分】综合计算代码行与文件路径与搜索词的匹配度得分。

    打分权重公式:
      - `matched * 4`: 命中的不同关键词种类数（鼓励多词同时命中）。
      - `occurrences`: 行内词频命中（最高封顶 4 次，防词频作弊）。
      - `path_hits * 2`: 路径命中关键词（如搜索 `router` 命中 `routes/repo.py` 加权）。
    """
    path_text = relative_path.lower()
    line_text = line.lower()
    matched = sum(1 for term in terms if term in line_text or term in path_text)
    occurrences = sum(min(line_text.count(term), 4) for term in terms)
    path_hits = sum(path_text.count(term) for term in terms)
    return float(matched * 4 + occurrences + path_hits * 2)


def _ripgrep_matches(checkout: Path, root: Path, terms: list[str], max_matches: int) -> list[tuple[float, str, int, str]]:
    """
    【高性能引擎】调用 ripgrep CLI 获取结构化 JSON 格式搜索结果。

    核心参数解析:
      - `--json`: 输出 JSON 行协议（按事件拆解包含 match/begin/end）。
      - `--max-columns 500`: 限制超长压缩行，防止单行爆炸。
      - `--glob !**/.env*`: 外部命令层前置拦截过滤敏感文件。
    """
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
    # rg 正常返回: 0=找到结果, 1=未找到结果；其他值代表异常
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
        # 每一个文件只保留最高分的一行匹配，增加结果多样性
        if current is None or candidate[0] > current[0]:
            best_by_file[relative] = candidate
        if len(best_by_file) >= max_matches:
            break
    return list(best_by_file.values())


def _python_matches(checkout: Path, root: Path, terms: list[str], max_files: int = 1000) -> list[tuple[float, str, int, str]]:
    """
    【回退引擎】纯 Python 实现的文件树递归搜索（在无 ripgrep 的环境下自动生效）。
    """
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
    """
    【高阶搜索入口】在本地检出目录中执行混合式代码检索并组装为标准结果。

    执行流程:
      1. 提取关键词，若无有效词直接返回空列表。
      2. 校验搜索根目录与沙箱边界。
      3. 检查系统环境是否存在 `rg` (ripgrep):
         - 若存在，优先调用 `_ripgrep_matches`。
         - 若发生超时或错误，自动平滑捕获异常并降级到 `_python_matches`。
      4. 按相关度得分降序排序，归一化得分 (score / max_score)。
      5. 组装为前端与智能体检索协议通用的标准结果字典。
    """
    terms = query_terms(query)
    if not terms:
        return []
    checkout, root = resolve_search_root(checkout_path, relative_path)
    capped_limit = min(max(int(limit), 1), 50)
    search_tool = "python_scan"

    # 优先 ripgrep 极速引擎，异常平滑降级
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

    # 归一化得分
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
    """
    【业务级统一接口】基于 Repository ORM 实体检索代码，自动解析其检出目录并捕获越界异常。
    """
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

