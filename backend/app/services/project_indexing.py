import fnmatch
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session

from app.db.models import ProjectIndexState, Repository
from app.db.session import SessionLocal
from app.services.knowledge_graph import rebuild_knowledge_graph
from app.services.rag.chunking import chunk_document
from app.services.code_analysis import _repo_checkout_path, _stored_checkout_path
from app.services.rag.vector_store import DocumentPayload, replace_documents, stable_source_id

PROJECT_INDEX_SOURCE_TYPES = ["project_overview", "project_doc", "project_manifest"]
PROJECT_INDEX_STATUSES = {"missing", "stale", "building", "ready", "failed"}
PROJECT_INDEX_CONFIG_PATH = Path(".devflow/index.yml")
PROJECT_INDEX_MODES = {"auto", "allowlist"}

AUTHORITATIVE_TOP_LEVEL = {"README", "ARCHITECTURE", "CONTRIBUTING"}
MANIFEST_FILES = {
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "poetry.lock",
    "pnpm-workspace.yaml",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
}
SOFT_CLUE_TOP_LEVEL = {"CHANGELOG", "HISTORY", "RELEASES"}
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
INDEXABLE_HIDDEN_DIRS = {".github"}
SECRET_NAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc", "credentials.json"}
SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


@dataclass(frozen=True)
class ProjectDocCandidate:
    path: Path
    relative_path: str
    source_type: str
    tier: str


@dataclass(frozen=True)
class ProjectIndexPolicy:
    mode: str = "auto"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    include_manifests: bool = True
    config_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "includeManifests": self.include_manifests,
            "configPath": self.config_path,
        }


@dataclass(frozen=True)
class ProjectDocDiscovery:
    candidates: list[ProjectDocCandidate]
    excluded_paths: list[str]
    policy: ProjectIndexPolicy
    truncated: bool = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _repo_root(repo: Repository) -> Path:
    stored = _stored_checkout_path(repo)
    if stored:
        return stored
    return _repo_checkout_path(repo)


def _safe_project_root(repo: Repository) -> Path | None:
    root = _repo_root(repo)
    if not root.exists() or not root.is_dir():
        return None
    try:
        return root.resolve()
    except OSError:
        return None


def _is_secret(path: Path) -> bool:
    name = path.name
    return name in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES or name.lower().startswith("credentials")


def _read_text(path: Path, max_bytes: int = 400_000) -> str | None:
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


def _stem_upper(path: Path) -> str:
    return path.stem.upper()


def _classify_file(path: Path, rel: str) -> tuple[str, str] | None:
    if _is_secret(path):
        return None
    name = path.name
    stem = _stem_upper(path)
    rel_posix = rel.replace("\\", "/")
    if path.suffix.lower() == ".md":
        if rel_posix.startswith("docs/") or stem in AUTHORITATIVE_TOP_LEVEL or stem.startswith("ADR"):
            return "project_doc", "authoritative"
        if stem in SOFT_CLUE_TOP_LEVEL or rel_posix.startswith(".github/ISSUE_TEMPLATE/"):
            return "project_doc", "soft_clue"
    if name in MANIFEST_FILES:
        return "project_manifest", "derived"
    return None


def _is_inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False


def _normalize_patterns(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of repository-relative glob patterns")
    patterns: list[str] = []
    for raw_pattern in value:
        if not isinstance(raw_pattern, str) or not raw_pattern.strip():
            raise ValueError(f"{field_name} entries must be non-empty strings")
        pattern = raw_pattern.strip().replace("\\", "/")
        while pattern.startswith("./"):
            pattern = pattern[2:]
        pattern = pattern.lstrip("/")
        if pattern == ".." or pattern.startswith("../"):
            raise ValueError(f"{field_name} patterns must stay inside the repository")
        if pattern.endswith("/"):
            pattern += "**"
        patterns.append(pattern)
    return tuple(dict.fromkeys(patterns))


def load_project_index_policy(root: Path) -> ProjectIndexPolicy:
    root = root.resolve()
    config_path = root / PROJECT_INDEX_CONFIG_PATH
    if not config_path.is_file():
        return ProjectIndexPolicy()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read {PROJECT_INDEX_CONFIG_PATH.as_posix()}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{PROJECT_INDEX_CONFIG_PATH.as_posix()} must contain a YAML object")
    version = raw.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported project index config version: {version}")
    project_docs = raw.get("project_docs") or {}
    if not isinstance(project_docs, dict):
        raise ValueError("project_docs must be a YAML object")
    mode = str(project_docs.get("mode") or "auto").strip().lower()
    if mode not in PROJECT_INDEX_MODES:
        raise ValueError(f"project_docs.mode must be one of: {', '.join(sorted(PROJECT_INDEX_MODES))}")
    include = _normalize_patterns(project_docs.get("include"), "project_docs.include")
    exclude = _normalize_patterns(project_docs.get("exclude"), "project_docs.exclude")
    if mode == "allowlist" and not include:
        raise ValueError("project_docs.include cannot be empty when mode is allowlist")
    include_manifests = raw.get("include_manifests", True)
    if not isinstance(include_manifests, bool):
        raise ValueError("include_manifests must be true or false")
    return ProjectIndexPolicy(
        mode=mode,
        include=include,
        exclude=exclude,
        include_manifests=include_manifests,
        config_path=PROJECT_INDEX_CONFIG_PATH.as_posix(),
    )


def _matches_any(relative_path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in patterns)


def _policy_allows(candidate: ProjectDocCandidate, policy: ProjectIndexPolicy) -> bool:
    if _matches_any(candidate.relative_path, policy.exclude):
        return False
    if candidate.source_type == "project_manifest":
        return policy.include_manifests
    if policy.mode == "allowlist":
        return _matches_any(candidate.relative_path, policy.include)
    return True


def _discover_project_docs(root: Path, max_files: int = 500) -> ProjectDocDiscovery:
    root = root.resolve()
    policy = load_project_index_policy(root)
    results: list[ProjectDocCandidate] = []
    excluded_paths: list[str] = []
    truncated = False
    for current_root, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(
            name
            for name in dir_names
            if name not in SKIP_DIRS and (not name.startswith(".") or name in INDEXABLE_HIDDEN_DIRS)
        )
        for filename in sorted(file_names):
            path = Path(current_root) / filename
            if not _is_inside(root, path):
                continue
            rel = path.relative_to(root).as_posix()
            classification = _classify_file(path, rel)
            if not classification:
                continue
            source_type, tier = classification
            candidate = ProjectDocCandidate(path=path, relative_path=rel, source_type=source_type, tier=tier)
            if not _policy_allows(candidate, policy):
                excluded_paths.append(rel)
                continue
            if len(results) >= max_files:
                truncated = True
                break
            results.append(candidate)
        if truncated:
            break
    return ProjectDocDiscovery(results, excluded_paths, policy, truncated)


def discover_project_docs(root: Path, max_files: int = 500) -> list[ProjectDocCandidate]:
    return _discover_project_docs(root, max_files=max_files).candidates


def project_fingerprint(repo: Repository) -> str:
    root = _safe_project_root(repo)
    if not root:
        return ""
    hasher = hashlib.sha256()
    discovery = _discover_project_docs(root, max_files=1000)
    hasher.update(json.dumps(discovery.policy.as_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8"))
    for candidate in discovery.candidates:
        try:
            stat = candidate.path.stat()
        except OSError:
            continue
        hasher.update(candidate.relative_path.encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("ascii"))
        hasher.update(str(stat.st_size).encode("ascii"))
    digest = hasher.hexdigest()
    return f"files:{digest}"


def _get_state_row(db: Session, repo_id: uuid.UUID) -> ProjectIndexState | None:
    return db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo_id).one_or_none()


def _get_or_create_state(db: Session, repo_id: uuid.UUID) -> ProjectIndexState:
    row = _get_state_row(db, repo_id)
    if row:
        return row
    row = ProjectIndexState(repo_id=repo_id, status="missing")
    db.add(row)
    db.flush()
    return row


def _state_payload(repo: Repository, row: ProjectIndexState | None, fingerprint: str | None = None) -> dict[str, Any]:
    current_fingerprint = fingerprint if fingerprint is not None else project_fingerprint(repo)
    if row is None:
        return {
            "repo_id": repo.id,
            "status": "missing",
            "fingerprint": current_fingerprint,
            "docs_indexed": 0,
            "docs_total": 0,
            "error_message": None,
            "summary_json": None,
            "snoozed_until": None,
            "last_scan_at": None,
        }
    status = row.status if row.status in PROJECT_INDEX_STATUSES else "missing"
    if status == "ready" and current_fingerprint and row.fingerprint != current_fingerprint:
        status = "stale"
    return {
        "repo_id": row.repo_id,
        "status": status,
        "fingerprint": row.fingerprint or current_fingerprint,
        "docs_indexed": row.docs_indexed or 0,
        "docs_total": row.docs_total or 0,
        "error_message": row.error_message,
        "summary_json": row.summary_json,
        "snoozed_until": row.snoozed_until,
        "last_scan_at": row.last_scan_at,
    }


def get_project_index_state(db: Session, repo: Repository) -> dict[str, Any]:
    return _state_payload(repo, _get_state_row(db, repo.id))


def mark_project_index_building(db: Session, repo: Repository) -> ProjectIndexState:
    row = _get_or_create_state(db, repo.id)
    row.status = "building"
    row.fingerprint = project_fingerprint(repo)
    row.error_message = None
    row.snoozed_until = None
    row.updated_at = _now()
    db.commit()
    return row


def snooze_project_index(db: Session, repo: Repository, days: int = 7) -> ProjectIndexState:
    row = _get_or_create_state(db, repo.id)
    row.snoozed_until = _now() + timedelta(days=days)
    row.updated_at = _now()
    if row.status not in {"ready", "building"}:
        row.status = "missing"
    db.commit()
    db.refresh(row)
    return row


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:500] or fallback
    return fallback


def _summary_from_text(text: str, limit: int = 360) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    paragraphs = []
    for block in cleaned.split("\n\n"):
        value = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if value and not value.startswith("#") and not value.startswith("|") and not value.startswith("```"):
            paragraphs.append(value)
    summary = paragraphs[0] if paragraphs else " ".join(cleaned.splitlines()[:4])
    return summary[: limit - 3].rstrip() + "..." if len(summary) > limit else summary


def _detect_tech_stack(root: Path) -> list[str]:
    detectors = {
        "package.json": "node",
        "tsconfig.json": "typescript",
        "pyproject.toml": "python",
        "requirements.txt": "python",
        "go.mod": "go",
        "Cargo.toml": "rust",
        "pom.xml": "java",
        "build.gradle": "java",
    }
    return [tech for filename, tech in detectors.items() if (root / filename).exists()]


def _safe_dirs(root: Path, limit: int = 12) -> list[str]:
    dirs: list[str] = []
    try:
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if len(dirs) >= limit:
                break
            if child.name.startswith(".") or child.name in SKIP_DIRS:
                continue
            if child.is_dir():
                dirs.append(child.name)
    except OSError:
        return []
    return dirs


def _build_summary(
    repo: Repository,
    root: Path,
    candidates: list[ProjectDocCandidate],
    docs_indexed: int,
    discovery: ProjectDocDiscovery,
) -> dict[str, Any]:
    tier_coverage: dict[str, int] = {}
    source_type_coverage: dict[str, int] = {}
    docs_list: list[dict[str, str]] = []
    for item in candidates:
        tier_coverage[item.tier] = tier_coverage.get(item.tier, 0) + 1
        source_type_coverage[item.source_type] = source_type_coverage.get(item.source_type, 0) + 1
        docs_list.append({"path": item.relative_path, "tier": item.tier, "source_type": item.source_type})
    return {
        "projectName": repo.full_name,
        "rootPath": str(root),
        "techStack": sorted(set(_detect_tech_stack(root))),
        "dirStructure": _safe_dirs(root),
        "docsList": docs_list[:80],
        "tierCoverage": tier_coverage,
        "sourceTypeCoverage": source_type_coverage,
        "docsIndexed": docs_indexed,
        "indexPolicy": discovery.policy.as_dict(),
        "excludedFiles": len(discovery.excluded_paths),
        "excludedPaths": discovery.excluded_paths[:80],
        "discoveryTruncated": discovery.truncated,
    }


def _payloads_for_candidates(repo: Repository, root: Path, candidates: list[ProjectDocCandidate], fingerprint: str) -> list[DocumentPayload]:
    payloads: list[DocumentPayload] = []
    scanned_at = _now().isoformat()
    for candidate in candidates:
        text = _read_text(candidate.path)
        if not text or not text.strip():
            continue
        source_id = stable_source_id(repo.id, candidate.source_type, candidate.relative_path)
        title = _extract_title(text, candidate.relative_path)
        chunks = chunk_document(text, candidate.relative_path, max_chars=1800, overlap=180)
        for index, chunk in enumerate(chunks):
            content = str(chunk["content"])
            section_title = chunk.get("section_title")
            payloads.append(
                DocumentPayload(
                    source_type=candidate.source_type,
                    source_id=source_id,
                    title=f"{title} · {section_title}" if section_title else (f"{title}#{index + 1}" if len(chunks) > 1 else title),
                    content=content,
                    metadata={
                        **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                        "display_title": title,
                        "path": candidate.relative_path,
                        "root_path": str(root),
                        "provenance_tier": candidate.tier,
                        "chunk_index": index,
                        "chunk_chars": len(content),
                        "total_chunks": len(chunks),
                        "chunk_strategy": "structure_aware_v1",
                        "fingerprint": fingerprint,
                        "scanned_at": scanned_at,
                    },
                )
            )
    return payloads


def scan_project_index(db: Session, repo: Repository) -> dict[str, Any]:
    row = _get_or_create_state(db, repo.id)
    fp = project_fingerprint(repo)
    row.status = "building"
    row.fingerprint = fp
    row.error_message = None
    row.snoozed_until = None
    row.updated_at = _now()
    db.commit()

    try:
        root = _safe_project_root(repo)
        if not root:
            raise RuntimeError("项目本地路径不存在，请先连接本地仓库或完成代码同步")

        discovery = _discover_project_docs(root)
        candidates = discovery.candidates
        payloads = _payloads_for_candidates(repo, root, candidates, fp)
        overview = {
            "repo": repo.full_name,
            "description": repo.description or "",
            "default_branch": repo.default_branch or "",
            "local_path": str(root),
            "indexed_files": [candidate.relative_path for candidate in candidates[:80]],
        }
        payloads.insert(
            0,
            DocumentPayload(
                source_type="project_overview",
                source_id=stable_source_id(repo.id, "project_overview", "overview"),
                title=f"项目概况：{repo.full_name}",
                content=json.dumps(overview, ensure_ascii=False, indent=2),
                metadata={"display_title": "项目概况", "fingerprint": fp, "root_path": str(root), "chunk_index": 0},
            ),
        )

        docs = replace_documents(
            db,
            repo.id,
            PROJECT_INDEX_SOURCE_TYPES,
            payloads,
            commit=False,
            generate_embeddings=False,
            sync_vector_store=False,
        )
        summary = _build_summary(repo, root, candidates, len(docs), discovery)
        row = _get_or_create_state(db, repo.id)
        row.status = "ready"
        row.fingerprint = fp
        row.docs_indexed = len(docs)
        row.docs_total = len(candidates)
        row.summary_json = summary
        row.last_scan_at = _now()
        row.error_message = None
        row.updated_at = _now()
        # EvidenceItems are a denormalized search surface used by ChatAgent's
        # memory tools. Refresh them in the same transaction so excluded
        # project documents cannot survive a successful rescan.
        from app.services.chat_memory import EvidenceStore

        EvidenceStore(db).rebuild_repo(repo)
        db.commit()
        db.refresh(row)
        rebuild_knowledge_graph(db, repo)
        return _state_payload(repo, row, fingerprint=fp)
    except Exception as exc:
        db.rollback()
        row = _get_or_create_state(db, repo.id)
        row.status = "failed"
        row.error_message = str(exc)
        row.updated_at = _now()
        db.commit()
        db.refresh(row)
        return _state_payload(repo, row, fingerprint=fp)


async def scan_project_index_by_id(repo_id: uuid.UUID) -> None:
    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return
        scan_project_index(db, repo)
    finally:
        db.close()


def is_snoozed(state: dict[str, Any]) -> bool:
    until = _aware(state.get("snoozed_until"))
    return bool(until and until > _now())
