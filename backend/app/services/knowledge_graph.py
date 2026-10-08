import json
import re
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import (
    AgentRun,
    ConversationMemory,
    Document,
    Issue,
    KnowledgeGraphEdge,
    PRFile,
    PullRequest,
    Repository,
    ThreadMemory,
    WorkflowRun,
)

GRAPH_NODE_LIMIT = 72
ISSUE_REF_RE = re.compile(r"(?<![A-Za-z])#(\d+)\b")
CLOSING_ISSUE_RE = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+(?:issue\s+)?#(\d+)\b",
    re.IGNORECASE,
)
PR_REF_RE = re.compile(r"\b(?:PR|pull request)\s*#?(\d+)\b", re.IGNORECASE)

NODE_TYPE_ALIASES = {
    "pr": "pull_request",
    "pull_request": "pull_request",
    "workflow_run": "ci_run",
    "ci": "ci_run",
    "ci_run": "ci_run",
    "memory": "memory_note",
    "memory_note": "memory_note",
    "knowledge_file": "document",
    "weekly_report": "document",
    "project_doc": "document",
    "project_manifest": "document",
    "project_overview": "document",
    "document": "document",
    "code_file": "code_file",
    "team_member": "team_member",
    "issue": "issue",
    "session": "session_decision",
    "session_decision": "session_decision",
}

DOCUMENT_NODE_TYPES = {"knowledge_file", "weekly_report", "project_doc", "project_manifest", "project_overview"}


@dataclass
class DocumentGroup:
    source_type: str
    source_id: str
    title: str
    content: str
    metadata: dict[str, Any]
    created_at: Any = None


def normalize_node_type(value: str | None) -> str:
    key = (value or "").strip()
    return NODE_TYPE_ALIASES.get(key, key)


def rebuild_knowledge_graph(db: Session, repo: Repository) -> dict[str, int]:
    db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).delete(synchronize_session=False)
    edges = _generate_edges(db, repo.id)
    db.add_all(
        KnowledgeGraphEdge(
            repo_id=repo.id,
            from_type=edge["from_type"],
            from_id=edge["from_id"],
            to_type=edge["to_type"],
            to_id=edge["to_id"],
            relation=edge["relation"],
            confidence=edge["confidence"],
            source=edge["source"],
            meta=edge["metadata"],
        )
        for edge in edges
    )
    db.commit()
    nodes = _collect_nodes(db, repo.id)
    return {"edges": len(edges), "nodes": len(nodes)}


def knowledge_graph_payload(
    db: Session,
    repo: Repository,
    *,
    center_type: str | None = None,
    center_id: str | None = None,
    depth: int = 1,
) -> dict[str, Any]:
    edge_count = db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).count()
    if edge_count == 0:
        rebuild_knowledge_graph(db, repo)

    nodes_by_key = _collect_nodes(db, repo.id)
    edge_rows = db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).all()
    edges = [_edge_payload(row) for row in edge_rows]
    center_key = None
    resolved_center_type = normalize_node_type(center_type)
    if resolved_center_type and center_id:
        center_key = _node_key(resolved_center_type, center_id)
        nodes_by_key.setdefault(
            center_key,
            _fallback_node(resolved_center_type, center_id, highlighted=True),
        )

    selected_keys, truncated = _select_subgraph(nodes_by_key, edges, center_key, max(1, min(depth, 2)))
    selected_edges = [
        edge for edge in edges if _node_key(edge["from_type"], edge["from_id"]) in selected_keys and _node_key(edge["to_type"], edge["to_id"]) in selected_keys
    ]
    for edge in selected_edges:
        if edge["relation"] == "used_as_evidence":
            nodes_by_key[_node_key(edge["to_type"], edge["to_id"])]["highlighted"] = True

    return {
        "repo_id": repo.id,
        "center_type": resolved_center_type or None,
        "center_id": center_id,
        "depth": depth,
        "nodes": [nodes_by_key[key] for key in selected_keys if key in nodes_by_key],
        "edges": selected_edges,
        "stats": {
            "total_nodes": len(nodes_by_key),
            "total_edges": len(edges),
            "visible_nodes": len(selected_keys),
            "visible_edges": len(selected_edges),
            "truncated": truncated,
        },
    }


def _generate_edges(db: Session, repo_id: uuid.UUID) -> list[dict[str, Any]]:
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    prs = db.query(PullRequest).filter(PullRequest.repo_id == repo_id).all()
    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    pr_files = db.query(PRFile).join(PullRequest, PRFile.pr_id == PullRequest.id).filter(PullRequest.repo_id == repo_id).all()
    docs = _document_groups(db, repo_id)

    issues_by_number = {issue.number: issue for issue in issues}
    prs_by_number = {pr.number: pr for pr in prs}
    team_by_name = _team_members_by_name(docs)
    code_path_to_id = _code_path_to_node_id(pr_files)
    edges: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    def add_edge(
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        relation: str,
        *,
        confidence: float = 1.0,
        source: str = "explicit",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        from_type = normalize_node_type(from_type)
        to_type = normalize_node_type(to_type)
        if not from_id or not to_id or (from_type == to_type and from_id == to_id):
            return
        key = (from_type, str(from_id), to_type, str(to_id), relation)
        existing = edges.get(key)
        if existing:
            existing["confidence"] = max(existing["confidence"], confidence)
            existing["metadata"].update(metadata or {})
            return
        edges[key] = {
            "id": _edge_id(from_type, str(from_id), to_type, str(to_id), relation),
            "from_type": from_type,
            "from_id": str(from_id),
            "to_type": to_type,
            "to_id": str(to_id),
            "relation": relation,
            "confidence": confidence,
            "source": source,
            "metadata": metadata or {},
        }

    for issue in issues:
        for assignee in issue.assignees or []:
            member_id = team_by_name.get(str(assignee).strip().lower())
            if member_id:
                add_edge("issue", str(issue.id), "team_member", member_id, "assigned_to", metadata={"assignee": assignee})
        _add_text_reference_edges(
            add_edge,
            "issue",
            str(issue.id),
            f"{issue.title}\n{issue.body or ''}",
            issues_by_number,
            prs_by_number,
            code_path_to_id,
            relation_for_issue="mentions",
            relation_for_pr="mentions",
            relation_for_code="mentions",
        )

    for pr in prs:
        text = f"{pr.title}\n{pr.body or ''}\n{pr.head_branch or ''}\n{pr.base_branch or ''}"
        closing_refs = _closing_issue_refs(text)
        for number in closing_refs:
            issue = issues_by_number.get(number)
            if issue:
                add_edge("pull_request", str(pr.id), "issue", str(issue.id), "resolves", metadata={"issue_number": number})
        for number in _issue_refs(text) - closing_refs:
            issue = issues_by_number.get(number)
            if issue:
                add_edge("pull_request", str(pr.id), "issue", str(issue.id), "mentions", metadata={"issue_number": number})
        author_member = team_by_name.get(str(pr.author or "").strip().lower())
        if author_member:
            add_edge("pull_request", str(pr.id), "team_member", author_member, "mentions", metadata={"author": pr.author})

    for pr_file in pr_files:
        code_id = code_path_to_id.get(pr_file.filename) or _path_node_id(pr_file.filename)
        add_edge("pull_request", str(pr_file.pr_id), "code_file", code_id, "touches_file", metadata={"path": pr_file.filename, "status": pr_file.status})

    for run in runs:
        if (run.conclusion or "").lower() != "failure":
            continue
        text = f"{run.name}\n{run.logs_text or ''}"
        for number in _pr_refs(text):
            pr = prs_by_number.get(number)
            if pr:
                add_edge("ci_run", str(run.id), "pull_request", str(pr.id), "failed_in_ci", confidence=0.85, metadata={"pr_number": number})
        for number in _issue_refs(text):
            issue = issues_by_number.get(number)
            if issue:
                add_edge("ci_run", str(run.id), "issue", str(issue.id), "failed_in_ci", confidence=0.75, metadata={"issue_number": number})

    for doc in docs:
        if doc.source_type == "team_member":
            continue
        from_type = "memory_note" if doc.source_type == "memory_note" else "document"
        relation = "cites" if doc.source_type == "memory_note" else "mentions"
        text = f"{doc.title}\n{doc.content}\n{json.dumps(doc.metadata, ensure_ascii=False, default=str)}"
        _add_text_reference_edges(
            add_edge,
            from_type,
            doc.source_id,
            text,
            issues_by_number,
            prs_by_number,
            code_path_to_id,
            relation_for_issue=relation,
            relation_for_pr=relation,
            relation_for_code="cites" if doc.source_type == "memory_note" else "documents",
        )

    for decision in _decision_nodes(db, repo_id):
        text = str(decision["metadata"].get("content") or decision["title"])
        _add_text_reference_edges(
            add_edge,
            "session_decision",
            decision["id"],
            text,
            issues_by_number,
            prs_by_number,
            code_path_to_id,
            relation_for_issue="cites",
            relation_for_pr="cites",
            relation_for_code="cites",
        )
        for citation in decision["metadata"].get("citations") or []:
            target_type = normalize_node_type(str(citation.get("type") or citation.get("source_type") or ""))
            target_id = str(citation.get("id") or citation.get("source_id") or "")
            if target_type and target_id:
                add_edge("session_decision", decision["id"], target_type, target_id, "cites", confidence=0.9, source="memory_citation")

    for run in db.query(AgentRun).filter(AgentRun.repo_id == repo_id).order_by(AgentRun.created_at.desc()).limit(25):
        source_id = f"agent_run:{run.id}"
        for citation in _extract_citation_like_values(run.tool_calls or []):
            target_type = normalize_node_type(str(citation.get("type") or citation.get("source_type") or ""))
            target_id = str(citation.get("id") or citation.get("source_id") or "")
            if target_type and target_id:
                add_edge("session_decision", source_id, target_type, target_id, "used_as_evidence", confidence=0.75, source="tool_call_trace")

    return list(edges.values())


def _add_text_reference_edges(
    add_edge: Any,
    from_type: str,
    from_id: str,
    text: str,
    issues_by_number: dict[int, Issue],
    prs_by_number: dict[int, PullRequest],
    code_path_to_id: dict[str, str],
    *,
    relation_for_issue: str,
    relation_for_pr: str,
    relation_for_code: str,
) -> None:
    for number in _issue_refs(text):
        issue = issues_by_number.get(number)
        if issue:
            add_edge(from_type, from_id, "issue", str(issue.id), relation_for_issue, confidence=0.75, metadata={"issue_number": number})
    for number in _pr_refs(text):
        pr = prs_by_number.get(number)
        if pr:
            add_edge(from_type, from_id, "pull_request", str(pr.id), relation_for_pr, confidence=0.8, metadata={"pr_number": number})
    for path, code_id in code_path_to_id.items():
        if len(path) >= 6 and path in text:
            add_edge(from_type, from_id, "code_file", code_id, relation_for_code, confidence=0.8, metadata={"path": path})


def _collect_nodes(db: Session, repo_id: uuid.UUID) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}

    def put(node_type: str, node_id: str, title: str, subtitle: str = "", metadata: dict[str, Any] | None = None, highlighted: bool = False) -> None:
        normalized = normalize_node_type(node_type)
        key = _node_key(normalized, str(node_id))
        if key in nodes:
            nodes[key]["metadata"].update(metadata or {})
            nodes[key]["highlighted"] = nodes[key]["highlighted"] or highlighted
            return
        nodes[key] = {
            "id": str(node_id),
            "type": normalized,
            "title": title or str(node_id),
            "subtitle": subtitle or "",
            "metadata": metadata or {},
            "highlighted": highlighted,
        }

    for issue in db.query(Issue).filter(Issue.repo_id == repo_id).all():
        put("issue", str(issue.id), f"#{issue.number} {issue.title}", issue.state, {"number": issue.number, "labels": issue.labels or [], "assignees": issue.assignees or []})
    for pr in db.query(PullRequest).filter(PullRequest.repo_id == repo_id).all():
        subtitle = "merged" if pr.merged_at else pr.state
        put("pull_request", str(pr.id), f"#{pr.number} {pr.title}", subtitle, {"number": pr.number, "author": pr.author, "head_branch": pr.head_branch})
    for run in db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all():
        put("ci_run", str(run.id), run.name, run.conclusion or run.status, {"status": run.status, "conclusion": run.conclusion, "html_url": run.html_url})

    docs = _document_groups(db, repo_id)
    for doc in docs:
        if doc.source_type == "team_member":
            put("team_member", doc.source_id, doc.metadata.get("name") or doc.title, doc.metadata.get("role") or "", doc.metadata)
        elif doc.source_type == "memory_note":
            put("memory_note", doc.source_id, doc.title, "memory note", doc.metadata)
        elif doc.source_type in DOCUMENT_NODE_TYPES:
            put("document", doc.source_id, doc.metadata.get("display_title") or doc.title, doc.source_type, doc.metadata | {"source_type": doc.source_type})

    for pr_file in db.query(PRFile).join(PullRequest, PRFile.pr_id == PullRequest.id).filter(PullRequest.repo_id == repo_id).all():
        put("code_file", _path_node_id(pr_file.filename), pr_file.filename, "PR touched file", {"path": pr_file.filename, "status": pr_file.status})

    for decision in _decision_nodes(db, repo_id):
        put("session_decision", decision["id"], decision["title"], decision["subtitle"], decision["metadata"])

    for run in db.query(AgentRun).filter(AgentRun.repo_id == repo_id).order_by(AgentRun.created_at.desc()).limit(25):
        put("session_decision", f"agent_run:{run.id}", f"AI answer: {run.route}", "tool evidence path", {"route": run.route, "created_at": run.created_at.isoformat() if run.created_at else None})

    for edge in db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo_id).all():
        from_key = _node_key(edge.from_type, edge.from_id)
        to_key = _node_key(edge.to_type, edge.to_id)
        nodes.setdefault(from_key, _fallback_node(edge.from_type, edge.from_id))
        nodes.setdefault(to_key, _fallback_node(edge.to_type, edge.to_id))

    return nodes


def _select_subgraph(
    nodes_by_key: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    center_key: str | None,
    depth: int,
) -> tuple[list[str], bool]:
    if center_key and center_key in nodes_by_key:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            a = _node_key(edge["from_type"], edge["from_id"])
            b = _node_key(edge["to_type"], edge["to_id"])
            adjacency[a].add(b)
            adjacency[b].add(a)
        selected: list[str] = []
        seen = {center_key}
        queue: deque[tuple[str, int]] = deque([(center_key, 0)])
        while queue and len(selected) < GRAPH_NODE_LIMIT:
            key, distance = queue.popleft()
            selected.append(key)
            if distance >= depth:
                continue
            for nxt in sorted(adjacency.get(key, []), key=lambda item: nodes_by_key.get(item, {}).get("title", item)):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, distance + 1))
        return selected, bool(queue)

    touched: set[str] = set()
    for edge in edges:
        touched.add(_node_key(edge["from_type"], edge["from_id"]))
        touched.add(_node_key(edge["to_type"], edge["to_id"]))
    ranked = sorted(
        nodes_by_key,
        key=lambda key: (0 if key in touched else 1, _node_priority(nodes_by_key[key]["type"]), nodes_by_key[key]["title"]),
    )
    selected = ranked[:GRAPH_NODE_LIMIT]
    return selected, len(ranked) > GRAPH_NODE_LIMIT


def _document_groups(db: Session, repo_id: uuid.UUID) -> list[DocumentGroup]:
    groups: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for doc in db.query(Document).filter(Document.repo_id == repo_id).order_by(Document.created_at.asc()).all():
        groups[(doc.source_type, str(doc.source_id))].append(doc)

    output: list[DocumentGroup] = []
    for (source_type, source_id), docs in groups.items():
        docs.sort(key=lambda item: ((item.meta or {}).get("chunk_index", 0), item.created_at))
        first = docs[0]
        metadata = dict(first.meta or {})
        title = str(metadata.get("display_title") or first.title)
        content = "\n".join(doc.content or "" for doc in docs[:4])
        output.append(DocumentGroup(source_type, source_id, title, content, metadata, first.created_at))
    return output


def _decision_nodes(db: Session, repo_id: uuid.UUID) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    thread_memories = db.query(ThreadMemory).filter(ThreadMemory.repo_id == repo_id).all()
    conversation_memories = db.query(ConversationMemory).filter(ConversationMemory.repo_id == repo_id).all()
    for prefix, memory in [("thread", item) for item in thread_memories] + [("conversation", item) for item in conversation_memories]:
        for index, decision in enumerate(memory.decisions or []):
            text = str(decision).strip()
            if not text:
                continue
            output.append(
                {
                    "id": f"{prefix}:{memory.id}:decision:{index}",
                    "title": _clip(text, 72),
                    "subtitle": "session decision",
                    "metadata": {
                        "content": text,
                        "memory_id": str(memory.id),
                        "citations": memory.citations or [],
                        "sessions_incorporated": memory.sessions_incorporated,
                    },
                }
            )
    return output


def _team_members_by_name(docs: list[DocumentGroup]) -> dict[str, str]:
    output: dict[str, str] = {}
    for doc in docs:
        if doc.source_type != "team_member":
            continue
        for value in [doc.metadata.get("name"), doc.title]:
            text = str(value or "").strip().lower()
            if text:
                output[text] = doc.source_id
    return output


def _code_path_to_node_id(pr_files: list[PRFile]) -> dict[str, str]:
    output: dict[str, str] = {}
    for pr_file in pr_files:
        output.setdefault(pr_file.filename, _path_node_id(pr_file.filename))
    return output


def _extract_citation_like_values(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if ("source_type" in value or "type" in value) and ("source_id" in value or "id" in value):
            found.append(value)
        for child in value.values():
            found.extend(_extract_citation_like_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_extract_citation_like_values(child))
    return found


def _edge_payload(row: KnowledgeGraphEdge) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "from_type": row.from_type,
        "from_id": row.from_id,
        "to_type": row.to_type,
        "to_id": row.to_id,
        "relation": row.relation,
        "confidence": row.confidence,
        "source": row.source,
        "metadata": row.meta or {},
    }


def _fallback_node(node_type: str, node_id: str, highlighted: bool = False) -> dict[str, Any]:
    return {
        "id": str(node_id),
        "type": normalize_node_type(node_type),
        "title": str(node_id),
        "subtitle": "referenced item",
        "metadata": {},
        "highlighted": highlighted,
    }


def _node_key(node_type: str, node_id: str) -> str:
    return f"{normalize_node_type(node_type)}:{node_id}"


def _edge_id(from_type: str, from_id: str, to_type: str, to_id: str, relation: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"devflow-graph:{from_type}:{from_id}:{relation}:{to_type}:{to_id}"))


def _path_node_id(path: str) -> str:
    return f"path:{path}"


def _issue_refs(text: str) -> set[int]:
    return {int(match.group(1)) for match in ISSUE_REF_RE.finditer(text or "")}


def _closing_issue_refs(text: str) -> set[int]:
    return {int(match.group(1)) for match in CLOSING_ISSUE_RE.finditer(text or "")}


def _pr_refs(text: str) -> set[int]:
    return {int(match.group(1)) for match in PR_REF_RE.finditer(text or "")}


def _node_priority(node_type: str) -> int:
    order = {
        "issue": 0,
        "pull_request": 1,
        "ci_run": 2,
        "team_member": 3,
        "memory_note": 4,
        "session_decision": 5,
        "document": 6,
        "code_file": 7,
    }
    return order.get(node_type, 20)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."
