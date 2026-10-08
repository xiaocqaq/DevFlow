from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import CodeRelation, CodeSymbol, PullRequest, Repository, User
from app.db.session import get_db
from app.schemas.code_graph import CodeGraphSearchResponse, CodeSymbolResponse
from app.services.code_analysis import sync_repository_code_analysis
from app.services.code_search import read_code_excerpt, repository_checkout_path
from app.services.permissions import require_permission
from app.services.worktree_manager import pr_worktree_path

router = APIRouter()


@router.post("/repos/{repo_id}/code-graph/rebuild")
async def rebuild_code_graph(
    repo_id: UUID,
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    symbol_count = await sync_repository_code_analysis(db, repo)
    return {"repo_id": str(repo.id), "indexed_symbols": symbol_count}


@router.get("/repos/{repo_id}/code-symbols", response_model=list[CodeSymbolResponse])
async def list_code_symbols(
    repo_id: UUID,
    path: str | None = None,
    pr_id: UUID | None = None,
    pr_number: int | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[CodeSymbol]:
    query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_id)
    if path:
        query = query.filter(CodeSymbol.path.contains(path))
    if pr_id:
        query = query.filter(CodeSymbol.pr_id == pr_id)
    if pr_number:
        query = query.filter(CodeSymbol.pr_number == pr_number)
    return query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(limit).all()


@router.get("/repos/{repo_id}/code-graph/search", response_model=CodeGraphSearchResponse)
async def search_code_graph(
    repo_id: UUID,
    q: str = Query(default=""),
    pr_id: UUID | None = None,
    pr_number: int | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> dict:
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    text = q.strip()
    query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo_id)
    if pr_id:
        query = query.filter(CodeSymbol.pr_id == pr_id)
    if pr_number:
        query = query.filter(CodeSymbol.pr_number == pr_number)
    if text:
        query = query.filter(or_(CodeSymbol.name.contains(text), CodeSymbol.path.contains(text)))
    symbols = query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(limit).all()
    symbol_ids = [item.id for item in symbols]
    relations = []
    if symbol_ids:
        rows = (
            db.query(CodeRelation)
            .filter(
                CodeRelation.repo_id == repo_id,
                CodeRelation.pr_id == pr_id if pr_id else True,
                CodeRelation.pr_number == pr_number if pr_number else True,
                or_(CodeRelation.source_symbol_id.in_(symbol_ids), CodeRelation.target_symbol_id.in_(symbol_ids), CodeRelation.target_name.in_([item.name for item in symbols])),
            )
            .limit(limit * 4)
            .all()
        )
        relations = [
            {
                "id": str(row.id),
                "source_symbol_id": str(row.source_symbol_id) if row.source_symbol_id else None,
                "target_symbol_id": str(row.target_symbol_id) if row.target_symbol_id else None,
                "source_name": row.source_name,
                "target_name": row.target_name,
                "relation_type": row.relation_type,
                "path": row.path,
                "branch": row.branch,
                "pr_id": str(row.pr_id) if row.pr_id else None,
                "pr_number": row.pr_number,
                "metadata": row.meta or {},
            }
            for row in rows
        ]
    documents = []
    checkout_path = repository_checkout_path(repo)
    selected_pr = None
    if pr_id:
        selected_pr = db.get(PullRequest, pr_id)
    elif pr_number:
        selected_pr = (
            db.query(PullRequest)
            .filter(PullRequest.repo_id == repo_id, PullRequest.number == pr_number)
            .one_or_none()
        )
    if selected_pr and selected_pr.repo_id == repo_id:
        snapshot_path = pr_worktree_path(repo, selected_pr)
        if snapshot_path.exists():
            checkout_path = snapshot_path
    if checkout_path.exists():
        for symbol in symbols[:limit]:
            snippet = read_code_excerpt(checkout_path, symbol.path, symbol.start_line, line_count=7)
            if not snippet:
                continue
            documents.append(
                {
                    "id": f"workspace:{symbol.path}:{symbol.start_line}",
                    "title": f"{symbol.path}:{symbol.start_line}",
                    "source_id": symbol.path,
                    "path": symbol.path,
                    "symbol": symbol.name,
                    "start_line": symbol.start_line,
                    "end_line": symbol.start_line + 6,
                    "pr_id": str(selected_pr.id) if selected_pr else None,
                    "pr_number": selected_pr.number if selected_pr else None,
                    "snippet": snippet[:1200],
                }
            )
    return {"query": text, "symbols": symbols, "relations": relations, "documents": documents}
