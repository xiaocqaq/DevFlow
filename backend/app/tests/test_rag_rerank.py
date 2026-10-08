from app.services.rag.rerank import rerank_documents


def test_rerank_prefers_exact_path_and_query_terms() -> None:
    candidates = [
        {
            "id": "general",
            "title": "README setup",
            "snippet": "General local development notes",
            "score": 0.8,
            "vector_score": 0.8,
            "keyword_score": 0.1,
            "metadata": {},
        },
        {
            "id": "auth",
            "title": "backend/app/auth/middleware.py#1",
            "snippet": "validate token and refresh session when login expires",
            "score": 0.5,
            "vector_score": 0.5,
            "keyword_score": 1.0,
            "metadata": {"path": "backend/app/auth/middleware.py"},
        },
    ]

    results = rerank_documents("login token auth middleware", candidates, limit=2)

    assert results[0]["id"] == "auth"
    assert results[0]["rerank_score"] >= results[1]["rerank_score"]
    assert "rank_reason" in results[0]
