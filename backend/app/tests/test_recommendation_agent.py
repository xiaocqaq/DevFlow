import pytest

from app.services.agents.recommendation_agent import RecommendationAgent


@pytest.mark.asyncio
async def test_recommendation_agent_avoids_excluded_prompts() -> None:
    agent = RecommendationAgent()
    payload = {
        "issues": [{"number": 12, "title": "Login returns 500", "state": "open"}],
        "pull_requests": [{"number": 8, "title": "Refactor auth middleware", "state": "open"}],
        "workflow_runs": [{"name": "backend-tests", "status": "completed", "conclusion": "failure"}],
        "limit": 5,
    }

    first = await agent.run(payload, {})
    second = await agent.run({**payload, "exclude": first["suggestions"]}, {})

    assert 4 <= len(first["suggestions"]) <= 5
    assert 4 <= len(second["suggestions"]) <= 5
    assert set(first["suggestions"]).isdisjoint(second["suggestions"])


@pytest.mark.asyncio
async def test_recommendation_agent_returns_contextual_prompts_for_empty_repo() -> None:
    result = await RecommendationAgent().run({"issues": [], "pull_requests": [], "workflow_runs": [], "limit": 5}, {})

    assert 4 <= len(result["suggestions"]) <= 5
    assert all("失败" not in prompt and "PR #" not in prompt and "Issue #" not in prompt for prompt in result["suggestions"])


@pytest.mark.asyncio
async def test_recommendation_agent_cycles_when_candidates_are_exhausted() -> None:
    agent = RecommendationAgent()
    payload = {"issues": [], "pull_requests": [], "workflow_runs": [], "limit": 5}
    first = await agent.run(payload, {})
    second = await agent.run({**payload, "exclude": first["suggestions"]}, {})
    third = await agent.run({**payload, "exclude": [*first["suggestions"], *second["suggestions"]]}, {})

    assert first["suggestions"]
    assert second["suggestions"]
    assert third["suggestions"]


@pytest.mark.asyncio
async def test_recommendation_agent_does_not_suggest_missing_pr_or_failed_ci() -> None:
    result = await RecommendationAgent().run(
        {
            "issues": [{"number": 3, "title": "Feature request", "state": "open"}],
            "pull_requests": [],
            "workflow_runs": [],
            "limit": 5,
        },
        {},
    )

    assert all("PR" not in prompt and "CI 为什么失败" not in prompt for prompt in result["suggestions"])
