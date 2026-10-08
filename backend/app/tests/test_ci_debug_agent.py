import pytest

from app.services.agents.ci_debug_agent import CIDebugAgent


@pytest.mark.asyncio
async def test_ci_debug_agent_includes_failed_job_context() -> None:
    result = await CIDebugAgent().run(
        {
            "name": "backend-test",
            "status": "completed",
            "conclusion": "failure",
            "jobs": [{"name": "pytest", "conclusion": "failure", "steps": [{"name": "Run tests", "conclusion": "failure"}]}],
            "logs_text": "pytest tests/test_api.py FAILED AssertionError",
        },
        {},
    )
    assert result["failure_type"] == "test"
    assert "pytest" in result["failure_summary"]
