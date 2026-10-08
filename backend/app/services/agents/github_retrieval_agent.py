from typing import Any

from app.services.agents.base import BaseAgent
from app.services.github.actions import get_workflow_run_logs, list_workflow_run_jobs, list_workflow_runs
from app.services.github.issues import list_issues
from app.services.github.pull_requests import list_pull_request_files, list_pull_request_review_comments, list_pull_requests


class GitHubRetrievalAgent(BaseAgent):
    name = "github_retrieval_agent"
    description = "检索 GitHub Issue、PR、diff、Review 评论、workflow run 和失败日志。"
    available_tools = [
        "list_issues",
        "list_pull_requests",
        "list_pull_request_files",
        "list_pull_request_review_comments",
        "list_workflow_runs",
        "list_workflow_run_jobs",
        "get_workflow_run_logs",
    ]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        owner = input_data["owner"]
        repo = input_data["repo"]
        token = input_data.get("token")
        base_url = input_data.get("base_url")
        limit = int(input_data.get("limit", 30))
        task = input_data.get("task", "repo_snapshot")

        if task == "issues":
            return {"issues": await list_issues(owner, repo, token=token, limit=limit, base_url=base_url)}
        if task == "pull_requests":
            return {"pull_requests": await list_pull_requests(owner, repo, token=token, limit=limit, base_url=base_url)}
        if task == "pull_request_detail":
            number = int(input_data["number"])
            files = await list_pull_request_files(owner, repo, number, token=token, base_url=base_url)
            comments = await list_pull_request_review_comments(owner, repo, number, token=token, base_url=base_url)
            return {"files": files, "review_comments": comments}
        if task == "workflow_runs":
            return {"workflow_runs": await list_workflow_runs(owner, repo, token=token, limit=limit, base_url=base_url)}
        if task == "workflow_run_failure":
            run_id = int(input_data["run_id"])
            jobs = await list_workflow_run_jobs(owner, repo, run_id, token=token, base_url=base_url)
            logs_text = await get_workflow_run_logs(owner, repo, run_id, token=token, base_url=base_url)
            return {"jobs": jobs, "logs_text": logs_text}

        issues = await list_issues(owner, repo, token=token, limit=limit, base_url=base_url)
        pull_requests = await list_pull_requests(owner, repo, token=token, limit=limit, base_url=base_url)
        workflow_runs = await list_workflow_runs(owner, repo, token=token, limit=limit, base_url=base_url)
        return {"issues": issues, "pull_requests": pull_requests, "workflow_runs": workflow_runs}
