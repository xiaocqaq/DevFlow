from io import BytesIO
from zipfile import BadZipFile, ZipFile

from app.services.github.client import GitHubClient


async def list_workflow_runs(owner: str, repo: str, token: str | None = None, limit: int = 30, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    payload = await client.get(f"/repos/{owner}/{repo}/actions/runs", {"per_page": limit})
    return payload.get("workflow_runs", [])


async def list_workflow_run_jobs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    client = GitHubClient(token, base_url)
    payload = await client.get(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs", {"per_page": 100})
    return payload.get("jobs", [])


async def get_workflow_run_logs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None, max_chars: int = 80000) -> str:
    client = GitHubClient(token, base_url)
    payload = await client.get_bytes(f"/repos/{owner}/{repo}/actions/runs/{run_id}/logs")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            parts: list[str] = []
            for name in sorted(archive.namelist()):
                if name.endswith("/"):
                    continue
                raw = archive.read(name)
                text = raw.decode("utf-8", errors="replace")
                parts.append(f"### {name}\n{text}")
                if sum(len(part) for part in parts) >= max_chars:
                    break
            return "\n\n".join(parts)[:max_chars]
    except BadZipFile:
        return payload.decode("utf-8", errors="replace")[:max_chars]
