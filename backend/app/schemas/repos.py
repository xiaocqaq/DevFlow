from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RepoConnectRequest(BaseModel):
    owner: str | None = None
    repo: str | None = None
    token: str | None = None
    github_token: str | None = None
    provider: str = "github"
    api_base_url: str | None = None
    local_path: str | None = None
    clone_parent_dir: str | None = None
    demo_mode: bool = False
    initial_sync_limit: int = Field(default=5, ge=1, le=50)
    sync_issues: bool = True
    sync_pull_requests: bool = True
    sync_workflow_runs: bool = True


class RepoResponse(BaseModel):
    id: UUID
    owner: str
    name: str
    full_name: str
    provider: str = "github"
    api_base_url: str | None = None
    clone_url: str | None = None
    local_path: str | None = None
    checkout_mode: str = "managed"
    description: str | None = None
    default_branch: str | None = None
    last_sync_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class RepoConnectResponse(BaseModel):
    repo_id: UUID
    full_name: str
    provider: str
    api_base_url: str | None = None
    clone_url: str | None = None
    local_path: str | None = None
    checkout_mode: str = "managed"
    default_branch: str | None
    connected: bool
    demo_mode: bool
    message: str
    sync_status: str = "not_started"
    synced: dict[str, int] | None = None


class RepoSyncRequest(BaseModel):
    sync_issues: bool = True
    sync_pull_requests: bool = True
    sync_workflow_runs: bool = True
    limit: int = 30


class RepoSyncResponse(BaseModel):
    repo_id: UUID
    status: str
    synced: dict[str, int]
