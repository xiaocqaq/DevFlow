from typing import Any

from pydantic import BaseModel, Field


class SkillManifestResponse(BaseModel):
    name: str
    title: str
    version: str
    description: str
    category: str
    entrypoint: str
    tools: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)
    output_contract: str
    safety_level: str


class SkillResponse(BaseModel):
    manifest: SkillManifestResponse
    path: str
    instructions: str | None = None


class SkillListResponse(BaseModel):
    skills: list[SkillResponse]


class McpToolResponse(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class McpRuntimeResponse(BaseModel):
    status: str
    server_name: str
    server_version: str
    protocol_version: str
    transport: str
    tools: list[McpToolResponse] = Field(default_factory=list)
