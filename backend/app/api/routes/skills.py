from fastapi import APIRouter, HTTPException

from app.schemas.skills import McpRuntimeResponse, McpToolResponse, SkillListResponse, SkillManifestResponse, SkillResponse
from app.services.mcp_client import McpClientError, StdioMcpClient
from app.services.skills import get_skill_registry

router = APIRouter()


def _skill_response(skill, include_instructions: bool = False) -> SkillResponse:
    return SkillResponse(
        manifest=SkillManifestResponse.model_validate(skill.manifest.model_dump()),
        path=skill.path,
        instructions=skill.instructions if include_instructions else None,
    )


@router.get("", response_model=SkillListResponse)
async def list_skills() -> SkillListResponse:
    registry = get_skill_registry()
    return SkillListResponse(skills=[_skill_response(skill) for skill in registry.list_skills()])


@router.get("/mcp-status", response_model=McpRuntimeResponse)
async def get_mcp_status() -> McpRuntimeResponse:
    try:
        inspection = await StdioMcpClient().inspect_server()
    except (McpClientError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"MCP Server 验证失败：{exc}") from exc

    initialize = inspection.get("initialize") or {}
    server_info = initialize.get("serverInfo") or {}
    tools = [
        McpToolResponse(
            name=str(item.get("name") or ""),
            description=str(item.get("description") or ""),
            input_schema=item.get("inputSchema") or {},
        )
        for item in inspection.get("tools") or []
    ]
    return McpRuntimeResponse(
        status="online",
        server_name=str(server_info.get("name") or "unknown"),
        server_version=str(server_info.get("version") or "unknown"),
        protocol_version=str(initialize.get("protocolVersion") or "unknown"),
        transport="stdio",
        tools=tools,
    )


@router.get("/{skill_name}", response_model=SkillResponse)
async def get_skill(skill_name: str) -> SkillResponse:
    registry = get_skill_registry()
    skill = registry.get_skill(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail="未找到 Skill")
    return _skill_response(skill, include_instructions=True)
