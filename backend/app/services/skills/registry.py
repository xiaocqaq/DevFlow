import ast
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SkillRegistryError(ValueError):
    pass


class SkillManifest(BaseModel):
    name: str
    title: str
    version: str
    description: str
    category: str = "general"
    entrypoint: str
    tools: list[str] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)
    output_contract: str = "freeform"
    safety_level: str = "read_only"


class SkillDefinition(BaseModel):
    manifest: SkillManifest
    path: str
    instructions: str

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "skill_name": self.manifest.name,
            "skill_title": self.manifest.title,
            "skill_version": self.manifest.version,
            "skill_steps": self.manifest.workflow_steps,
        }

    def prompt_summary(self) -> str:
        steps = ", ".join(self.manifest.workflow_steps) if self.manifest.workflow_steps else "在 SKILL.md 中定义"
        return (
            f"- {self.manifest.name} v{self.manifest.version} -> {self.manifest.entrypoint}: "
            f"{self.manifest.description} 步骤：{steps}。安全级别：{self.manifest.safety_level}。"
        )


class SkillActivation(BaseModel):
    skill_name: str
    skill_title: str
    skill_version: str
    entrypoint: str
    tools: list[str] = Field(default_factory=list)
    workflow_steps: list[str] = Field(default_factory=list)
    output_contract: str
    safety_level: str
    activation_mode: str
    activation_reason: str
    instructions: str
    instruction_digest: str
    resources: list[dict[str, Any]] = Field(default_factory=list)

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "skill_title": self.skill_title,
            "skill_version": self.skill_version,
            "skill_steps": self.workflow_steps,
            "skill_activation_mode": self.activation_mode,
            "skill_activation_reason": self.activation_reason,
            "skill_instruction_digest": self.instruction_digest,
        }

    def prompt_block(self) -> str:
        resource_text = ""
        if self.resources:
            sections = []
            for resource in self.resources:
                content = str(resource.get("content") or "").strip()
                if content:
                    sections.append(f"### 资源：{resource['path']}\n{content}")
            if sections:
                resource_text = "\n\n按需加载的 Skill 资源：\n\n" + "\n\n".join(sections)
        return (
            f"## {self.skill_title} ({self.skill_name} v{self.skill_version})\n"
            f"激活方式：{self.activation_mode}；原因：{self.activation_reason}\n"
            f"允许工具：{', '.join(self.tools) or '无'}\n"
            f"输出契约：{self.output_contract}；安全级别：{self.safety_level}\n\n"
            f"{self.instructions.strip()}"
            f"{resource_text}"
        )


class SkillRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._skills = self._load_skills()
        self._validate_unique_keys()
        self._by_name = {skill.manifest.name: skill for skill in self._skills}
        self._by_entrypoint = {skill.manifest.entrypoint: skill for skill in self._skills}

    def list_skills(self) -> list[SkillDefinition]:
        return sorted(self._skills, key=lambda skill: skill.manifest.name)

    def get_skill(self, name: str) -> SkillDefinition | None:
        return self._by_name.get(name)

    def skill_for_entrypoint(self, tool_name: str) -> SkillDefinition | None:
        return self._by_entrypoint.get(tool_name)

    def trace_metadata_by_tool(self) -> dict[str, dict[str, Any]]:
        return {tool_name: skill.trace_metadata() for tool_name, skill in self._by_entrypoint.items()}

    def annotate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(tool_call.get("tool_name") or "")
        skill = self.skill_for_entrypoint(tool_name)
        if not skill:
            return dict(tool_call)
        return {**tool_call, **skill.trace_metadata()}

    def prompt_context(self) -> str:
        skills = self.list_skills()
        if not skills:
            return "当前没有注册 DevFlow 技能。"
        return "\n".join(skill.prompt_summary() for skill in skills)

    def tool_description_suffix(self, tool_name: str) -> str:
        skill = self.skill_for_entrypoint(tool_name)
        if not skill:
            return ""
        return (
            f" 注册技能：{skill.manifest.title} v{skill.manifest.version}。"
            f"{skill.manifest.description}"
        )

    def tool_description_suffixes(self) -> dict[str, str]:
        return {tool_name: self.tool_description_suffix(tool_name) for tool_name in self._by_entrypoint}

    def activate(
        self,
        message: str,
        *,
        requested_name: str | None = None,
        max_auto_skills: int = 3,
    ) -> list[SkillActivation]:
        explicit = self._explicit_skills(message, requested_name)
        if explicit:
            return [self._activation(skill, "explicit", "用户明确指定 Skill") for skill in explicit]

        scored = [
            (self._activation_score(message, skill), skill)
            for skill in self._skills
        ]
        selected = [
            (score, skill)
            for score, skill in scored
            if score > 0
        ]
        selected.sort(key=lambda item: (-item[0], item[1].manifest.name))
        return [
            self._activation(skill, "automatic", f"消息命中 Skill 触发条件，匹配分 {score}")
            for score, skill in selected[:max_auto_skills]
        ]

    def activate_entrypoint(self, tool_name: str, *, reason: str = "专用 Agent 执行入口") -> SkillActivation | None:
        skill = self.skill_for_entrypoint(tool_name)
        if skill is None:
            return None
        return self._activation(skill, "entrypoint", reason)

    def _activation(self, skill: SkillDefinition, mode: str, reason: str) -> SkillActivation:
        instructions = skill.instructions.strip()
        return SkillActivation(
            skill_name=skill.manifest.name,
            skill_title=skill.manifest.title,
            skill_version=skill.manifest.version,
            entrypoint=skill.manifest.entrypoint,
            tools=skill.manifest.tools,
            workflow_steps=skill.manifest.workflow_steps,
            output_contract=skill.manifest.output_contract,
            safety_level=skill.manifest.safety_level,
            activation_mode=mode,
            activation_reason=reason,
            instructions=instructions,
            instruction_digest=hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16],
            resources=self._load_referenced_resources(skill),
        )

    def _explicit_skills(self, message: str, requested_name: str | None) -> list[SkillDefinition]:
        if requested_name:
            skill = self._skill_by_name_or_title(requested_name)
            if skill is None:
                raise SkillRegistryError(f"未找到用户指定的 Skill：{requested_name}")
            return [skill]

        lowered = message.lower()
        if "skill" not in lowered and "技能" not in message:
            return []
        selected = [
            skill
            for skill in self._skills
            if skill.manifest.name.lower() in lowered or skill.manifest.title.lower() in lowered
        ]
        if selected:
            return selected

        match = re.search(
            r"(?:use|using|用|使用|调用|启用)\s*([A-Za-z][A-Za-z0-9_-]+)\s*(?:skill|技能)",
            message,
            flags=re.IGNORECASE,
        )
        if match:
            raise SkillRegistryError(f"未找到用户指定的 Skill：{match.group(1)}")
        return []

    def _skill_by_name_or_title(self, value: str) -> SkillDefinition | None:
        normalized = value.strip().lower()
        return next(
            (
                skill
                for skill in self._skills
                if skill.manifest.name.lower() == normalized or skill.manifest.title.lower() == normalized
            ),
            None,
        )

    def _activation_score(self, message: str, skill: SkillDefinition) -> int:
        lowered = message.lower()
        score = 0
        for trigger in skill.manifest.triggers:
            normalized = trigger.strip().lower()
            if not normalized:
                continue
            if re.fullmatch(r"[a-z0-9_+#.-]+", normalized):
                matched = re.search(rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])", lowered)
            else:
                matched = normalized in lowered
            if matched:
                score += max(1, min(len(normalized), 8))
        return score

    def _load_referenced_resources(self, skill: SkillDefinition) -> list[dict[str, Any]]:
        references = sorted(
            {
                match.group(1).strip()
                for match in re.finditer(
                    r"`((?:references|templates|scripts|assets)/[^`]+)`",
                    skill.instructions,
                )
            }
        )
        if len(references) > 8:
            raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源超过 8 个")

        skill_root = Path(skill.path).resolve().parent
        resources: list[dict[str, Any]] = []
        text_suffixes = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".py", ".sh", ".ps1"}
        for relative in references:
            resource_path = (skill_root / relative).resolve()
            try:
                resource_path.relative_to(skill_root)
            except ValueError as exc:
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用了目录外资源：{relative}") from exc
            if not resource_path.is_file():
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源不存在：{relative}")
            if resource_path.stat().st_size > 64 * 1024:
                raise SkillRegistryError(f"Skill {skill.manifest.name} 引用的资源超过 64 KiB：{relative}")
            content = resource_path.read_text(encoding="utf-8") if resource_path.suffix.lower() in text_suffixes else ""
            resources.append(
                {
                    "path": relative.replace("\\", "/"),
                    "content": content,
                    "size": resource_path.stat().st_size,
                    "digest": hashlib.sha256(resource_path.read_bytes()).hexdigest()[:16],
                }
            )
        return resources

    def _load_skills(self) -> list[SkillDefinition]:
        if not self.root.exists():
            return []
        skills: list[SkillDefinition] = []
        for skill_file in sorted(self.root.glob("*/SKILL.md")):
            skills.append(_load_skill(skill_file))
        return skills

    def _validate_unique_keys(self) -> None:
        seen_names: set[str] = set()
        seen_entrypoints: set[str] = set()
        for skill in self._skills:
            if skill.manifest.name in seen_names:
                raise SkillRegistryError(f"Skill 名称重复：{skill.manifest.name}")
            if skill.manifest.entrypoint in seen_entrypoints:
                raise SkillRegistryError(f"Skill entrypoint 重复：{skill.manifest.entrypoint}")
            seen_names.add(skill.manifest.name)
            seen_entrypoints.add(skill.manifest.entrypoint)


def _load_skill(path: Path) -> SkillDefinition:
    text = path.read_text(encoding="utf-8")
    manifest_data, instructions = _split_frontmatter(text, path)
    manifest = SkillManifest.model_validate(manifest_data)
    return SkillDefinition(manifest=manifest, path=str(path), instructions=instructions)


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        raise SkillRegistryError(f"{path} 必须以 frontmatter 开头")

    frontmatter_lines: list[str] = []
    body_start = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = index + 1
            break
        frontmatter_lines.append(line)
    if body_start is None:
        raise SkillRegistryError(f"{path} 缺少 frontmatter 结束标记")

    return _parse_frontmatter(frontmatter_lines), "\n".join(lines[body_start:]).strip()


def _parse_frontmatter(lines: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if current_list_key is None:
                raise SkillRegistryError("列表项缺少父级 key")
            data.setdefault(current_list_key, [])
            if not isinstance(data[current_list_key], list):
                raise SkillRegistryError(f"{current_list_key} 不是列表")
            data[current_list_key].append(_parse_value(stripped[2:]))
            continue
        if ":" not in raw_line:
            raise SkillRegistryError(f"frontmatter 行无效：{raw_line}")
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SkillRegistryError("frontmatter key 不能为空")
        if value == "":
            data[key] = []
            current_list_key = key
        else:
            data[key] = _parse_value(value)
            current_list_key = None
    return data


def _parse_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
        return parsed if isinstance(parsed, list) else value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


@lru_cache(maxsize=1)
def get_skill_registry() -> SkillRegistry:
    root = Path(__file__).resolve().parents[2] / "skills"
    return SkillRegistry(root)
