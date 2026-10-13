# ==============================================================================
# 模块名称: 智能对话交互数据模型 (Chat Schemas)
# 架构定位: 数据传输对象层 (DTO / Pydantic Contracts Layer)
# 核心技术与语法知识点:
#   1. Pydantic v2 数据验证与序列化:
#      所有模型继承自 `BaseModel`，在运行时自动执行类型强制转换与格式合法性校验。
#   2. 可变默认参数防御 (`Field(default_factory=...)`):
#      在 Python 类属性中，若直接写 `context: dict = {}` 会导致所有实例共享同一个全局字典对象（Mutable Default Trap）。
#      使用 `Field(default_factory=dict)` 或 `Field(default_factory=list)` 确保每个实例均拥有独立的全新对象。
#   3. 面向对象继承 (`ChatPlanExecuteRequest(ChatRequest)`):
#      子类自动复用父类全部校验规则并扩展专属字段。
#   4. 联合类型与可选性 (`UUID | str`, `UUID | None = None`):
#      采用现代 Python 3.10+ PEP 604 语法，替代传统的 `Union[UUID, str]` 与 `Optional[UUID]`。
# ==============================================================================

from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    【请求模式】用户对话请求标准输入载荷。

    Attributes:
        repo_id (UUID): 关联的仓库主键。
        conversation_id (UUID | None): 对话会话 UUID，为空时系统自动路由到默认会话。
        message (str): 用户发送的自然语言文本。
        context (dict): 客户端附带的上下文参数（如当前选中文件、临时设置等）。
    """
    repo_id: UUID
    conversation_id: UUID | None = None
    message: str
    context: dict = Field(default_factory=dict)


class ChatPlanExecuteRequest(ChatRequest):
    """
    【继承扩展模式】确认执行特定工作流规格 (Workflow Spec) 的对话请求。

    继承关系:
      复用 `ChatRequest` 的全部字段，并额外附加结构化的工作流定义 `workflow_spec`。
    """
    workflow_spec: dict


class ToolCallTrace(BaseModel):
    """
    【追踪模式】单个工具调用或技能激活的审计跟踪实体。

    用于向前端透明化展示 Agent 在背后调用了哪个工具、入参是什么、是否出错以及耗时重试信息。
    """
    tool_name: str
    arguments: dict
    status: str
    error: str | None = None
    error_kind: str | None = None
    retryable: bool | None = None
    attempts: int | None = None
    tool_use_id: str | None = None
    skill_name: str | None = None
    skill_title: str | None = None
    skill_version: str | None = None
    skill_steps: list[str] = Field(default_factory=list)
    skill_activation_mode: str | None = None
    skill_activation_reason: str | None = None
    skill_instruction_digest: str | None = None


class ChatResponse(BaseModel):
    """
    【响应模式】同步对话接口返回的标准响应载荷。

    包含最终回答、路由决策、工具执行迹线、引用的知识文档、Token 用量与上下文压缩统计。
    """
    run_id: UUID | str
    message_id: UUID | str
    route: str
    tool_calls: list[ToolCallTrace]
    agent_events: list[dict] = []
    answer: str
    citations: list[dict] = []
    completion: dict = {}
    compression_stats: dict = {}
    progressive_compaction: dict = {}
    model_usage: list[dict] = []
    context_diagnostics: dict = {}


class ChatHistoryMessage(BaseModel):
    """
    【历史消息模式】用于在前端对话窗口渲染的历史气泡消息。
    """
    id: UUID | str
    conversation_id: UUID | str | None = None
    role: str
    content: str
    route: str | None = None
    tool_calls: list[ToolCallTrace] = []
    agent_events: list[dict] = []
    compression_stats: dict = {}
    progressive_compaction: dict = {}
    model_usage: list[dict] = []
    context_diagnostics: dict = {}
    created_at: str | None = None


class PromptRecommendationRequest(BaseModel):
    """
    【请求模式】智能 Prompt 推荐请求。

    数值范围约束:
      `limit`: 限制在 4 到 5 条之间 (`ge=4, le=5`)，保持前端界面排版紧凑美观。
    """
    repo_id: UUID
    exclude: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=4, le=5)


class PromptRecommendationResponse(BaseModel):
    """
    【响应模式】针对当前仓库上下文生成的推荐提示词列表。
    """
    agent: str
    suggestions: list[str]

