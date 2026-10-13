# ==============================================================================
# 模块名称: 多智能体协同工作流数据契约 (Multi-Agent Workflow Schemas)
# 架构定位: 数据契约层 (DTO / Pydantic Layer) / 工作流编排规范
# 核心架构与语法知识点:
#   1. 类型字面量枚举 (`typing.Literal`):
#      使用 `Literal["pending", "running", ...]` 取代传统 Python Enum 类，
#      在 Pydantic 校验与 OpenAPI 文档生成中天然序列化为字符串，且静态类型检查极为严格。
#   2. 有向无环图 (DAG) 拓扑任务建模:
#      - 每个 `WorkflowTask` 拥有唯一的 `id` 与 `dependencies: list[str]` 前置依赖列表。
#      - 工作流编排器据此分析依赖拓扑，并发调度无依赖项的任务，最大并发受 `max_parallel_tasks` 限制。
#   3. 意图声明与资源界限 (`WorkflowClaim`):
#      任务执行前需声明 `write_intent` (是否涉及外部写操作)，用于安全沙箱前置审计。
#   4. 动态时间戳与 ID 工厂:
#      `Field(default_factory=lambda: datetime.now(timezone.utc))` 避免时间戳停滞在模块导入时。
# ==============================================================================

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# 任务生命周期状态字面量类型
WorkflowTaskStatus = Literal["pending", "running", "success", "skipped", "error"]

# 观察者发现问题严重度级别字面量
WorkflowFindingSeverity = Literal["info", "warning", "blocker"]


def _workflow_id() -> str:
    """
    【ID 生成器】生成包含统一前缀且截取 12 位 UUID 十六进制字符的紧凑工作流标识符。
    """
    return f"workflow_{uuid.uuid4().hex[:12]}"


class WorkflowClaim(BaseModel):
    """
    【资源声明契约】Agent 执行任务前声明的资源访问与意图范围。

    Attributes:
        target_type (str): 目标资源类型（如 repo, issue, pull_request, file）。
        target_id (str | None): 具体操作的资源 ID。
        allowed_sources (list[str]): 允许检索的数据源列表。
        write_intent (bool): 是否具有修改外部状态的意图（为 True 时需安全审查）。
        reason (str): 请求该操作的合理解释。
    """
    target_type: str
    target_id: str | None = None
    allowed_sources: list[str] = Field(default_factory=list)
    write_intent: bool = False
    reason: str = ""


class WorkflowTask(BaseModel):
    """
    【DAG 任务单元契约】工作流中的单个独立执行节点。

    Attributes:
        id (str): 任务节点唯一标识（如 "task_1", "task_2"）。
        agent_name (str): 负责执行该任务的专用智能体名称（如 "github_retrieval_agent"）。
        task_type (str): 任务类型标识符。
        objective (str): 该任务的核心目标描述。
        dependencies (list[str]): 必须在此任务之前执行完成的前置任务 ID 列表。
        claim (WorkflowClaim): 资源与写操作意图声明。
        input_hint (dict[str, Any]): 任务初始输入上下文提示。
        critical (bool): 是否为关键阻断节点（若为 True 且失败，整个工作流直接中止）。
    """
    id: str
    agent_name: str
    task_type: str
    objective: str
    dependencies: list[str] = Field(default_factory=list)
    claim: WorkflowClaim
    input_hint: dict[str, Any] = Field(default_factory=dict)
    critical: bool = False


class WorkflowSpec(BaseModel):
    """
    【工作流规格定义】规划智能体 (Planner Agent) 生成的完整多智能体协同执行蓝图。

    Attributes:
        workflow_id (str): 工作流规格 ID。
        goal (str): 整体业务宏观目标。
        trigger_message (str): 触发该工作流的用户原始提问。
        acceptance_criteria (list[str]): 验收标准条目清单。
        tasks (list[WorkflowTask]): 任务节点集合（形成 DAG 结构）。
        max_parallel_tasks (int): 最大允许并发执行的任务线程数（1~8）。
        requires_human_confirmation (bool): 是否包含高危动作需要人工事前审批。
        created_at (datetime): 蓝图生成 UTC 时间戳。
    """
    workflow_id: str = Field(default_factory=_workflow_id)
    goal: str
    trigger_message: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    tasks: list[WorkflowTask] = Field(default_factory=list)
    max_parallel_tasks: int = Field(default=3, ge=1, le=8)
    requires_human_confirmation: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowTaskResult(BaseModel):
    """
    【任务执行结果契约】单个任务执行完毕后的输出报告。

    包含状态、执行摘要、结构化输出、收集的证据数、置信度以及耗时时间戳。
    """
    task_id: str
    agent_name: str
    task_type: str
    status: WorkflowTaskStatus
    summary: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = 0
    confidence: float | None = Field(default=None, ge=0, le=1)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class WorkflowFinding(BaseModel):
    """
    【独立观察发现】观察者智能体 (Observer Agent) 评估过程中发现的风险或异常项。
    """
    finding_type: str
    severity: WorkflowFindingSeverity
    message: str
    task_ids: list[str] = Field(default_factory=list)
    recommendation: str = ""


class WorkflowObservation(BaseModel):
    """
    【质检与反思评价契约】由 Observer Agent 对整个工作流执行过程与产出质量做出的监督评估。

    Attributes:
        observer_agent (str): 观察者身份标识。
        findings (list[WorkflowFinding]): 发现的问题与隐患清单。
        overall_confidence (float): 综合置信度得分（0.0 ~ 1.0）。
        human_review_required (bool): 是否建议转交人工最终复核。
        summary (str): 整体质量评估总结。
    """
    observer_agent: str = "observer_agent"
    findings: list[WorkflowFinding] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0, le=1)
    human_review_required: bool = False
    summary: str = ""


class WorkflowReport(BaseModel):
    """
    【最终工作流交付报告】整合规格蓝图、全部子任务输出、观察者评估及汇总回答的最终完整产物。
    """
    spec: WorkflowSpec
    task_results: list[WorkflowTaskResult]
    observation: WorkflowObservation
    final_answer: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    agent_events: list[dict[str, Any]] = Field(default_factory=list)

