# ==============================================================================
# 模块名称: 智能体分析与评估结构化输出契约 (Agent Analysis Schemas)
# 架构定位: 数据传输对象层 (DTO / Pydantic Layer) / LLM 结构化输出协议
# 核心技术与语法知识点:
#   1. LLM 结构化输出 (Structured Outputs / JSON Schema):
#      大模型（如 Gemini / OpenAI）通过 `with_structured_output(Schema)` 原生绑定 Pydantic 类，
#      自动将模型推导出的非结构化自然语言强制转换为符合这些字段约束的结构化 JSON。
#   2. 严格业务字面量约束 (`typing.Literal`):
#      - 结论类型限定: `Literal["进入开发", "等待澄清", "关闭", "拆分", "合并到已有 Issue"]`
#      - 优先级与复杂度限定: `Literal["P0", "P1", "P2", "P3"]`、`Literal["S", "M", "L", "XL"]`
#      - 合入判定限定: `Literal["建议合入", "修改后合入", "暂缓", "拒绝"]`
#   3. 数值区间硬约束 (`Field(ge=0, le=1)`):
#      用于置信度 (confidence) 归一化校验，防止模型返回越界的异常浮点数。
#   4. 嵌套模型默认工厂 (`Field(default_factory=IssueDrafts)`):
#      在父模型中嵌套子模型时，避免默认实例在模块级别单例共享。
# ==============================================================================

from typing import Literal

from pydantic import BaseModel, Field


class DuplicateCandidate(BaseModel):
    """
    【查重候选实体】向量库中匹配到的可能重复的历史 Issue。

    Attributes:
        issue_id (str): 历史 Issue ID。
        title (str): 历史 Issue 标题。
        score (float): 向量余弦相似度分数 (0.0~1.0)。
    """
    issue_id: str
    title: str
    score: float = Field(ge=0, le=1)


class IssueEvidence(BaseModel):
    """
    【证据卡片契约】支持 Agent 结论的依据项。
    """
    source_type: str
    title: str
    snippet: str
    reference: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class IssueDrafts(BaseModel):
    """
    【动作草案载荷】Agent 提议的可选文本草案（如待发布的澄清留言、任务拆解大纲）。
    """
    clarification_comment: str | None = None
    task_breakdown: str | None = None


class IssueAnalysisOutput(BaseModel):
    """
    【Issue 分析输出规范】IssueAgent 的全量结构化评估结果。

    包含优先级判定、复杂度预估、智能推荐负责人、排查清单以及草案提议。
    """
    summary: str
    conclusion: Literal["进入开发", "等待澄清", "关闭", "拆分", "合并到已有 Issue"]
    conclusion_reason: str
    category: Literal["Bug", "Feature", "Question", "Documentation", "Refactor", "Test", "Ops"]
    priority: Literal["P0", "P1", "P2", "P3"]
    complexity: Literal["S", "M", "L", "XL"]
    suggested_owner: str
    owner_reason: str
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)
    evidence: list[IssueEvidence] = Field(default_factory=list)
    checklist: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    drafts: IssueDrafts = Field(default_factory=IssueDrafts)
    confidence: float = Field(ge=0, le=1)


class PRReviewFinding(BaseModel):
    """
    【PR 审查独立发现】代码审查过程中捕获的具体隐患点。
    """
    severity: Literal["P1", "P2", "P3"]
    title: str
    evidence: str
    required_action: str
    blocking: bool = False


class PRReviewOutput(BaseModel):
    """
    【PR 审查输出规范】PRReviewAgent 的结构化终审报告。

    包含合入建议裁决（如“建议合入”、“修改后合入”）、风险清单、测试建议与 Review Checklist。
    """
    summary: str
    plan: list[str] = []
    executed_steps: list[str] = []
    merge_recommendation: Literal["建议合入", "修改后合入", "暂缓", "拒绝"] = "暂缓"
    recommendation_reason: str = ""
    review_findings: list[PRReviewFinding] = Field(default_factory=list)
    key_changes: list[str] = []
    risk_points: list[str] = []
    blocking_issues: list[str] = []
    review_checklist: list[str] = []
    test_suggestions: list[str] = []
    files_need_attention: list[str] = []
    review_comments: list[str] = []
    confidence: float = Field(ge=0, le=1)


class CIDebugOutput(BaseModel):
    """
    【CI 诊断输出规范】CIDebugAgent 的故障排查诊断报告。

    包含失败类型归类、首个关键错误、根因分析、分步修复指南与合并阻断判定。
    """
    failure_summary: str
    failure_type: Literal["test", "build", "lint", "dependency", "permission", "environment", "unknown"]
    plan: list[str] = []
    executed_steps: list[str] = []
    first_error: str | None = None
    root_cause: str = ""
    possible_causes: list[str] = []
    fix_steps: list[str] = []
    debug_steps: list[str] = []
    related_files: list[str] = []
    is_merge_blocking: bool = True
    blocking_reason: str = ""
    confidence: float = Field(ge=0, le=1)


class SafetyOutput(BaseModel):
    """
    【安全审计判定】SafetyAgent 针对提议操作的高危拦截与审批裁决。
    """
    requires_confirmation: bool
    risk_level: Literal["low", "medium", "high"]
    reason: str
    draft: str | None = None

