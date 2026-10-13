"""安全风险审查与人工确认门禁智能体模块.

本模块负责对智能体意图执行的动作（如发评论、创建 Issue、修改 Label、关闭 PR）
进行风险分级（Risk Level Classification），实施 Human-in-the-loop 人机协同安全防线。
"""

from typing import Any

from app.schemas.analysis import SafetyOutput
from app.services.agents.base import BaseAgent


class SafetyAgent(BaseAgent):
    """安全审查智能体 (Safety Agent).

    【Python 语法要点 - 关键词包含性匹配与生成器表达式】:
    - 使用 `any(token in action for token in [...])`:
      在布尔判断中利用生成器表达式配合 `any()` 具有短路求值（Short-circuit Evaluation）特性，
      只要匹配到第一个高危动词便立即返回 True，无需遍历剩余列表。
    """

    name = "safety_agent"
    description = "检查某个操作是否需要人工确认。"
    available_tools = ["classify_action_risk"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """评估动作风险等级并决定是否要求人工确认审批."""
        action = input_data.get("action", "").lower()
        risky = any(
            token in action
            for token in [
                "comment",
                "create_issue",
                "label",
                "close_issue",
                "send_report",
                "write",
                "post",
                "send",
                "评论",
                "创建",
                "新建",
                "发给",
                "发送",
                "改标签",
                "关闭",
                "写回",
            ]
        )
        return SafetyOutput(
            requires_confirmation=risky,
            risk_level="medium" if risky else "low",
            reason="MVP 阶段所有外部写操作只生成草稿。" if risky else "只读分析操作不需要人工确认。",
            draft=input_data.get("draft"),
        ).model_dump()
