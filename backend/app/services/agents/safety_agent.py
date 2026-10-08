from typing import Any

from app.schemas.analysis import SafetyOutput
from app.services.agents.base import BaseAgent


class SafetyAgent(BaseAgent):
    name = "safety_agent"
    description = "检查某个操作是否需要人工确认。"
    available_tools = ["classify_action_risk"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
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
