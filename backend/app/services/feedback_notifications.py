import logging
from typing import Any

import httpx

from app.core.config import settings
from app.db.models import AgentRun, ChatFeedback, ChatMessage, Repository, User

logger = logging.getLogger(__name__)

REASON_LABELS = {
    "inaccurate": "内容不准确",
    "not_relevant": "没有解决问题",
    "missing_context": "缺少关键上下文",
    "unreliable_citation": "引用或证据不可靠",
    "tool_error": "工具执行异常",
    "other": "其他",
}


def _clip(value: str | None, limit: int) -> str:
    text = " ".join((value or "").split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _trace_url(run_id: Any) -> str:
    path = f"/api/chat/feedback/trace/{run_id}"
    base_url = (settings.feedback_trace_base_url or "").strip().rstrip("/")
    return f"{base_url}{path}" if base_url else path


async def send_negative_feedback_notification(
    *,
    feedback: ChatFeedback,
    repo: Repository,
    message: ChatMessage,
    run: AgentRun | None,
    reporter: User | None,
) -> tuple[str, str | None]:
    webhook_url = (settings.feishu_feedback_webhook_url or "").strip()
    if not webhook_url:
        return "not_configured", None

    run_id = feedback.run_id or "未关联"
    lines = [
        "DevFlow AI 收到一条负反馈",
        f"项目：{repo.full_name}",
        f"反馈人：{reporter.name if reporter else '内部用户'}",
        f"trace_id：{run_id}",
        f"message_id：{feedback.assistant_message_id}",
        f"路由：{run.route if run else message.route or 'unknown'}",
        f"原因：{REASON_LABELS.get(feedback.reason or '', '未填写')}",
    ]
    if feedback.comment:
        lines.append(f"补充说明：{_clip(feedback.comment, 300)}")
    if run:
        lines.extend(
            [
                f"用户问题：{_clip(run.user_message, 300)}",
                f"Agent 回答：{_clip(run.final_answer, 500)}",
            ]
        )
    lines.append(f"执行轨迹：{_trace_url(run_id)}" if feedback.run_id else "执行轨迹：该历史回答未关联 run_id")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                webhook_url,
                json={"msg_type": "text", "content": {"text": "\n".join(lines)}},
            )
            response.raise_for_status()
            payload = response.json()
            status_code = payload.get("StatusCode", payload.get("code", 0)) if isinstance(payload, dict) else 0
            if status_code not in {0, "0", None}:
                raise RuntimeError(str(payload.get("StatusMessage") or payload.get("msg") or payload))
    except Exception as exc:  # Notification failure must not lose the persisted feedback.
        logger.warning("发送飞书负反馈通知失败：%s", exc)
        return "failed", _clip(str(exc), 1000)
    return "sent", None
