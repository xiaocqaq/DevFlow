from __future__ import annotations

from typing import Any

from app.schemas.workflow import WorkflowClaim, WorkflowObservation, WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "planner_agent"
    description = "根据用户目标和可用仓库信号，构建有边界的工程工作流规格。"
    available_tools = ["create_workflow_spec"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        message = str(input_data.get("message") or input_data.get("goal") or "").strip()
        lowered = message.lower()
        tasks: list[WorkflowTask] = []
        issue_label = f"Issue #{context['issue_number']}" if context.get("issue_number") is not None else "当前 Issue"
        pr_label = f"PR #{context['pr_number']}" if context.get("pr_number") is not None else "当前 PR"
        ci_label = (
            f"CI/Workflow Run #{context['workflow_run_number']}"
            if context.get("workflow_run_number") is not None
            else "最新失败的 CI 运行"
        )
        has_ci_run = context.get("has_ci_run", context.get("has_failed_ci"))

        if context.get("has_repo", True):
            tasks.append(
                WorkflowTask(
                    id="repo_health",
                    agent_name="repo_health_agent",
                    task_type="repo_health",
                    objective="在专用分析前捕获仓库基线。",
                    claim=WorkflowClaim(
                        target_type="repository",
                        target_id=context.get("repo_id"),
                        allowed_sources=["issues", "pull_requests", "workflow_runs"],
                        reason="共享仓库快照可以让后续 Agent 结论有证据支撑。",
                    ),
                )
            )

        broad = self._is_broad_engineering_question(lowered)
        if context.get("has_issue") and (broad or self._mentions_issue(lowered)):
            tasks.append(
                WorkflowTask(
                    id="issue_analysis",
                    agent_name="issue_analyst_agent",
                    task_type="issue_analysis",
                    objective=f"分析 {issue_label} 的分诊结论、优先级、负责人、证据和下一步动作。",
                    dependencies=["repo_health"] if self._has_task(tasks, "repo_health") else [],
                    claim=WorkflowClaim(
                        target_type="issue",
                        target_id=context.get("issue_id"),
                        allowed_sources=["issue", "similar_issues", "code", "documents", "team_members"],
                        reason="判断负责人和优先级需要明确 Issue 范围。",
                    ),
                    input_hint={"issue_number": context.get("issue_number")},
                )
            )

        if context.get("has_pr") and (broad or self._mentions_pr(lowered)):
            tasks.append(
                WorkflowTask(
                    id="pr_review",
                    agent_name="pr_review_agent",
                    task_type="pr_review",
                    objective=f"审查 {pr_label} diff、风险、阻塞问题和测试建议。",
                    dependencies=["repo_health"] if self._has_task(tasks, "repo_health") else [],
                    claim=WorkflowClaim(
                        target_type="pull_request",
                        target_id=context.get("pr_id"),
                        allowed_sources=["pull_request", "diff", "review_comments", "ci_summary", "documents"],
                        reason="合并就绪度和风险判断需要明确 PR 范围。",
                    ),
                    input_hint={"pr_number": context.get("pr_number")},
                    critical=self._mentions_merge(lowered),
                )
            )

        if has_ci_run and (broad or self._mentions_ci(lowered) or self._mentions_merge(lowered)):
            tasks.append(
                WorkflowTask(
                    id="ci_debug",
                    agent_name="ci_debug_agent",
                    task_type="ci_debug",
                    objective=f"分析 {ci_label}，并判断是否阻塞合并或发布。",
                    dependencies=["repo_health"] if self._has_task(tasks, "repo_health") else [],
                    claim=WorkflowClaim(
                        target_type="workflow_run",
                        target_id=context.get("workflow_run_id"),
                        allowed_sources=["workflow_run", "jobs", "logs", "recent_prs", "code"],
                        reason="任何合并或就绪度结论都需要先看 CI 失败上下文。",
                    ),
                    input_hint={"workflow_run_number": context.get("workflow_run_number")},
                    critical=True,
                )
            )

        if context.get("enable_rag_task") or self._mentions_history(lowered):
            tasks.append(
                WorkflowTask(
                    id="rag_search",
                    agent_name="rag_context_agent",
                    task_type="rag_search",
                    objective="检索与目标相关的历史 Issue、PR、失败 CI 日志、项目文档和已批准知识。",
                    dependencies=["repo_health"] if self._has_task(tasks, "repo_health") else [],
                    claim=WorkflowClaim(
                        target_type="knowledge",
                        target_id=context.get("repo_id"),
                        allowed_sources=["rag_documents", "approved_memory", "uploaded_knowledge"],
                        reason="历史证据可以避免重复分析和遗漏回归风险。",
                    ),
                )
            )

        if context.get("requires_safety"):
            tasks.append(
                WorkflowTask(
                    id="safety_review",
                    agent_name="safety_agent",
                    task_type="safety_review",
                    objective="判断写入或外部发送风险，并生成安全草稿而不是执行副作用。",
                    claim=WorkflowClaim(
                        target_type="external_action",
                        target_id=context.get("repo_id"),
                        allowed_sources=["user_request", "repository_snapshot"],
                        write_intent=True,
                        reason="写入类操作必须经过人工确认。",
                    ),
                    critical=True,
                )
            )

        if len(tasks) == 1 and tasks[0].task_type == "repo_health":
            if context.get("has_pr"):
                tasks.append(self._fallback_pr_task(context))
            elif has_ci_run:
                tasks.append(self._fallback_ci_task(context))
            elif context.get("has_issue"):
                tasks.append(self._fallback_issue_task(context))

        spec = WorkflowSpec(
            goal=message or "运行一次工程协作工作流。",
            trigger_message=message,
            acceptance_criteria=[
                "所有被选中的 Agent 都必须停留在声明的 claim 范围内。",
                "每个专用结果都应包含证据、置信度，或在证据不足时说明明确原因。",
                "Observer 必须在最终综合前标出跨 Agent 冲突。",
                "最终综合必须保留 What、Why、Tradeoff、Open Questions、Next Action。",
                "PR 审查必须按 P1/P2/P3 分级；P1/P2 未解决前不能建议直接合入。",
                "外部写操作必须返回草稿，并要求人工确认。",
            ],
            tasks=tasks,
            requires_human_confirmation=bool(context.get("requires_safety")),
        )
        return spec.model_dump(mode="json")

    async def replan(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        previous_spec = WorkflowSpec.model_validate(input_data["spec"])
        results = [WorkflowTaskResult.model_validate(item) for item in input_data.get("task_results") or []]
        observation = WorkflowObservation.model_validate(input_data["observation"])
        tasks = [task.model_copy(deep=True) for task in previous_spec.tasks]
        task_types = {task.task_type for task in tasks}
        successful_task_ids = {result.task_id for result in results if result.status == "success"}
        finding_types = {finding.finding_type for finding in observation.findings}

        if "missing_safety_review" in finding_types and "safety_review" not in task_types:
            tasks.append(
                WorkflowTask(
                    id=self._unique_task_id(tasks, "safety_review"),
                    agent_name="safety_agent",
                    task_type="safety_review",
                    objective="Review write or external-send risk before producing a final recommendation.",
                    dependencies=self._dependency_if_successful(successful_task_ids, "repo_health"),
                    claim=WorkflowClaim(
                        target_type="external_action",
                        target_id=context.get("repo_id"),
                        allowed_sources=["user_request", "repository_snapshot"],
                        write_intent=True,
                        reason="Replan added a missing safety gate after Observer flagged it.",
                    ),
                    critical=True,
                )
            )

        needs_context_recovery = bool(
            finding_types
            & {
                "task_error",
                "task_skipped",
                "evidence_gap",
                "low_confidence",
                "owner_needs_review",
                "team_profile_missing",
            }
        )
        if needs_context_recovery and "rag_search" not in task_types:
            tasks.append(
                WorkflowTask(
                    id=self._unique_task_id(tasks, "rag_search"),
                    agent_name="rag_context_agent",
                    task_type="rag_search",
                    objective="Recover missing evidence by searching historical issues, pull requests, failed CI logs, project documents, and approved knowledge.",
                    dependencies=self._dependency_if_successful(successful_task_ids, "repo_health"),
                    claim=WorkflowClaim(
                        target_type="knowledge",
                        target_id=context.get("repo_id"),
                        allowed_sources=["rag_documents", "approved_memory", "uploaded_knowledge"],
                        reason="Replan added historical context because Observer found weak evidence or confidence.",
                    ),
                )
            )

        if context.get("has_ci_run", context.get("has_failed_ci")) and "ci_debug" not in task_types and self._replan_needs_ci(previous_spec, results, observation):
            tasks.append(
                WorkflowTask(
                    id=self._unique_task_id(tasks, "ci_debug"),
                    agent_name="ci_debug_agent",
                    task_type="ci_debug",
                    objective="Check the latest failed CI run before final merge or release guidance.",
                    dependencies=self._dependency_if_successful(successful_task_ids, "repo_health"),
                    claim=WorkflowClaim(
                        target_type="workflow_run",
                        target_id=context.get("workflow_run_id"),
                        allowed_sources=["workflow_run", "jobs", "logs", "recent_prs", "code"],
                        reason="Replan added CI debugging because the current decision may depend on failed automation.",
                    ),
                    input_hint={"workflow_run_number": context.get("workflow_run_number")},
                    critical=True,
                )
            )

        replanned = previous_spec.model_copy(deep=True)
        replanned.tasks = tasks
        return replanned.model_dump(mode="json")

    def _fallback_issue_task(self, context: dict[str, Any]) -> WorkflowTask:
        label = f"Issue #{context['issue_number']}" if context.get("issue_number") is not None else "最新 Issue"
        return WorkflowTask(
            id="issue_analysis",
            agent_name="issue_analyst_agent",
            task_type="issue_analysis",
            objective=f"分析 {label}，因为它是当前最强的可用工程信号。",
            dependencies=["repo_health"],
            claim=WorkflowClaim(target_type="issue", target_id=context.get("issue_id"), allowed_sources=["issue", "team_members"]),
            input_hint={"issue_number": context.get("issue_number")},
        )

    def _fallback_pr_task(self, context: dict[str, Any]) -> WorkflowTask:
        label = f"PR #{context['pr_number']}" if context.get("pr_number") is not None else "最新 PR"
        return WorkflowTask(
            id="pr_review",
            agent_name="pr_review_agent",
            task_type="pr_review",
            objective=f"审查 {label}，因为它是当前最强的可用工程信号。",
            dependencies=["repo_health"],
            claim=WorkflowClaim(target_type="pull_request", target_id=context.get("pr_id"), allowed_sources=["pull_request", "diff", "ci_summary"]),
            input_hint={"pr_number": context.get("pr_number")},
        )

    def _fallback_ci_task(self, context: dict[str, Any]) -> WorkflowTask:
        label = (
            f"CI/Workflow Run #{context['workflow_run_number']}"
            if context.get("workflow_run_number") is not None
            else "最新失败的 CI 运行"
        )
        return WorkflowTask(
            id="ci_debug",
            agent_name="ci_debug_agent",
            task_type="ci_debug",
            objective=f"分析 {label}，因为它是当前最强的可用工程信号。",
            dependencies=["repo_health"],
            claim=WorkflowClaim(target_type="workflow_run", target_id=context.get("workflow_run_id"), allowed_sources=["workflow_run", "jobs", "logs"]),
            input_hint={"workflow_run_number": context.get("workflow_run_number")},
            critical=True,
        )

    def _has_task(self, tasks: list[WorkflowTask], task_id: str) -> bool:
        return any(task.id == task_id for task in tasks)

    def _unique_task_id(self, tasks: list[WorkflowTask], base: str) -> str:
        existing = {task.id for task in tasks}
        if base not in existing:
            return base
        index = 2
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    def _dependency_if_successful(self, successful_task_ids: set[str], task_id: str) -> list[str]:
        return [task_id] if task_id in successful_task_ids else []

    def _replan_needs_ci(
        self,
        spec: WorkflowSpec,
        results: list[WorkflowTaskResult],
        observation: WorkflowObservation,
    ) -> bool:
        text = " ".join(
            [
                spec.goal,
                spec.trigger_message,
                *(result.summary for result in results),
                *(finding.message for finding in observation.findings),
            ]
        ).lower()
        return self._mentions_ci(text) or self._mentions_merge(text)

    def _mentions_issue(self, text: str) -> bool:
        return any(token in text for token in ["issue", "bug", "需求", "问题", "负责人", "分配", "优先级"])

    def _mentions_pr(self, text: str) -> bool:
        return any(token in text for token in ["pr", "pull request", "diff", "review", "合并", "代码审查", "风险"])

    def _mentions_ci(self, text: str) -> bool:
        return any(token in text for token in ["ci", "workflow", "action", "build", "test", "失败", "流水线", "构建"])

    def _mentions_merge(self, text: str) -> bool:
        return any(token in text for token in ["merge", "合并", "能不能上", "能不能发", "readiness", "release"])

    def _mentions_history(self, text: str) -> bool:
        return any(token in text for token in ["history", "similar", "rag", "knowledge", "历史", "相似", "知识库", "记忆"])

    def _is_broad_engineering_question(self, text: str) -> bool:
        return any(
            token in text
            for token in [
                "multi-agent",
                "workflow",
                "协作",
                "工程状态",
                "阻塞",
                "风险",
                "优先处理",
                "下一步",
                "整体",
                "全局",
                "当前仓库",
                "能不能合并",
            ]
        )
