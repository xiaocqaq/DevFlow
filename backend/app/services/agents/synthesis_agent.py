from __future__ import annotations

from typing import Any

from app.schemas.workflow import WorkflowObservation, WorkflowSpec, WorkflowTaskResult
from app.services.agents.base import BaseAgent


class SynthesisAgent(BaseAgent):
    name = "synthesis_agent"
    description = "将多 Agent 工作流综合成一份工程决策备忘录。"
    available_tools = ["synthesize_workflow_report"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        spec = WorkflowSpec.model_validate(input_data["spec"])
        results = [WorkflowTaskResult.model_validate(item) for item in input_data.get("task_results") or []]
        observation = WorkflowObservation.model_validate(input_data["observation"])
        answer = self._build_answer(spec, results, observation)
        return {"answer": answer}

    def _build_answer(self, spec: WorkflowSpec, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> str:
        blockers = [item for item in observation.findings if item.severity == "blocker"]
        warnings = [item for item in observation.findings if item.severity == "warning"]
        decision = self._decision(blockers, warnings, results)

        lines = [
            "## 多 Agent 协作结论",
            "",
            f"目标：{spec.goal}",
            f"总体判断：{decision}",
            f"Observer 置信度：{observation.overall_confidence:.2f}",
            "",
            "## 协作任务图",
        ]
        for task in spec.tasks:
            deps = f"，依赖：{', '.join(task.dependencies)}" if task.dependencies else ""
            lines.append(f"- {task.id}: {task.agent_name} 负责 {task.objective}{deps}")

        lines.extend(["", "## Agent 输出摘要"])
        for result in results:
            lines.append(f"- {result.agent_name} [{result.status}]: {result.summary or '无摘要。'}")

        lines.extend(["", "## Observer 检查"])
        if observation.findings:
            for finding in observation.findings:
                lines.append(f"- {finding.severity.upper()}: {finding.message} 建议：{finding.recommendation}")
        else:
            lines.append("- 未发现跨 Agent 结论冲突。")

        lines.extend(["", "## 决策依据与待确认项"])
        lines.extend(f"- {item}" for item in self._decision_basis(results, observation))

        next_steps = self._next_steps(blockers, warnings, results)
        lines.extend(["", "## 下一步"])
        lines.extend(f"- {item}" for item in next_steps)
        return "\n".join(lines).strip()

    def _decision(
        self,
        blockers: list[Any],
        warnings: list[Any],
        results: list[WorkflowTaskResult],
    ) -> str:
        if blockers:
            if any(item.finding_type in {"ci_merge_gate", "cross_agent_conflict"} for item in blockers):
                return "暂缓合并或发布，先处理 CI/跨 Agent 冲突。"
            if any(item.finding_type == "human_confirmation_required" for item in blockers):
                return "只生成草稿，等待人工确认后再执行外部写操作。"
            return "存在阻塞项，当前不建议直接推进。"
        pr_result = self._by_type(results, "pr_review")
        if pr_result and pr_result.output.get("merge_recommendation"):
            return f"按 PR Agent 建议推进：{pr_result.output.get('merge_recommendation')}。"
        if warnings:
            return "可以继续推进，但需要补齐 Observer 标出的证据或团队信息。"
        return "可以继续推进，未发现明显跨域阻塞。"

    def _next_steps(
        self,
        blockers: list[Any],
        warnings: list[Any],
        results: list[WorkflowTaskResult],
    ) -> list[str]:
        steps: list[str] = []
        ci_result = self._by_type(results, "ci_debug")
        pr_result = self._by_type(results, "pr_review")
        issue_result = self._by_type(results, "issue_analysis")

        if ci_result and ci_result.output.get("is_merge_blocking"):
            debug_steps = [str(item) for item in ci_result.output.get("debug_steps") or [] if str(item).strip()]
            steps.extend(debug_steps[:3] or ["先定位并修复最新失败 CI。"])
        if pr_result:
            blocking = [str(item) for item in pr_result.output.get("blocking_issues") or [] if str(item).strip()]
            tests = [str(item) for item in pr_result.output.get("test_suggestions") or [] if str(item).strip()]
            steps.extend(blocking[:2])
            steps.extend(tests[:2])
        if issue_result:
            actions = [str(item) for item in issue_result.output.get("checklist") or issue_result.output.get("action_items") or [] if str(item).strip()]
            steps.extend(actions[:2])
        for finding in [*blockers, *warnings]:
            if finding.recommendation:
                steps.append(str(finding.recommendation))
        if not steps:
            steps.append("把本次 workflow 结论回填到 PR/Issue 描述或团队同步中。")
        return list(dict.fromkeys(steps))[:6]

    def _decision_basis(self, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> list[str]:
        basis: list[str] = []
        for result in results:
            confidence = f"，置信度 {result.confidence:.2f}" if result.confidence is not None else ""
            evidence = f"，证据 {result.evidence_count} 条" if result.evidence_count else ""
            basis.append(f"{result.agent_name}：{result.summary or '无摘要'}{evidence}{confidence}。")
        if observation.findings:
            for finding in observation.findings:
                basis.append(f"待确认：{finding.message} 处理建议：{finding.recommendation}")
        else:
            basis.append("待确认：最终合入、评论或外部发送仍需要人工按团队流程确认。")
        return basis[:8]

    def _by_type(self, results: list[WorkflowTaskResult], task_type: str) -> WorkflowTaskResult | None:
        return next((item for item in results if item.task_type == task_type and item.status == "success"), None)
