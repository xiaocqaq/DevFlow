from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.schemas.workflow import WorkflowObservation, WorkflowReport, WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.agents.base import BaseAgent
from app.services.agents.observer_agent import ObserverAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.synthesis_agent import SynthesisAgent


WorkflowTaskRunner = Callable[[WorkflowTask, WorkflowSpec, dict[str, Any]], Awaitable[WorkflowTaskResult | dict[str, Any]]]


class WorkflowOrchestrator(BaseAgent):
    name = "workflow_orchestrator"
    description = "协调 Planner、专用 Agent、Observer 和 Synthesis 完成工程决策。"
    available_tools = ["run_engineering_workflow"]

    def __init__(
        self,
        task_runners: dict[str, WorkflowTaskRunner],
        planner: PlannerAgent | None = None,
        observer: ObserverAgent | None = None,
        synthesizer: SynthesisAgent | None = None,
    ) -> None:
        self.task_runners = task_runners
        self.planner = planner or PlannerAgent()
        self.observer = observer or ObserverAgent()
        self.synthesizer = synthesizer or SynthesisAgent()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        spec_data = await self.planner.run(input_data, context)
        return await self.run_spec(spec_data, context)

    async def run_spec(
        self,
        spec_data: WorkflowSpec | dict[str, Any],
        context: dict[str, Any],
        *,
        allow_replan: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        spec = spec_data if isinstance(spec_data, WorkflowSpec) else WorkflowSpec.model_validate(spec_data)
        current_spec = spec
        task_results: list[WorkflowTaskResult] = []
        observations: list[WorkflowObservation] = []
        round_records: list[dict[str, Any]] = []
        replan_events: list[dict[str, Any]] = []
        max_replans = self._max_replans(context, allow_replan)

        for attempt in range(max_replans + 1):
            round_results = await self._run_task_graph(current_spec, context, previous_results=task_results)
            task_results.extend(round_results)
            observation = await self._observe(current_spec, task_results, context)
            observations.append(observation)
            round_records.append({"spec": current_spec, "results": round_results, "observation": observation})

            if attempt >= max_replans:
                break

            replan_reason = self._replan_reason(task_results, observation)
            if replan_reason is None:
                break
            if not hasattr(self.planner, "replan"):
                break

            next_spec_data = await self.planner.replan(
                {
                    "spec": current_spec.model_dump(mode="json"),
                    "task_results": [item.model_dump(mode="json") for item in task_results],
                    "observation": observation.model_dump(mode="json"),
                    "reason": replan_reason,
                    "replan_iteration": attempt + 1,
                },
                context,
            )
            next_spec = self._normalize_replanned_spec(
                original_spec=spec,
                current_spec=current_spec,
                next_spec=WorkflowSpec.model_validate(next_spec_data),
            )
            if self._same_task_graph(current_spec, next_spec):
                break

            replan_events.append(
                {
                    "type": "workflow_replan",
                    "workflow_id": spec.workflow_id,
                    "iteration": attempt + 1,
                    "reason": replan_reason,
                    "from_task_count": len(current_spec.tasks),
                    "to_task_count": len(next_spec.tasks),
                    "added_tasks": [
                        task.id
                        for task in next_spec.tasks
                        if task.id not in {existing.id for existing in current_spec.tasks}
                    ],
                }
            )
            current_spec = next_spec

        observation = observations[-1] if observations else WorkflowObservation(summary="Workflow did not run.")
        synthesis = await self.synthesizer.run(
            {
                "spec": current_spec.model_dump(mode="json"),
                "task_results": [item.model_dump(mode="json") for item in task_results],
                "observation": observation.model_dump(mode="json"),
            },
            context,
        )
        report = WorkflowReport(
            spec=current_spec,
            task_results=task_results,
            observation=observation,
            final_answer=str(synthesis.get("answer") or ""),
            tool_calls=self._tool_calls(current_spec, task_results, replan_events),
            citations=self._citations(task_results),
            metrics={
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "task_count": len(task_results),
                "plan_task_count": len(current_spec.tasks),
                "execution_rounds": len(observations),
                "replan_count": len(replan_events),
                "success_count": len([item for item in task_results if item.status == "success"]),
                "error_count": len([item for item in task_results if item.status == "error"]),
                "skipped_count": len([item for item in task_results if item.status == "skipped"]),
                "observer_findings": len(observation.findings),
            },
            agent_events=self._agent_events(round_records, replan_events),
        )
        return report.model_dump(mode="json")

    async def _observe(
        self,
        spec: WorkflowSpec,
        task_results: list[WorkflowTaskResult],
        context: dict[str, Any],
    ) -> WorkflowObservation:
        observation_data = await self.observer.run(
            {
                "spec": spec.model_dump(mode="json"),
                "task_results": [item.model_dump(mode="json") for item in task_results],
            },
            context,
        )
        return WorkflowObservation.model_validate(observation_data)

    async def _run_task_graph(
        self,
        spec: WorkflowSpec,
        context: dict[str, Any],
        previous_results: list[WorkflowTaskResult] | None = None,
    ) -> list[WorkflowTaskResult]:
        pending = {task.id: task for task in spec.tasks}
        completed: dict[str, WorkflowTaskResult] = {
            result.task_id: result
            for result in previous_results or []
            if result.status == "success"
        }
        blocked_dependencies: dict[str, WorkflowTaskResult] = {
            result.task_id: result
            for result in previous_results or []
            if result.status in {"error", "skipped"}
        }
        for task_id in completed:
            pending.pop(task_id, None)
        results: list[WorkflowTaskResult] = []

        while pending:
            blocked = [
                task
                for task in pending.values()
                if any(dependency in blocked_dependencies for dependency in task.dependencies)
            ]
            for task in blocked:
                failed_dependencies = [
                    dependency
                    for dependency in task.dependencies
                    if dependency in blocked_dependencies
                ]
                result = self._dependency_blocked_result(task, failed_dependencies)
                blocked_dependencies[task.id] = result
                results.append(result)
                pending.pop(task.id, None)
            if not pending:
                break

            ready = [
                task
                for task in pending.values()
                if all(dependency in completed for dependency in task.dependencies)
            ]
            if not ready:
                for task in pending.values():
                    result = self._error_result(task, "任务依赖存在循环或缺失依赖。", error_kind="dependency_cycle_or_missing")
                    blocked_dependencies[task.id] = result
                    results.append(result)
                break

            batch = ready[: spec.max_parallel_tasks]
            batch_results = await asyncio.gather(
                *(self._run_single_task(task, spec, context) for task in batch),
                return_exceptions=False,
            )
            for task, result in zip(batch, batch_results):
                if result.status == "success":
                    completed[task.id] = result
                else:
                    blocked_dependencies[task.id] = result
                results.append(result)
                pending.pop(task.id, None)

        return results

    async def _run_single_task(
        self,
        task: WorkflowTask,
        spec: WorkflowSpec,
        context: dict[str, Any],
    ) -> WorkflowTaskResult:
        started_at = datetime.now(timezone.utc)
        runner = self.task_runners.get(task.task_type)
        if runner is None:
            return WorkflowTaskResult(
                task_id=task.id,
                agent_name=task.agent_name,
                task_type=task.task_type,
                status="skipped",
                summary=f"没有为 {task.task_type} 注册任务执行器。",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        try:
            raw_result = await asyncio.wait_for(
                runner(task, spec, context),
                timeout=self._task_timeout_seconds(context),
            )
            result = raw_result if isinstance(raw_result, WorkflowTaskResult) else WorkflowTaskResult.model_validate(raw_result)
            result.started_at = result.started_at or started_at
            result.completed_at = result.completed_at or datetime.now(timezone.utc)
            return result
        except TimeoutError:
            return self._error_result(
                task,
                f"{task.agent_name} 执行超过 {self._task_timeout_seconds(context):g} 秒。",
                started_at=started_at,
                error_kind="timeout",
            )
        except Exception as exc:
            return self._error_result(task, str(exc), started_at=started_at, error_kind="task_runtime_error")

    def _error_result(
        self,
        task: WorkflowTask,
        error: str,
        started_at: datetime | None = None,
        *,
        error_kind: str = "task_runtime_error",
    ) -> WorkflowTaskResult:
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="error",
            summary=f"{task.agent_name} 执行失败。",
            output={"error_kind": error_kind},
            error=error,
            started_at=started_at or datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

    def _dependency_blocked_result(self, task: WorkflowTask, failed_dependencies: list[str]) -> WorkflowTaskResult:
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="skipped",
            summary=f"{task.agent_name} 未执行，因为依赖任务未成功：{', '.join(failed_dependencies)}。",
            output={"error_kind": "dependency_failed", "failed_dependencies": failed_dependencies},
            confidence=0.0,
            completed_at=datetime.now(timezone.utc),
        )

    def _task_timeout_seconds(self, context: dict[str, Any]) -> float:
        try:
            value = float(context.get("workflow_task_timeout_seconds", settings.workflow_task_timeout_seconds))
        except (TypeError, ValueError):
            value = settings.workflow_task_timeout_seconds
        return max(0.001, value)

    def _max_replans(self, context: dict[str, Any], allow_replan: bool) -> int:
        if not allow_replan or context.get("disable_replan"):
            return 0
        try:
            requested = int(context.get("max_replans", 1))
        except (TypeError, ValueError):
            requested = 1
        return max(0, min(2, requested))

    def _replan_reason(self, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> str | None:
        fixable_findings = {
            "task_error",
            "task_skipped",
            "evidence_gap",
            "low_confidence",
            "missing_safety_review",
            "owner_needs_review",
            "team_profile_missing",
        }
        for finding in observation.findings:
            if finding.finding_type in fixable_findings:
                return f"{finding.finding_type}: {finding.recommendation or finding.message}"
        if any(result.status == "error" for result in results):
            return "task_error: at least one workflow task failed"
        low_confidence_results = [
            result
            for result in results
            if result.status == "success" and result.confidence is not None and result.confidence < 0.45
        ]
        if low_confidence_results:
            return "low_confidence: one or more successful task results have weak confidence"
        if results and observation.overall_confidence < 0.45:
            return "low_confidence: observer confidence is below the workflow threshold"
        return None

    def _normalize_replanned_spec(
        self,
        *,
        original_spec: WorkflowSpec,
        current_spec: WorkflowSpec,
        next_spec: WorkflowSpec,
    ) -> WorkflowSpec:
        normalized = next_spec.model_copy(deep=True)
        normalized.workflow_id = original_spec.workflow_id
        normalized.goal = normalized.goal or current_spec.goal
        normalized.trigger_message = normalized.trigger_message or current_spec.trigger_message
        if not normalized.acceptance_criteria:
            normalized.acceptance_criteria = list(current_spec.acceptance_criteria)
        normalized.max_parallel_tasks = current_spec.max_parallel_tasks
        normalized.requires_human_confirmation = (
            current_spec.requires_human_confirmation or normalized.requires_human_confirmation
        )
        return normalized

    def _same_task_graph(self, left: WorkflowSpec, right: WorkflowSpec) -> bool:
        def signature(spec: WorkflowSpec) -> list[tuple[str, str, tuple[str, ...]]]:
            return [(task.id, task.task_type, tuple(task.dependencies)) for task in spec.tasks]

        return signature(left) == signature(right)

    def _tool_calls(
        self,
        spec: WorkflowSpec,
        results: list[WorkflowTaskResult],
        replan_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        calls = [
            {
                "tool_name": f"workflow.{result.task_type}",
                "arguments": {
                    "workflow_id": spec.workflow_id,
                    "task_id": result.task_id,
                    "agent": result.agent_name,
                },
                "status": result.status,
                "error": result.error,
            }
            for result in results
        ]
        calls.extend(
            {
                "tool_name": "planner_agent.replan_workflow_spec",
                "arguments": {
                    "workflow_id": spec.workflow_id,
                    "iteration": event.get("iteration"),
                    "reason": event.get("reason"),
                },
                "status": "success",
                "error": None,
            }
            for event in replan_events
        )
        return calls

    def _citations(self, results: list[WorkflowTaskResult]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for result in results:
            for citation in result.citations:
                key = (
                    str(citation.get("type") or ""),
                    str(citation.get("id") or citation.get("source_id") or ""),
                    str(citation.get("title") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                citations.append(citation)
        return citations

    def _agent_events(
        self,
        round_records: list[dict[str, Any]],
        replan_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for index, record in enumerate(round_records):
            if index > 0 and index - 1 < len(replan_events):
                events.append(replan_events[index - 1])
            spec = record["spec"]
            events.append(
                {
                    "type": "workflow_spec",
                    "workflow_id": spec.workflow_id,
                    "goal": spec.goal,
                    "iteration": index,
                    "is_replan": index > 0,
                    "tasks": [
                        {
                            "id": task.id,
                            "agent_name": task.agent_name,
                            "task_type": task.task_type,
                            "dependencies": task.dependencies,
                            "claim": task.claim.model_dump(mode="json"),
                        }
                        for task in spec.tasks
                    ],
                }
            )
            for result in record["results"]:
                skill_runtime = result.output.get("_skill_runtime") if isinstance(result.output, dict) else None
                events.append(
                    {
                        "type": "workflow_task_result",
                        "workflow_id": spec.workflow_id,
                        "iteration": index,
                        "task_id": result.task_id,
                        "agent_name": result.agent_name,
                        "task_type": result.task_type,
                        "status": result.status,
                        "summary": result.summary,
                        "confidence": result.confidence,
                        "evidence_count": result.evidence_count,
                        **(
                            {
                                key: value
                                for key, value in skill_runtime.items()
                                if key
                                in {
                                    "skill_name",
                                    "skill_title",
                                    "skill_version",
                                    "skill_steps",
                                    "skill_activation_mode",
                                    "skill_activation_reason",
                                    "skill_instruction_digest",
                                    "validation",
                                }
                            }
                            if isinstance(skill_runtime, dict)
                            else {}
                        ),
                    }
                )
            observation = record["observation"]
            events.append(
                {
                    "type": "workflow_observer_result",
                    "workflow_id": spec.workflow_id,
                    "iteration": index,
                    "is_final": index == len(round_records) - 1,
                    "overall_confidence": observation.overall_confidence,
                    "human_review_required": observation.human_review_required,
                    "findings": [finding.model_dump(mode="json") for finding in observation.findings],
                }
            )
        return events
