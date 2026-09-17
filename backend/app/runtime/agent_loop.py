"""GeoAgent 的执行循环。

``MainAgent`` 负责理解目标和选择恢复策略，``AgentLoop`` 负责把已经决定好的
Plan 逐步执行。两者分开后，离线规则路径和未来的模型路径可以共享同一套
``step -> observe -> checkpoint``运行语义。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.core.models import Plan, PlanStep, TaskStatus, ToolResult, ToolStatus

StepExecutor = Callable[[PlanStep, dict[str, Any]], Awaitable[ToolResult]]
StepVerifier = Callable[[PlanStep, ToolResult], Awaitable[list[str]]]
CheckpointWriter = Callable[[set[str], dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class PlanLoopOutcome:
    completed_steps: frozenset[str]
    findings: tuple[Any, ...]
    output_ids: tuple[str, ...]
    artifacts: tuple[str, ...]
    errors: tuple[str, ...]
    step_outputs: dict[str, dict[str, Any]]
    failed_step: PlanStep | None = None
    failed_result: ToolResult | None = None
    verification_problems: tuple[str, ...] = ()


class AgentLoop:
    """执行有依赖关系的 GIS Plan，不负责猜测用户意图。"""

    def __init__(self, main_agent=None) -> None:
        # 保留一个很薄的兼容入口；真正的计划循环仍然由本类负责。
        self.main_agent = main_agent

    async def run(self, request, **kwargs):
        """兼容旧的 Runtime 门面，避免外部调用者绕过 MainAgent。"""

        if self.main_agent is None:
            raise RuntimeError("AgentLoop.run 需要注入 MainAgent；计划执行请调用 execute_plan。")
        return await self.main_agent.run(request, **kwargs)

    async def execute_plan(
        self,
        plan: Plan,
        *,
        completed_steps: set[str] | None = None,
        findings: list[Any] | None = None,
        output_ids: list[str] | None = None,
        artifacts: list[str] | None = None,
        errors: list[str] | None = None,
        step_outputs: dict[str, dict[str, Any]] | None = None,
        execute_step: StepExecutor,
        verify_step: StepVerifier,
        checkpoint: CheckpointWriter,
    ) -> PlanLoopOutcome:
        completed = set(completed_steps or ())
        collected_findings = list(findings or ())
        collected_outputs = list(output_ids or ())
        collected_artifacts = list(artifacts or ())
        collected_errors = list(errors or ())
        outputs = dict(step_outputs or {})

        for step in plan.steps:
            if step.id in completed:
                step.status = TaskStatus.SUCCEEDED
                continue
            missing_dependencies = [dependency for dependency in step.depends_on if dependency not in completed]
            if missing_dependencies:
                step.status = TaskStatus.FAILED
                message = f"步骤 {step.id} 依赖未完成：{', '.join(missing_dependencies)}"
                collected_errors.append(message)
                await checkpoint(completed, _state(collected_findings, collected_outputs, collected_artifacts, collected_errors, outputs))
                return PlanLoopOutcome(frozenset(completed), tuple(collected_findings), tuple(collected_outputs), tuple(collected_artifacts), tuple(collected_errors), outputs, failed_step=step)

            if step.tool_name is None:
                step.status = TaskStatus.SUCCEEDED
                completed.add(step.id)
                await checkpoint(completed, _state(collected_findings, collected_outputs, collected_artifacts, collected_errors, outputs))
                continue

            step.status = TaskStatus.RUNNING
            arguments = resolve_plan_arguments(step.arguments, outputs)
            result = await execute_step(step, arguments)
            finding = {
                "step_id": step.id,
                "tool": step.tool_name,
                "status": result.status.value,
                "output": result.output,
                "datasets": result.datasets,
                "artifacts": result.artifacts,
                "error": result.error.model_dump(mode="json") if result.error else None,
            }
            collected_findings.append(finding)
            _extend_unique(collected_outputs, result.datasets)
            _extend_unique(collected_artifacts, result.artifacts)
            outputs[step.id] = {
                "dataset_id": result.datasets[-1] if result.datasets else None,
                "dataset_ids": list(result.datasets),
                "artifact_ids": list(result.artifacts),
                "output": result.output,
            }

            if result.status not in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS}:
                step.status = TaskStatus.FAILED
                message = result.error.message if result.error else f"{step.title}失败。"
                collected_errors.append(message)
                await checkpoint(completed, _state(collected_findings, collected_outputs, collected_artifacts, collected_errors, outputs))
                if step.required:
                    return PlanLoopOutcome(frozenset(completed), tuple(collected_findings), tuple(collected_outputs), tuple(collected_artifacts), tuple(collected_errors), outputs, failed_step=step, failed_result=result)
                continue

            problems = await verify_step(step, result)
            if problems:
                step.status = TaskStatus.FAILED
                collected_errors.extend(problems)
                await checkpoint(completed, _state(collected_findings, collected_outputs, collected_artifacts, collected_errors, outputs))
                if step.required:
                    return PlanLoopOutcome(frozenset(completed), tuple(collected_findings), tuple(collected_outputs), tuple(collected_artifacts), tuple(collected_errors), outputs, failed_step=step, failed_result=result, verification_problems=tuple(problems))
                continue

            step.status = TaskStatus.SUCCEEDED
            completed.add(step.id)
            await checkpoint(completed, _state(collected_findings, collected_outputs, collected_artifacts, collected_errors, outputs))

        return PlanLoopOutcome(frozenset(completed), tuple(collected_findings), tuple(collected_outputs), tuple(collected_artifacts), tuple(collected_errors), outputs)


def resolve_plan_arguments(arguments: dict[str, Any], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """解析 ``${step.dataset_id}`` 形式的步骤间引用。"""

    def resolve(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            reference = value[2:-1]
            step_id, _, field = reference.partition(".")
            snapshot = outputs.get(step_id, {})
            return snapshot.get(field, value)
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(arguments)


def _state(findings: list[Any], output_ids: list[str], artifacts: list[str], errors: list[str], step_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "findings": findings,
        "output_ids": output_ids,
        "artifacts": artifacts,
        "errors": errors,
        "step_outputs": step_outputs,
    }


def _extend_unique(values: list[str], additions: list[str]) -> None:
    for value in additions:
        if value not in values:
            values.append(value)


__all__ = ["AgentLoop", "PlanLoopOutcome", "resolve_plan_arguments"]
