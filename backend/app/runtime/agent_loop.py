"""GeoAgent 的执行循环。

``MainAgent`` 负责理解目标和选择恢复策略，``AgentLoop`` 负责把已经决定好的
Plan 逐步执行。两者分开后，离线规则路径和未来的模型路径可以共享同一套
``step -> observe -> checkpoint``运行语义。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.core.models import LoopDirective, Plan, PlanStep, TaskStatus
from app.runtime.tool_execution_cycle import ExecutionOutcome

StepExecutor = Callable[[PlanStep, dict[str, Any]], Awaitable[ExecutionOutcome]]
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
    failed_outcome: ExecutionOutcome | None = None
    directive: LoopDirective = LoopDirective.CONTINUE


class AgentLoop:
    """执行有依赖关系的 GIS Plan，不负责猜测用户意图。"""

    @staticmethod
    def next_executable_step(plan: Plan, completed_steps: set[str] | None = None) -> PlanStep | None:
        """返回当前确定可执行的下一步，不执行整张 Plan。"""

        completed = set(completed_steps or ())
        for step in plan.steps:
            if step.id in completed:
                continue
            if all(dependency in completed for dependency in step.depends_on):
                return step
        return None

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
            outcome = await execute_step(step, arguments)
            result = outcome.result
            finding = {
                "step_id": step.id,
                "tool": step.tool_name,
                "status": result.status.value,
                "accepted": outcome.accepted,
                "verified": outcome.verified,
                "verification_problems": list(outcome.verification_problems),
                "recovery_action": outcome.recovery_action.value if outcome.recovery_action else None,
                "directive": outcome.directive.value,
                "attempts": outcome.attempts,
                "output": result.output,
                "datasets": result.datasets,
                "artifacts": result.artifacts,
                "error": result.error.model_dump(mode="json") if result.error else None,
            }
            collected_findings.append(finding)

            if not outcome.accepted:
                step.status = TaskStatus.FAILED
                message = _outcome_message(step, outcome)
                collected_errors.append(message)
                await checkpoint(completed, _state(collected_findings, collected_outputs, collected_artifacts, collected_errors, outputs))
                if step.required:
                    return PlanLoopOutcome(
                        frozenset(completed),
                        tuple(collected_findings),
                        tuple(collected_outputs),
                        tuple(collected_artifacts),
                        tuple(collected_errors),
                        outputs,
                        failed_step=step,
                        failed_outcome=outcome,
                        directive=outcome.directive,
                    )
                continue

            _extend_unique(collected_outputs, result.datasets)
            _extend_unique(collected_artifacts, result.artifacts)
            outputs[step.id] = {
                "dataset_id": result.datasets[-1] if result.datasets else None,
                "dataset_ids": list(result.datasets),
                "artifact_ids": list(result.artifacts),
                "output": result.output,
            }
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


def _outcome_message(step: PlanStep, outcome: ExecutionOutcome) -> str:
    if outcome.verification_problems:
        return "；".join(outcome.verification_problems)
    if outcome.result.error is not None:
        return outcome.result.error.message
    if outcome.rationale:
        return outcome.rationale
    return f"{step.title}失败。"
