"""有界、确定性的 Plan Revision 基础。

Replanner 不替代 Planner，也不调用模型。它只负责用当前事实重新编译计划，
保留已完成步骤，并拒绝与上一版剩余执行语义完全相同的计划。
"""

from __future__ import annotations

import json
from typing import Any

from app.core.models import Dataset, IntentResult, Plan, ReplanContext

from .planner import Planner


class ReplanNotPossible(RuntimeError):
    """当前失败上下文没有安全的确定性重规划方案。"""


class Replanner:
    """根据失败上下文重新编译剩余计划，不负责执行。"""

    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or Planner()

    def replan(self, context: ReplanContext, intent: IntentResult, datasets: list[Dataset]) -> Plan:
        rebuilt = self.planner.build(context.goal, intent, datasets)
        if rebuilt.clarification:
            raise ReplanNotPossible("REPLAN_NOT_SUPPORTED: " + rebuilt.clarification)

        revision = context.current_revision + 1
        completed = set(context.completed_steps)
        steps = _revision_steps(context, rebuilt.steps, revision, completed)
        _validate_plan_references(steps, completed)
        revised = rebuilt.model_copy(
            update={
                "revision": revision,
                "metadata": {
                    **rebuilt.metadata,
                    "source": "replan",
                    "previous_revision": context.current_revision,
                    "completed_steps": sorted(completed),
                },
                "steps": steps,
            }
        )
        if remaining_plan_fingerprint(revised, completed) == remaining_plan_fingerprint(context.current_plan, completed):
            raise ReplanNotPossible("REPLAN_NO_PROGRESS")
        return revised


def plan_fingerprint(plan: Plan) -> str:
    """仅基于稳定执行语义计算计划指纹。"""

    payload = [
        {
            "tool_name": step.tool_name,
            "arguments": step.arguments,
            "depends_on": step.depends_on,
            "required": step.required,
        }
        for step in plan.steps
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def remaining_plan_fingerprint(plan: Plan, completed_steps: set[str] | list[str]) -> str:
    completed = set(completed_steps)
    payload = [
        {
            "tool_name": step.tool_name,
            "arguments": step.arguments,
            "depends_on": [dependency for dependency in step.depends_on if dependency not in completed],
            "required": step.required,
        }
        for step in plan.steps
        if step.id not in completed
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _revision_steps(context: ReplanContext, steps: list[Any], revision: int, completed: set[str]) -> list[Any]:
    """为语义变化的失败步骤建立可辨识 ID，并同步依赖和输出引用。"""

    revised = list(steps)
    mapping: dict[str, str] = {}
    failed = context.failed_step
    if failed is not None:
        current_ids = {step.id for step in revised}
        replacement = next((step for step in revised if step.id == failed.id), None)
        if replacement is not None and _step_signature(replacement) != _step_signature(failed) and replacement.id not in completed:
            mapping[failed.id] = f"{failed.id}__rev{revision}"
        elif replacement is None and failed.id not in current_ids:
            # Planner 若按位置生成了新的 step id，优先把失败步骤的旧引用映射到
            # 同位置步骤；无法建立可靠映射时由引用校验明确拒绝，而不是猜算法。
            try:
                index = next(index for index, step in enumerate(context.current_plan.steps) if step.id == failed.id)
            except StopIteration:
                index = -1
            if 0 <= index < len(revised) and revised[index].id not in completed:
                mapping[failed.id] = revised[index].id

    normalized = []
    for step in revised:
        new_id = mapping.get(step.id, step.id)
        normalized.append(
            step.model_copy(
                update={
                    "id": new_id,
                    "depends_on": [mapping.get(dependency, dependency) for dependency in step.depends_on],
                    "arguments": _rewrite_references(step.arguments, mapping),
                }
            )
        )
    return normalized


def _step_signature(step: Any) -> tuple[Any, ...]:
    return (step.tool_name, step.arguments, tuple(step.depends_on), step.required)


def _rewrite_references(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        for old, new in mapping.items():
            value = value.replace("${" + old + ".", "${" + new + ".")
        return value
    if isinstance(value, dict):
        return {key: _rewrite_references(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_references(item, mapping) for item in value]
    return value


def _validate_plan_references(steps: list[Any], completed: set[str]) -> None:
    valid_ids = completed | {step.id for step in steps}
    for step in steps:
        missing = [dependency for dependency in step.depends_on if dependency not in valid_ids]
        if missing:
            raise ReplanNotPossible(f"REPLAN_INVALID_DEPENDENCY: {step.id} -> {', '.join(missing)}")
        for value in _walk_strings(step.arguments):
            if value.startswith("${") and value.endswith("}"):
                reference = value[2:-1].partition(".")[0]
                if reference and reference not in valid_ids:
                    raise ReplanNotPossible(f"REPLAN_INVALID_OUTPUT_REFERENCE: {value}")


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


__all__ = ["ReplanNotPossible", "Replanner", "plan_fingerprint", "remaining_plan_fingerprint"]
