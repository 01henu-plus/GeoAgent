"""运行时计划的无状态执行辅助函数。

AgentRuntime 负责循环控制；本模块只负责寻找满足依赖的真实工具步骤，
以及解析步骤之间的数据引用。
"""

from __future__ import annotations

from typing import Any

from app.core.models import Plan, PlanStep


def next_executable_step(plan: Plan, completed_steps: set[str] | None = None) -> PlanStep | None:
    completed = set(completed_steps or ())
    for step in plan.steps:
        if step.id in completed:
            continue
        if all(dependency in completed for dependency in step.depends_on):
            return step
    return None


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


__all__ = ["next_executable_step", "resolve_plan_arguments"]
