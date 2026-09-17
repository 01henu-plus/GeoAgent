"""决策层的轻量工具选择器。"""

from app.core.models import IntentResult, Plan


def select_tool(intent: IntentResult, plan: Plan | None = None) -> str | None:
    """优先从已生成的 Plan 取下一步，规则映射只作为兼容兜底。"""

    if plan is not None:
        step = next((item for item in plan.steps if item.tool_name), None)
        if step is not None:
            return step.tool_name
    if intent.entities.get("buffer_requested"):
        return "vector.buffer"
    if intent.entities.get("distance_analysis_requested"):
        return "analysis.distance"
    if intent.intent.value == "DATA_INSPECTION":
        return "dataset.inspect"
    return None
