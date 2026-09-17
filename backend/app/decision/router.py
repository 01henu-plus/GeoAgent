"""Main Agent 自执行、委派与询问用户的统一路由。"""

from app.core.models import (
    AgentDecision,
    Dataset,
    DecisionType,
    IntentResult,
    Plan,
    SubTask,
    ToolCall,
)
from app.decision.tool_selector import select_tool


class AgentRouter:
    def route(self, intent: IntentResult, plan: Plan, datasets: list[Dataset], *, subtasks: list[SubTask] | None = None) -> AgentDecision:
        """根据已生成的计划选择下一步，不让执行层再次猜测意图。"""

        if plan.clarification:
            return AgentDecision(
                type=DecisionType.ASK_USER,
                reasoning_summary=plan.clarification,
                final_response=plan.clarification,
            )
        if self.should_delegate(intent, datasets):
            return AgentDecision(
                type=DecisionType.DELEGATE,
                reasoning_summary="任务包含多个相互独立的 GIS 主题，交给临时智能体并行处理。",
                subtasks=list(subtasks or []),
            )
        if intent.intent.value in {"UNKNOWN", "KNOWLEDGE_QUERY", "RESULT_INTERPRETATION", "RUN_DIAGNOSIS"}:
            return AgentDecision(
                type=DecisionType.FINAL,
                reasoning_summary="当前回合不需要直接执行 GIS 工具。",
            )
        tool_name = select_tool(intent, plan)
        executable = next((step for step in plan.steps if step.tool_name == tool_name), None)
        if executable is None:
            return AgentDecision(
                type=DecisionType.FINAL,
                reasoning_summary="计划没有可执行的 GIS 工具步骤。",
            )
        return AgentDecision(
            type=DecisionType.TOOL,
            reasoning_summary=f"按计划从 {executable.title} 开始执行。",
            tool_call=ToolCall(name=executable.tool_name or executable.action, arguments=executable.arguments),
        )

    def should_delegate(self, intent: IntentResult, datasets: list[Dataset]) -> bool:
        return intent.intent.value == "SPATIAL_ANALYSIS" and len(datasets) >= 2 and sum(bool(intent.entities.get(key)) for key in ("road_requested", "population_requested", "terrain_requested")) >= 2
