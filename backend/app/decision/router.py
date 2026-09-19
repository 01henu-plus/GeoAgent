"""Main Agent 自执行、委派与询问用户的统一路由。"""

from app.core.models import (
    AgentDecision,
    Dataset,
    DecisionType,
    InteractionMode,
    Plan,
    RequestFrame,
    SubTask,
    ToolCall,
)


class AgentRouter:
    def route(self, request_frame: RequestFrame, plan: Plan, datasets: list[Dataset], *, subtasks: list[SubTask] | None = None) -> AgentDecision:
        """根据已生成的计划选择下一步，不让执行层再次猜测意图。"""

        if plan.clarification:
            return AgentDecision(
                type=DecisionType.ASK_USER,
                reasoning_summary=plan.clarification,
                final_response=plan.clarification,
            )
        if self.should_delegate(request_frame, datasets) or plan.metadata.get("delegated_roles"):
            return AgentDecision(
                type=DecisionType.DELEGATE,
                reasoning_summary="任务包含多个相互独立的 GIS 主题，交给临时智能体并行处理。",
                subtasks=list(subtasks or []),
            )
        if request_frame.mode in {InteractionMode.CHAT, InteractionMode.CANCEL_TASK, InteractionMode.QUERY}:
            return AgentDecision(
                type=DecisionType.FINAL,
                reasoning_summary="当前回合不需要直接执行 GIS 工具。",
            )
        executable = next((step for step in plan.steps if step.tool_name), None)
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

    def should_delegate(self, request_frame: RequestFrame, datasets: list[Dataset]) -> bool:
        text = request_frame.goal.casefold()
        roles = sum(
            any(term in text for term in terms)
            for terms in (("道路", "路网", "road"), ("人口", "population"), ("dem", "高程", "地形", "terrain"))
        )
        return roles >= 2 and len(datasets) >= 2 and bool(set(request_frame.capabilities) & {"raster_analysis", "vector_analysis"})
