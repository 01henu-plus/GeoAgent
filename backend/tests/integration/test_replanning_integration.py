import asyncio
from types import SimpleNamespace

from app.core.models import (
    AgentDecision,
    AgentRequest,
    DecisionType,
    FailureAction,
    IntentType,
    LoopDirective,
    Plan,
    PlanStep,
    RequestFrame,
    Run,
    ToolError,
    ToolResult,
    ToolStatus,
)
from app.runtime.session import AgentRuntimeSession
from app.runtime.tool_execution_cycle import ExecutionOutcome


def _outcome(result, *, accepted, directive):
    return ExecutionOutcome(
        result=result,
        verified=accepted,
        verification_problems=[] if accepted else ["算法不适用"],
        recovery_action=None if accepted else FailureAction.REPLAN,
        attempts=1,
        accepted=accepted,
        directive=directive,
        rationale=None if accepted else "当前算法不适用",
    )


def test_planner_replan_increments_revision_and_skips_completed_step(application):
    task = application.task_service.create("重规划测试", conversation_id="replan-conversation")
    run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(run)
    plan = Plan(
        goal="重规划测试",
        intent=IntentType.SPATIAL_ANALYSIS,
        steps=[
            PlanStep(id="inspect", title="检查数据", action="dataset.inspect", tool_name="dataset.inspect"),
            PlanStep(id="slope", title="计算坡度", action="raster.slope", tool_name="raster.slope", depends_on=["inspect"]),
        ],
    )
    calls = []

    async def execute(current_run, tool_name, arguments, **kwargs):
        calls.append(tool_name)
        if tool_name == "raster.slope":
            return _outcome(
                ToolResult(call_id="failed", status=ToolStatus.FAILED, error=ToolError(code="ALGORITHM_NOT_APPLICABLE", message="算法不适用")),
                accepted=False,
                directive=LoopDirective.REPLAN,
            )
        return _outcome(ToolResult(call_id="ok", status=ToolStatus.SUCCESS, output={"ok": True}, datasets=["slope-result"] if tool_name == "raster.slope__rev2" else []), accepted=True, directive=LoopDirective.CONTINUE)

    class FakeReplanner:
        def replan(self, context, request_frame, datasets):
            return Plan(
                goal=context.goal,
                intent=IntentType.SPATIAL_ANALYSIS,
                revision=context.current_revision + 1,
                steps=[
                    PlanStep(id="inspect", title="检查数据", action="dataset.inspect", tool_name="dataset.inspect"),
                    PlanStep(id="slope__rev2", title="计算坡度", action="raster.slope", tool_name="raster.slope__rev2", depends_on=["inspect"]),
                ],
                metadata={"source": "replan"},
            )

    request = AgentRequest(user_input="重规划", conversation_id=task.conversation_id)
    frame = RequestFrame(mode="new_task", goal="重规划测试")
    session = AgentRuntimeSession(
        run=run,
        datasets=[],
        current_plan=plan,
        original_plan=plan.model_copy(deep=True),
    )
    handlers = application.main_agent.runtime_action_handlers
    original_cycle = application.main_agent.tool_execution_cycle
    original_replanner = application.main_agent.replanner
    application.main_agent.tool_execution_cycle = SimpleNamespace(execute=execute)
    application.main_agent.replanner = FakeReplanner()
    try:
        inspect = asyncio.run(handlers.execute_plan_step(plan.steps[0], request=request, run=run, task=task, request_frame=frame, session=session))
        failed = asyncio.run(handlers.execute_plan_step(plan.steps[1], request=request, run=run, task=task, request_frame=frame, session=session))
        replanned = asyncio.run(
            handlers.handle_replan(
                AgentDecision(type=DecisionType.REPLAN, reasoning_summary="当前算法不适用"),
                object(),
                request=request,
                run=run,
                task=task,
                request_frame=frame,
                session=session,
            )
        )
        completed = asyncio.run(handlers.execute_plan_step(session.current_plan.steps[1], request=request, run=run, task=task, request_frame=frame, session=session))
    finally:
        application.main_agent.tool_execution_cycle = original_cycle
        application.main_agent.replanner = original_replanner

    assert inspect.directive is LoopDirective.CONTINUE
    assert failed.directive is LoopDirective.REPLAN
    assert replanned.current_plan is not None
    assert completed.directive is LoopDirective.CONTINUE
    assert completed.completed_steps == ("slope__rev2",)
    assert calls == ["dataset.inspect", "raster.slope", "raster.slope__rev2"]
    assert application.store.get_run(run.id).replan_count == 1
    checkpoint = application.checkpoints.latest(run.id)
    assert checkpoint is not None
    assert checkpoint.state["plan"]["revision"] == 2
