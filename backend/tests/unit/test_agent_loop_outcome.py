import asyncio

from app.core.models import LoopDirective, Plan, PlanStep, ToolError, ToolResult, ToolStatus
from app.runtime.agent_loop import AgentLoop
from app.runtime.tool_execution_cycle import ExecutionOutcome


def _outcome(*, accepted: bool, directive: LoopDirective, status: ToolStatus = ToolStatus.SUCCESS) -> ExecutionOutcome:
    result = ToolResult(
        call_id="call-1",
        status=status,
        output="完成" if accepted else None,
        error=None if accepted else ToolError(code="TEST_FAILURE", message="测试失败"),
    )
    return ExecutionOutcome(
        result=result,
        verified=accepted,
        verification_problems=[] if accepted else ["验证失败"],
        recovery_action=None,
        attempts=1,
        accepted=accepted,
        directive=directive,
        rationale=None if accepted else "测试执行失败",
    )


def _plan() -> Plan:
    return Plan(
        goal="测试计划",
        intent="DATA_INSPECTION",
        steps=[PlanStep(id="step-1", title="检查数据", action="inspect", tool_name="dataset.inspect")],
    )


def _run(executor: ExecutionOutcome):
    checkpoints = []

    async def execute_step(step, arguments):
        return executor

    async def checkpoint(completed, state):
        checkpoints.append((completed, state))

    outcome = asyncio.run(
        AgentLoop().execute_plan(
            _plan(),
            execute_step=execute_step,
            checkpoint=checkpoint,
        )
    )
    return outcome, checkpoints


def test_agent_loop_consumes_execution_outcome_without_second_verifier():
    outcome, _ = _run(_outcome(accepted=True, directive=LoopDirective.CONTINUE))

    assert outcome.completed_steps == {"step-1"} or outcome.completed_steps == frozenset({"step-1"})
    assert outcome.failed_outcome is None
    assert outcome.directive is LoopDirective.CONTINUE
    assert outcome.findings[0]["accepted"] is True


def test_agent_loop_does_not_accept_success_status_when_outcome_rejected():
    outcome, _ = _run(_outcome(accepted=False, directive=LoopDirective.ABORT))

    assert not outcome.completed_steps
    assert outcome.failed_outcome is not None
    assert outcome.failed_outcome.accepted is False
    assert outcome.directive is LoopDirective.ABORT
    assert outcome.findings[0]["status"] == ToolStatus.SUCCESS.value


def test_agent_loop_propagates_ask_user_and_replan_directives():
    ask_user, _ = _run(_outcome(accepted=False, directive=LoopDirective.ASK_USER))
    replan, _ = _run(_outcome(accepted=False, directive=LoopDirective.REPLAN))

    assert ask_user.directive is LoopDirective.ASK_USER
    assert ask_user.failed_outcome is not None
    assert replan.directive is LoopDirective.REPLAN
    assert replan.failed_outcome is not None
