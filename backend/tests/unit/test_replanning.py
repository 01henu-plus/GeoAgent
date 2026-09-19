import pytest

from app.core.models import (
    FailureAction,
    IntentType,
    InteractionMode,
    LoopDirective,
    Plan,
    PlanStep,
    ReplanContext,
    RequestFrame,
    Run,
    RunBudget,
)
from app.decision.replanner import Replanner, ReplanNotPossible, remaining_plan_fingerprint
from app.runtime.budget import BudgetExceeded, BudgetGuard


def _plan(*, revision=1, failed_id="operate"):
    return Plan(
        goal="处理数据",
        intent=IntentType.SPATIAL_ANALYSIS,
        revision=revision,
        steps=[
            PlanStep(id="inspect", title="检查", action="dataset.inspect", tool_name="dataset.inspect"),
            PlanStep(id=failed_id, title="处理", action="vector.buffer", tool_name="vector.buffer", depends_on=["inspect"], arguments={"dataset_id": "${inspect.dataset_id}"}),
            PlanStep(id="publish", title="发布", action="map.render", tool_name="map.render", depends_on=[failed_id], arguments={"dataset_id": f"${{{failed_id}.dataset_id}}"}),
        ],
    )


def _context(plan, *, failed_step=None):
    return ReplanContext(
        goal=plan.goal,
        original_plan=plan,
        current_plan=plan,
        current_revision=plan.revision,
        completed_steps=["inspect"],
        step_outputs={"inspect": {"dataset_id": "dem"}},
        failed_step=failed_step or plan.steps[1],
        failed_tool_name=plan.steps[1].tool_name,
        failed_arguments=plan.steps[1].arguments,
        error_code="ALGORITHM_NOT_APPLICABLE",
        error_message="算法不适用",
        recovery_action=FailureAction.REPLAN,
        directive=LoopDirective.REPLAN,
        attempts=1,
        current_dataset_ids=["dem"],
        replan_count=1,
        previous_replan_reasons=["operate:ALGORITHM_NOT_APPLICABLE"],
    )


class RevisedPlanner:
    def build(self, request_frame, datasets):
        plan = _plan(revision=2, failed_id="operate__rev2")
        return plan


class SamePlanner:
    def build(self, request_frame, datasets):
        return _plan()


def test_replanner_increments_revision_and_reuses_completed_steps():
    plan = _plan()
    frame = RequestFrame(mode=InteractionMode.NEW_TASK, goal=plan.goal, capabilities=["vector_analysis"], needs_planning=True, needs_tool=True)
    revised = Replanner(RevisedPlanner()).replan(_context(plan), frame, [])

    assert revised.revision == 2
    assert revised.metadata["source"] == "replan"
    assert revised.metadata["completed_steps"] == ["inspect"]
    assert revised.steps[0].id == "inspect"
    assert revised.steps[1].id == "operate__rev2"
    assert revised.steps[2].arguments["dataset_id"] == "${operate__rev2.dataset_id}"


def test_replanner_rejects_same_remaining_plan():
    plan = _plan()

    with pytest.raises(ReplanNotPossible, match="REPLAN_NO_PROGRESS"):
        Replanner(SamePlanner()).replan(_context(plan), RequestFrame(mode=InteractionMode.NEW_TASK, goal=plan.goal, capabilities=["vector_analysis"], needs_planning=True, needs_tool=True), [])


def test_remaining_fingerprint_ignores_completed_steps_but_keeps_dependencies():
    plan = _plan()
    fingerprint = remaining_plan_fingerprint(plan, {"inspect"})

    assert '"tool_name":"dataset.inspect"' not in fingerprint
    assert "operate" in fingerprint


def test_replan_budget_is_independent_from_action_retry_budget():
    run = Run(replan_count=1, agent_id="main")
    guard = BudgetGuard(RunBudget(max_retry_per_action=2, max_replans=1))

    with pytest.raises(BudgetExceeded, match="REPLAN_BUDGET_EXCEEDED"):
        guard.check_replan(run)
