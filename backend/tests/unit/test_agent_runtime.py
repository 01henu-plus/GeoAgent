import asyncio

from app.core.models import (
    AgentDecision,
    AgentRequest,
    AgentResultStatus,
    DecisionType,
    InteractionMode,
    Plan,
    PlanStep,
    RequestFrame,
    Run,
    WorkingMemory,
)
from app.decision import DecisionEngine
from app.models import ModelResponse
from app.runtime.agent_runtime import AgentRuntime, RuntimeTransition
from app.runtime.agent_state import AgentState, AgentStateBuilder
from app.runtime.context_manager import ContextManager


def _state() -> AgentState:
    request = AgentRequest(user_input="检查 DEM", conversation_id="conversation-runtime")
    frame = RequestFrame(mode=InteractionMode.NEW_TASK, goal="检查 DEM")
    return AgentState(request=request, request_frame=frame, run_id="run-runtime", goal="检查 DEM")


def test_decision_engine_adapts_model_text_and_multi_tool_calls():
    engine = DecisionEngine()
    final = engine.from_model_response(ModelResponse(model="fake", content="已完成。"))
    assert final.type is DecisionType.FINAL
    assert final.final_response == "已完成。"

    decision = engine.from_model_response(
        ModelResponse(
            model="fake",
            tool_calls=[
                {"id": "call-a", "function": {"name": "dataset.inspect", "arguments": '{"dataset_id":"a"}'}},
                {"id": "call-b", "function": {"name": "dataset.inspect", "arguments": '{"dataset_id":"b"}'}},
            ],
        )
    )
    assert decision.type is DecisionType.TOOL
    assert [item.id for item in decision.normalized_tool_calls()] == ["call-a", "call-b"]
    assert decision.tool_call is not None and decision.tool_call.id == "call-a"


def test_decision_engine_keeps_invalid_call_observable_without_executing_it():
    decision = DecisionEngine().from_model_response(
        ModelResponse(tool_calls=[{"id": "bad", "function": {"name": "dataset.inspect", "arguments": "not-json"}}])
    )
    assert decision.type is DecisionType.TOOL
    assert decision.normalized_tool_calls()[0].arguments == {}
    assert decision.metadata["invalid_tool_calls"][0]["id"] == "bad"


def test_decision_engine_state_gate_emits_plan_without_executing_it():
    state = _state().model_copy(
        update={
            "request_frame": RequestFrame(mode=InteractionMode.NEW_TASK, goal="分析 DEM", needs_planning=True),
            "goal": "分析 DEM",
        }
    )
    decision = DecisionEngine().decide(state)
    assert decision.type is DecisionType.PLAN
    assert decision.plan_goal == "分析 DEM"


def test_decision_engine_returns_none_when_no_deterministic_action_exists():
    assert DecisionEngine().decide(_state()) is None


def test_decision_engine_maps_internal_control_capabilities():
    engine = DecisionEngine()
    plan = engine.from_model_response(
        ModelResponse(tool_calls=[{"id": "plan", "function": {"name": "agent.plan", "arguments": '{"goal":"分析 DEM"}'}}])
    )
    ask_user = engine.from_model_response(
        ModelResponse(tool_calls=[{"id": "ask", "function": {"name": "agent.ask_user", "arguments": '{"question":"请提供距离"}'}}])
    )
    assert plan.type is DecisionType.PLAN
    assert plan.plan_goal == "分析 DEM"
    assert ask_user.type is DecisionType.ASK_USER
    assert ask_user.final_response == "请提供距离"


def test_decision_engine_rejects_mixed_control_and_gis_batch():
    decision = DecisionEngine().from_model_response(
        ModelResponse(
            tool_calls=[
                {"id": "plan", "function": {"name": "agent.plan", "arguments": "{}"}},
                {"id": "inspect", "function": {"name": "dataset.inspect", "arguments": "{}"}},
            ]
        )
    )
    assert decision.type is DecisionType.ABORT
    assert decision.metadata["error_code"] == "INVALID_AGENT_DECISION_BATCH"


def test_decision_engine_maps_runtime_failure_directives_only_when_deterministic():
    engine = DecisionEngine()
    state = _state().model_copy(update={"latest_failure": {"directive": "REPLAN", "deterministic": True, "error": "失败"}})
    assert engine.decide(state).type is DecisionType.REPLAN

    non_deterministic = state.model_copy(update={"latest_failure": {"directive": "REPLAN", "deterministic": False}})
    assert engine.decide(non_deterministic) is None


def test_agent_state_builder_reads_fresh_working_memory_without_mutating_store(application):
    task = application.task_service.create("分析 DEM", conversation_id="conversation-state")
    run = Run(id="run-state", task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(run)
    memory = WorkingMemory(task_id=task.id, conversation_id=task.conversation_id, active_dataset_ids=["dem"], constraints=["范围=上海"])
    application.store.save_working_memory(memory)
    request = AgentRequest(user_input="继续", conversation_id=task.conversation_id)
    frame = RequestFrame(mode=InteractionMode.CONTINUE_TASK, goal="继续分析 DEM", target_task_id=task.id)

    state = AgentStateBuilder(application.store).build(request, frame, run, task=task)
    state.active_dataset_ids.append("mutated-only-in-view")
    state.working_memory.constraints.append("不应写回")
    stored = application.store.get_working_memory(task.id)
    assert stored is not None
    assert stored.active_dataset_ids == ["dem"]
    assert stored.constraints == ["范围=上海"]

    updated = stored.model_copy(update={"active_dataset_ids": ["dem", "slope"]})
    application.store.save_working_memory(updated)
    refreshed = AgentStateBuilder(application.store).build(request, frame, run, task=task)
    assert refreshed.active_dataset_ids == ["dem", "slope"]


def test_agent_runtime_dispatches_final_ask_user_and_abort():
    async def run_case(decision: AgentDecision):
        return await AgentRuntime().run(
            _state(),
            decide=lambda _state: decision,
            dispatch=lambda item, _state: RuntimeTransition(
                terminal=True,
                status={
                    DecisionType.FINAL: AgentResultStatus.SUCCESS,
                    DecisionType.ASK_USER: AgentResultStatus.BLOCKED,
                    DecisionType.ABORT: AgentResultStatus.FAILED,
                }[item.type],
                final_response=item.final_response,
                error="WAITING_USER" if item.type is DecisionType.ASK_USER else None,
            ),
        )

    final = asyncio.run(run_case(AgentDecision(type=DecisionType.FINAL, reasoning_summary="完成", final_response="完成")))
    ask = asyncio.run(run_case(AgentDecision(type=DecisionType.ASK_USER, reasoning_summary="需要信息", final_response="请补充")))
    abort = asyncio.run(run_case(AgentDecision(type=DecisionType.ABORT, reasoning_summary="终止")))
    assert final.status is AgentResultStatus.SUCCESS
    assert ask.status is AgentResultStatus.BLOCKED
    assert abort.status is AgentResultStatus.FAILED


def test_agent_runtime_plan_fast_path_executes_one_transition_before_decision():
    plan = Plan(goal="执行计划", intent="DATA_INSPECTION", steps=[PlanStep(id="step-1", title="检查", action="inspect")])
    state = _state().model_copy(update={"current_plan": plan})
    calls: list[str] = []

    async def fast_path(current):
        calls.append("fast")
        return RuntimeTransition(current_plan=current.current_plan, completed_steps=("step-1",), clear_plan=True)

    async def decide(_state):
        calls.append("decide")
        return AgentDecision(type=DecisionType.FINAL, reasoning_summary="完成", final_response="计划已完成")

    async def dispatch(_decision, _state):
        return RuntimeTransition(terminal=True, status=AgentResultStatus.SUCCESS, final_response="计划已完成")

    result = asyncio.run(AgentRuntime().run(state, decide=decide, dispatch=dispatch, fast_path=fast_path))
    assert result.status is AgentResultStatus.SUCCESS
    assert calls == ["fast", "decide"]


def test_agent_runtime_separates_transition_safety_from_model_turns():
    state = _state().model_copy(update={"current_plan": Plan(goal="持续执行", intent="DATA_INSPECTION", steps=[])})
    fast_calls = 0
    decision_calls = 0

    async def fast_path(current):
        nonlocal fast_calls
        fast_calls += 1
        if fast_calls == 4:
            return RuntimeTransition(clear_plan=True)
        return RuntimeTransition(current_plan=current.current_plan)

    async def decide(_state):
        nonlocal decision_calls
        decision_calls += 1
        return AgentDecision(type=DecisionType.FINAL, reasoning_summary="完成", final_response="完成")

    async def dispatch(_decision, _state):
        return RuntimeTransition(terminal=True, status=AgentResultStatus.SUCCESS, final_response="完成")

    result = asyncio.run(
        AgentRuntime(max_runtime_transitions=8).run(
            state,
            decide=decide,
            dispatch=dispatch,
            fast_path=fast_path,
        )
    )
    assert result.status is AgentResultStatus.SUCCESS
    assert result.iterations == 5
    assert fast_calls == 4
    assert decision_calls == 1
    assert result.state is not None and result.state.turn_count == 0


def test_agent_runtime_carries_subagent_results_into_next_decision():
    decision_count = 0

    async def decide(state):
        nonlocal decision_count
        decision_count += 1
        if state.subagent_results:
            return AgentDecision(type=DecisionType.FINAL, reasoning_summary="汇总", final_response="汇总完成")
        return AgentDecision(type=DecisionType.DELEGATE, reasoning_summary="委派")

    async def dispatch(decision, _state):
        if decision.type is DecisionType.DELEGATE:
            return RuntimeTransition(
                observation={"type": "delegation", "completed": 1, "total": 1},
                subagent_results=({"subtask_id": "sub-1", "status": "SUCCESS", "summary": "已完成"},),
            )
        return RuntimeTransition(terminal=True, status=AgentResultStatus.SUCCESS, final_response=decision.final_response)

    result = asyncio.run(AgentRuntime().run(_state(), decide=decide, dispatch=dispatch))
    assert result.status is AgentResultStatus.SUCCESS
    assert decision_count == 2
    assert result.state is not None
    assert result.state.subagent_results[0]["subtask_id"] == "sub-1"


def test_context_exposes_plan_summary_progress_and_latest_failure():
    plan = Plan(
        goal="先检查再分析",
        intent="DATA_INSPECTION",
        steps=[
            PlanStep(id="inspect", title="检查数据", action="inspect", tool_name="dataset.inspect"),
            PlanStep(id="analyze", title="分析数据", action="analyze", tool_name="raster.slope", depends_on=["inspect"]),
        ],
    )
    context = ContextManager(max_tokens=2000).main_context(
        AgentRequest(user_input="继续分析"),
        [],
        plan,
        [],
        plan_progress={"completed_steps": ["inspect"], "step_outputs": {"inspect": {"dataset_id": "dem"}}},
        latest_failure={"action": "REPLAN", "error": "结果验证失败"},
    )
    assert context["plan_state"]["completed_steps"] == ["inspect"]
    assert context["plan_state"]["steps"][1]["status"] != "SUCCEEDED"
    assert context["latest_failure"]["action"] == "REPLAN"
