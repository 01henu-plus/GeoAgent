import asyncio
from types import SimpleNamespace

from app.core.models import (
    AgentDecision,
    AgentRequest,
    AgentResultStatus,
    DecisionType,
    InteractionMode,
    Plan,
    RequestFrame,
    Run,
    WorkingMemory,
)
from app.decision.decision_engine import DecisionEngine
from app.decision.model_provider import ModelDecisionProvider
from app.models import ModelAdapter, ModelResponse
from app.runtime.action_dispatcher import RuntimeActionDispatcher
from app.runtime.agent_runtime import RuntimeTransition
from app.runtime.checkpoint_codec import RuntimeCheckpointCodec
from app.runtime.session import AgentRuntimeSession


class _TextModel(ModelAdapter):
    async def complete(self, request):
        return ModelResponse(content="已完成", model="test-model")


def test_runtime_session_is_typed_and_restores_canonical_state_without_mutating_working_memory():
    run = Run(id="run-session", task_id="task-session", agent_id="main")
    memory = WorkingMemory(task_id="task-session", active_dataset_ids=["dem"], constraints=["范围=上海"])
    plan = Plan(goal="分析 DEM", intent="DATA_INSPECTION", steps=[])
    session = AgentRuntimeSession.from_checkpoint(
        run=run,
        datasets=[],
        plan=plan,
        working_memory=memory,
        resume=None,
    )

    session.working_memory.constraints.append("仅输出摘要")

    assert session.run is run
    assert session.current_plan.goal == "分析 DEM"
    assert memory.constraints == ["范围=上海"]
    assert session is not memory


def test_checkpoint_codec_round_trips_canonical_state_and_reads_legacy_aliases():
    run = Run(id="run-codec", task_id="task-codec", agent_id="main", replan_count=2)
    session = AgentRuntimeSession(
        run=run,
        current_plan=Plan(goal="分析", intent="DATA_INSPECTION", steps=[]),
        original_plan=Plan(goal="原始分析", intent="DATA_INSPECTION", steps=[]),
        findings=[{"kind": "finding"}],
        dataset_ids={"dataset-a"},
        artifact_ids={"artifact-a"},
        completed_steps={"step-a"},
        decision_provider="offline",
    )
    request = AgentRequest(user_input="继续", conversation_id="conversation-codec")
    frame = RequestFrame(mode=InteractionMode.CONTINUE_TASK, goal="继续分析")
    payload = RuntimeCheckpointCodec.encode(request, None, frame, session)

    assert all(key in payload for key in RuntimeCheckpointCodec.CANONICAL_FIELDS)
    assert "model_findings" not in payload
    assert "model_dataset_ids" not in payload
    assert "model_artifact_ids" not in payload
    restored = RuntimeCheckpointCodec.decode(payload)
    assert restored.dataset_ids == ["dataset-a"]
    assert restored.current_plan is not None

    legacy = RuntimeCheckpointCodec.decode(
        {
            "messages": [
                {"role": "system", "content": "系统"},
                {"role": "user", "content": "继续"},
                {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {"name": "dataset.inspect", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
            ],
            "model_findings": ["旧观察"],
            "model_dataset_ids": ["legacy-dataset"],
            "model_artifact_ids": ["legacy-artifact"],
            "delegation_result": {"status": "SUCCESS", "summary": "旧委派"},
        }
    )
    assert legacy.protocol_messages[0]["role"] == "assistant"
    assert legacy.findings == ["旧观察"]
    assert legacy.dataset_ids == ["legacy-dataset"]
    assert legacy.legacy_delegation_result["summary"] == "旧委派"


def test_model_decision_provider_returns_only_agent_decision():
    request = AgentRequest(user_input="检查数据", conversation_id="conversation-provider")
    run = Run(id="run-provider", agent_id="main")
    state = SimpleNamespace(run_id=run.id)
    session = AgentRuntimeSession(run=run)
    provider = ModelDecisionProvider(
        decision_engine=DecisionEngine(),
        build_messages=lambda *args, **kwargs: ([{"role": "user", "content": "检查数据"}], [], []),
        refresh_datasets=lambda *args: [],
        resolve_request_resources=lambda _request: SimpleNamespace(),
        max_tokens=128,
    )

    decision = asyncio.run(
        provider.decide(
            state,
            session,
            request=request,
            run=run,
            task=None,
            intent=None,
            request_frame=None,
            model_adapter=_TextModel(),
        )
    )

    assert isinstance(decision, AgentDecision)
    assert decision.type is DecisionType.FINAL
    assert session.last_response is not None


def test_action_dispatcher_returns_transition_without_finalizing_run():
    called = []

    async def handle_plan(decision, state, *, session, **context):
        called.append((decision.type, session.run.id))
        return RuntimeTransition(current_plan=Plan(goal="计划", intent="DATA_INSPECTION", steps=[]))

    dispatcher = RuntimeActionDispatcher(handlers={DecisionType.PLAN: handle_plan})
    run = Run(id="run-dispatch", agent_id="main")
    session = AgentRuntimeSession(run=run)
    decision = AgentDecision(type=DecisionType.PLAN, reasoning_summary="需要计划", plan_goal="计划")

    transition = asyncio.run(dispatcher.dispatch(decision, SimpleNamespace(), session))

    assert transition.terminal is False
    assert called == [(DecisionType.PLAN, "run-dispatch")]
    assert run.status.value == "CREATED"


def test_default_main_agent_uses_controller_as_canonical_entry(application):
    assert not hasattr(application.main_agent, "_model_loop")
    assert not hasattr(application.main_agent, "_execute_plan")
    assert not hasattr(application.main_agent, "_delegate")
    assert application.main_agent.runtime_controller is not None


def test_runtime_controller_lifecycle_records_running_and_waiting_tool(application):
    from app.demo import seed_demo

    ids = seed_demo(application)
    statuses = []
    original_save_run = application.store.save_run

    def record_save(run):
        statuses.append(run.status)
        return original_save_run(run)

    application.store.save_run = record_save
    result = asyncio.run(application.ask("检查道路数据", dataset_ids=[ids["roads"]]))

    assert result.status is AgentResultStatus.SUCCESS
    assert any(status.value == "RUNNING" for status in statuses)
    assert any(status.value == "WAITING_TOOL" for status in statuses)
