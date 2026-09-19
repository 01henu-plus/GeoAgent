from app.core.models import (
    AgentRequest,
    Dataset,
    DatasetKind,
    DecisionType,
    Plan,
    PlanStep,
    RequestFrame,
    Run,
)
from app.decision.planner import Planner
from app.decision.router import AgentRouter
from app.runtime.checkpoint_codec import RuntimeCheckpointCodec
from app.runtime.context_manager import ContextManager
from app.runtime.session import AgentRuntimeSession


def test_planner_consumes_request_frame_without_legacy_intent():
    roads = Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson")
    frame = RequestFrame(
        mode="new_task",
        goal="给 roads 生成 500 米缓冲区",
        capabilities=["vector_analysis", "artifact_write"],
        needs_planning=True,
        needs_tool=True,
    )

    plan = Planner().build(frame, [roads])

    assert plan.metadata["operation"] == "buffer"
    assert plan.steps[1].tool_name == "vector.buffer"
    assert plan.steps[1].arguments["distance"] == 500


def test_router_consumes_request_frame_directly():
    frame = RequestFrame(mode="new_task", goal="检查数据", capabilities=["dataset_inspection"], needs_planning=True, needs_tool=True)
    plan = Plan(
        goal=frame.goal,
        intent="DATA_INSPECTION",
        steps=[PlanStep(id="inspect", title="检查数据", action="dataset.inspect", tool_name="dataset.inspect")],
    )

    decision = AgentRouter().route(frame, plan, [])

    assert decision.type is DecisionType.TOOL
    assert decision.tool_call is not None
    assert decision.tool_call.name == "dataset.inspect"


def test_new_checkpoint_does_not_write_intent_result():
    request = AgentRequest(user_input="检查数据", conversation_id="conversation-convergence")
    frame = RequestFrame(mode="new_task", goal=request.user_input, capabilities=["dataset_inspection"])
    run = Run(task_id="task-convergence", conversation_id=request.conversation_id, agent_id="main")
    session = AgentRuntimeSession(run=run)

    payload = RuntimeCheckpointCodec.encode(request, frame, session)

    assert "intent" not in payload
    assert payload["request_frame"]["goal"] == request.user_input


def test_context_does_not_duplicate_legacy_intent_view():
    request = AgentRequest(user_input="继续检查数据")
    frame = RequestFrame(mode="continue_task", goal="检查数据", capabilities=["dataset_inspection"])

    context = ContextManager().main_context(request, [], None, [], request_frame=frame)

    assert "request_frame" in context
    assert "legacy_intent" not in context
    assert "intent" not in context.get("deterministic_hint", {})
