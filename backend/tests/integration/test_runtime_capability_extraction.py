"""Runtime capability extraction 的架构证明测试。"""

import asyncio
import json

from app.core.models import (
    AgentDecision,
    AgentRequest,
    AgentResultStatus,
    DecisionType,
    IntentResult,
    IntentType,
    LoopDirective,
    Plan,
    PlanStep,
    RequestFrame,
    Run,
    RunStatus,
)
from app.demo import seed_demo
from app.models import ModelAdapter, ModelResponse
from app.runtime.action_dispatcher import RuntimeActionDispatcher
from app.runtime.agent_runtime import RuntimeTransition
from app.runtime.session import AgentRuntimeSession


def _forbid_legacy_runtime_methods(application) -> None:
    main_agent = application.main_agent

    def forbidden(*_args, **_kwargs):
        raise AssertionError("canonical runtime 不应调用 MainAgent 旧动作方法")

    for name in (
        "_dispatch_runtime_decision",
        "_execute_runtime_plan_step",
        "_dispatch_runtime_replan",
        "_execute_delegation",
    ):
        setattr(main_agent, name, forbidden)
    # Dispatcher 在初始化时保存过兼容回调；显式替换它才能证明 canonical
    # handler 没有在缺少注册时静默回退到旧路径。
    main_agent.runtime_action_dispatcher.compatibility_handler = forbidden


def test_dispatcher_uses_registered_handler_before_compatibility_fallback():
    called: list[str] = []

    async def canonical(decision, state, *, session, **context):
        called.append("canonical")
        return RuntimeTransition(observation={"source": "runtime"})

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("不应调用 compatibility_handler")

    run = Run(id="runtime-dispatch-proof", agent_id="main")
    session = AgentRuntimeSession(run=run)
    dispatcher = RuntimeActionDispatcher(
        handlers={DecisionType.TOOL: canonical},
        compatibility_handler=forbidden,
    )
    decision = AgentDecision(type=DecisionType.TOOL, reasoning_summary="执行工具")

    transition = asyncio.run(dispatcher.dispatch(decision, object(), session))

    assert transition.observation == {"source": "runtime"}
    assert called == ["canonical"]


def test_offline_tool_and_plan_use_runtime_handlers_not_mainagent_actions(application):
    ids = seed_demo(application)
    _forbid_legacy_runtime_methods(application)

    tool_result = asyncio.run(application.ask("检查道路数据", dataset_ids=[ids["roads"]]))
    plan_result = asyncio.run(application.ask("检查道路并生成 500 米缓冲区", dataset_ids=[ids["roads"]]))

    assert tool_result.status is AgentResultStatus.SUCCESS
    assert plan_result.status is AgentResultStatus.SUCCESS


def test_offline_delegation_uses_runtime_handler_not_mainagent_delegation(application):
    ids = seed_demo(application)
    _forbid_legacy_runtime_methods(application)

    result = asyncio.run(application.ask("综合道路、人口和 DEM，从三个方面评价当前区域", dataset_ids=list(ids.values())))

    assert result.status is AgentResultStatus.SUCCESS
    assert any(event.event_type == "DelegationCompleted" for event in application.store.list_events(result.trace_id))


def test_model_tool_path_uses_runtime_handler_not_mainagent_dispatch(application):
    ids = seed_demo(application)

    class Model(ModelAdapter):
        def __init__(self):
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    model="proof-model",
                    tool_calls=[
                        {
                            "id": "proof-inspect",
                            "function": {
                                "name": "dataset.inspect",
                                "arguments": json.dumps({"dataset_id": ids["roads"]}),
                            },
                        }
                    ],
                )
            return ModelResponse(model="proof-model", content="模型路径已完成。")

    application.main_agent.model_adapter = Model()
    _forbid_legacy_runtime_methods(application)

    result = asyncio.run(application.ask("请检查道路数据", dataset_ids=[ids["roads"]]))

    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary == "模型路径已完成。"


def test_replan_handler_is_runtime_owned(application):
    task = application.task_service.create("运行时重规划证明", conversation_id="runtime-replan-proof")
    run = Run(id="runtime-replan-run", task_id=task.id, conversation_id=task.conversation_id, agent_id="main", status=RunStatus.RUNNING)
    application.store.save_run(run)
    plan = Plan(
        goal="运行时重规划证明",
        intent=IntentType.SPATIAL_ANALYSIS,
        steps=[PlanStep(id="failed-step", title="失败步骤", action="分析", tool_name="raster.slope")],
    )
    session = AgentRuntimeSession(
        run=run,
        datasets=[],
        current_plan=plan,
        original_plan=plan.model_copy(deep=True),
        latest_failure={
            "step_id": "failed-step",
            "tool_name": "raster.slope",
            "error_code": "ALGORITHM_NOT_APPLICABLE",
            "error": "当前算法不适用",
            "directive": LoopDirective.REPLAN.value,
            "deterministic": True,
        },
    )

    class Replanner:
        def replan(self, context, intent, datasets):
            return Plan(
                goal=context.goal,
                intent=intent.intent,
                revision=context.current_revision + 1,
                steps=[PlanStep(id="replacement", title="替代步骤", action="检查", tool_name="dataset.inspect")],
            )

    application.main_agent.replanner = Replanner()
    _forbid_legacy_runtime_methods(application)
    request = AgentRequest(user_input="请重新规划", conversation_id=task.conversation_id)
    frame = RequestFrame(mode="modify_task", goal="运行时重规划证明")
    decision = AgentDecision(type=DecisionType.REPLAN, reasoning_summary="当前步骤不适用")

    transition = asyncio.run(
        application.main_agent.runtime_action_handlers.handle_replan(
            decision,
            object(),
            request=request,
            run=run,
            task=task,
            intent=IntentResult(intent=IntentType.SPATIAL_ANALYSIS, confidence=1),
            request_frame=frame,
            session=session,
        )
    )

    assert transition.current_plan is not None
    assert transition.current_plan.revision == 2
    assert transition.directive is LoopDirective.CONTINUE
