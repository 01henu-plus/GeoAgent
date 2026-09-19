import asyncio
import json

from app.application import Application
from app.checkpoint.context import make_checkpoint
from app.config import Settings
from app.core.models import (
    AgentRequest,
    AgentResultStatus,
    IntentType,
    LoopDirective,
    Plan,
    PlanStep,
    Run,
    RunStatus,
    TaskStatus,
    ToolResult,
    ToolStatus,
)
from app.demo import seed_demo
from app.models import ModelAdapter, ModelRequest, ModelResponse, ModelStreamChunk
from app.models.config import ModelProfile
from app.runtime.checkpoint_codec import RuntimeCheckpointCodec
from app.runtime.context_assembler import estimate_tokens
from app.runtime.session import AgentRuntimeSession
from app.runtime.tool_execution_cycle import ExecutionOutcome


class FakeToolModel(ModelAdapter):
    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                model="fake",
                tool_calls=[
                    {
                        "id": "call_inspect",
                        "type": "function",
                        "function": {"name": "dataset.inspect", "arguments": json.dumps({"dataset_id": self.dataset_id})},
                    }
                ],
            )
        return ModelResponse(model="fake", content="模型已根据工具结果完成检查。")


def test_model_loop_passes_tools_and_executes_tool(application):
    ids = seed_demo(application)
    fake = FakeToolModel(ids["roads"])
    application.main_agent.model_adapter = fake

    result = asyncio.run(application.ask("请检查道路数据", dataset_ids=[ids["roads"]]))

    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary == "模型已根据工具结果完成检查。"
    assert fake.requests[0].tools
    assert any(item["function"]["name"] == "dataset.inspect" for item in fake.requests[0].tools)
    assert "input_schema" not in fake.requests[0].messages[1]["content"]
    assert estimate_tokens({"messages": fake.requests[0].messages, "tools": fake.requests[0].tools}) <= application.budget.model_input_tokens
    assert any(message.get("role") == "tool" for message in fake.requests[1].messages)
    assert application.store.get_run(result.trace_id).turn_count == 2

    first_context = json.loads(fake.requests[0].messages[1]["content"].split("\n", 1)[1])
    second_context = json.loads(fake.requests[1].messages[1]["content"].split("\n", 1)[1])
    assert first_context["run_state"]["turn_count"] == 1
    assert second_context["run_state"]["turn_count"] == 2


def test_model_multi_tool_observation_keeps_every_execution_outcome(application):
    ids = seed_demo(application)

    class MultiToolModel(ModelAdapter):
        def __init__(self):
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    model="fake",
                    tool_calls=[
                        {"id": "call-a", "function": {"name": "dataset.inspect", "arguments": json.dumps({"dataset_id": ids["roads"]})}},
                        {"id": "call-b", "function": {"name": "dataset.inspect", "arguments": json.dumps({"dataset_id": ids["population"]})}},
                        {"id": "call-c", "function": {"name": "dataset.inspect", "arguments": json.dumps({"dataset_id": ids["roads"]})}},
                    ],
                )
            return ModelResponse(model="fake", content="三个工具结果都已收到。")

    model = MultiToolModel()
    application.main_agent.model_adapter = model

    result = asyncio.run(application.ask("检查道路和建筑数据"))

    assert result.status is AgentResultStatus.SUCCESS
    second_context = json.loads(model.requests[1].messages[1]["content"].split("\n", 1)[1])
    observations = second_context["current_observation"]["tool_observations"]
    assert [item["call_id"] for item in observations] == ["call-a", "call-b", "call-c"]
    assert all(item["accepted"] is True and item["verified"] is True for item in observations)
    tool_messages = [message for message in model.requests[1].messages if message.get("role") == "tool"]
    assert len(tool_messages) == 3
    assert all(json.loads(message["content"])["accepted"] is True for message in tool_messages)
    checkpoint = application.checkpoints.latest(result.trace_id)
    assert checkpoint is not None
    assert len(checkpoint.state["latest_observation"]["tool_observations"]) == 3


def test_model_multi_tool_observation_keeps_failed_and_successful_results_together(application):
    ids = seed_demo(application)

    class MixedToolModel(ModelAdapter):
        def __init__(self):
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    model="fake",
                    tool_calls=[
                        {"id": "call-failed", "function": {"name": "raster.slope", "arguments": "{}"}},
                        {"id": "call-ok-a", "function": {"name": "dataset.inspect", "arguments": json.dumps({"dataset_id": ids["roads"]})}},
                        {"id": "call-ok-b", "function": {"name": "dataset.inspect", "arguments": json.dumps({"dataset_id": ids["population"]})}},
                    ],
                )
            return ModelResponse(model="fake", content="已收到混合执行结果。")

    model = MixedToolModel()
    application.main_agent.model_adapter = model
    original_raw_executor = application.main_agent.tool_execution_cycle.raw_executor

    async def fake_raw_tool(current_run, name, arguments, *, call_id=None, attempt=1):
        if name == "raster.slope":
            return ToolResult(call_id=call_id or "call-failed", status=ToolStatus.SUCCESS, datasets=["missing-output"])
        return ToolResult(call_id=call_id or name, status=ToolStatus.SUCCESS, output={"ok": True})

    application.main_agent.tool_execution_cycle.raw_executor = fake_raw_tool
    try:
        result = asyncio.run(application.ask("同时检查数据并计算坡度"))
    finally:
        application.main_agent.tool_execution_cycle.raw_executor = original_raw_executor

    assert result.status is AgentResultStatus.SUCCESS
    second_context = json.loads(model.requests[1].messages[1]["content"].split("\n", 1)[1])
    observations = second_context["current_observation"]["tool_observations"]
    assert observations[0]["call_id"] == "call-failed"
    assert observations[0]["accepted"] is False
    assert observations[0]["verified"] is False
    assert observations[1]["accepted"] is True
    assert observations[2]["accepted"] is True
    assert "missing-output" not in application.store.get_working_memory(result.task_id).active_dataset_ids


def test_planner_path_uses_tool_execution_cycle(application):
    ids = seed_demo(application)
    calls = []
    original = application.main_agent.tool_execution_cycle.execute

    async def wrapped(*args, **kwargs):
        calls.append(args[1])
        return await original(*args, **kwargs)

    application.main_agent.tool_execution_cycle.execute = wrapped
    result = asyncio.run(application.ask("检查道路数据", dataset_ids=[ids["roads"]]))

    assert result.status is AgentResultStatus.SUCCESS
    assert "dataset.inspect" in calls


def test_model_path_verification_failure_is_not_accepted_into_working_memory(application):
    class VerificationModel(ModelAdapter):
        def __init__(self):
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    model="fake",
                    tool_calls=[
                        {
                            "id": "bad-output",
                            "function": {"name": "raster.slope", "arguments": "{}"},
                        }
                    ],
                )
            return ModelResponse(model="fake", content="输出未通过验证，需要重新确认数据。")

    model = VerificationModel()
    application.main_agent.model_adapter = model
    original_raw_executor = application.main_agent.tool_execution_cycle.raw_executor

    async def fake_raw_tool(current_run, name, arguments, *, call_id=None, attempt=1):
        return ToolResult(call_id=call_id or "bad-output", status=ToolStatus.SUCCESS, datasets=["missing-output"])

    application.main_agent.tool_execution_cycle.raw_executor = fake_raw_tool
    try:
        result = asyncio.run(application.ask("计算坡度"))
    finally:
        application.main_agent.tool_execution_cycle.raw_executor = original_raw_executor

    assert result.summary == "输出未通过验证，需要重新确认数据。"
    second_context = model.requests[1].messages[1]["content"]
    assert "accepted" in second_context
    assert "结果引用了未知 Dataset" in second_context
    assert "missing-output" not in application.store.get_working_memory(result.task_id).active_dataset_ids


def test_main_agent_planner_path_maps_loop_directives_without_replanning(application):
    task = application.task_service.create("执行控制信号测试", conversation_id="directive-conversation")
    run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(run)
    request = AgentRequest(user_input="执行控制信号测试", conversation_id=task.conversation_id)
    plan = Plan(
        goal="执行控制信号测试",
        intent=IntentType.DATA_INSPECTION,
        steps=[PlanStep(id="step-directive", title="检查", action="inspect", tool_name="dataset.inspect")],
    )

    async def run_with_directive(directive):
        outcome = ExecutionOutcome(
            result=ToolResult(call_id="directive-call", status=ToolStatus.FAILED),
            verified=False,
            verification_problems=[],
            recovery_action=None,
            attempts=1,
            accepted=False,
            directive=directive,
            rationale="执行层返回控制信号",
        )
        original = application.main_agent.tool_execution_cycle.execute
        async def execute(*args, **kwargs):
            return outcome

        application.main_agent.tool_execution_cycle.execute = execute
        current_plan = plan.model_copy(deep=True)
        session = AgentRuntimeSession(
            run=run,
            datasets=[],
            current_plan=current_plan,
            original_plan=current_plan.model_copy(deep=True),
        )
        try:
            return await application.main_agent.runtime_action_handlers.execute_plan_step(
                current_plan.steps[0],
                request=request,
                run=run,
                task=task,
                request_frame=None,
                session=session,
            )
        finally:
            application.main_agent.tool_execution_cycle.execute = original

    ask_user = asyncio.run(run_with_directive(LoopDirective.ASK_USER))
    replan = asyncio.run(run_with_directive(LoopDirective.REPLAN))
    abort = asyncio.run(run_with_directive(LoopDirective.ABORT))

    assert ask_user.directive is LoopDirective.ASK_USER
    assert replan.directive is LoopDirective.REPLAN
    assert abort.directive is LoopDirective.ABORT
    assert all(item.latest_failure is not None for item in (ask_user, replan, abort))


def test_model_runtime_gets_first_chance_when_offline_rules_would_ask(application):
    class ClarifyingModel(ModelAdapter):
        def __init__(self):
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            return ModelResponse(model="fake", content="请告诉我需要计算的距离，或说明使用哪个字段。")

    model = ClarifyingModel()
    application.main_agent.model_adapter = model

    result = asyncio.run(application.ask("帮我计算两个数据集之间的距离"))

    assert result.summary.startswith("请告诉我")
    assert len(model.requests) == 1
    assert not any(event.event_type == "ToolStarted" for event in application.store.list_events(result.trace_id))


def test_model_profiles_api_is_available_without_exposing_key(application, authenticated_client):
    profile = ModelProfile(
        id="local-qwen",
        label="本地千问",
        base_url="http://127.0.0.1:11434/v1",
        api_key="secret-key",
        model="qwen2.5:7b",
        default=True,
    )

    class ProfileModel(ModelAdapter):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(model=profile.model, content="连接配置已读取。")

    adapter = ProfileModel()
    application.model_profiles[profile.id] = profile
    application.model_adapters[profile.id] = adapter
    application.default_model_profile = profile.id
    application.model_adapter = adapter
    application.main_agent.model_adapters = application.model_adapters
    application.main_agent.default_model_profile = profile.id
    application.main_agent.model_adapter = adapter

    with authenticated_client as client:
        response = client.get("/api/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["default_profile"] == profile.id
    assert payload["profiles"][0]["label"] == profile.label
    assert payload["profiles"][0]["has_api_key"] is True
    assert "api_key" not in payload["profiles"][0]


def test_request_uses_selected_model_profile(application):
    class ReplyModel(ModelAdapter):
        def __init__(self, model: str) -> None:
            self.model = model
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            return ModelResponse(model=self.model, content=f"当前使用{self.model}。")

    first = ReplyModel("模型一")
    second = ReplyModel("模型二")
    application.model_adapters.update({"one": first, "two": second})
    application.default_model_profile = "one"
    application.model_adapter = first
    application.main_agent.model_adapters = application.model_adapters
    application.main_agent.default_model_profile = "one"
    application.main_agent.model_adapter = first

    result = asyncio.run(application.ask(AgentRequest(user_input="你好", model_profile="two")))

    assert result.summary == "当前使用模型二。"
    assert not first.requests
    assert len(second.requests) == 1


def test_model_profiles_load_from_environment_settings(tmp_path):
    settings = Settings(
        root=tmp_path,
        database=tmp_path / "state.sqlite3",
        workspace=tmp_path / "workspace",
        model_profiles=json.dumps(
            [
                {
                    "id": "one",
                    "label": "模型一",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "model": "qwen2.5:7b",
                    "default": True,
                },
                {
                    "id": "two",
                    "label": "模型二",
                    "base_url": "http://127.0.0.1:11434/v1",
                    "model": "llama3.2:3b",
                },
            ]
        ),
    )
    application = Application(settings)
    try:
        status = application.model_status()
    finally:
        asyncio.run(application.close())

    assert [item["id"] for item in status["profiles"]] == ["one", "two"]
    assert status["default_profile"] == "one"


def test_resume_uses_saved_checkpoint_plan(application, authenticated_client):
    ids = seed_demo(application)
    with authenticated_client:
        user_id = application.store.get_user_by_username("test-user").id
    conversation = application.conversations.create("恢复测试", user_id=user_id)
    request = AgentRequest(user_input="检查 roads", conversation_id=conversation.id, user_id=user_id, dataset_ids=[ids["roads"]])
    prepared = asyncio.run(application.main_agent.prepare_request(request))
    assert prepared.task is not None and prepared.run is not None
    old_run = prepared.run
    old_run = old_run.model_copy(update={"status": RunStatus.CANCELLED})
    application.store.save_run(old_run)
    datasets = application.main_agent._resolve_datasets(request)
    frame = prepared.frame
    plan = application.main_agent.planner.build(frame, datasets)
    session = AgentRuntimeSession(
        run=old_run,
        datasets=datasets,
        current_plan=plan,
        original_plan=plan.model_copy(deep=True),
    )
    checkpoint_state = RuntimeCheckpointCodec.encode(request, prepared.frame, session)
    checkpoint = make_checkpoint(old_run.id, "plan_created", checkpoint_state)
    application.checkpoints.save(checkpoint)

    with authenticated_client as client:
        response = client.post(f"/api/v1/runs/{old_run.id}/resume")

    assert response.status_code == 200
    payload = response.json()
    assert payload["resumed_from"] == old_run.id
    assert payload["run_id"] != old_run.id
    assert any(event.event_type == "ResumeStarted" for event in application.store.list_events(payload["run_id"]))


def test_run_manager_cancels_active_run(application):
    seed_demo(application)

    class WaitingModel(ModelAdapter):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            await asyncio.Event().wait()

    application.main_agent.model_adapter = WaitingModel()
    request = AgentRequest(user_input="等待取消", conversation_id="conv_cancel")

    async def run_case():
        run = await application.conversations.submit(request)
        await asyncio.sleep(0)
        assert await application.run_manager.cancel(run.id)
        return run.id, await application.run_manager.wait(run.id)

    run_id, result = asyncio.run(run_case())

    assert result.status is AgentResultStatus.CANCELLED
    assert application.store.get_run(run_id).status is RunStatus.CANCELLED


def test_run_manager_cancels_persisted_active_run_without_local_task(application):
    task = application.task_service.create("持久化取消", conversation_id="conv-db-cancel")
    run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main", status=RunStatus.RUNNING)
    application.store.save_run(run)

    assert asyncio.run(application.run_manager.cancel(run.id)) is True
    assert application.store.get_run(run.id).status is RunStatus.CANCELLED
    assert application.store.get_task(task.id).status is TaskStatus.CANCELLED


def test_run_manager_does_not_cancel_completed_run(application):
    run = Run(agent_id="main", status=RunStatus.COMPLETED)
    application.store.save_run(run)

    assert asyncio.run(application.run_manager.cancel(run.id)) is False
    assert application.store.get_run(run.id).status is RunStatus.COMPLETED


def test_websocket_streams_run_events_before_result(application, authenticated_client):
    seed_demo(application)
    with authenticated_client as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "ask", "message": "检查 roads"})
            messages = []
            while True:
                item = socket.receive_json()
                messages.append(item)
                if item["type"] == "result":
                    break

    assert messages[0]["type"] == "run"
    assert any(item["type"] == "event" for item in messages)
    assert messages[-1]["type"] == "result"
    assert messages[-1]["data"]["status"] == "SUCCESS"


def test_websocket_streams_model_deltas(application, authenticated_client):
    class StreamingModel(ModelAdapter):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(model="fake", content="第一段第二段")

        async def stream(self, request: ModelRequest):
            yield ModelStreamChunk(model="fake", content="第一段")
            yield ModelStreamChunk(model="fake", content="第二段", done=True)

    application.main_agent.model_adapter = StreamingModel()
    with authenticated_client as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "ask", "message": "你好"})
            messages = []
            while True:
                item = socket.receive_json()
                messages.append(item)
                if item["type"] == "result":
                    break

    assert [item["content"] for item in messages if item["type"] == "delta"] == ["第一段", "第二段"]
