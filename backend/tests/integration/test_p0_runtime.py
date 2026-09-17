import asyncio
import json

from fastapi.testclient import TestClient

from app.api import create_app
from app.application import Application
from app.checkpoint.context import make_checkpoint
from app.config import Settings
from app.core.models import AgentRequest, AgentResultStatus, RunStatus
from app.demo import seed_demo
from app.models import ModelAdapter, ModelRequest, ModelResponse, ModelStreamChunk
from app.models.config import ModelProfile


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
    assert any(message.get("role") == "tool" for message in fake.requests[1].messages)
    assert application.store.get_run(result.trace_id).turn_count == 2


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


def test_model_profiles_api_is_available_without_exposing_key(application):
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

    with TestClient(create_app(application)) as client:
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


def test_resume_uses_saved_checkpoint_plan(application):
    ids = seed_demo(application)
    request = AgentRequest(user_input="检查 roads", conversation_id="conv_resume", dataset_ids=[ids["roads"]])
    task, old_run = application.main_agent.prepare(request)
    old_run = old_run.model_copy(update={"status": RunStatus.CANCELLED})
    application.store.save_run(old_run)
    datasets = application.main_agent._resolve_datasets(request)
    intent = application.main_agent.intent_resolver.resolve(request, datasets)
    plan = application.main_agent.planner.build(request.user_input, intent, datasets)
    checkpoint = make_checkpoint(old_run.id, "plan_created", application.main_agent._checkpoint_state(request, intent, plan, datasets))
    application.checkpoints.save(checkpoint)

    with TestClient(create_app(application)) as client:
        response = client.post(f"/api/v1/runs/{old_run.id}/resume")

    assert response.status_code == 200
    payload = response.json()
    assert payload["resumed_from"] == old_run.id
    assert payload["run_id"] != old_run.id
    assert any(event.event_type == "ResumeStarted" for event in application.store.list_events(payload["run_id"]))


def test_run_manager_cancels_active_run(application):
    seed_demo(application)
    stopped = asyncio.Event()

    async def wait_forever(*args, **kwargs):
        await stopped.wait()

    application.main_agent._model_loop = wait_forever
    request = AgentRequest(user_input="等待取消", conversation_id="conv_cancel")

    async def run_case():
        run = application.conversations.submit(request)
        await asyncio.sleep(0)
        assert await application.run_manager.cancel(run.id)
        return run.id, await application.run_manager.wait(run.id)

    run_id, result = asyncio.run(run_case())

    assert result.status is AgentResultStatus.CANCELLED
    assert application.store.get_run(run_id).status is RunStatus.CANCELLED


def test_websocket_streams_run_events_before_result(application):
    seed_demo(application)
    with TestClient(create_app(application)) as client:
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


def test_websocket_streams_model_deltas(application):
    class StreamingModel(ModelAdapter):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(model="fake", content="第一段第二段")

        async def stream(self, request: ModelRequest):
            yield ModelStreamChunk(model="fake", content="第一段")
            yield ModelStreamChunk(model="fake", content="第二段", done=True)

    application.main_agent.model_adapter = StreamingModel()
    with TestClient(create_app(application)) as client:
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "ask", "message": "你好"})
            messages = []
            while True:
                item = socket.receive_json()
                messages.append(item)
                if item["type"] == "result":
                    break

    assert [item["content"] for item in messages if item["type"] == "delta"] == ["第一段", "第二段"]
