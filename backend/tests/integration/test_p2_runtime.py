import asyncio
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from app.core.models import AgentRequest, AgentResultStatus
from app.demo import seed_demo
from app.models import ModelAdapter, ModelRequest, ModelResponse
from app.runtime.context_manager import ContextManager


def test_context_manager_keeps_large_context_bounded():
    from app.core.models import AgentRequest, Dataset, DatasetKind

    context = ContextManager(max_tokens=120).main_context(
        AgentRequest(user_input="request " + "x" * 1000),
        [Dataset(name="roads", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson", metadata={"notes": "y" * 1000})],
        None,
        [],
    )

    assert context["truncated"] is True
    assert context["context_meta"]["over_budget"] is True
    assert context["context_meta"]["overflow_tokens"] > 0
    assert "request_frame" in context
    assert "working_memory" in context


def test_api_exposes_tool_schemas_memory_and_metrics(application, authenticated_client):
    ids = seed_demo(application)
    with authenticated_client as client:
        tools = client.get("/api/v1/tools")
        assert tools.status_code == 200
        buffer_tool = next(item for item in tools.json() if item["name"] == "vector.buffer")
        assert buffer_tool["input_schema"]["required"] == ["dataset_id", "distance"]

        saved = client.post("/api/v1/memories", json={"key": "default_crs", "value": "EPSG:4547"})
        assert saved.status_code == 200
        assert client.get("/api/v1/memories").json()[0]["key"] == "default_crs"

        result = client.post("/api/v1/ask", json={"message": "检查 roads", "dataset_ids": [ids["roads"]]})
        assert result.status_code == 200
        metrics = client.get("/api/v1/metrics").json()
        assert metrics["trace_events"] > 0
        assert metrics["tool_calls.completed"] > 0


def test_api_uploads_file_registers_dataset_and_accepts_attachment_reference(application, authenticated_client):
    content = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "起点"},
                    "geometry": {"type": "Point", "coordinates": [116.4, 39.9]},
                }
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")

    with authenticated_client as client:
        first = client.post(
            "/api/v1/attachments",
            files={"file": ("roads.geojson", content, "application/geo+json")},
        )
        second = client.post(
            "/api/v1/attachments",
            files={"file": ("roads.geojson", content, "application/geo+json")},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        first_dataset = first.json()["dataset"]
        second_dataset = second.json()["dataset"]
        assert first.json()["attachment_id"] == first_dataset["id"]
        assert first_dataset["name"] == "roads"
        assert first_dataset["path"] != second_dataset["path"]
        assert Path(first_dataset["path"]).exists()
        assert Path(second_dataset["path"]).exists()

        result = client.post(
            "/api/v1/ask",
            json={"message": "检查上传的数据", "attachment_ids": [first.json()["attachment_id"]]},
        )

    assert result.status_code == 200
    assert first_dataset["id"] in result.json()["datasets"]


def test_request_references_previous_run_for_diagnosis(application):
    ids = seed_demo(application)
    conversation_id = "conversation-reference"
    first = asyncio.run(application.ask("检查 roads", conversation_id=conversation_id, dataset_ids=[ids["roads"]]))
    second = asyncio.run(application.ask("为什么刚才失败？", conversation_id=conversation_id))

    assert first.status is AgentResultStatus.SUCCESS
    assert second.status is AgentResultStatus.SUCCESS
    assert second.findings[0]["run_id"] == first.trace_id


def test_basic_conversation_returns_a_chat_reply(application):
    result = asyncio.run(application.ask("你好"))

    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary.startswith("你好！")


def test_api_preserves_basic_conversation_messages(application, authenticated_client):
    conversation_id = "chat-history-check"
    with authenticated_client as client:
        response = client.post(f"/api/v1/conversations/{conversation_id}/messages", json={"message": "你好"})
        messages = client.get(f"/api/v1/conversations/{conversation_id}/messages")

    assert response.status_code == 200
    assert messages.status_code == 200
    assert [item["role"] for item in messages.json()] == ["user", "assistant"]
    assert messages.json()[1]["content"].startswith("你好！")


def test_distance_analysis_does_not_run_buffer(application):
    first_path = application.workspace.input_dir / "source_projected.geojson"
    second_path = application.workspace.input_dir / "target_projected.geojson"
    gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:3857").to_file(first_path, driver="GeoJSON")
    gpd.GeoDataFrame({"id": [1]}, geometry=[Point(100, 0)], crs="EPSG:3857").to_file(second_path, driver="GeoJSON")
    source = application.register_dataset(first_path, name="source").id
    target = application.register_dataset(second_path, name="target").id

    result = asyncio.run(application.ask("计算两个数据集的 500 米距离分布", dataset_ids=[source, target]))

    assert result.status is AgentResultStatus.SUCCESS
    events = application.store.list_events(result.trace_id)
    assert any(event.payload.get("tool") == "analysis.distance" for event in events)
    assert not any(event.payload.get("tool") == "vector.buffer" for event in events)


def test_model_checkpoint_resume_keeps_tool_outputs(application, authenticated_client):
    ids = seed_demo(application)
    with authenticated_client:
        user_id = application.store.get_user_by_username("test-user").id
    conversation = application.conversations.create("模型恢复测试", user_id=user_id)

    class ResumableModel(ModelAdapter):
        def __init__(self):
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    model="fake",
                    tool_calls=[
                        {
                            "id": "inspect-once",
                            "function": {"name": "dataset.inspect", "arguments": json.dumps({"dataset_id": ids["roads"]})},
                        }
                    ],
                )
            return ModelResponse(model="fake", content="已从恢复的工具结果完成检查。")

    model = ResumableModel()
    application.main_agent.model_adapter = model
    original_checkpoint = application.main_agent._checkpoint

    async def stop_after_model_checkpoint(run_id, phase, state):
        await original_checkpoint(run_id, phase, state)
        if phase == "model_tool_completed":
            raise asyncio.CancelledError()

    application.main_agent._checkpoint = stop_after_model_checkpoint
    cancelled = asyncio.run(application.ask(AgentRequest(user_input="请检查道路数据", conversation_id=conversation.id, user_id=user_id, dataset_ids=[ids["roads"]])))
    application.main_agent._checkpoint = original_checkpoint

    assert cancelled.status is AgentResultStatus.CANCELLED
    checkpoint = application.checkpoints.latest(cancelled.trace_id)
    assert checkpoint is not None
    assert checkpoint.state.get("protocol_messages")
    assert checkpoint.state["latest_observation"]["call_id"] == "inspect-once"
    assert "messages" not in checkpoint.state
    with authenticated_client as client:
        response = client.post(f"/api/v1/runs/{cancelled.trace_id}/resume")

    assert response.status_code == 200
    resumed = response.json()["result"]
    assert resumed["status"] == "SUCCESS"
    assert ids["roads"] in resumed["datasets"]
    assert "inspect-once" in model.requests[1].messages[1]["content"]
    assert len(model.requests) == 2
