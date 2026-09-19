import asyncio
import json

from app.core.models import (
    AgentRequest,
    ConversationMemory,
    ConversationMemoryEntry,
    CRSInfo,
    Dataset,
    DatasetKind,
    InteractionMode,
    MemoryItem,
    RequestFrame,
    Run,
    RunStatus,
    ToolResult,
    ToolStatus,
    UserProfile,
    WorkingMemory,
    WorkingMemoryItem,
)
from app.models import ModelAdapter, ModelRequest, ModelResponse
from app.run.lifecycle import LifecycleAction, PreparedRequest
from app.runtime.context_manager import ContextManager


def test_required_sections_survive_budget_compression():
    context = ContextManager(max_tokens=180).main_context(
        AgentRequest(user_input="请分析当前数据"),
        [],
        None,
        [],
        request_frame=RequestFrame(mode=InteractionMode.NEW_TASK, goal="分析数据"),
        request_resources=None,
        working_memory=WorkingMemory(task_id="task-1", active_dataset_ids=["dem-1"], constraints=["范围=上海"], unresolved_questions=["需要确认 CRS"]),
        current_observation={"status": "SUCCESS", "output": "已检查"},
    )

    for key in ("user_request", "request_frame", "task_goal", "request_resources", "working_memory", "current_observation"):
        assert key in context
    assert context["context_meta"]["budget_tokens"] == 180


def test_working_memory_core_is_not_removed_with_many_intermediate_results():
    memory = WorkingMemory(
        task_id="task-core",
        active_dataset_ids=["dem"],
        active_artifact_ids=["artifact-1"],
        constraints=["范围=上海"],
        unresolved_questions=["需要确认输出格式"],
        intermediate_results=[WorkingMemoryItem(kind="tool_result", reference_id=f"run-{index}", summary="x" * 300) for index in range(20)],
    )
    context = ContextManager(max_tokens=260).main_context(AgentRequest(user_input="继续"), [], None, [], working_memory=memory)

    core = context["working_memory"]
    assert core["active_dataset_ids"] == ["dem"]
    assert core["active_artifact_ids"] == ["artifact-1"]
    assert core["constraints"] == ["范围=上海"]
    assert core["unresolved_questions"] == ["需要确认输出格式"]


def test_project_memory_keeps_relevance_order_when_compressed():
    memories = [MemoryItem(key=f"memory-{index}", value="重要事实 " * 80) for index in range(5)]
    context = ContextManager(max_tokens=220).main_context(AgentRequest(user_input="查询项目规则"), [], None, memories)

    assert context["project_memory"]
    assert context["project_memory"][0]["key"] == "memory-0"


def test_recent_messages_are_limited_to_latest_eight():
    messages = [{"role": "user", "content": str(index)} for index in range(20)]
    context = ContextManager(max_tokens=2000).main_context(AgentRequest(user_input="继续"), [], None, [], conversation=messages)

    assert len(context["recent_messages"]) == 8
    assert context["recent_messages"][0]["content"] == "12"
    assert context["recent_messages"][-1]["content"] == "19"


def test_dataset_context_is_a_lightweight_view():
    dataset = Dataset(
        id="dataset-view",
        name="dem.tif",
        kind=DatasetKind.RASTER,
        path="D:/private/secret/dem.tif",
        format="tif",
        crs=CRSInfo(authority="EPSG:4326"),
        owner_user_id="user-secret",
        metadata={"driver": "GTiff", "raw_private_note": "不要进入模型"},
    )
    context = ContextManager(max_tokens=2000).main_context(AgentRequest(user_input="检查 DEM"), [dataset], None, [])
    serialized = json.dumps(context, ensure_ascii=False)

    assert "dataset-view" in serialized
    assert "dem.tif" in serialized
    assert "EPSG:4326" in serialized
    assert "D:/private/secret/dem.tif" not in serialized
    assert "user-secret" not in serialized
    assert "raw_private_note" not in serialized


def test_tool_schema_is_not_duplicated_in_textual_context():
    context = ContextManager(max_tokens=2000).main_context(
        AgentRequest(user_input="计算坡度"),
        [],
        None,
        [],
        tool_definitions=[{"name": "raster.slope", "description": "计算坡度", "input_schema": {"properties": {"dataset_id": {"type": "string"}}}}],
    )

    assert context["tool_capabilities"] == [{"name": "raster.slope", "summary": "计算坡度"}]
    assert "input_schema" not in json.dumps(context, ensure_ascii=False)


def test_profile_and_conversation_memory_use_model_views():
    profile = UserProfile(user_id="user-1", response_style="concise")
    conversation_memory = ConversationMemory(
        conversation_id="conversation-1",
        user_id="user-1",
        summary="当前研究区域为上海",
        key_facts=[ConversationMemoryEntry(content="使用米制")],
    )
    context = ContextManager(max_tokens=3000).main_context(
        AgentRequest(user_id="user-1", user_input="详细解释"),
        [],
        None,
        [],
        user_profile=profile,
        conversation_memory=conversation_memory,
        conversation=[{"role": "user", "content": "上一条消息"}],
    )

    assert context["user_profile"]["response_style"] == "concise"
    assert context["user_profile"]["measurement_system"] == "metric"
    assert context["conversation_memory"]["summary"] == "当前研究区域为上海"
    assert context["recent_messages"]
    assert "user_id" not in context["user_profile"]
    assert "updated_at" not in context["user_profile"]


def test_compressor_does_not_modify_authoritative_memory():
    memory = WorkingMemory(
        task_id="task-immutable",
        constraints=["范围=北京"],
        intermediate_results=[WorkingMemoryItem(kind="result", reference_id="result-1", summary="a" * 500)],
    )
    before = memory.model_dump(mode="json")
    ContextManager(max_tokens=120).main_context(AgentRequest(user_input="继续"), [], None, [], working_memory=memory)

    assert memory.model_dump(mode="json") == before


def test_latest_tool_result_is_current_observation():
    observation = ToolResult(call_id="call-1", status=ToolStatus.SUCCESS, datasets=["slope-1"], output={"count": 10})
    context = ContextManager(max_tokens=2000).main_context(AgentRequest(user_input="继续"), [], None, [], current_observation=observation)

    assert context["current_observation"]["call_id"] == "call-1"
    assert context["current_observation"]["datasets"] == ["slope-1"]
    assert context["current_observation"]["output"] == {"count": 10}


def test_model_second_turn_reads_updated_working_memory(application):
    task = application.task_service.create("动态上下文", conversation_id="context-refresh")
    run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main", status=RunStatus.RUNNING)
    application.store.save_run(run)
    memory = WorkingMemory(task_id=task.id, conversation_id=task.conversation_id, active_dataset_ids=["source"])
    application.store.save_working_memory(memory)
    generated = Dataset(id="generated-after-tool", name="slope.tif", kind=DatasetKind.RASTER, path="slope.tif", format="tif")
    application.store.save_dataset(generated)

    class TwoTurnModel(ModelAdapter):
        def __init__(self):
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(tool_calls=[{"id": "call-refresh", "function": {"name": "dataset.inspect", "arguments": "{}"}}])
            return ModelResponse(content="已读取最新工作状态。")

    model = TwoTurnModel()
    application.main_agent.model_adapter = model

    async def fake_tool(current_run, name, arguments, *, call_id=None):
        result = ToolResult(call_id=call_id or "call-refresh", status=ToolStatus.SUCCESS, datasets=[generated.id], output={"ok": True})
        application.main_agent.working_memory_updater.update_from_tool_result(current_run.task_id, result, run_id=current_run.id)
        return result

    original_raw_executor = application.main_agent.tool_execution_cycle.raw_executor
    application.main_agent.tool_execution_cycle.raw_executor = fake_tool
    request = AgentRequest(user_input="分析结果", conversation_id=task.conversation_id)
    frame = RequestFrame(mode=InteractionMode.CONTINUE_TASK, goal="分析结果", target_task_id=task.id)
    prepared = PreparedRequest(
        request=request,
        frame=frame,
        action=LifecycleAction.BIND_TASK,
        task=task,
        run=run,
        working_memory=memory,
    )
    try:
        result = asyncio.run(application.main_agent.run(request, prepared=prepared))
    finally:
        application.main_agent.tool_execution_cycle.raw_executor = original_raw_executor

    assert result is not None
    second_context = model.requests[1].messages[1]["content"]
    assert generated.id in second_context
