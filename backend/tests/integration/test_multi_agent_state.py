import asyncio

from app.agent.manager import AgentManager
from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Dataset,
    DatasetKind,
    ErrorCategory,
    Run,
    SubAgentExecutionResult,
    SubTask,
    ToolError,
    ToolResult,
    ToolStatus,
    WorkingMemory,
    WorkingMemoryDelta,
    WorkingMemoryItem,
)
from app.demo import seed_demo
from app.state import WorkingMemoryUpdater


def test_delegated_subagents_use_parent_task_and_merge_results_into_working_memory(application):
    ids = seed_demo(application)

    result = asyncio.run(application.ask("综合道路、人口和 DEM，从三个方面评价当前区域", dataset_ids=list(ids.values())))

    main_run = application.store.get_run(result.trace_id)
    child_runs = [item for item in application.store.list_runs() if item.parent_run_id == result.trace_id]
    task = application.store.get_task(result.task_id)
    memory = application.store.get_working_memory(result.task_id)

    assert main_run is not None
    assert task is not None
    assert len(application.store.list_tasks()) == 1
    assert len(child_runs) == 3
    assert all(item.task_id == task.id for item in child_runs)
    assert all(item.metadata.get("subtask_id") in task.subtasks for item in child_runs)
    assert memory is not None
    assert set(result.datasets).issubset(memory.active_dataset_ids)


def test_subagent_receives_deep_working_memory_snapshot_without_writing_parent_memory(application):
    dataset = Dataset(id="dataset-subagent-snapshot", name="dem.tif", kind=DatasetKind.RASTER, path="dem.tif", format="tif")
    task = application.task_service.create("分析 DEM", conversation_id="conv-subagent-snapshot")
    parent_run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(parent_run)
    memory = WorkingMemory(task_id=task.id, conversation_id=task.conversation_id, active_dataset_ids=[dataset.id], constraints=["范围=上海"])
    application.store.save_working_memory(memory)
    captured: dict[str, object] = {}

    class CaptureContext:
        def sub_context(self, request, subtask, datasets, **kwargs):
            captured["working_memory"] = kwargs["working_memory"]
            return {"datasets": [item.model_dump(mode="json") for item in datasets]}

    async def fake_call(run, name, arguments):
        return ToolResult(call_id=f"call-{name}", status=ToolStatus.SUCCESS, output={"checked": True})

    application.sub_agent.context_manager = CaptureContext()
    application.sub_agent._call = fake_call
    execution = asyncio.run(
        application.sub_agent.run(
            AgentRequest(user_input="检查这个数据", conversation_id=task.conversation_id),
            SubTask(goal="检查 DEM", description="检查输入数据", dataset_ids=[dataset.id]),
            [dataset],
            parent_task_id=task.id,
            parent_run_id=parent_run.id,
            working_memory_snapshot=memory,
        )
    )

    snapshot = captured["working_memory"]
    assert isinstance(execution, SubAgentExecutionResult)
    assert isinstance(snapshot, WorkingMemory)
    assert snapshot is not memory
    assert snapshot.active_dataset_ids == [dataset.id]
    assert snapshot.constraints == ["范围=上海"]
    assert application.store.get_working_memory(task.id).model_dump() == memory.model_dump()


def test_agent_manager_passes_independent_snapshots_and_returns_deltas(application):
    class FakeSubAgent:
        def __init__(self):
            self.snapshots = []

        async def run(self, request, subtask, datasets, *, parent_task_id, parent_run_id, working_memory_snapshot):
            self.snapshots.append(working_memory_snapshot)
            return SubAgentExecutionResult(
                result=AgentResult(
                    agent_id=f"agent-{subtask.id}",
                    task_id=parent_task_id,
                    status=AgentResultStatus.SUCCESS,
                    summary=subtask.goal,
                    trace_id=parent_run_id,
                ),
                working_memory_delta=WorkingMemoryDelta(added_dataset_ids=[subtask.id], source_run_id=subtask.id),
            )

    fake = FakeSubAgent()
    manager = AgentManager(fake, max_parallel=2)
    memory = WorkingMemory(task_id="task-manager", constraints=["范围=上海"], active_dataset_ids=["dem"])
    tasks = [
        SubTask(id="sub-a", goal="坡度", description="计算坡度"),
        SubTask(id="sub-b", goal="阴影", description="生成阴影"),
    ]

    executions = asyncio.run(
        manager.run(
            AgentRequest(user_input="并行处理", conversation_id="conv-manager"),
            tasks,
            [],
            parent_task_id="task-manager",
            parent_run_id="run-manager",
            working_memory_snapshot=memory,
        )
    )

    assert [item.result.task_id for item in executions] == ["task-manager", "task-manager"]
    assert [item.working_memory_delta.added_dataset_ids for item in executions] == [["sub-a"], ["sub-b"]]
    assert len(fake.snapshots) == 2
    assert fake.snapshots[0] is not memory
    assert fake.snapshots[1] is not memory
    assert fake.snapshots[0] is not fake.snapshots[1]
    assert all(snapshot.constraints == ["范围=上海"] for snapshot in fake.snapshots)
    assert all(snapshot.active_dataset_ids == ["dem"] for snapshot in fake.snapshots)


def test_parallel_deltas_are_merged_deterministically_and_idempotently(application):
    memory = WorkingMemory(task_id="task-merge", active_dataset_ids=["dem"])
    item_a = WorkingMemoryItem(kind="tool_result", reference_id="call-a", summary="坡度", source_run_id="run-a")
    item_b = WorkingMemoryItem(kind="tool_result", reference_id="call-b", summary="阴影", source_run_id="run-b")
    deltas = [
        WorkingMemoryDelta(added_dataset_ids=["slope"], added_artifact_ids=["art-a"], intermediate_results=[item_a], source_run_id="run-a"),
        WorkingMemoryDelta(added_dataset_ids=["hillshade"], added_artifact_ids=["art-b"], intermediate_results=[item_b], source_run_id="run-b"),
    ]
    updater = WorkingMemoryUpdater(application.store)

    merged = updater.merge_deltas(memory, deltas)
    merged_again = updater.merge_deltas(merged, deltas)

    assert merged.active_dataset_ids == ["dem", "slope", "hillshade"]
    assert merged.active_artifact_ids == ["art-a", "art-b"]
    assert len(merged.intermediate_results) == 2
    assert merged_again.active_dataset_ids == merged.active_dataset_ids
    assert merged_again.active_artifact_ids == merged.active_artifact_ids
    assert len(merged_again.intermediate_results) == 2


def test_partial_subagent_delta_keeps_outputs_created_before_failure(application):
    updater = WorkingMemoryUpdater(application.store)
    success = ToolResult(call_id="call-project", status=ToolStatus.SUCCESS, datasets=["projected-dem"])
    failure = ToolResult(
        call_id="call-slope",
        status=ToolStatus.FAILED,
        error=ToolError(code="SLOPE_FAILED", category=ErrorCategory.DATA, message="坡度计算失败"),
    )
    current = WorkingMemoryDelta(source_run_id="run-subagent")
    current = updater.merge_delta(current, updater.build_delta_from_tool_result(success, run_id="run-subagent"))
    current = updater.merge_delta(current, updater.build_delta_from_tool_result(failure, run_id="run-subagent"))
    merged = updater.apply_delta(WorkingMemory(task_id="task-partial"), current)

    assert merged.active_dataset_ids == ["projected-dem"]
    assert "需要补充工具输入：SLOPE_FAILED" in merged.unresolved_questions


def test_failed_subagent_without_outputs_does_not_add_resources(application):
    updater = WorkingMemoryUpdater(application.store)
    failure = ToolResult(
        call_id="call-failed",
        status=ToolStatus.FAILED,
        error=ToolError(code="INPUT_MISSING", category=ErrorCategory.INPUT, message="缺少输入"),
    )
    delta = updater.build_delta_from_tool_result(failure, run_id="run-failed")
    merged = updater.apply_delta(WorkingMemory(task_id="task-failed", active_dataset_ids=["dem"]), delta)

    assert delta.added_dataset_ids == []
    assert delta.added_artifact_ids == []
    assert merged.active_dataset_ids == ["dem"]
    assert merged.active_artifact_ids == []


def test_subagent_partial_execution_returns_delta_without_persisting_parent_memory(application):
    dataset = Dataset(id="dataset-subagent-partial", name="dem.tif", kind=DatasetKind.RASTER, path="dem.tif", format="tif")
    task = application.task_service.create("计算坡度", conversation_id="conv-subagent-partial")
    parent_run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(parent_run)
    memory = WorkingMemory(task_id=task.id, conversation_id=task.conversation_id, active_dataset_ids=[dataset.id])
    application.store.save_working_memory(memory)
    responses = {
        "dataset.inspect": ToolResult(call_id="inspect", status=ToolStatus.SUCCESS, output={"ok": True}),
        "raster.slope": ToolResult(
            call_id="slope",
            status=ToolStatus.FAILED,
            error=ToolError(code="CRS_UNIT_MISMATCH", category=ErrorCategory.CRS, message="需要投影坐标系"),
        ),
        "crs.reproject": ToolResult(call_id="reproject", status=ToolStatus.SUCCESS, datasets=["projected-dem"]),
    }

    async def fake_call(run, name, arguments):
        if name == "raster.slope" and arguments["dataset_id"] == "projected-dem":
            return ToolResult(
                call_id="slope-retry",
                status=ToolStatus.FAILED,
                error=ToolError(code="SLOPE_FAILED", category=ErrorCategory.DATA, message="计算失败"),
            )
        return responses[name]

    application.sub_agent._call = fake_call
    execution = asyncio.run(
        application.sub_agent.run(
            AgentRequest(user_input="计算坡度", conversation_id=task.conversation_id),
            SubTask(goal="地形坡度", description="计算 DEM 坡度", operation="raster.slope", dataset_ids=[dataset.id]),
            [dataset],
            parent_task_id=task.id,
            parent_run_id=parent_run.id,
            working_memory_snapshot=memory,
        )
    )

    assert execution.result.status is AgentResultStatus.PARTIAL
    assert "projected-dem" in execution.working_memory_delta.added_dataset_ids
    assert application.store.get_working_memory(task.id).active_dataset_ids == [dataset.id]


def test_main_agent_tool_outputs_are_also_reflected_in_working_memory(application):
    ids = seed_demo(application)

    result = asyncio.run(application.ask("检查 roads 并生成 500 米缓冲区", dataset_ids=[ids["roads"]]))

    memory = application.store.get_working_memory(result.task_id)
    assert memory is not None
    assert set(result.datasets).issubset(memory.active_dataset_ids)
    assert set(result.artifacts).issubset(memory.active_artifact_ids)


def test_delegation_does_not_write_subagent_results_to_project_memory(application):
    ids = seed_demo(application)

    asyncio.run(application.ask("综合道路、人口和 DEM，从三个方面评价当前区域", dataset_ids=list(ids.values())))

    assert application.memory.list() == []
