import asyncio
import json

from app.agent.manager import AgentManager
from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    CRSInfo,
    Dataset,
    DatasetKind,
    ErrorCategory,
    LoopDirective,
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
from app.models import ModelAdapter, ModelRequest, ModelResponse
from app.runtime.tool_execution_cycle import ExecutionOutcome
from app.state import WorkingMemoryUpdater


class _DelegationModel(ModelAdapter):
    def __init__(self, decisions: list[ModelResponse]) -> None:
        self.decisions = list(decisions)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.decisions.pop(0)


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
    assert memory.unresolved_questions == []


def test_runtime_delegation_is_observation_then_mainagent_final(application):
    ids = seed_demo(application)
    model = _DelegationModel(
        [
            ModelResponse(model="fake", tool_calls=[{"id": "delegate", "function": {"name": "agent.delegate", "arguments": "{\"reason\":\"三个主题需要并行处理\"}"}}]),
            ModelResponse(model="fake", content="已根据子任务结果完成综合判断。"),
        ]
    )
    calls = []

    class FakeManager:
        async def run(self, request, tasks, datasets, **kwargs):
            calls.append([item.id for item in tasks])
            return [
                SubAgentExecutionResult(
                    result=AgentResult(agent_id=f"agent-{item.id}", task_id=kwargs["parent_task_id"], status=AgentResultStatus.SUCCESS, summary=f"完成：{item.goal}", trace_id=f"run-{item.id}"),
                    working_memory_delta=WorkingMemoryDelta(added_dataset_ids=[f"derived-{item.id}"], source_run_id=f"run-{item.id}"),
                )
                for item in tasks
            ]

    original_manager = application.main_agent.agent_manager
    application.main_agent.model_adapter = model
    application.main_agent.agent_manager = FakeManager()
    try:
        result = asyncio.run(application.ask("综合道路、人口和 DEM，从三个方面评价当前区域", dataset_ids=list(ids.values())))
    finally:
        application.main_agent.agent_manager = original_manager

    assert result.status is AgentResultStatus.SUCCESS
    assert result.summary == "已根据子任务结果完成综合判断。"
    assert len(calls) == 1
    assert len(model.requests) == 2
    second_context = json.loads(model.requests[1].messages[1]["content"].split("\n", 1)[1])
    assert second_context["current_observation"]["type"] == "delegation"
    assert second_context["current_observation"]["results"][0]["status"] == "SUCCESS"
    assert second_context["current_observation"]["results"][0]["subtask_id"]
    memory = application.store.get_working_memory(result.task_id)
    assert memory is not None
    assert all(item.startswith("derived-") for item in memory.active_dataset_ids if item.startswith("derived-"))
    events = application.store.list_events(result.trace_id)
    assert any(item.event_type == "DelegationCompleted" for item in events)


def test_runtime_delegation_same_fingerprint_does_not_spawn_twice(application):
    ids = seed_demo(application)
    model = _DelegationModel(
        [
            ModelResponse(model="fake", tool_calls=[{"id": "delegate-a", "function": {"name": "agent.delegate", "arguments": "{}"}}]),
            ModelResponse(model="fake", tool_calls=[{"id": "delegate-b", "function": {"name": "agent.delegate", "arguments": "{}"}}]),
            ModelResponse(model="fake", content="已停止重复委派并完成判断。"),
        ]
    )
    spawn_count = 0

    class FakeManager:
        async def run(self, request, tasks, datasets, **kwargs):
            nonlocal spawn_count
            spawn_count += 1
            return [
                SubAgentExecutionResult(
                    result=AgentResult(agent_id=f"agent-{item.id}", task_id=kwargs["parent_task_id"], status=AgentResultStatus.SUCCESS, summary="已完成", trace_id=f"run-{item.id}"),
                    working_memory_delta=WorkingMemoryDelta(source_run_id=f"run-{item.id}"),
                )
                for item in tasks
            ]

    original_manager = application.main_agent.agent_manager
    application.main_agent.model_adapter = model
    application.main_agent.agent_manager = FakeManager()
    try:
        result = asyncio.run(application.ask("综合道路、人口和 DEM，从三个方面评价当前区域", dataset_ids=list(ids.values())))
    finally:
        application.main_agent.agent_manager = original_manager

    assert result.status is AgentResultStatus.SUCCESS
    assert spawn_count == 1
    assert len(model.requests) == 3


def test_required_subagent_directives_converge_through_parent_runtime(application):
    ids = seed_demo(application)
    for directive, expected_status, expected_error in (
        (LoopDirective.ASK_USER, AgentResultStatus.BLOCKED, "WAITING_USER"),
        (LoopDirective.REPLAN, AgentResultStatus.BLOCKED, "REPLAN_REQUIRED"),
        (LoopDirective.ABORT, AgentResultStatus.FAILED, "上一步执行要求终止当前运行。"),
    ):
        model = _DelegationModel(
            [ModelResponse(model="fake", tool_calls=[{"id": "delegate", "function": {"name": "agent.delegate", "arguments": "{}"}}])]
        )

        class FakeManager:
            async def run(self, request, tasks, datasets, **kwargs):
                return [
                    SubAgentExecutionResult(
                        result=AgentResult(agent_id=f"agent-{item.id}", task_id=kwargs["parent_task_id"], status=AgentResultStatus.BLOCKED if directive is not LoopDirective.ABORT else AgentResultStatus.FAILED, summary="子任务未完成", error="需要补充信息" if directive is LoopDirective.ASK_USER else "算法不适用" if directive is LoopDirective.REPLAN else "工具失败", trace_id=f"run-{item.id}"),
                        directive=directive,
                        failure_rationale="需要补充信息" if directive is LoopDirective.ASK_USER else "算法不适用" if directive is LoopDirective.REPLAN else "工具失败",
                    )
                    for item in tasks
                ]

        original_manager = application.main_agent.agent_manager
        application.main_agent.model_adapter = model
        application.main_agent.agent_manager = FakeManager()
        try:
            result = asyncio.run(application.ask("综合道路、人口和 DEM，从三个方面评价当前区域", dataset_ids=list(ids.values())))
        finally:
            application.main_agent.agent_manager = original_manager

        assert result.status is expected_status
        assert result.error == expected_error


def test_optional_subagent_abort_remains_partial_observation(application):
    task = application.task_service.create("可选委派", conversation_id="conv-optional-delegation")
    run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(run)
    request = AgentRequest(user_input="可选委派", conversation_id=task.conversation_id)
    subtask = SubTask(id="optional-subtask", goal="可选检查", description="非必需检查", required=False)
    original_manager = application.main_agent.agent_manager

    class FakeManager:
        async def run(self, request, tasks, datasets, **kwargs):
            return [
                SubAgentExecutionResult(
                    result=AgentResult(agent_id="optional-agent", task_id=kwargs["parent_task_id"], status=AgentResultStatus.FAILED, summary="可选任务失败", error="工具失败", trace_id="optional-run"),
                    directive=LoopDirective.ABORT,
                    failure_rationale="工具失败",
                )
            ]

    application.main_agent.agent_manager = FakeManager()
    try:
        delegation = asyncio.run(
            application.main_agent.runtime_action_handlers.execute_delegation(
                request,
                run,
                task,
                [],
                [subtask],
                    plan=None,
                request_frame=None,
                working_memory=None,
                fingerprint="optional-delegation",
            )
        )
    finally:
        application.main_agent.agent_manager = original_manager

    assert delegation.directive is LoopDirective.CONTINUE
    assert delegation.latest_failure is None
    assert delegation.observation["results"][0]["directive"] == LoopDirective.ABORT.value


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
    snapshot_input = memory.model_copy(deep=True)
    execution = asyncio.run(
        application.sub_agent.run(
            AgentRequest(user_input="检查这个数据", conversation_id=task.conversation_id),
            SubTask(goal="检查 DEM", description="检查输入数据", dataset_ids=[dataset.id]),
            [dataset],
            parent_task_id=task.id,
            parent_run_id=parent_run.id,
            working_memory_snapshot=snapshot_input,
        )
    )

    snapshot = captured["working_memory"]
    assert isinstance(execution, SubAgentExecutionResult)
    assert isinstance(snapshot, WorkingMemory)
    assert snapshot is snapshot_input
    assert snapshot is not memory
    assert snapshot.active_dataset_ids == [dataset.id]
    assert snapshot.constraints == ["范围=上海"]
    assert application.store.get_working_memory(task.id).model_dump() == memory.model_dump()


def test_deterministic_subagent_counts_tools_without_cognitive_turns(application):
    dataset = Dataset(id="dataset-subagent-budget", name="roads.geojson", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson")
    application.registry.register(dataset)
    task = application.task_service.create("检查道路", conversation_id="conv-subagent-budget")
    parent_run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(parent_run)

    async def fake_execute(call, agent_id, services):
        return ToolResult(call_id=f"call-{call.name}", status=ToolStatus.SUCCESS, output={"ok": True})

    application.sub_agent.executor.execute = fake_execute
    execution = asyncio.run(
        application.sub_agent.run(
            AgentRequest(user_input="检查道路", conversation_id=task.conversation_id),
            SubTask(goal="道路质量", description="检查道路数据", operation="vector.validate", dataset_ids=[dataset.id]),
            [dataset],
            parent_task_id=task.id,
            parent_run_id=parent_run.id,
            working_memory_snapshot=None,
        )
    )

    child_run = application.store.get_run(execution.result.trace_id)
    assert child_run is not None
    assert child_run.turn_count == 0
    assert child_run.tool_call_count >= 1


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
    assert merged.unresolved_questions == []


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
    dataset = Dataset(id="dataset-subagent-partial", name="dem.tif", kind=DatasetKind.RASTER, path="dem.tif", format="tif", crs=CRSInfo(authority="EPSG:4326"))
    application.registry.register(dataset)
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
            "raster.reproject": ToolResult(call_id="reproject", status=ToolStatus.SUCCESS, datasets=["projected-dem"]),
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

    assert execution.result.status is AgentResultStatus.FAILED
    assert execution.directive is LoopDirective.ABORT
    assert "projected-dem" not in execution.working_memory_delta.added_dataset_ids
    assert application.store.get_working_memory(task.id).active_dataset_ids == [dataset.id]


def test_subagent_retry_uses_shared_cycle_and_accepts_only_final_result(application):
    dataset = Dataset(id="dataset-subagent-retry", name="roads.geojson", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson")
    application.registry.register(dataset)
    task = application.task_service.create("检查道路", conversation_id="conv-subagent-retry")
    parent_run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(parent_run)
    calls = []
    responses = [
        ToolResult(call_id="inspect", status=ToolStatus.SUCCESS, output={"ok": True}),
        ToolResult(call_id="validate-failed", status=ToolStatus.FAILED, retryable=True, error=ToolError(code="EXECUTION_TIMEOUT", message="临时超时")),
        ToolResult(call_id="validate-ok", status=ToolStatus.SUCCESS, output={"valid": True}),
    ]

    async def fake_call(run, name, arguments):
        calls.append(name)
        return responses.pop(0)

    application.sub_agent._call = fake_call
    execution = asyncio.run(
        application.sub_agent.run(
            AgentRequest(user_input="检查道路", conversation_id=task.conversation_id),
            SubTask(goal="道路质量", description="检查道路数据", operation="vector.validate", dataset_ids=[dataset.id]),
            [dataset],
            parent_task_id=task.id,
            parent_run_id=parent_run.id,
            working_memory_snapshot=None,
        )
    )

    assert execution.result.status is AgentResultStatus.SUCCESS
    assert execution.directive is LoopDirective.CONTINUE
    assert calls == ["dataset.inspect", "vector.validate", "vector.validate"]
    assert execution.working_memory_delta.added_dataset_ids == []


def test_subagent_repair_accepts_final_dataset_without_internal_repair_dataset(application):
    dataset = Dataset(id="dataset-subagent-repair", name="dem.tif", kind=DatasetKind.RASTER, path="dem.tif", format="tif", crs=CRSInfo(authority="EPSG:4326"))
    application.registry.register(dataset)
    task = application.task_service.create("计算坡度", conversation_id="conv-subagent-repair")
    parent_run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(parent_run)
    responses = {
        "dataset.inspect": ToolResult(call_id="inspect", status=ToolStatus.SUCCESS, output={"ok": True}),
        "raster.slope": [
            ToolResult(call_id="slope-failed", status=ToolStatus.FAILED, error=ToolError(code="CRS_UNIT_MISMATCH", message="需要投影坐标系")),
            ToolResult(call_id="slope-final", status=ToolStatus.SUCCESS, datasets=["slope-final"], output={"ok": True}),
        ],
        "raster.reproject": ToolResult(call_id="reproject", status=ToolStatus.SUCCESS, datasets=["projected-dem"]),
    }

    async def fake_call(run, name, arguments):
        value = responses[name]
        return value.pop(0) if isinstance(value, list) else value

    application.sub_agent._call = fake_call
    application.sub_agent.tool_execution_cycle.verifier = type("Verifier", (), {"verify": lambda self, result, datasets: (True, [])})()
    execution = asyncio.run(
        application.sub_agent.run(
            AgentRequest(user_input="计算坡度", conversation_id=task.conversation_id),
            SubTask(goal="地形坡度", description="计算 DEM 坡度", operation="raster.slope", dataset_ids=[dataset.id]),
            [dataset],
            parent_task_id=task.id,
            parent_run_id=parent_run.id,
            working_memory_snapshot=None,
        )
    )

    assert execution.result.status is AgentResultStatus.SUCCESS
    assert execution.working_memory_delta.added_dataset_ids == ["slope-final"]
    assert "projected-dem" not in execution.working_memory_delta.added_dataset_ids
    assert execution.result.datasets == ["slope-final"]


def test_subagent_directives_stop_local_execution_without_planner(application):
    dataset = Dataset(id="dataset-subagent-directive", name="roads.geojson", kind=DatasetKind.VECTOR, path="roads.geojson", format="geojson")
    application.registry.register(dataset)
    task = application.task_service.create("执行子任务", conversation_id="conv-subagent-directive")
    parent_run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    original = application.sub_agent.tool_execution_cycle.execute

    async def run_directive(directive):
        calls = []

        async def execute(*args, **kwargs):
            calls.append(args[1])
            return ExecutionOutcome(
                result=ToolResult(call_id="failed", status=ToolStatus.FAILED, error=ToolError(code="INPUT", message="需要补充条件")),
                verified=False,
                verification_problems=[],
                recovery_action=None,
                attempts=1,
                accepted=False,
                directive=directive,
                rationale="执行层返回控制信号",
            )

        application.sub_agent.tool_execution_cycle.execute = execute
        try:
            execution = await application.sub_agent.run(
                AgentRequest(user_input="执行子任务", conversation_id=task.conversation_id),
                SubTask(goal="道路质量", description="检查道路", operation="vector.validate", dataset_ids=[dataset.id]),
                [dataset],
                parent_task_id=task.id,
                parent_run_id=parent_run.id,
                working_memory_snapshot=None,
            )
        finally:
            application.sub_agent.tool_execution_cycle.execute = original
        return execution, calls

    ask, ask_calls = asyncio.run(run_directive(LoopDirective.ASK_USER))
    replan, replan_calls = asyncio.run(run_directive(LoopDirective.REPLAN))
    abort, abort_calls = asyncio.run(run_directive(LoopDirective.ABORT))

    assert ask.result.status is AgentResultStatus.BLOCKED and ask.result.error == "WAITING_USER"
    assert replan.result.status is AgentResultStatus.BLOCKED and replan.result.error == "REPLAN_REQUIRED"
    assert abort.result.status is AgentResultStatus.FAILED
    assert ask.directive is LoopDirective.ASK_USER
    assert replan.directive is LoopDirective.REPLAN
    assert abort.directive is LoopDirective.ABORT
    assert ask_calls == replan_calls == abort_calls == ["dataset.inspect"]


def test_mainagent_delegation_aggregates_required_directives(application):
    task = application.task_service.create("聚合子任务", conversation_id="conv-delegation-directive")
    parent_run = Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main")
    application.store.save_run(parent_run)
    request = AgentRequest(user_input="聚合子任务", conversation_id=task.conversation_id)
    subtask = SubTask(id="required-subtask", goal="检查", description="检查数据", required=True)
    original_manager = application.main_agent.agent_manager

    def execution(directive, status, error):
        return SubAgentExecutionResult(
            result=AgentResult(agent_id="subagent", task_id=task.id, status=status, summary="子任务结果", error=error, trace_id="sub-run"),
            directive=directive,
            failure_rationale=error,
        )

    async def run_with(value):
        class FakeManager:
            async def run(self, *args, **kwargs):
                return [value]

        application.main_agent.agent_manager = FakeManager()
        try:
            return await application.main_agent.runtime_action_handlers.execute_delegation(
                request,
                parent_run,
                task,
                [],
                [subtask],
                    plan=None,
                request_frame=None,
                working_memory=None,
                fingerprint=f"directive-{value.directive.value}",
            )
        finally:
            application.main_agent.agent_manager = original_manager

    ask = asyncio.run(run_with(execution(LoopDirective.ASK_USER, AgentResultStatus.BLOCKED, "缺少字段")))
    replan = asyncio.run(run_with(execution(LoopDirective.REPLAN, AgentResultStatus.BLOCKED, "算法不适用")))
    abort = asyncio.run(run_with(execution(LoopDirective.ABORT, AgentResultStatus.FAILED, "工具失败")))

    assert ask.directive is LoopDirective.ASK_USER
    assert replan.directive is LoopDirective.REPLAN
    assert abort.directive is LoopDirective.ABORT


def test_agent_manager_preserves_subagent_directive(application):
    class FakeSubAgent:
        async def run(self, request, subtask, datasets, **kwargs):
            return SubAgentExecutionResult(
                result=AgentResult(agent_id="subagent", task_id=kwargs["parent_task_id"], status=AgentResultStatus.BLOCKED, summary="等待用户", error="WAITING_USER", trace_id="sub-run"),
                directive=LoopDirective.ASK_USER,
                failure_rationale="缺少输入",
            )

    manager = AgentManager(FakeSubAgent(), max_parallel=1)
    executions = asyncio.run(
        manager.run(
            AgentRequest(user_input="检查", conversation_id="conv-manager-directive"),
            [SubTask(id="sub-directive", goal="检查", description="检查")],
            [],
            parent_task_id="task-manager-directive",
            parent_run_id="run-manager-directive",
            working_memory_snapshot=None,
        )
    )

    assert executions[0].directive is LoopDirective.ASK_USER
    assert executions[0].failure_rationale == "缺少输入"


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
