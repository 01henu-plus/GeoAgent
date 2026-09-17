import asyncio

from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Checkpoint,
    Dataset,
    DatasetKind,
    InteractionMode,
    RequestFrame,
    Run,
    RunStatus,
    TaskStatus,
    ToolResult,
    ToolStatus,
    WorkingMemory,
)
from app.memory import MemoryExtractor
from app.memory.models import MemoryCandidate
from app.state import WorkingMemoryUpdater


def test_new_task_initializes_working_memory_with_request_dataset(application):
    dataset = Dataset(id="dataset-wm-new", name="dem.tif", kind=DatasetKind.RASTER, path="dem.tif", format="tif")
    application.store.save_dataset(dataset)
    request = AgentRequest(user_input="分析这个 DEM", conversation_id="conv-wm-new", dataset_ids=[dataset.id])

    prepared = asyncio.run(application.main_agent.prepare_request(request))

    assert prepared.task is not None
    assert prepared.working_memory is not None
    assert prepared.working_memory.task_id == prepared.task.id
    assert prepared.working_memory.active_dataset_ids == [dataset.id]
    assert application.store.get_working_memory(prepared.task.id).task_id == prepared.task.id


def test_continue_reuses_task_working_memory(application):
    dataset = Dataset(id="dataset-wm-continue", name="dem.tif", kind=DatasetKind.RASTER, path="dem.tif", format="tif")
    application.store.save_dataset(dataset)
    first = asyncio.run(
        application.main_agent.prepare_request(
            AgentRequest(user_input="分析这个 DEM", conversation_id="conv-wm-continue", dataset_ids=[dataset.id])
        )
    )
    second = asyncio.run(application.main_agent.prepare_request(AgentRequest(user_input="继续", conversation_id="conv-wm-continue")))

    assert first.task is not None and second.task is not None
    assert second.task.id == first.task.id
    assert second.working_memory is not None
    assert second.working_memory.task_id == first.working_memory.task_id
    assert second.working_memory.active_dataset_ids == [dataset.id]


def test_modify_merges_constraint_into_existing_working_memory(application):
    first = asyncio.run(application.main_agent.prepare_request(AgentRequest(user_input="分析 DEM", conversation_id="conv-wm-modify")))
    second = asyncio.run(application.main_agent.prepare_request(AgentRequest(user_input="不对，把范围改成上海", conversation_id="conv-wm-modify")))

    assert first.task is not None and second.task is not None
    assert second.task.id == first.task.id
    assert second.working_memory is not None
    assert any("范围改成上海" in constraint for constraint in second.working_memory.constraints)


def test_retry_reuses_working_memory_identity(application):
    task = application.task_service.create("分析栅格", conversation_id="conv-wm-retry")
    failed = Run(task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.FAILED, error="工具失败")
    application.store.save_run(failed)
    memory = WorkingMemory(task_id=task.id, conversation_id=task.conversation_id, active_dataset_ids=["dataset-a"])
    application.store.save_working_memory(memory)

    prepared = asyncio.run(application.main_agent.prepare_request(AgentRequest(user_input="再试一次", conversation_id=task.conversation_id)))

    assert prepared.task is not None and prepared.task.id == task.id
    assert prepared.working_memory is not None
    assert prepared.working_memory.task_id == memory.task_id
    assert prepared.working_memory.active_dataset_ids == ["dataset-a"]


def test_tool_result_updates_working_memory_with_dataset_and_artifact(application):
    task = application.task_service.create("处理数据", conversation_id="conv-wm-tool")
    memory = WorkingMemory(task_id=task.id, conversation_id=task.conversation_id)
    application.store.save_working_memory(memory)
    updater = WorkingMemoryUpdater(application.store)

    result = ToolResult(call_id="call-wm-tool", status=ToolStatus.SUCCESS, datasets=["dataset-b"], artifacts=["artifact-a"])
    updated = updater.update_from_tool_result(task.id, result, run_id="run-wm-tool")

    assert updated is not None
    assert updated.active_dataset_ids == ["dataset-b"]
    assert updated.active_artifact_ids == ["artifact-a"]
    assert updated.intermediate_results[0].source_run_id == "run-wm-tool"
    assert updated.intermediate_results[0].reference_id == result.call_id


def test_blocked_request_records_unresolved_question_only_after_clarification(application):
    task = application.task_service.create("等待数据", conversation_id="conv-wm-clarification")
    application.task_service.update(task, status=TaskStatus.WAITING)
    application.store.save_run(Run(task_id=task.id, conversation_id=task.conversation_id, agent_id="main", status=RunStatus.WAITING_USER))
    application.store.save_working_memory(WorkingMemory(task_id=task.id, conversation_id=task.conversation_id))

    result = asyncio.run(application.ask(AgentRequest(user_input="用刚才的数据继续", conversation_id=task.conversation_id)))

    assert result.status is AgentResultStatus.BLOCKED
    memory = application.store.get_working_memory(task.id)
    assert memory is not None
    assert memory.unresolved_questions


def test_working_memory_is_task_scoped(application):
    first = WorkingMemory(task_id="task-a", conversation_id="conv-a", active_dataset_ids=["dataset-a"])
    second = WorkingMemory(task_id="task-b", conversation_id="conv-b", active_dataset_ids=["dataset-b"])
    application.store.save_working_memory(first)
    application.store.save_working_memory(second)

    assert application.store.get_working_memory("task-a").active_dataset_ids == ["dataset-a"]
    assert application.store.get_working_memory("task-b").active_dataset_ids == ["dataset-b"]


def test_deleting_one_run_does_not_delete_shared_task_working_memory(application):
    task = application.task_service.create("共享任务", conversation_id="conv-wm-delete")
    first = Run(task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.COMPLETED)
    second = Run(task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.COMPLETED)
    application.store.save_run(first)
    application.store.save_run(second)
    application.store.save_working_memory(WorkingMemory(task_id=task.id, conversation_id=task.conversation_id, active_dataset_ids=["dataset-shared"]))

    application.store.delete_run(first.id)

    assert application.store.get_task(task.id) is not None
    assert application.store.get_working_memory(task.id).active_dataset_ids == ["dataset-shared"]


def test_main_agent_does_not_write_runtime_state_to_project_memory(application):
    result = asyncio.run(application.ask("你好"))

    assert result.status is AgentResultStatus.SUCCESS
    assert {item.key for item in application.memory.list()}.isdisjoint({"last_run_id", "last_result_summary"})


def test_memory_write_policy_rejects_temporary_run_state(application):
    candidate = MemoryCandidate(key="last_run_id", value="run-temporary", category="run_state", durability="project")

    assert application.memory.write_candidate(candidate) is None
    assert application.memory.list() == []


def test_memory_write_policy_accepts_stable_project_fact_and_deduplicates(application):
    candidate = MemoryCandidate(
        key="project_default_crs",
        value="EPSG:3857",
        category="project_constraint",
        durability="durable",
        source_task_id="task-memory",
        source_run_id="run-memory",
    )

    first = application.memory.write_candidate(candidate)
    second = application.memory.write_candidate(candidate)

    assert first is not None
    assert second is not None
    assert len(application.memory.list()) == 1
    assert application.memory.get("project_default_crs").value == "EPSG:3857"


def test_memory_extractor_only_extracts_explicit_durable_fact(application):
    extractor = MemoryExtractor()
    run = Run(task_id="task-extractor", agent_id="main", status=RunStatus.COMPLETED)
    result = application.memory.write_candidates(
        extractor.extract(
            AgentRequest(user_input="项目默认 CRS 使用 EPSG:3857"),
            None,
            run,
            result=AgentResult(
                agent_id="main",
                task_id=run.task_id,
                status=AgentResultStatus.SUCCESS,
                summary="已记录",
                trace_id=run.id,
            ),
        )
    )

    assert len(result) == 1
    assert result[0].key == "project_default_crs"
    assert application.memory.write_candidates(
        extractor.extract(
            AgentRequest(user_input="检查当前数据"),
            None,
            run,
            result=AgentResult(
                agent_id="main",
                task_id=run.task_id,
                status=AgentResultStatus.SUCCESS,
                summary="检查完成",
                trace_id=run.id,
            ),
        )
    ) == []


def test_context_uses_structured_working_memory_without_run_state(application):
    from app.core.models import WorkingMemory

    context = application.context_manager.main_context(
        AgentRequest(user_input="继续"),
        [],
        None,
        [],
        working_memory=WorkingMemory(task_id="task-context", active_dataset_ids=["dataset-context"]),
    )

    assert context["working_memory"]["task_id"] == "task-context"
    assert context["working_memory"]["active_dataset_ids"] == ["dataset-context"]
    assert "run_id" not in context["working_memory"]
    assert "turn_count" not in context["working_memory"]


def test_resume_uses_current_task_working_memory_instead_of_old_checkpoint(application):
    task = application.task_service.create("恢复任务", conversation_id="conv-wm-resume-current")
    old_run = Run(id="run-wm-resume-current", task_id=task.id, conversation_id=task.conversation_id, agent_id="main", status=RunStatus.CANCELLED)
    application.store.save_run(old_run)
    version_one = WorkingMemory(task_id=task.id, conversation_id=task.conversation_id, active_dataset_ids=["dataset-v1"])
    application.store.save_working_memory(version_one)
    request = AgentRequest(user_input="继续", conversation_id=task.conversation_id)
    frame = RequestFrame(mode=InteractionMode.CONTINUE_TASK, goal="恢复任务", target_task_id=task.id, target_run_id=old_run.id, needs_planning=True)
    checkpoint = Checkpoint(
        run_id=old_run.id,
        phase="plan_created",
        state={"request": request.model_dump(mode="json"), "request_frame": frame.model_dump(mode="json"), "working_memory": version_one.model_dump(mode="json")},
    )
    version_two = version_one.model_copy(update={"active_dataset_ids": ["dataset-v2"]})
    application.store.save_working_memory(version_two)
    seen: dict[str, WorkingMemory] = {}

    async def fake_model_loop(*args, **kwargs):
        seen["memory"] = kwargs["working_memory"]
        return AgentResult(agent_id="main", task_id=task.id, status=AgentResultStatus.SUCCESS, summary="恢复完成", trace_id=args[1].id)

    application.main_agent._model_loop = fake_model_loop
    prepared = asyncio.run(application.main_agent.prepare_request(request, resume_from=checkpoint))
    asyncio.run(application.main_agent.run(request, prepared=prepared, resume_from=checkpoint))

    assert seen["memory"].active_dataset_ids == ["dataset-v2"]
    assert application.store.get_working_memory(task.id).active_dataset_ids == ["dataset-v2"]


def test_resume_restores_checkpoint_working_memory_when_store_is_missing(application):
    task = application.task_service.create("恢复缺失状态", conversation_id="conv-wm-resume-fallback")
    old_run = Run(id="run-wm-resume-fallback", task_id=task.id, conversation_id=task.conversation_id, agent_id="main", status=RunStatus.CANCELLED)
    application.store.save_run(old_run)
    version_one = WorkingMemory(task_id=task.id, conversation_id=task.conversation_id, active_dataset_ids=["dataset-checkpoint"])
    request = AgentRequest(user_input="继续", conversation_id=task.conversation_id)
    frame = RequestFrame(mode=InteractionMode.CONTINUE_TASK, goal="恢复缺失状态", target_task_id=task.id, target_run_id=old_run.id, needs_planning=True)
    checkpoint = Checkpoint(
        run_id=old_run.id,
        phase="plan_created",
        state={"request": request.model_dump(mode="json"), "request_frame": frame.model_dump(mode="json"), "working_memory": version_one.model_dump(mode="json")},
    )
    seen: dict[str, WorkingMemory] = {}

    async def fake_model_loop(*args, **kwargs):
        seen["memory"] = kwargs["working_memory"]
        return AgentResult(agent_id="main", task_id=task.id, status=AgentResultStatus.SUCCESS, summary="恢复完成", trace_id=args[1].id)

    application.main_agent._model_loop = fake_model_loop
    prepared = asyncio.run(application.main_agent.prepare_request(request, resume_from=checkpoint))
    asyncio.run(application.main_agent.run(request, prepared=prepared, resume_from=checkpoint))

    assert seen["memory"].active_dataset_ids == ["dataset-checkpoint"]
    assert application.store.get_working_memory(task.id).active_dataset_ids == ["dataset-checkpoint"]
