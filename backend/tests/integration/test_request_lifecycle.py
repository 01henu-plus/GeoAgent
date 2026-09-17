import asyncio
import logging

from app.core.models import (
    AgentRequest,
    Artifact,
    ArtifactKind,
    Dataset,
    DatasetKind,
    RequestFrame,
    RequestResolutionStatus,
    RequestResources,
    Run,
    RunStatus,
    StateSnapshot,
    TaskStatus,
)
from app.models import ModelAdapter, ModelRequest
from app.understanding.interpreter import RequestInterpreter
from app.understanding.models import ReferenceResolution
from app.understanding.pipeline import RequestUnderstandingPipeline


def _await(coro):
    return asyncio.run(coro)


def _create_waiting_task(application, conversation_id: str = "conv-lifecycle"):
    task = application.task_service.create("分析栅格", conversation_id=conversation_id)
    return application.task_service.update(task, status=TaskStatus.WAITING)


def test_continue_binds_existing_task_without_creating_new_task(application):
    task = _create_waiting_task(application)
    previous = Run(id="run-continue-source", task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.WAITING_USER)
    application.store.save_run(previous)

    result = _await(application.ask(AgentRequest(user_input="继续", conversation_id=task.conversation_id)))

    assert result.task_id == task.id
    assert len(application.store.list_tasks(task.conversation_id)) == 1
    assert application.store.get_run(result.trace_id).task_id == task.id
    assert application.store.get_run(result.trace_id).metadata["interaction_mode"] == "continue_task"
    assert application.store.get_run(result.trace_id).parent_run_id is None
    assert application.store.get_run(result.trace_id).metadata["continued_from"] == previous.id


def test_modify_binds_existing_task_without_replacing_goal(application):
    task = _create_waiting_task(application)
    previous = Run(id="run-modify-source", task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.WAITING_USER)
    application.store.save_run(previous)

    result = _await(application.ask(AgentRequest(user_input="不对，把范围改成上海", conversation_id=task.conversation_id)))

    assert result.task_id == task.id
    assert len(application.store.list_tasks(task.conversation_id)) == 1
    assert application.store.get_task(task.id).goal == "分析栅格"
    assert application.store.get_run(result.trace_id).task_id == task.id
    assert application.store.get_run(result.trace_id).parent_run_id is None
    assert application.store.get_run(result.trace_id).metadata["continued_from"] == previous.id


def test_request_dataset_has_priority_over_historical_dataset(application):
    old_run = Run(id="run-old-resource", task_id="task-old-resource", agent_id="main", conversation_id="conv-resource", status=RunStatus.COMPLETED)
    old_dataset = Dataset(id="dataset-old", name="old.tif", kind=DatasetKind.RASTER, path="old.tif", format="tif", created_by_run_id=old_run.id)
    new_dataset = Dataset(id="dataset-new", name="new.tif", kind=DatasetKind.RASTER, path="new.tif", format="tif")
    application.store.save_run(old_run)
    application.store.save_dataset(old_dataset)
    resources = RequestResources(datasets=[new_dataset])

    frame = _await(
        application.main_agent.request_understanding.understand(
            "conv-resource",
            "检查这个数据",
            request_resources=resources,
        )
    )

    assert frame.references[0].target_id == new_dataset.id


def test_request_attachment_has_priority_over_historical_dataset(application):
    task = _create_waiting_task(application, "conv-attachment")
    old_run = Run(id="run-old-attachment", task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.COMPLETED)
    old_dataset = Dataset(id="dataset-old-attachment", name="old.tif", kind=DatasetKind.RASTER, path="old.tif", format="tif", created_by_run_id=old_run.id)
    attachment = Dataset(id="dataset-attachment", name="A.tif", kind=DatasetKind.RASTER, path="A.tif", format="tif")
    application.store.save_run(old_run)
    application.store.save_dataset(old_dataset)
    resources = RequestResources(datasets=[attachment])

    frame = _await(
        application.main_agent.request_understanding.understand(
            task.conversation_id,
            "用这个数据继续",
            request_resources=resources,
        )
    )

    assert frame.references[0].target_id == attachment.id


def test_recent_dataset_is_used_when_request_has_no_resource(application):
    task = _create_waiting_task(application, "conv-recent-resource")
    old_run = Run(id="run-recent-resource", task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.COMPLETED)
    recent = Dataset(id="dataset-recent", name="recent.tif", kind=DatasetKind.RASTER, path="recent.tif", format="tif", created_by_run_id=old_run.id)
    application.store.save_run(old_run)
    application.store.save_dataset(recent)

    frame = _await(
        application.main_agent.request_understanding.understand(
            task.conversation_id,
            "用刚才的数据继续",
        )
    )

    assert frame.references[0].target_id == recent.id


def test_explicit_referenced_run_has_priority_over_recent_run(application):
    task = _create_waiting_task(application, "conv-run-resource")
    recent = Run(id="run-recent", task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.COMPLETED)
    selected = Run(id="run-selected", task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.COMPLETED)
    application.store.save_run(selected)
    application.store.save_run(recent)
    resources = RequestResources(runs=[selected])

    frame = _await(
        application.main_agent.request_understanding.understand(
            task.conversation_id,
            "查看上一轮运行",
            request_resources=resources,
        )
    )

    assert any(item.type == "run" and item.target_id == selected.id for item in frame.references)


def test_explicit_run_reference_requires_conversation_ownership(application):
    same = Run(id="run-same-conversation", task_id="task-same", agent_id="main", conversation_id="conversation-same", status=RunStatus.COMPLETED)
    other = Run(id="run-other-conversation", task_id="task-other", agent_id="main", conversation_id="conversation-other", status=RunStatus.COMPLETED)
    application.store.save_run(same)
    application.store.save_run(other)
    resolver = application.main_agent.request_understanding.reference_resolver

    assert [item.id for item in resolver.resolve_runs([same.id], conversation_id="conversation-same")] == [same.id]
    assert resolver.resolve_runs([other.id], conversation_id="conversation-same") == []
    assert [item.id for item in resolver.resolve_runs(["latest"], conversation_id="conversation-same")] == [same.id]
    assert resolver.resolve_runs(["latest"], conversation_id="conversation-same", exclude_run_id=same.id) == []


def test_blocked_reason_does_not_drop_existing_blocking_issue(application):
    other_task = application.task_service.create("其他会话任务", conversation_id="conversation-other")
    frame = RequestFrame(
        mode="query",
        goal="查看结果",
        target_task_id=other_task.id,
        resolution_status=RequestResolutionStatus.RESOLVED,
        blocking_issues=["原有阻塞原因"],
    )

    prepared = application.main_agent.lifecycle_binder.bind(
        AgentRequest(user_input="查看结果", conversation_id="conversation-current"),
        frame,
    )

    assert "原有阻塞原因" in prepared.frame.blocking_issues
    assert "目标任务不属于当前会话" in prepared.frame.blocking_issues


def test_retry_creates_run_on_failed_task_without_new_task(application):
    task = _create_waiting_task(application)
    failed = Run(task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.FAILED, error="工具失败")
    application.store.save_run(failed)

    result = _await(application.ask(AgentRequest(user_input="再试一次", conversation_id=task.conversation_id)))

    retry = application.store.get_run(result.trace_id)
    assert retry.id != failed.id
    assert retry.task_id == task.id
    assert retry.parent_run_id is None
    assert retry.metadata["retry_of"] == failed.id
    assert len(application.store.list_tasks(task.conversation_id)) == 1


def test_retry_without_failed_run_is_blocked_without_business_task(application):
    result = _await(application.ask(AgentRequest(user_input="再试一次", conversation_id="conv-empty")))

    assert result.status.value == "BLOCKED"
    assert result.error == "NEEDS_CLARIFICATION"
    assert application.store.list_tasks("conv-empty") == []
    assert application.store.get_run(result.trace_id).task_id is None


def test_cancel_cancels_real_active_run_and_task(application):
    application.main_agent._model_loop = _wait_forever
    request = AgentRequest(user_input="分析这份数据", conversation_id="conv-cancel")

    async def scenario():
        run = await application.conversations.submit(request)
        await asyncio.sleep(0)
        result = await application.ask(AgentRequest(user_input="取消", conversation_id=request.conversation_id))
        return run, result

    run, result = _await(scenario())

    assert result.status.value == "CANCELLED"
    assert application.store.get_run(run.id).status is RunStatus.CANCELLED
    assert application.store.get_task(run.task_id).status is TaskStatus.CANCELLED


def test_unresolved_reference_blocks_before_execution(application):
    result = _await(application.ask(AgentRequest(user_input="用刚才的数据继续", conversation_id="conv-no-data")))

    assert result.status.value == "BLOCKED"
    assert application.store.get_run(result.trace_id).metadata["interaction_mode"] == "continue_task"
    assert application.store.list_tasks("conv-no-data") == []


def test_conversation_isolation_does_not_resolve_other_conversation_artifact(application):
    run = Run(task_id="task-a", agent_id="main", conversation_id="conversation-a", status=RunStatus.COMPLETED)
    application.store.save_run(run)
    application.store.save_artifact(Artifact(id="artifact-a", name="A.tif", kind=ArtifactKind.DATASET, run_id=run.id))

    frame = _await(RequestUnderstandingPipeline(application.store).understand("conversation-b", "用刚才的数据继续"))

    assert frame.references == []
    assert "刚才的数据" in frame.unresolved_references
    assert frame.resolution_status.value == "needs_clarification"


def test_new_task_does_not_inherit_active_target(application):
    task = _create_waiting_task(application)

    frame = _await(RequestUnderstandingPipeline(application.store).understand(task.conversation_id, "新建一个上海范围分析"))

    assert frame.mode.value == "new_task"
    assert frame.target_task_id is None
    assert frame.target_run_id is None


def test_overlapping_reference_uses_longest_mention(application):
    task = _create_waiting_task(application)
    run = Run(task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.COMPLETED)
    application.store.save_run(run)
    application.store.save_artifact(Artifact(id="artifact-a", name="A.tif", kind=ArtifactKind.DATASET, run_id=run.id))

    frame = _await(RequestUnderstandingPipeline(application.store).understand(task.conversation_id, "使用这个任务继续"))

    mentions = [item.mention for item in frame.references]
    assert "这个任务" in mentions
    assert "这个" not in mentions


def test_structured_interpreter_fallback_is_observable(caplog):
    class BrokenAdapter(ModelAdapter):
        supports_structured_output = True

        async def complete(self, request: ModelRequest):
            raise RuntimeError("模型不可用")

    caplog.set_level(logging.WARNING)
    frame = _await(
        RequestInterpreter().interpret(
            message="请分析栅格数据",
            state=StateSnapshot(conversation_id="conv-fallback"),
            resolution=ReferenceResolution(),
            model_adapter=BrokenAdapter(),
        )
    )

    assert frame.mode.value == "new_task"
    assert "回退确定性解析" in caplog.text


async def _wait_forever(*args, **kwargs):
    await asyncio.Event().wait()
