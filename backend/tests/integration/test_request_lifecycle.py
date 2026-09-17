import asyncio
import logging

from app.core.models import (
    AgentRequest,
    Artifact,
    ArtifactKind,
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

    result = _await(application.ask(AgentRequest(user_input="继续", conversation_id=task.conversation_id)))

    assert result.task_id == task.id
    assert len(application.store.list_tasks(task.conversation_id)) == 1
    assert application.store.get_run(result.trace_id).task_id == task.id
    assert application.store.get_run(result.trace_id).metadata["interaction_mode"] == "continue_task"


def test_modify_binds_existing_task_without_replacing_goal(application):
    task = _create_waiting_task(application)

    result = _await(application.ask(AgentRequest(user_input="不对，把范围改成上海", conversation_id=task.conversation_id)))

    assert result.task_id == task.id
    assert len(application.store.list_tasks(task.conversation_id)) == 1
    assert application.store.get_task(task.id).goal == "分析栅格"
    assert application.store.get_run(result.trace_id).task_id == task.id


def test_retry_creates_run_on_failed_task_without_new_task(application):
    task = _create_waiting_task(application)
    failed = Run(task_id=task.id, agent_id="main", conversation_id=task.conversation_id, status=RunStatus.FAILED, error="工具失败")
    application.store.save_run(failed)

    result = _await(application.ask(AgentRequest(user_input="再试一次", conversation_id=task.conversation_id)))

    retry = application.store.get_run(result.trace_id)
    assert retry.id != failed.id
    assert retry.task_id == task.id
    assert retry.parent_run_id == failed.id
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
    assert application.store.get_run(result.trace_id).metadata["interaction_mode"] == "new_task"
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
