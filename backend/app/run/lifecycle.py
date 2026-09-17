"""把已验证的 RequestFrame 绑定到真实 Task/Run 生命周期。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from app.core.models import (
    AgentRequest,
    InteractionMode,
    RequestFrame,
    RequestResolutionStatus,
    RequestResources,
    Run,
    Task,
    TaskStatus,
    WorkingMemory,
)
from app.run.predicates import is_active_run, is_retryable_failed_run
from app.runtime.lifecycle import start_run
from app.state import StateStore, WorkingMemoryUpdater
from app.task.service import TaskService


class LifecycleAction(StrEnum):
    CREATE_TASK = "create_task"
    BIND_TASK = "bind_task"
    RETRY_RUN = "retry_run"
    CANCEL_RUN = "cancel_run"
    COMMAND_RUN = "command_run"
    BLOCKED = "blocked"


class PreparedRequest:
    """请求理解和生命周期绑定后的单一执行上下文。"""

    def __init__(
        self,
        *,
        request: AgentRequest,
        frame: RequestFrame,
        action: LifecycleAction,
        task: Task | None = None,
        run: Run | None = None,
        target_task: Task | None = None,
        target_run: Run | None = None,
        working_memory: WorkingMemory | None = None,
        working_memory_created: bool = False,
        blocked_reason: str | None = None,
    ) -> None:
        self.request = request
        self.frame = frame
        self.action = action
        self.task = task
        self.run = run
        self.target_task = target_task
        self.target_run = target_run
        self.working_memory = working_memory
        self.working_memory_created = working_memory_created
        self.blocked_reason = blocked_reason


class RequestLifecycleBinder:
    """只负责决定本轮使用哪个 Task/Run，不负责规划和工具执行。"""

    def __init__(self, store: StateStore, task_service: TaskService) -> None:
        self.store = store
        self.task_service = task_service
        self.working_memory_updater = WorkingMemoryUpdater(store)

    def bind(
        self,
        request: AgentRequest,
        frame: RequestFrame,
        *,
        request_resources: RequestResources | None = None,
        metadata: dict[str, object] | None = None,
    ) -> PreparedRequest:
        if frame.resolution_status is not RequestResolutionStatus.RESOLVED:
            return self._blocked(request, frame, metadata=metadata)

        target_task = self.store.get_task(frame.target_task_id) if frame.target_task_id else None
        target_run = self.store.get_run(frame.target_run_id) if frame.target_run_id else None
        if target_task and target_task.conversation_id not in {None, request.conversation_id}:
            return self._blocked(request, frame, "目标任务不属于当前会话", metadata=metadata)
        if target_run and target_run.conversation_id not in {None, request.conversation_id}:
            return self._blocked(request, frame, "目标运行不属于当前会话", metadata=metadata)

        if frame.mode is InteractionMode.NEW_TASK:
            task = self.task_service.create(frame.goal, conversation_id=request.conversation_id)
            task = self.task_service.update(task, status=TaskStatus.RUNNING)
            working_memory, working_memory_created = self._bind_working_memory(task, frame, request_resources)
            run = self._new_run(request, task, frame, metadata=metadata)
            return PreparedRequest(request=request, frame=frame, action=LifecycleAction.CREATE_TASK, task=task, run=run, working_memory=working_memory, working_memory_created=working_memory_created)

        if frame.mode in {InteractionMode.CONTINUE_TASK, InteractionMode.MODIFY_TASK}:
            if target_task is None:
                return self._blocked(request, frame, "没有找到要继续或修改的任务", metadata=metadata)
            task = self.task_service.update(target_task, status=TaskStatus.RUNNING)
            lineage = dict(metadata or {})
            if target_run is not None:
                lineage["continued_from"] = target_run.id
            working_memory, working_memory_created = self._bind_working_memory(task, frame, request_resources)
            run = self._new_run(
                request,
                task,
                frame,
                metadata=lineage,
            )
            return PreparedRequest(request=request, frame=frame, action=LifecycleAction.BIND_TASK, task=task, run=run, target_task=target_task, target_run=target_run, working_memory=working_memory, working_memory_created=working_memory_created)

        if frame.mode is InteractionMode.RETRY_TASK:
            if target_run is None or not is_retryable_failed_run(target_run):
                return self._blocked(request, frame, "没有找到可重试的失败运行", metadata=metadata)
            task = target_task or (self.store.get_task(target_run.task_id) if target_run.task_id else None)
            if task is None:
                return self._blocked(request, frame, "失败运行没有关联任务", metadata=metadata)
            task = self.task_service.update(task, status=TaskStatus.RUNNING)
            working_memory, working_memory_created = self._bind_working_memory(task, frame, request_resources)
            run = self._new_run(request, task, frame, metadata={"retry_of": target_run.id, **(metadata or {})})
            return PreparedRequest(request=request, frame=frame, action=LifecycleAction.RETRY_RUN, task=task, run=run, target_task=task, target_run=target_run, working_memory=working_memory, working_memory_created=working_memory_created)

        if frame.mode is InteractionMode.CANCEL_TASK:
            if target_run is None or not is_active_run(target_run):
                return self._blocked(request, frame, "没有找到可取消的运行", metadata=metadata)
            working_memory = self.store.get_working_memory(target_task.id) if target_task else None
            return PreparedRequest(request=request, frame=frame, action=LifecycleAction.CANCEL_RUN, task=target_task, run=target_run, target_task=target_task, target_run=target_run, working_memory=working_memory)

        # QUERY/CHAT 只产生 command/audit Run，不创建业务 Task。
        command_run = self._new_run(request, target_task, frame, metadata={"command_run": True, **(metadata or {})})
        working_memory = self.store.get_working_memory(target_task.id) if target_task else None
        return PreparedRequest(request=request, frame=frame, action=LifecycleAction.COMMAND_RUN, task=target_task, run=command_run, target_task=target_task, target_run=target_run, working_memory=working_memory)

    def _bind_working_memory(
        self,
        task: Task,
        frame: RequestFrame,
        request_resources: RequestResources | None,
    ) -> tuple[WorkingMemory, bool]:
        created = self.store.get_working_memory(task.id) is None
        memory = self.working_memory_updater.load_or_create(task.id, task.conversation_id)
        memory = self.working_memory_updater.apply_request(memory, frame, request_resources)
        self.store.save_working_memory(memory)
        return memory, created

    def _new_run(
        self,
        request: AgentRequest,
        task: Task | None,
        frame: RequestFrame,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Run:
        run_metadata: dict[str, Any] = {
            "goal": frame.goal,
            "request_id": request.request_id,
            "interaction_mode": frame.mode.value,
            "request_frame": frame.model_dump(mode="json"),
            **(metadata or {}),
        }
        run = start_run(
            Run(
                conversation_id=request.conversation_id,
                task_id=task.id if task else None,
                agent_id="main",
                metadata=run_metadata,
            )
        )
        self.store.save_run(run)
        return run

    def _blocked(self, request: AgentRequest, frame: RequestFrame, reason: str | None = None, *, metadata: dict[str, object] | None = None) -> PreparedRequest:
        issues = [*frame.blocking_issues, *([reason] if reason else [])]
        issues = list(dict.fromkeys(issue for issue in issues if issue)) or ["请求需要澄清"]
        blocked_frame = frame.model_copy(update={"blocking_issues": issues})
        target_task = self.store.get_task(frame.target_task_id) if frame.target_task_id else None
        run = self._new_run(request, target_task, blocked_frame, metadata={"command_run": True, "blocked": True, **(metadata or {})})
        working_memory = self.store.get_working_memory(target_task.id) if target_task else None
        return PreparedRequest(
            request=request,
            frame=blocked_frame,
            action=LifecycleAction.BLOCKED,
            task=target_task,
            run=run,
            target_task=target_task,
            working_memory=working_memory,
            blocked_reason="；".join(blocked_frame.blocking_issues),
        )


__all__ = ["LifecycleAction", "PreparedRequest", "RequestLifecycleBinder"]
