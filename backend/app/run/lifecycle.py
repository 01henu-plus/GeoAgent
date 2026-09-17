"""把已验证的 RequestFrame 绑定到真实 Task/Run 生命周期。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from app.core.models import (
    AgentRequest,
    InteractionMode,
    RequestFrame,
    RequestResolutionStatus,
    Run,
    RunStatus,
    Task,
    TaskStatus,
)
from app.runtime.lifecycle import start_run
from app.state import StateStore
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
        blocked_reason: str | None = None,
    ) -> None:
        self.request = request
        self.frame = frame
        self.action = action
        self.task = task
        self.run = run
        self.target_task = target_task
        self.target_run = target_run
        self.blocked_reason = blocked_reason


class RequestLifecycleBinder:
    """只负责决定本轮使用哪个 Task/Run，不负责规划和工具执行。"""

    def __init__(self, store: StateStore, task_service: TaskService) -> None:
        self.store = store
        self.task_service = task_service

    def bind(
        self,
        request: AgentRequest,
        frame: RequestFrame,
        *,
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
            run = self._new_run(request, task, frame, metadata=metadata)
            return PreparedRequest(request=request, frame=frame, action=LifecycleAction.CREATE_TASK, task=task, run=run)

        if frame.mode in {InteractionMode.CONTINUE_TASK, InteractionMode.MODIFY_TASK}:
            if target_task is None:
                return self._blocked(request, frame, "没有找到要继续或修改的任务", metadata=metadata)
            task = self.task_service.update(target_task, status=TaskStatus.RUNNING)
            run = self._new_run(
                request,
                task,
                frame,
                parent_run_id=target_run.id if target_run else None,
                metadata=metadata,
            )
            return PreparedRequest(request=request, frame=frame, action=LifecycleAction.BIND_TASK, task=task, run=run, target_task=target_task, target_run=target_run)

        if frame.mode is InteractionMode.RETRY_TASK:
            if target_run is None or not _is_failed(target_run):
                return self._blocked(request, frame, "没有找到可重试的失败运行", metadata=metadata)
            task = target_task or (self.store.get_task(target_run.task_id) if target_run.task_id else None)
            if task is None:
                return self._blocked(request, frame, "失败运行没有关联任务", metadata=metadata)
            task = self.task_service.update(task, status=TaskStatus.RUNNING)
            run = self._new_run(request, task, frame, parent_run_id=target_run.id, metadata={"retry_of": target_run.id, **(metadata or {})})
            return PreparedRequest(request=request, frame=frame, action=LifecycleAction.RETRY_RUN, task=task, run=run, target_task=task, target_run=target_run)

        if frame.mode is InteractionMode.CANCEL_TASK:
            if target_run is None or not _is_active(target_run):
                return self._blocked(request, frame, "没有找到可取消的运行", metadata=metadata)
            return PreparedRequest(request=request, frame=frame, action=LifecycleAction.CANCEL_RUN, task=target_task, run=target_run, target_task=target_task, target_run=target_run)

        # QUERY/CHAT 只产生 command/audit Run，不创建业务 Task。
        command_run = self._new_run(request, target_task, frame, metadata={"command_run": True, **(metadata or {})})
        return PreparedRequest(request=request, frame=frame, action=LifecycleAction.COMMAND_RUN, task=target_task, run=command_run, target_task=target_task, target_run=target_run)

    def _new_run(
        self,
        request: AgentRequest,
        task: Task | None,
        frame: RequestFrame,
        *,
        parent_run_id: str | None = None,
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
                parent_run_id=parent_run_id,
                agent_id="main",
                metadata=run_metadata,
            )
        )
        self.store.save_run(run)
        return run

    def _blocked(self, request: AgentRequest, frame: RequestFrame, reason: str | None = None, *, metadata: dict[str, object] | None = None) -> PreparedRequest:
        issues = frame.blocking_issues or ([reason] if reason else ["请求需要澄清"])
        blocked_frame = frame.model_copy(update={"blocking_issues": list(dict.fromkeys(issues))})
        target_task = self.store.get_task(frame.target_task_id) if frame.target_task_id else None
        run = self._new_run(request, target_task, blocked_frame, metadata={"command_run": True, "blocked": True, **(metadata or {})})
        return PreparedRequest(
            request=request,
            frame=blocked_frame,
            action=LifecycleAction.BLOCKED,
            task=target_task,
            run=run,
            target_task=target_task,
            blocked_reason="；".join(blocked_frame.blocking_issues),
        )


def _is_failed(run: Run) -> bool:
    return run.status is RunStatus.FAILED or (bool(run.error) and run.status not in {RunStatus.WAITING_USER, RunStatus.WAITING_APPROVAL, RunStatus.CANCELLED})


def _is_active(run: Run) -> bool:
    return run.status in {
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.RUNNING,
        RunStatus.WAITING_TOOL,
        RunStatus.WAITING_SUBAGENT,
        RunStatus.WAITING_USER,
        RunStatus.WAITING_APPROVAL,
        RunStatus.RETRYING,
        RunStatus.REPLANNING,
        RunStatus.VALIDATING,
    }


__all__ = ["LifecycleAction", "PreparedRequest", "RequestLifecycleBinder"]
