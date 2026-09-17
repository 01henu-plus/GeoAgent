"""RunManager：管理后台执行、查询和取消入口。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from app.core.models import AgentRequest, AgentResult, Checkpoint, Run, RunStatus, TaskStatus
from app.run.lifecycle import LifecycleAction, PreparedRequest
from app.run.predicates import is_active_run


class RunManager:
    def __init__(self, main_agent, store, metrics=None) -> None:
        self.main_agent = main_agent
        self.store = store
        self.metrics = metrics
        self._active: dict[str, asyncio.Task[AgentResult]] = {}
        self._finished: dict[str, AgentResult] = {}

    async def submit(
        self,
        request: AgentRequest,
        *,
        resume_from: Checkpoint | None = None,
        metadata: dict[str, object] | None = None,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> Run:
        prepared = await self.main_agent.prepare_request(request, metadata=metadata, resume_from=resume_from)
        if prepared.action is LifecycleAction.CANCEL_RUN:
            if prepared.target_run is None or not await self.cancel(prepared.target_run.id):
                raise RuntimeError("目标运行当前不可取消")
            return self.store.get_run(prepared.target_run.id) or prepared.target_run
        if prepared.run is None:
            raise RuntimeError(prepared.blocked_reason or "请求没有可执行的 Run")
        execution = asyncio.create_task(self._execute(prepared, resume_from, on_model_delta=on_model_delta))
        self._active[prepared.run.id] = execution
        if self.metrics:
            self.metrics.increment("runs.submitted")
        return prepared.run

    async def _execute(self, prepared: PreparedRequest, checkpoint: Checkpoint | None, *, on_model_delta: Callable[[str], Awaitable[None]] | None = None) -> AgentResult:
        run = prepared.run
        if run is None:
            raise RuntimeError("PreparedRequest 缺少 Run")
        try:
            result = await self.main_agent.run(prepared.request, prepared=prepared, resume_from=checkpoint, on_model_delta=on_model_delta)
            self._finished[run.id] = result
            if self.metrics:
                self.metrics.increment(f"runs.finished.{result.status.value.casefold()}")
            return result
        except asyncio.CancelledError:
            current = self.store.get_run(run.id) or run
            if current.status not in {RunStatus.COMPLETED, RunStatus.PARTIAL_COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.BUDGET_EXCEEDED}:
                current = current.model_copy(update={"status": RunStatus.CANCELLED, "error": "CANCELLED"})
                self.store.save_run(current)
            if prepared.task is not None and prepared.action in {LifecycleAction.CREATE_TASK, LifecycleAction.BIND_TASK, LifecycleAction.RETRY_RUN}:
                self.main_agent.task_service.update(prepared.task, status=TaskStatus.CANCELLED, result="运行已取消")
            result = AgentResult(agent_id="main", task_id=current.task_id, status="CANCELLED", summary="运行已取消。", error="CANCELLED", trace_id=run.id)
            self._finished[run.id] = result
            if self.metrics:
                self.metrics.increment("runs.finished.cancelled")
            return result
        finally:
            self._active.pop(run.id, None)

    async def execute(self, request: AgentRequest) -> AgentResult:
        run = await self.submit(request)
        return await self.wait(run.id)

    async def wait(self, run_id: str) -> AgentResult:
        task = self._active.get(run_id)
        if task is not None:
            return await task
        if run_id in self._finished:
            return self._finished[run_id]
        run = self.store.get_run(run_id)
        if run and isinstance(run.metadata.get("result"), dict):
            result = AgentResult.model_validate(run.metadata["result"])
            self._finished[run_id] = result
            return result
        raise KeyError(f"运行不存在或尚未提交：{run_id}")

    def get(self, run_id: str):
        return self.store.get_run(run_id)

    def list(self, limit: int = 50):
        return self.store.list_runs(limit)

    def is_active(self, run_id: str) -> bool:
        task = self._active.get(run_id)
        return task is not None and not task.done()

    def forget(self, run_id: str) -> None:
        self._finished.pop(run_id, None)

    async def cancel(self, run_id: str) -> bool:
        task = self._active.get(run_id)
        if task and not task.done():
            self.main_agent.executor.cancel_run(run_id)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if self.metrics:
                self.metrics.increment("runs.cancelled")
            return True
        run = self.store.get_run(run_id)
        if run and is_active_run(run):
            cancelled = run.model_copy(update={"status": RunStatus.CANCELLED, "error": "CANCELLED"})
            self.store.save_run(cancelled)
            self._mark_task_cancelled(cancelled)
            self._finished[run_id] = AgentResult(agent_id="main", task_id=run.task_id, status="CANCELLED", summary="运行已取消。", error="CANCELLED", trace_id=run_id)
            if self.metrics:
                self.metrics.increment("runs.cancelled")
            return True
        return False

    def _mark_task_cancelled(self, run: Run) -> None:
        if not run.task_id:
            return
        task = self.store.get_task(run.task_id)
        if task and task.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING, TaskStatus.WAITING, TaskStatus.BLOCKED}:
            self.main_agent.task_service.update(task, status=TaskStatus.CANCELLED, result="运行已取消")
