"""RunManager：管理后台执行、查询和取消入口。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from app.core.models import AgentRequest, AgentResult, Checkpoint, Run, RunStatus, TaskStatus


class RunManager:
    def __init__(self, main_agent, store, metrics=None) -> None:
        self.main_agent = main_agent
        self.store = store
        self.metrics = metrics
        self._active: dict[str, asyncio.Task[AgentResult]] = {}
        self._finished: dict[str, AgentResult] = {}

    def submit(
        self,
        request: AgentRequest,
        *,
        resume_from: Checkpoint | None = None,
        metadata: dict[str, object] | None = None,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> Run:
        task, run = self.main_agent.prepare(request, metadata=metadata)
        execution = asyncio.create_task(self._execute(run, request, task, resume_from, on_model_delta=on_model_delta))
        self._active[run.id] = execution
        if self.metrics:
            self.metrics.increment("runs.submitted")
        return run

    async def _execute(self, run: Run, request: AgentRequest, task, checkpoint: Checkpoint | None, *, on_model_delta: Callable[[str], Awaitable[None]] | None = None) -> AgentResult:
        try:
            result = await self.main_agent.run(request, prepared=(task, run), resume_from=checkpoint, on_model_delta=on_model_delta)
            self._finished[run.id] = result
            if self.metrics:
                self.metrics.increment(f"runs.finished.{result.status.value.casefold()}")
            return result
        except asyncio.CancelledError:
            current = self.store.get_run(run.id) or run
            if current.status not in {RunStatus.COMPLETED, RunStatus.PARTIAL_COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.BUDGET_EXCEEDED}:
                self.store.save_run(current.model_copy(update={"status": RunStatus.CANCELLED, "error": "CANCELLED"}))
            self.main_agent.task_service.update(task, status=TaskStatus.CANCELLED, result="运行已取消")
            result = AgentResult(agent_id="main", task_id=run.task_id, status="CANCELLED", summary="运行已取消。", error="CANCELLED", trace_id=run.id)
            self._finished[run.id] = result
            if self.metrics:
                self.metrics.increment("runs.finished.cancelled")
            return result
        finally:
            self._active.pop(run.id, None)

    async def execute(self, request: AgentRequest) -> AgentResult:
        run = self.submit(request)
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
            current = self.store.get_run(run_id)
            if current and current.status in {RunStatus.RUNNING, RunStatus.WAITING_TOOL, RunStatus.WAITING_SUBAGENT, RunStatus.PLANNING}:
                self.store.save_run(current.model_copy(update={"status": RunStatus.CANCELLED, "error": "CANCELLED"}))
            if self.metrics:
                self.metrics.increment("runs.cancelled")
            return True
        run = self.store.get_run(run_id)
        if run and run.status in {RunStatus.RUNNING, RunStatus.WAITING_TOOL, RunStatus.WAITING_SUBAGENT, RunStatus.PLANNING}:
            self.store.save_run(run.model_copy(update={"status": RunStatus.CANCELLED}))
            if self.metrics:
                self.metrics.increment("runs.cancelled")
            return True
        return False
