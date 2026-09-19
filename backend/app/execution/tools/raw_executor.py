"""ToolExecutionCycle 使用的单一 Raw Tool 执行边界。"""

from __future__ import annotations

from typing import Any

from app.core.models import Run, RunStatus, ToolCall, ToolResult, new_id
from app.execution.tools.executor import ToolExecutor
from app.runtime.budget import BudgetGuard
from app.state import StateStore


class RawToolExecutor:
    """只负责一次底层 Tool 调用，不处理重试、修复或结果验收。"""

    def __init__(
        self,
        *,
        executor: ToolExecutor,
        store: StateStore,
        guard: BudgetGuard,
        services_factory=None,
    ) -> None:
        self.executor = executor
        self.store = store
        self.guard = guard
        self.services_factory = services_factory

    async def execute(
        self,
        run: Run,
        name: str,
        arguments: dict[str, Any],
        *,
        call_id: str | None = None,
        attempt: int = 1,
    ) -> ToolResult:
        current = self.store.get_run(run.id) or run
        self.guard.check_turn(current)
        self.guard.check_execution_time(current)
        self.guard.check_tool(current)
        current = current.model_copy(update={"tool_call_count": current.tool_call_count + 1, "status": RunStatus.WAITING_TOOL})
        self.store.save_run(current)
        agent_id = current.agent_id or "main"
        call = ToolCall(
            id=call_id or new_id("call"),
            name=name,
            arguments=arguments,
            run_id=current.id,
            agent_id=agent_id,
            attempt=attempt,
        )
        user_id = self.store.user_id_for_run(current.id)
        services = self.services_factory(user_id) if self.services_factory else getattr(self.executor, "services", {})
        return await self.executor.execute(call, agent_id=agent_id, services=services)


__all__ = ["RawToolExecutor"]
