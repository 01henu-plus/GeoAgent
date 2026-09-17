"""临时 SubAgent 管理器。"""

from __future__ import annotations

from app.agent.scheduler import AgentScheduler
from app.agent.sub_agent import SubAgent
from app.core.models import (
    AgentRequest,
    AgentResult,
    Dataset,
    SubAgentExecutionResult,
    SubTask,
    WorkingMemory,
    WorkingMemoryDelta,
)
from app.task.graph import TaskGraph


class AgentManager:
    def __init__(self, sub_agent: SubAgent, *, max_parallel: int = 3, max_subagents: int = 5, timeout_seconds: int = 180) -> None:
        self.sub_agent = sub_agent
        self.scheduler = AgentScheduler(max_parallel=max_parallel, timeout_seconds=timeout_seconds)
        self.max_subagents = max_subagents

    async def run(
        self,
        request: AgentRequest,
        tasks: list[SubTask],
        datasets: list[Dataset],
        *,
        parent_task_id: str,
        parent_run_id: str,
        working_memory_snapshot: WorkingMemory | None,
    ) -> list[SubAgentExecutionResult]:
        if len(tasks) > self.max_subagents:
            raise ValueError(f"SubAgent 数量超过预算：{self.max_subagents}")
        by_id: dict[str, SubAgentExecutionResult] = {}
        for batch in TaskGraph(tasks).parallel_batches():
            jobs = [
                lambda task=task: self.sub_agent.run(
                    request,
                    task,
                    datasets,
                    parent_task_id=parent_task_id,
                    parent_run_id=parent_run_id,
                    working_memory_snapshot=working_memory_snapshot.model_copy(deep=True) if working_memory_snapshot else None,
                )
                for task in batch
            ]
            values = await self.scheduler.run_parallel(jobs)
            for value, task in zip(values, batch, strict=True):
                if isinstance(value, SubAgentExecutionResult):
                    by_id[task.id] = value
                else:
                    by_id[task.id] = SubAgentExecutionResult(
                        result=AgentResult(agent_id="scheduler", task_id=parent_task_id, status="FAILED", summary=f"{task.goal} 调度失败", error=str(value), trace_id=parent_run_id),
                        working_memory_delta=WorkingMemoryDelta(),
                    )
        return [by_id[task.id] for task in tasks]
