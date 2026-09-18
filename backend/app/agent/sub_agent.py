"""一次性 GIS SubAgent。

SubAgent 使用最小 Context，只能围绕一个主题读取数据和生成局部发现；它不能
创建更多 Agent，也不直接改变项目 Memory。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Dataset,
    Run,
    RunBudget,
    RunStatus,
    SubAgentExecutionResult,
    SubTask,
    ToolCall,
    ToolStatus,
    WorkingMemory,
    WorkingMemoryDelta,
    new_id,
)
from app.events import EventType
from app.execution.tools import ToolExecutor
from app.gis.crs.service import CRSService
from app.gis.errors import as_tool_error
from app.observability import TraceRecorder
from app.runtime.budget import BudgetExceeded, BudgetGuard
from app.runtime.context_manager import ContextManager
from app.runtime.lifecycle import finish_run, start_run
from app.state import StateStore, WorkingMemoryUpdater


class SubAgent:
    def __init__(self, executor: ToolExecutor, store: StateStore, trace: TraceRecorder, *, context_manager: ContextManager | None = None, budget: RunBudget | None = None, services_factory=None) -> None:
        self.executor = executor
        self.store = store
        self.trace = trace
        self.context_manager = context_manager or ContextManager(max_chars=10000)
        self.budget = budget or RunBudget(max_agent_turns=10)
        self.guard = BudgetGuard(self.budget)
        self.services_factory = services_factory

    async def run(
        self,
        request: AgentRequest,
        subtask: SubTask,
        datasets: list[Dataset],
        *,
        parent_task_id: str,
        parent_run_id: str,
        working_memory_snapshot: WorkingMemory | None,
    ) -> SubAgentExecutionResult:
        agent_id = new_id("subagent")
        run = start_run(
            Run(
                parent_run_id=parent_run_id,
                conversation_id=request.conversation_id,
                task_id=parent_task_id,
                agent_id=agent_id,
                metadata={"goal": subtask.goal, "subtask_id": subtask.id, "subtask_goal": subtask.goal},
            )
        )
        self.store.save_run(run)
        await self.trace.emit(parent_run_id, EventType.SUBAGENT_SPAWNED, f"启动 {subtask.goal}", payload={"agent_id": agent_id, "subtask_id": subtask.id}, agent_id=agent_id)
        selected = _dataset_for_subtask(subtask, datasets)
        local_datasets = [selected] if selected is not None else []
        allowed_tools = _allowed_tools(subtask)
        local = self.context_manager.sub_context(request, subtask.model_dump(mode="json"), local_datasets, working_memory=working_memory_snapshot, allowed_tools=allowed_tools)
        findings: list[Any] = [{"scope": "subtask", "goal": subtask.goal, "context_dataset_count": len(local["datasets"])}]
        result_datasets: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []
        budget_exceeded = False
        updater = WorkingMemoryUpdater(self.store)
        delta = WorkingMemoryDelta(source_run_id=run.id)

        def record(tool_result):
            nonlocal delta
            delta = updater.merge_delta(delta, updater.build_delta_from_tool_result(tool_result, run_id=run.id))
            return tool_result

        try:
            self.guard.check_execution_time(run)
            if selected is None:
                raise ValueError("没有可用于该 SubTask 的数据集。")
            inspect = record(await self._call(run, "dataset.inspect", {"dataset_id": selected.id}))
            if inspect.status not in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS}:
                raise RuntimeError(inspect.error.message if inspect.error else "数据检查失败")
            findings.append({"dataset": selected.name, "inspection": inspect.output})
            result_datasets.append(selected.id)
            if subtask.operation == "vector.validate" or "road" in subtask.goal.casefold() or "道路" in subtask.goal:
                validation = record(await self._call(run, "vector.validate", {"dataset_id": selected.id}))
                if validation.status not in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS}:
                    raise RuntimeError(validation.error.message if validation.error else "道路质量检查失败")
                findings.append({"road_quality": validation.output})
                warnings.extend(validation.warnings)
            elif subtask.operation == "raster.slope" or "terrain" in subtask.goal.casefold() or any(word in subtask.goal for word in ("地形", "坡度", "DEM")):
                if selected.kind.value != "RASTER":
                    raise ValueError("terrain SubTask 需要 Raster DEM。")
                slope = record(await self._call(run, "raster.slope", {"dataset_id": selected.id}))
                if slope.status is ToolStatus.FAILED and slope.error and slope.error.code == "CRS_UNIT_MISMATCH":
                    target_crs = CRSService(default_crs=self.executor.services["settings"].default_crs).choose_projected_crs(selected)
                    projected = record(await self._call(run, "crs.reproject", {"dataset_id": selected.id, "target_crs": target_crs}))
                    if projected.status is not ToolStatus.SUCCESS or not projected.datasets:
                        raise RuntimeError(projected.error.message if projected.error else "DEM 重投影失败")
                    result_datasets.extend(projected.datasets)
                    slope = record(await self._call(run, "raster.slope", {"dataset_id": projected.datasets[0]}))
                if slope.status not in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS}:
                    raise RuntimeError(slope.error.message if slope.error else "坡度计算失败")
                result_datasets.extend(slope.datasets)
                findings.append({"terrain": slope.output})
            elif "population" in subtask.goal.casefold() or "人口" in subtask.goal:
                fields = selected.schema.fields if selected.schema else {}
                population_fields = [field for field in fields if any(word in field.casefold() for word in ("pop", "人口", "count"))]
                findings.append({"population_fields": population_fields, "feature_count": selected.schema.feature_count if selected.schema else None})
        except asyncio.CancelledError:
            self.executor.cancel_run(run.id)
            cancelled = finish_run(self.store.get_run(run.id) or run, RunStatus.CANCELLED, error="CANCELLED")
            self.store.save_run(cancelled)
            await self.trace.emit(run.id, EventType.RUN_CANCELLED, "SubAgent 运行已取消", payload={"status": RunStatus.CANCELLED.value}, agent_id=agent_id)
            raise
        except BudgetExceeded as exc:
            errors.append(str(exc))
            budget_exceeded = True
        except Exception as exc:
            error = as_tool_error(exc)
            errors.append(error.message)
        status = AgentResultStatus.BLOCKED if budget_exceeded else AgentResultStatus.FAILED if errors and not findings[1:] else AgentResultStatus.PARTIAL if errors else AgentResultStatus.SUCCESS
        summary = f"{subtask.goal}：{'因预算限制停止' if budget_exceeded else '完成' if status is AgentResultStatus.SUCCESS else '部分完成' if status is AgentResultStatus.PARTIAL else '失败'}"
        final = AgentResult(agent_id=agent_id, task_id=parent_task_id, status=status, summary=summary, findings=findings, datasets=result_datasets, warnings=warnings, error="；".join(errors) if errors else None, trace_id=run.id, evidence=[{"run_id": run.id, "events": len(self.store.list_events(run.id)), "subtask_id": subtask.id}])
        finished = finish_run(
            self.store.get_run(run.id) or run,
            RunStatus.BUDGET_EXCEEDED if budget_exceeded else RunStatus.COMPLETED if status is AgentResultStatus.SUCCESS else RunStatus.PARTIAL_COMPLETED if status is AgentResultStatus.PARTIAL else RunStatus.FAILED,
            error=final.error,
        )
        self.store.save_run(finished.model_copy(update={"metadata": {**finished.metadata, "result": final.model_dump(mode="json")}}))
        await self.trace.emit(parent_run_id, EventType.SUBAGENT_COMPLETED, summary, payload={"agent_id": agent_id, "status": status.value, "result": final.model_dump(mode="json")}, agent_id=agent_id)
        return SubAgentExecutionResult(result=final, working_memory_delta=delta)

    async def _call(self, run: Run, name: str, arguments: dict[str, Any]):
        current = self.store.get_run(run.id) or run
        self.guard.check_turn(current)
        self.guard.check_tool(current)
        self.guard.check_execution_time(current)
        current = current.model_copy(update={"turn_count": current.turn_count + 1, "tool_call_count": current.tool_call_count + 1})
        self.store.save_run(current)
        call = ToolCall(name=name, arguments=arguments, run_id=current.id, agent_id=current.agent_id)
        user_id = self.store.user_id_for_run(current.id)
        services = self.services_factory(user_id) if self.services_factory else self.executor.services if hasattr(self.executor, "services") else {}
        return await self.executor.execute(call, agent_id=run.agent_id, services=services)


def _dataset_for_subtask(subtask: SubTask, datasets: list[Dataset]) -> Dataset | None:
    if subtask.dataset_ids:
        selected = next((item for item in datasets if item.id in subtask.dataset_ids), None)
        if selected is not None:
            return selected
    text = f"{subtask.goal} {subtask.description}".casefold()
    for dataset in datasets:
        if any(word in dataset.name.casefold() for word in ("road", "道路")) and any(word in text for word in ("road", "道路")):
            return dataset
        if any(word in dataset.name.casefold() for word in ("population", "人口", "pop")) and any(word in text for word in ("population", "人口")):
            return dataset
        if any(word in dataset.name.casefold() for word in ("dem", "elevation", "terrain", "高程")) and any(word in text for word in ("terrain", "地形", "dem", "坡度")):
            return dataset
    return datasets[0] if datasets else None


def _allowed_tools(subtask: SubTask) -> list[str]:
    operation = subtask.operation
    if operation == "raster.slope":
        return ["dataset.inspect", "raster.inspect", "raster.slope", "raster.reproject", "crs.reproject"]
    if operation == "vector.validate":
        return ["dataset.inspect", "vector.validate", "vector.repair", "crs.reproject"]
    return ["dataset.inspect", "raster.inspect"]
