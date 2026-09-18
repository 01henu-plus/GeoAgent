"""一次性 GIS SubAgent。

SubAgent 使用最小 Context，只围绕一个主题读取数据和生成局部发现。它共享
MainAgent 的 ToolExecutionCycle，但只产生 WorkingMemoryDelta，不直接写父任务状态。
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Dataset,
    LoopDirective,
    Run,
    RunBudget,
    RunStatus,
    SubAgentExecutionResult,
    SubTask,
    ToolCall,
    ToolResult,
    WorkingMemory,
    WorkingMemoryDelta,
    new_id,
)
from app.decision import FailureAnalyzer, ResultVerifier
from app.events import EventType
from app.execution.tools import ToolExecutor
from app.gis.errors import as_tool_error
from app.observability import TraceRecorder
from app.runtime.budget import BudgetExceeded, BudgetGuard
from app.runtime.context_manager import ContextManager
from app.runtime.lifecycle import finish_run, start_run
from app.runtime.tool_execution_cycle import ExecutionOutcome, ToolExecutionCycle
from app.state import StateStore, WorkingMemoryUpdater


class _SubAgentOutcomeStop(Exception):
    def __init__(self, outcome: ExecutionOutcome, label: str) -> None:
        super().__init__(label)
        self.outcome = outcome
        self.label = label


class SubAgent:
    def __init__(
        self,
        executor: ToolExecutor,
        store: StateStore,
        trace: TraceRecorder,
        *,
        context_manager: ContextManager | None = None,
        budget: RunBudget | None = None,
        services_factory=None,
        registry=None,
        failure_analyzer: FailureAnalyzer | None = None,
        verifier: ResultVerifier | None = None,
        default_crs: str | None = None,
    ) -> None:
        self.executor = executor
        self.store = store
        self.trace = trace
        self.budget = budget or RunBudget(max_agent_turns=10)
        self.context_manager = context_manager or ContextManager(max_tokens=self.budget.subagent_context_tokens)
        self.guard = BudgetGuard(self.budget)
        self.services_factory = services_factory
        executor_services = getattr(executor, "services", {})
        self.registry = registry or executor_services.get("registry")
        self.default_crs = default_crs or getattr(executor_services.get("settings"), "default_crs", "EPSG:3857")
        if self.registry is None:
            raise ValueError("SubAgent 需要用户作用域 Dataset Registry")
        self.tool_execution_cycle = ToolExecutionCycle(
            raw_executor=self._raw_tool_for_cycle,
            tool_registry=self.executor.registry,
            registry=self.registry,
            trace=self.trace,
            failure_analyzer=failure_analyzer or FailureAnalyzer(),
            verifier=verifier or ResultVerifier(),
            budget=self.budget,
            default_crs=self.default_crs,
        )

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
        updater = WorkingMemoryUpdater(self.store)
        delta = WorkingMemoryDelta(source_run_id=run.id)
        terminal_outcome: ExecutionOutcome | None = None
        failure_rationale: str | None = None
        directive = LoopDirective.CONTINUE
        budget_exceeded = False

        async def execute_action(name: str, arguments: dict[str, Any]) -> ExecutionOutcome:
            nonlocal delta
            outcome = await self.tool_execution_cycle.execute(
                run,
                name,
                arguments,
                user_id=self.store.user_id_for_run(run.id),
            )
            if outcome.accepted:
                # 只在本地构造 Delta；SubAgent 绝不调用 update_from_tool_result。
                delta = updater.merge_delta(delta, updater.build_delta_from_tool_result(outcome.result, run_id=run.id))
                warnings.extend(outcome.result.warnings)
            return outcome

        try:
            self.guard.check_execution_time(run)
            if selected is None:
                raise ValueError("没有可用于该 SubTask 的数据集。")

            inspect = await execute_action("dataset.inspect", {"dataset_id": selected.id})
            _require_accepted(inspect, "数据检查")
            findings.append({"dataset": selected.name, "inspection": inspect.result.output})

            if subtask.operation == "vector.validate" or "road" in subtask.goal.casefold() or "道路" in subtask.goal:
                validation = await execute_action("vector.validate", {"dataset_id": selected.id})
                _require_accepted(validation, "道路质量检查")
                findings.append({"road_quality": validation.result.output})
            elif subtask.operation == "raster.slope" or "terrain" in subtask.goal.casefold() or any(word in subtask.goal for word in ("地形", "坡度", "DEM")):
                if selected.kind.value != "RASTER":
                    raise ValueError("terrain SubTask 需要 Raster DEM。")
                slope = await execute_action("raster.slope", {"dataset_id": selected.id})
                _require_accepted(slope, "坡度计算")
                result_datasets.extend(slope.result.datasets)
                findings.append({"terrain": slope.result.output})
            elif "population" in subtask.goal.casefold() or "人口" in subtask.goal:
                fields = selected.schema.fields if selected.schema else {}
                population_fields = [field for field in fields if any(word in field.casefold() for word in ("pop", "人口", "count"))]
                findings.append({"population_fields": population_fields, "feature_count": selected.schema.feature_count if selected.schema else None})
        except _SubAgentOutcomeStop as stop:
            terminal_outcome = stop.outcome
            directive = terminal_outcome.directive
            failure_rationale = terminal_outcome.rationale or _outcome_message(terminal_outcome, stop.label)
            errors.append(failure_rationale)
        except asyncio.CancelledError:
            self.executor.cancel_run(run.id)
            cancelled = finish_run(self.store.get_run(run.id) or run, RunStatus.CANCELLED, error="CANCELLED")
            self.store.save_run(cancelled)
            await self.trace.emit(run.id, EventType.RUN_CANCELLED, "SubAgent 运行已取消", payload={"status": RunStatus.CANCELLED.value}, agent_id=agent_id)
            raise
        except BudgetExceeded as exc:
            errors.append(str(exc))
            failure_rationale = str(exc)
            directive = LoopDirective.ABORT
            budget_exceeded = True
        except Exception as exc:
            error = as_tool_error(exc)
            errors.append(error.message)
            failure_rationale = error.message
            directive = LoopDirective.ABORT

        if terminal_outcome is not None:
            if directive in {LoopDirective.ASK_USER, LoopDirective.REPLAN}:
                status = AgentResultStatus.BLOCKED
                result_error = "WAITING_USER" if directive is LoopDirective.ASK_USER else "REPLAN_REQUIRED"
            else:
                status = AgentResultStatus.FAILED
                result_error = errors[-1] if errors else "SubAgent 执行失败"
        elif budget_exceeded:
            status = AgentResultStatus.BLOCKED
            result_error = errors[-1]
        elif errors:
            status = AgentResultStatus.PARTIAL if findings[1:] else AgentResultStatus.FAILED
            result_error = errors[-1]
        else:
            status = AgentResultStatus.SUCCESS
            result_error = None

        summary = f"{subtask.goal}：{'因预算限制停止' if budget_exceeded else '完成' if status is AgentResultStatus.SUCCESS else '部分完成' if status is AgentResultStatus.PARTIAL else '失败'}"
        if directive is LoopDirective.ASK_USER:
            summary = f"{subtask.goal}：需要补充信息。"
        elif directive is LoopDirective.REPLAN:
            summary = f"{subtask.goal}：当前执行策略不适用，需要重新规划。"
        final = AgentResult(
            agent_id=agent_id,
            task_id=parent_task_id,
            status=status,
            summary=summary,
            findings=findings,
            datasets=list(dict.fromkeys(result_datasets)),
            warnings=warnings,
            error=result_error,
            trace_id=run.id,
            evidence=[{"run_id": run.id, "events": len(self.store.list_events(run.id)), "subtask_id": subtask.id}],
        )
        finished = finish_run(
            self.store.get_run(run.id) or run,
            RunStatus.BUDGET_EXCEEDED if budget_exceeded else RunStatus.WAITING_USER if status is AgentResultStatus.BLOCKED else RunStatus.COMPLETED if status is AgentResultStatus.SUCCESS else RunStatus.PARTIAL_COMPLETED if status is AgentResultStatus.PARTIAL else RunStatus.FAILED,
            error=final.error,
        )
        self.store.save_run(finished.model_copy(update={"metadata": {**finished.metadata, "result": final.model_dump(mode="json"), "directive": directive.value}}))
        await self.trace.emit(parent_run_id, EventType.SUBAGENT_COMPLETED, summary, payload={"agent_id": agent_id, "status": status.value, "directive": directive.value, "result": final.model_dump(mode="json")}, agent_id=agent_id)
        return SubAgentExecutionResult(result=final, working_memory_delta=delta, directive=directive, failure_rationale=failure_rationale)

    async def _execute_tool_raw(self, run: Run, name: str, arguments: dict[str, Any], *, call_id: str | None = None, attempt: int = 1) -> ToolResult:
        current = self.store.get_run(run.id) or run
        self.guard.check_turn(current)
        self.guard.check_tool(current)
        self.guard.check_execution_time(current)
        # SubAgent 当前是确定性执行器，没有独立认知回合；这里只累计工具调用。
        current = current.model_copy(update={"tool_call_count": current.tool_call_count + 1, "status": RunStatus.WAITING_TOOL})
        self.store.save_run(current)
        call = ToolCall(id=call_id or new_id("call"), name=name, arguments=arguments, run_id=current.id, agent_id=current.agent_id, attempt=attempt)
        user_id = self.store.user_id_for_run(current.id)
        services = self.services_factory(user_id) if self.services_factory else self.executor.services if hasattr(self.executor, "services") else {}
        return await self.executor.execute(call, agent_id=current.agent_id, services=services)

    async def _call(self, run: Run, name: str, arguments: dict[str, Any], *, call_id: str | None = None, attempt: int = 1) -> ToolResult:
        """Raw Tool hook，保留给测试和旧调用方；业务执行统一经过 Cycle。"""

        return await self._execute_tool_raw(run, name, arguments, call_id=call_id, attempt=attempt)

    async def _raw_tool_for_cycle(self, run: Run, name: str, arguments: dict[str, Any], *, call_id: str | None = None, attempt: int = 1) -> ToolResult:
        try:
            return await self._call(run, name, arguments, call_id=call_id, attempt=attempt)
        except TypeError as exc:
            # 保留现有测试和外部 hook 的三参数兼容形式；正式 executor 仍会记录 attempt。
            if "unexpected keyword argument" not in str(exc):
                raise
            return await self._call(run, name, arguments)


def _require_accepted(outcome: ExecutionOutcome, label: str) -> ToolResult:
    if not outcome.accepted:
        raise _SubAgentOutcomeStop(outcome, label)
    return outcome.result


def _outcome_message(outcome: ExecutionOutcome, label: str) -> str:
    if outcome.verification_problems:
        return "；".join(outcome.verification_problems)
    if outcome.result.error is not None:
        return outcome.result.error.message
    return outcome.rationale or f"{label}失败。"


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
