"""Runtime-owned implementations for the actions selected by the controller.

The decision source is deliberately outside this module.  Once a decision says
"call a tool", "build a plan", "replan", or "delegate", this module owns the
execution semantics and returns a :class:`RuntimeTransition` to the runtime.
MainAgent still contains compatibility wrappers for older callers, but the
canonical runtime does not call them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.agent.manager import AgentManager
from app.core.models import (
    AgentDecision,
    AgentRequest,
    AgentResultStatus,
    ErrorCategory,
    FailureAction,
    LoopDirective,
    Plan,
    ReplanContext,
    RequestFrame,
    Run,
    RunBudget,
    RunStatus,
    SubAgentExecutionResult,
    SubTask,
    Task,
    TaskStatus,
    ToolError,
    ToolResult,
    ToolStatus,
    WorkingMemory,
)
from app.decision import (
    Planner,
    Replanner,
    ReplanNotPossible,
    TaskDecomposer,
)
from app.events import EventType
from app.gis.crs.service import CRSService
from app.runtime.agent_loop import PlanLoopOutcome, resolve_plan_arguments
from app.runtime.agent_runtime import RuntimeTransition
from app.runtime.budget import BudgetExceeded, BudgetGuard
from app.runtime.checkpoint_codec import RuntimeCheckpointCodec
from app.runtime.session import AgentRuntimeSession
from app.runtime.tool_execution_cycle import ExecutionOutcome, ToolExecutionCycle
from app.state import StateStore, WorkingMemoryUpdater
from app.task.service import TaskService

CheckpointWriter = Callable[[str, str, dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class DelegationExecution:
    """一次运行时委派的结构化观察，不是最终 AgentResult。"""

    tasks: tuple[SubTask, ...]
    executions: tuple[SubAgentExecutionResult, ...]
    directive: LoopDirective
    findings: tuple[Any, ...]
    dataset_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    observation: dict[str, Any]
    subagent_results: tuple[dict[str, Any], ...]
    latest_failure: dict[str, Any] | None
    fingerprint: str


class RuntimeActionHandlers:
    """Runtime action capability implementations.

    This class receives concrete domain services instead of the whole
    ``MainAgent``.  It therefore remains usable by the canonical controller
    without making MainAgent the hidden execution service locator.
    """

    def __init__(
        self,
        *,
        store: StateStore,
        trace,
        budget: RunBudget,
        guard: BudgetGuard,
        planner: Planner | Callable[[], Planner],
        replanner: Replanner | Callable[[], Replanner],
        decomposer: TaskDecomposer,
        agent_manager: AgentManager | Callable[[], AgentManager],
        task_service: TaskService,
        plan_loop,
        tool_execution_cycle: ToolExecutionCycle | Callable[[], ToolExecutionCycle],
        working_memory_updater: WorkingMemoryUpdater,
        registry,
        default_crs: str,
        checkpoint: Callable[[], CheckpointWriter],
        step_checkpoint: Callable[..., Awaitable[None]] | None = None,
        checkpoint_codec: RuntimeCheckpointCodec,
    ) -> None:
        self.store = store
        self.trace = trace
        self.budget = budget
        self.guard = guard
        self.planner = planner
        self.replanner = replanner
        self.decomposer = decomposer
        self.agent_manager = agent_manager
        self.task_service = task_service
        self.plan_loop = plan_loop
        self.tool_execution_cycle = tool_execution_cycle
        self.working_memory_updater = working_memory_updater
        self.registry = registry
        self.default_crs = default_crs
        self.checkpoint = checkpoint
        self.step_checkpoint = step_checkpoint
        self.checkpoint_codec = checkpoint_codec

    async def handle_tool(
        self,
        decision: AgentDecision,
        _state,
        *,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        request_frame: RequestFrame | None,
        session: AgentRuntimeSession,
    ) -> RuntimeTransition:
        """执行一批模型工具调用，并只接受通过 Cycle 的结果。"""

        raw_calls = decision.metadata.get("raw_tool_calls") or []
        session.protocol_messages.append(
            {"role": "assistant", "content": decision.metadata.get("content", ""), "tool_calls": raw_calls}
        )
        invalid_calls = {
            item.get("id"): item
            for item in decision.metadata.get("invalid_tool_calls", [])
            if isinstance(item, dict)
        }
        observations: list[dict[str, Any]] = []
        batch_failure: dict[str, Any] | None = None
        for call in decision.normalized_tool_calls():
            invalid = invalid_calls.get(call.id)
            if invalid is not None:
                result = ToolResult(
                    call_id=call.id,
                    status=ToolStatus.FAILED,
                    error=ToolError(
                        code="INVALID_TOOL_ARGUMENTS",
                        category=ErrorCategory.INPUT,
                        message=f"模型工具参数不是有效 JSON：{invalid.get('error', '未知错误')}",
                    ),
                )
                outcome = None
                observation = _invalid_tool_call_observation(call.id, result, str(invalid.get("error", "未知错误")))
            else:
                outcome = await self._tool_cycle().execute(
                    session.run,
                    call.name,
                    call.arguments,
                    user_id=self.store.user_id_for_run(session.run.id),
                    call_id=call.id,
                )
                result = outcome.result
                observation = _execution_observation(outcome)
            observations.append(observation)
            session.findings.append(
                {
                    "tool": call.name,
                    "status": result.status.value,
                    "accepted": outcome.accepted if outcome else False,
                    "verification_problems": outcome.verification_problems if outcome else [],
                    "recovery_action": outcome.recovery_action.value if outcome and outcome.recovery_action else None,
                    "directive": outcome.directive.value if outcome else LoopDirective.ABORT.value,
                    "verified": outcome.verified if outcome else False,
                    "attempts": outcome.attempts if outcome else 1,
                    "output": result.output,
                    "error": result.error.model_dump(mode="json") if result.error else None,
                }
            )
            if outcome is not None and outcome.accepted:
                self._accept_tool_result(session, result)
                session.dataset_ids.update(result.datasets)
                session.artifact_ids.update(result.artifacts)
            elif outcome is not None:
                batch_failure = _failure_from_outcome(outcome, call.name, deterministic=False)
            session.protocol_messages.append(_protocol_tool_message(outcome or observation))

        session.latest_failure = batch_failure
        session.latest_observation = _batch_observation(observations)
        current_run = self.store.get_run(session.run.id) or session.run
        session.run = current_run.model_copy(update={"status": RunStatus.RUNNING})
        self.store.save_run(session.run)
        session.protocol_messages = _compact_protocol_messages(
            session.protocol_messages,
            max_tokens=self.budget.protocol_history_tokens,
        )
        await self._checkpoint_runtime(request, request_frame, session, "model_tool_completed")
        return RuntimeTransition(
            observation=session.latest_observation,
            latest_failure=session.latest_failure,
            clear_failure=session.latest_failure is None,
            findings=tuple(session.findings),
            dataset_ids=tuple(sorted(session.dataset_ids)),
            artifact_ids=tuple(sorted(session.artifact_ids)),
        )

    async def handle_plan(
        self,
        decision: AgentDecision,
        state,
        *,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        request_frame: RequestFrame | None,
        session: AgentRuntimeSession,
    ) -> RuntimeTransition:
        """构建计划并开启 runtime-owned fast path。"""

        current_run = self.store.get_run(session.run.id) or session.run
        self.store.save_run(current_run.model_copy(update={"status": RunStatus.PLANNING}))
        frame = request_frame or RequestFrame(mode="new_task", goal=decision.plan_goal or state.goal)
        if decision.plan_goal and decision.plan_goal != frame.goal:
            frame = frame.model_copy(update={"goal": decision.plan_goal})
        new_plan = self._planner().build(frame, session.datasets)
        session.current_plan = new_plan
        session.original_plan = new_plan.model_copy(deep=True)
        session.fast_path_enabled = True
        session.completed_steps = set()
        session.step_outputs = {}
        session.latest_failure = None
        session.run = (self.store.get_run(session.run.id) or session.run).model_copy(update={"status": RunStatus.RUNNING})
        self.store.save_run(session.run)
        await self.trace.emit(
            session.run.id,
            EventType.PLAN_CREATED,
            f"生成运行时计划：{len(new_plan.steps)} 步",
            payload={**new_plan.model_dump(mode="json"), "source": "agent_runtime"},
            agent_id="main",
        )
        await self._checkpoint_runtime(request, request_frame, session, "plan_created")
        return RuntimeTransition(current_plan=new_plan, clear_failure=True)

    async def execute_plan_step(
        self,
        step,
        *,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        request_frame: RequestFrame | None,
        session: AgentRuntimeSession,
    ) -> RuntimeTransition:
        """执行 Plan fast path 的一个步骤。"""

        if step.tool_name is None:
            session.completed_steps.add(step.id)
            step.status = TaskStatus.SUCCEEDED
            return RuntimeTransition(completed_steps=(step.id,), current_plan=session.current_plan)

        arguments = self._complete_plan_arguments(
            step.tool_name,
            resolve_plan_arguments(step.arguments, session.step_outputs),
            user_id=request.user_id,
        )
        outcome = await self._tool_cycle().execute(
            session.run,
            step.tool_name,
            arguments,
            user_id=request.user_id,
        )
        current_run = self.store.get_run(session.run.id) or session.run
        session.run = current_run.model_copy(update={"status": RunStatus.RUNNING})
        self.store.save_run(session.run)
        session.findings.append(
            {
                "step_id": step.id,
                "tool": step.tool_name,
                "status": outcome.result.status.value,
                "accepted": outcome.accepted,
                "verified": outcome.verified,
                "verification_problems": list(outcome.verification_problems),
                "directive": outcome.directive.value,
                "attempts": outcome.attempts,
                "datasets": list(outcome.result.datasets),
                "artifacts": list(outcome.result.artifacts),
            }
        )
        if outcome.accepted:
            self._accept_tool_result(session, outcome.result)
            session.dataset_ids.update(outcome.result.datasets)
            session.artifact_ids.update(outcome.result.artifacts)
            session.latest_failure = None
            session.completed_steps.add(step.id)
            step.status = TaskStatus.SUCCEEDED
            session.step_outputs[step.id] = {
                "dataset_id": outcome.result.datasets[-1] if outcome.result.datasets else None,
                "dataset_ids": list(outcome.result.datasets),
                "artifact_ids": list(outcome.result.artifacts),
                "output": outcome.result.output,
            }
        else:
            session.fast_path_enabled = False
            session.latest_failure = _failure_from_outcome(outcome, step.tool_name, deterministic=True, step_id=step.id)
            step.status = TaskStatus.FAILED
        observation = _execution_observation(outcome)
        session.latest_observation = observation
        await self._step_checkpoint(
            request,
            request_frame,
            session,
            "plan_step_completed" if outcome.accepted else "plan_step_failed",
        )
        return RuntimeTransition(
            observation=observation,
            latest_failure=session.latest_failure,
            clear_failure=session.latest_failure is None,
            directive=outcome.directive,
            findings=tuple(session.findings),
            dataset_ids=tuple(sorted(session.dataset_ids)),
            artifact_ids=tuple(sorted(session.artifact_ids)),
            current_plan=session.current_plan,
            completed_steps=(step.id,) if outcome.accepted else (),
            step_outputs={step.id: session.step_outputs[step.id]} if outcome.accepted else {},
        )

    async def handle_replan(
        self,
        decision: AgentDecision,
        _state,
        *,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        request_frame: RequestFrame | None,
        session: AgentRuntimeSession,
    ) -> RuntimeTransition:
        """将 REPLAN 决策接到 Replanner，不在执行层自行扩展决策循环。"""

        current_plan = session.current_plan
        if current_plan is None:
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.BLOCKED,
                error="REPLAN_REQUIRED",
                final_response="当前没有可供重新规划的执行计划。",
                directive=LoopDirective.REPLAN,
            )
        failure = session.latest_failure or {}
        failed_step = next((item for item in current_plan.steps if item.id == failure.get("step_id")), None)
        error_message = str(failure.get("error") or decision.reasoning_summary or "当前执行失败，需要重新规划。")
        error_code = str(failure.get("error_code") or "REPLAN_REQUIRED")
        verification_problems = [str(item) for item in failure.get("verification_problems", [])]
        failed_result = ToolResult(
            call_id=f"replan_{failed_step.id if failed_step else 'runtime'}",
            status=ToolStatus.SUCCESS if verification_problems and not failure.get("error_code") else ToolStatus.FAILED,
            error=None
            if verification_problems and not failure.get("error_code")
            else ToolError(code=error_code, category=ErrorCategory.EXECUTION, message=error_message),
        )
        failed_outcome = ExecutionOutcome(
            result=failed_result,
            verified=False,
            verification_problems=verification_problems,
            recovery_action=FailureAction.REPLAN,
            attempts=int(failure.get("attempts") or 1),
            accepted=False,
            directive=LoopDirective.REPLAN,
            rationale=error_message,
        )
        outcome = PlanLoopOutcome(
            completed_steps=frozenset(session.completed_steps),
            findings=tuple(session.findings),
            output_ids=tuple(sorted(session.dataset_ids)),
            artifacts=tuple(sorted(session.artifact_ids)),
            errors=(error_message,),
            step_outputs=dict(session.step_outputs),
            failed_step=failed_step,
            failed_outcome=failed_outcome,
            directive=LoopDirective.REPLAN,
        )
        frame = request_frame or RequestFrame(mode="new_task", goal=request.user_input)
        original_plan = session.original_plan or current_plan.model_copy(deep=True)
        current_run = self.store.get_run(run.id) or run
        self.store.save_run(current_run.model_copy(update={"status": RunStatus.REPLANNING}))
        try:
            revised, next_state = await self._replan(
                request,
                run,
                session.datasets,
                frame,
                original_plan,
                current_plan,
                outcome,
                {"previous_replan_reasons": list(session.previous_replan_reasons)},
            )
        except ReplanNotPossible as exc:
            running = self.store.get_run(run.id) or run
            if running.status is RunStatus.REPLANNING:
                self.store.save_run(running.model_copy(update={"status": RunStatus.RUNNING}))
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.BLOCKED,
                error="REPLAN_REQUIRED",
                final_response=str(exc),
                directive=LoopDirective.REPLAN,
                latest_failure=failure,
            )
        except BudgetExceeded as exc:
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.BLOCKED,
                error=str(exc),
                directive=LoopDirective.ABORT,
                latest_failure=failure,
            )

        session.current_plan = revised
        session.original_plan = original_plan
        session.completed_steps = set(next_state["completed_steps"])
        session.step_outputs = dict(next_state["step_outputs"])
        session.findings = list(next_state["findings"])
        session.previous_replan_reasons = list(next_state["previous_replan_reasons"])
        session.latest_failure = None
        session.fast_path_enabled = True
        session.run = (self.store.get_run(run.id) or run).model_copy(update={"status": RunStatus.RUNNING})
        self.store.save_run(session.run)
        return RuntimeTransition(
            current_plan=revised,
            clear_failure=True,
            directive=LoopDirective.CONTINUE,
            observation={"replanned": True, "revision": revised.revision},
            findings=tuple(session.findings),
            dataset_ids=tuple(sorted(session.dataset_ids)),
            artifact_ids=tuple(sorted(session.artifact_ids)),
        )

    async def handle_delegate(
        self,
        decision: AgentDecision,
        _state,
        *,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        request_frame: RequestFrame | None,
        session: AgentRuntimeSession,
    ) -> RuntimeTransition:
        if task is None:
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.BLOCKED,
                error="WAITING_USER",
                final_response="当前请求没有可委派的业务任务。",
            )
        tasks = decision.subtasks or self.decomposer.decompose(request, session.datasets)
        fingerprint = _delegation_fingerprint(tasks)
        if fingerprint in session.completed_delegation_fingerprints:
            observation = {
                "type": "delegation",
                "code": "DELEGATION_NO_PROGRESS",
                "completed": 0,
                "total": len(tasks),
                "directive": LoopDirective.CONTINUE.value,
                "fingerprint": fingerprint,
                "message": "相同委派已经执行过，本轮没有新的资源或失败条件变化。",
            }
            session.latest_observation = observation
            return RuntimeTransition(
                observation=observation,
                directive=LoopDirective.CONTINUE,
                subagent_results=tuple(session.subagent_results),
            )
        delegation = await self._execute_delegation(
            request,
            run,
            task,
            session.datasets,
            tasks,
            plan=session.current_plan,
            request_frame=request_frame,
            working_memory=session.working_memory,
            fingerprint=fingerprint,
        )
        session.fast_path_enabled = False
        session.completed_delegation_fingerprints.add(delegation.fingerprint)
        session.subagent_results = _merge_subagent_views(session.subagent_results, delegation.subagent_results)
        session.findings.extend(delegation.findings)
        session.dataset_ids.update(delegation.dataset_ids)
        session.artifact_ids.update(delegation.artifact_ids)
        session.latest_observation = delegation.observation
        session.latest_failure = delegation.latest_failure
        return RuntimeTransition(
            observation=delegation.observation,
            directive=delegation.directive,
            latest_failure=delegation.latest_failure,
            clear_failure=delegation.latest_failure is None,
            findings=tuple(session.findings),
            dataset_ids=tuple(sorted(session.dataset_ids)),
            artifact_ids=tuple(sorted(session.artifact_ids)),
            subagent_results=tuple(session.subagent_results),
        )

    async def execute_delegation(
        self,
        request: AgentRequest,
        run: Run,
        task: Task,
        datasets,
        tasks=None,
        *,
        plan: Plan | None,
        request_frame: RequestFrame | None,
        working_memory: WorkingMemory | None,
        fingerprint: str,
        completed_fingerprints: set[str] | None = None,
        legacy_plan_progress: bool = False,
    ) -> DelegationExecution:
        """兼容旧委派调用方的薄包装，实际逻辑仍在 runtime 内。"""

        return await self._execute_delegation(
            request,
            run,
            task,
            datasets,
            tasks,
            plan=plan,
            request_frame=request_frame,
            working_memory=working_memory,
            fingerprint=fingerprint,
            completed_fingerprints=completed_fingerprints,
            legacy_plan_progress=legacy_plan_progress,
        )

    async def _execute_delegation(
        self,
        request: AgentRequest,
        run: Run,
        task: Task,
        datasets,
        tasks=None,
        *,
        plan: Plan | None,
        request_frame: RequestFrame | None,
        working_memory: WorkingMemory | None,
        fingerprint: str,
        completed_fingerprints: set[str] | None = None,
        legacy_plan_progress: bool = False,
    ) -> DelegationExecution:
        tasks = list(tasks or self.decomposer.decompose(request, datasets))
        self.guard.check_subagents(len(tasks))
        attached = self.task_service.attach_subtasks(task, tasks)
        task.subtasks = attached.subtasks
        task.updated_at = attached.updated_at
        for subtask in tasks:
            await self.trace.emit(
                run.id,
                EventType.SUBTASK_CREATED,
                subtask.goal,
                payload=subtask.model_dump(mode="json"),
                agent_id="main",
            )
        if legacy_plan_progress and plan is not None:
            _set_plan_step_status(plan, "decompose", TaskStatus.SUCCEEDED)
            _set_plan_step_status(plan, "parallel", TaskStatus.RUNNING)
        await self._checkpoint_writer()(
            run.id,
            "delegation_started",
            {
                **_checkpoint_state(request, plan, datasets, request_frame),
                "subtask_ids": [item.id for item in tasks],
                "delegation_fingerprint": fingerprint,
                "completed_delegation_fingerprints": sorted(completed_fingerprints or set()),
                "runtime_delegation": not legacy_plan_progress,
            },
        )
        self.store.save_run(run.model_copy(update={"status": RunStatus.WAITING_SUBAGENT}))
        parent_memory = working_memory or self.store.get_working_memory(task.id)
        executions = await self._agent_manager().run(
            request,
            tasks,
            datasets,
            parent_task_id=task.id,
            parent_run_id=run.id,
            working_memory_snapshot=parent_memory,
        )
        deltas = [item.working_memory_delta for item in executions]
        if parent_memory is not None:
            parent_memory = self.working_memory_updater.merge_deltas(parent_memory, deltas)
            self.store.save_working_memory(parent_memory)
        current_run = self.store.get_run(run.id) or run
        self.store.save_run(current_run.model_copy(update={"status": RunStatus.RUNNING}))

        views = tuple(
            _subagent_result_view(subtask, execution)
            for subtask, execution in zip(tasks, executions, strict=True)
        )
        findings = tuple(dict(item) for item in views)
        output_ids = tuple(sorted({item for delta in deltas for item in delta.added_dataset_ids}))
        artifact_ids = tuple(sorted({item for delta in deltas for item in delta.added_artifact_ids}))
        required = [execution for execution, subtask in zip(executions, tasks, strict=True) if subtask.required]
        directive = _aggregate_subagent_directive(required)
        latest_failure = _delegation_failure(executions, tasks)
        observation = {
            "type": "delegation",
            "completed": sum(1 for item in executions if item.result.status is AgentResultStatus.SUCCESS),
            "total": len(executions),
            "directive": directive.value,
            "fingerprint": fingerprint,
            "results": [dict(item) for item in views],
        }
        await self.trace.emit(
            run.id,
            EventType.DELEGATION_COMPLETED,
            f"委派完成：{observation['completed']}/{observation['total']}",
            payload={
                "subtask_count": len(tasks),
                "success_count": observation["completed"],
                "blocked_count": sum(1 for item in executions if item.result.status is AgentResultStatus.BLOCKED),
                "failed_count": sum(1 for item in executions if item.result.status is AgentResultStatus.FAILED),
                "directive": directive.value,
                "dataset_count": len(output_ids),
                "artifact_count": len(artifact_ids),
                "fingerprint": fingerprint,
            },
            agent_id="main",
        )
        await self._checkpoint_writer()(
            run.id,
            "delegation_completed",
            {
                **_checkpoint_state(request, plan, datasets, request_frame),
                "subtask_ids": [item.id for item in tasks],
                "delegation_fingerprint": fingerprint,
                "subagent_results": [dict(item) for item in views],
                "dataset_ids": list(output_ids),
                "artifact_ids": list(artifact_ids),
                "directive": directive.value,
                "working_memory_refs": _working_memory_refs(parent_memory),
                "runtime_delegation": not legacy_plan_progress,
            },
        )
        return DelegationExecution(
            tasks=tuple(tasks),
            executions=tuple(executions),
            directive=directive,
            findings=findings,
            dataset_ids=output_ids,
            artifact_ids=artifact_ids,
            observation=observation,
            subagent_results=views,
            latest_failure=latest_failure,
            fingerprint=fingerprint,
        )

    async def _replan(
        self,
        request: AgentRequest,
        run: Run,
        datasets,
        request_frame: RequestFrame | None,
        original_plan: Plan,
        current_plan: Plan,
        outcome: PlanLoopOutcome,
        state: dict[str, Any],
    ) -> tuple[Plan, dict[str, Any]]:
        current_run = self.store.get_run(run.id) or run
        self.guard.check_replan(current_run)
        next_count = current_run.replan_count + 1
        failed = outcome.failed_outcome
        failed_step = outcome.failed_step
        reasons = list(state.get("previous_replan_reasons", []))
        reasons.append(_replan_reason(failed_step, failed))
        current_memory = self.store.get_working_memory(run.task_id) if run.task_id else None
        current_dataset_ids = list(
            dict.fromkeys([*(current_memory.active_dataset_ids if current_memory else []), *outcome.output_ids])
        )
        current_artifact_ids = list(
            dict.fromkeys([*(current_memory.active_artifact_ids if current_memory else []), *outcome.artifacts])
        )
        context = ReplanContext(
            goal=current_plan.goal,
            original_plan=original_plan,
            current_plan=current_plan,
            current_revision=current_plan.revision,
            completed_steps=sorted(outcome.completed_steps),
            step_outputs=_compact_replan_step_outputs(outcome.step_outputs),
            failed_step=failed_step,
            failed_tool_name=failed_step.tool_name if failed_step else None,
            failed_arguments=dict(failed_step.arguments) if failed_step else {},
            error_code=failed.result.error.code if failed and failed.result.error else None,
            error_message=failed.result.error.message if failed and failed.result.error else None,
            verification_problems=list(failed.verification_problems) if failed else [],
            recovery_action=failed.recovery_action if failed else None,
            directive=outcome.directive,
            attempts=failed.attempts if failed else 1,
            current_dataset_ids=current_dataset_ids,
            current_artifact_ids=current_artifact_ids,
            replan_count=next_count,
            previous_replan_reasons=reasons,
        )
        replanning_run = current_run.model_copy(update={"status": RunStatus.REPLANNING, "replan_count": next_count})
        self.store.save_run(replanning_run)
        await self._checkpoint_writer()(
            run.id,
            "replan_started",
            {
                "request": request.model_dump(mode="json"),
                "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
                "plan": current_plan.model_dump(mode="json"),
                "original_plan": original_plan.model_dump(mode="json"),
                "replan_context": context.model_dump(mode="json"),
                "replan_count": next_count,
                **_plan_state_from_outcome(outcome, reasons),
            },
        )
        revised = self._replanner().replan(context, request_frame or RequestFrame(mode="new_task", goal=context.goal), datasets)
        self.store.save_run(replanning_run.model_copy(update={"status": RunStatus.RUNNING}))
        next_state = _plan_state_from_outcome(outcome, reasons)
        next_state["errors"] = []
        await self._checkpoint_writer()(
            run.id,
            "replan_completed",
            {
                "request": request.model_dump(mode="json"),
                "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
                "plan": revised.model_dump(mode="json"),
                "original_plan": original_plan.model_dump(mode="json"),
                "replan_count": next_count,
                **next_state,
            },
        )
        await self.trace.emit(
            run.id,
            EventType.PLAN_CREATED,
            f"生成 Replan revision {revised.revision}",
            payload={"revision": revised.revision, "source": "replan", "previous_revision": current_plan.revision},
            agent_id="main",
        )
        return revised, next_state

    def _accept_tool_result(self, session: AgentRuntimeSession, result: ToolResult) -> None:
        updated = self.working_memory_updater.update_from_tool_result(
            session.run.task_id,
            result,
            run_id=session.run.id,
        )
        if updated is not None:
            session.working_memory = updated

    def _planner(self) -> Planner:
        value = self.planner
        return value() if callable(value) else value

    def _replanner(self) -> Replanner:
        value = self.replanner
        return value() if callable(value) else value

    def _tool_cycle(self) -> ToolExecutionCycle:
        value = self.tool_execution_cycle
        return value() if callable(value) else value

    def _agent_manager(self) -> AgentManager:
        value = self.agent_manager
        return value() if callable(value) else value

    def _complete_plan_arguments(self, tool_name: str, arguments: dict[str, Any], *, user_id: str | None) -> dict[str, Any]:
        completed = dict(arguments)
        if tool_name in {"crs.reproject", "raster.reproject"} and completed.get("target_crs") == "auto":
            dataset = self.registry.for_user(user_id).resolve(str(completed.get("dataset_id", "")))
            if dataset is not None:
                completed["target_crs"] = CRSService(default_crs=self.default_crs).choose_projected_crs(dataset)
        return completed

    async def _checkpoint_runtime(
        self,
        request: AgentRequest,
        request_frame: RequestFrame | None,
        session: AgentRuntimeSession,
        phase: str,
    ) -> None:
        await self._checkpoint_writer()(
            session.run.id,
            phase,
            self.checkpoint_codec.encode(request, request_frame, session),
        )

    async def _step_checkpoint(
        self,
        request: AgentRequest,
        request_frame: RequestFrame | None,
        session: AgentRuntimeSession,
        phase: str,
    ) -> None:
        if self.step_checkpoint is not None:
            await self.step_checkpoint(request, request_frame, session, phase)
            return
        await self._checkpoint_runtime(request, request_frame, session, phase)

    def _checkpoint_writer(self) -> CheckpointWriter:
        return self.checkpoint()


def _checkpoint_state(request, plan, datasets, request_frame):
    return {
        "request": request.model_dump(mode="json"),
        "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
        "plan": plan.model_dump(mode="json") if plan else None,
        "dataset_ids": [item.id for item in datasets],
    }


def _set_plan_step_status(plan: Plan, step_id: str, status: TaskStatus) -> None:
    step = next((item for item in plan.steps if item.id == step_id), None)
    if step is not None:
        step.status = status


def _execution_observation(outcome: ExecutionOutcome) -> dict[str, Any]:
    observation = outcome.result.model_dump(mode="json")
    observation.update(
        {
            "accepted": outcome.accepted,
            "verified": outcome.verified,
            "verification_problems": list(outcome.verification_problems),
            "recovery_action": outcome.recovery_action.value if outcome.recovery_action else None,
            "directive": outcome.directive.value,
            "attempts": outcome.attempts,
            "rationale": outcome.rationale,
        }
    )
    return {key: value for key, value in observation.items() if value not in (None, [], {})}


def _invalid_tool_call_observation(call_id: str, result: ToolResult, detail: str) -> dict[str, Any]:
    observation = result.model_dump(mode="json")
    observation.update(
        {
            "call_id": call_id,
            "accepted": False,
            "verified": False,
            "verification_problems": [f"模型工具参数无效：{detail}"],
            "recovery_action": None,
            "directive": LoopDirective.ABORT.value,
            "attempts": 1,
        }
    )
    return {key: value for key, value in observation.items() if value not in (None, [], {})}


def _batch_observation(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {"tool_observations": items}
    if len(items) == 1:
        batch["call_id"] = items[0].get("call_id")
    return batch


def _failure_from_outcome(outcome: ExecutionOutcome, tool_name: str, *, deterministic: bool, step_id: str | None = None) -> dict[str, Any]:
    return {
        "action": outcome.directive.value,
        "directive": outcome.directive.value,
        "deterministic": deterministic,
        "step_id": step_id,
        "tool_name": tool_name,
        "error_code": outcome.result.error.code if outcome.result.error else None,
        "error": outcome.result.error.message if outcome.result.error else outcome.rationale,
        "verification_problems": list(outcome.verification_problems),
        "recovery_action": outcome.recovery_action.value if outcome.recovery_action else None,
        "attempts": outcome.attempts,
    }


def _plan_state_from_outcome(outcome: PlanLoopOutcome, reasons: list[str]) -> dict[str, Any]:
    return {
        "completed_steps": sorted(outcome.completed_steps),
        "findings": list(outcome.findings),
        "output_ids": list(outcome.output_ids),
        "artifacts": list(outcome.artifacts),
        "errors": list(outcome.errors),
        "step_outputs": dict(outcome.step_outputs),
        "previous_replan_reasons": reasons,
    }


def _replan_reason(step, outcome: ExecutionOutcome | None) -> str:
    code = outcome.result.error.code if outcome and outcome.result.error else "VERIFICATION_FAILED" if outcome and outcome.verification_problems else "REPLAN"
    return f"{step.id if step else 'unknown_step'}:{code}"


def _compact_replan_step_outputs(step_outputs: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for step_id, value in step_outputs.items():
        if isinstance(value, dict):
            compact[step_id] = {key: value[key] for key in ("dataset_id", "dataset_ids", "artifact_ids", "status") if key in value}
    return compact


def _aggregate_subagent_directive(executions: list[Any]) -> LoopDirective:
    directives = {item.directive for item in executions}
    if LoopDirective.ASK_USER in directives:
        return LoopDirective.ASK_USER
    if LoopDirective.REPLAN in directives:
        return LoopDirective.REPLAN
    if LoopDirective.ABORT in directives:
        return LoopDirective.ABORT
    return LoopDirective.CONTINUE


def _subagent_result_view(subtask: SubTask, execution: SubAgentExecutionResult) -> dict[str, Any]:
    result = execution.result
    return {
        "subtask_id": subtask.id,
        "agent_id": result.agent_id,
        "goal": subtask.goal,
        "operation": subtask.operation,
        "required": subtask.required,
        "status": result.status.value,
        "summary": result.summary,
        "directive": execution.directive.value,
        "dataset_ids": list(execution.working_memory_delta.added_dataset_ids),
        "artifact_ids": list(execution.working_memory_delta.added_artifact_ids),
        "key_findings": _compact_subagent_findings(result.findings),
        "failure_rationale": execution.failure_rationale or result.error,
    }


def _compact_subagent_findings(findings: list[Any]) -> list[Any]:
    return [_compact_subagent_value(item) for item in findings[:6]]


def _compact_subagent_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 2:
        return str(value)[:240]
    if isinstance(value, dict):
        return {str(key): _compact_subagent_value(item, depth=depth + 1) for key, item in list(value.items())[:8]}
    if isinstance(value, (list, tuple)):
        return [_compact_subagent_value(item, depth=depth + 1) for item in list(value)[:8]]
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:499] + "…"
    return value


def _delegation_failure(executions: list[SubAgentExecutionResult], tasks: list[SubTask]) -> dict[str, Any] | None:
    for subtask, execution in zip(tasks, executions, strict=True):
        if not subtask.required or execution.directive is LoopDirective.CONTINUE:
            continue
        return {
            "directive": execution.directive.value,
            "action": execution.directive.value,
            "deterministic": True,
            "subtask_id": subtask.id,
            "tool_name": subtask.operation,
            "error": execution.failure_rationale or execution.result.error or execution.result.summary,
            "recovery_action": execution.directive.value,
            "attempts": 1,
        }
    return None


def _merge_subagent_views(current: list[Any], incoming: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    merged = [dict(item) for item in current if isinstance(item, dict)]
    positions = {(item.get("subtask_id"), item.get("agent_id")): index for index, item in enumerate(merged)}
    for item in incoming:
        key = (item.get("subtask_id"), item.get("agent_id"))
        if key in positions:
            merged[positions[key]] = dict(item)
        else:
            positions[key] = len(merged)
            merged.append(dict(item))
    return merged


def _delegation_fingerprint(tasks: list[SubTask]) -> str:
    payload = sorted(
        [
            {
                "goal": task.goal,
                "operation": task.operation,
                "dataset_ids": sorted(task.dataset_ids),
                "dependencies": sorted(task.dependencies),
                "required": task.required,
            }
            for task in tasks
        ],
        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
    )
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _working_memory_refs(memory: WorkingMemory | None) -> dict[str, Any] | None:
    if memory is None:
        return None
    return {
        "task_id": memory.task_id,
        "conversation_id": memory.conversation_id,
        "active_dataset_ids": list(memory.active_dataset_ids),
        "active_artifact_ids": list(memory.active_artifact_ids),
        "unresolved_questions": list(memory.unresolved_questions),
    }


def _protocol_tool_message(value: Any) -> dict[str, Any]:
    # Local import avoids making protocol compatibility part of the action API.
    from app.runtime.protocol_history import protocol_tool_message

    return protocol_tool_message(value)


def _compact_protocol_messages(messages: list[dict[str, Any]], *, max_tokens: int) -> list[dict[str, Any]]:
    from app.runtime.protocol_history import compact_protocol_messages

    return compact_protocol_messages(messages, max_tokens=max_tokens)


__all__ = ["DelegationExecution", "RuntimeActionHandlers"]
