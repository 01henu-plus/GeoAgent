"""统一 AgentRuntime 的控制器。

Controller 负责把 authoritative state、typed session、Decision Provider 和动作
分派接到最小 AgentRuntime。它不负责最终 Run/Task/Memories 结算。
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.models import (
    AgentRequest,
    IntentResult,
    RequestFrame,
    Run,
    RunBudget,
    RunStatus,
    Task,
    WorkingMemory,
)
from app.decision.decision_engine import DecisionEngine
from app.decision.model_provider import ModelDecisionProvider
from app.decision.offline_provider import OfflineDecisionProvider
from app.models import ModelAdapter
from app.runtime.action_dispatcher import RuntimeActionDispatcher
from app.runtime.agent_loop import AgentLoop
from app.runtime.agent_runtime import AgentRuntime, RuntimeOutcome, RuntimeTransition
from app.runtime.agent_state import AgentStateBuilder
from app.runtime.budget import BudgetGuard
from app.runtime.checkpoint_codec import RuntimeCheckpointCodec
from app.runtime.session import AgentRuntimeSession, RuntimeResumeState
from app.state import StateStore


class AgentRuntimeController:
    """正式的 refresh → fast path → decide → dispatch 控制边界。"""

    def __init__(
        self,
        *,
        store: StateStore,
        budget: RunBudget,
        guard: BudgetGuard,
        runtime: AgentRuntime,
        state_builder: AgentStateBuilder,
        decision_engine: DecisionEngine,
        model_provider: ModelDecisionProvider,
        offline_provider: OfflineDecisionProvider,
        dispatcher: RuntimeActionDispatcher,
        plan_loop: AgentLoop,
        model_adapter_for: Callable[[AgentRequest], ModelAdapter | None],
        trace_decision: Callable[..., Awaitable[None]],
        checkpoint: Callable[[str, str, dict[str, Any]], Awaitable[None]],
        checkpoint_codec: RuntimeCheckpointCodec,
        plan_fast_path: Callable[..., Awaitable[RuntimeTransition] | RuntimeTransition],
    ) -> None:
        self.store = store
        self.budget = budget
        self.guard = guard
        self.runtime = runtime
        self.state_builder = state_builder
        self.decision_engine = decision_engine
        self.model_provider = model_provider
        self.offline_provider = offline_provider
        self.dispatcher = dispatcher
        self.plan_loop = plan_loop
        self.model_adapter_for = model_adapter_for
        self.trace_decision = trace_decision
        self.checkpoint = checkpoint
        self.checkpoint_codec = checkpoint_codec
        self.plan_fast_path = plan_fast_path

    async def run(
        self,
        request: AgentRequest,
        *,
        run: Run,
        task: Task | None,
        datasets: list[Any],
        intent: IntentResult | None,
        plan,
        request_frame: RequestFrame | None,
        working_memory: WorkingMemory | None,
        resume: RuntimeResumeState | None = None,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> RuntimeOutcome:
        model_adapter = self.model_adapter_for(request)
        session = AgentRuntimeSession.from_resume(
            run=run,
            datasets=datasets,
            plan=plan,
            working_memory=working_memory,
            resume=resume,
            decision_provider="model" if model_adapter is not None else "offline",
        )
        current = self.store.get_run(run.id) or run
        if current.status is not RunStatus.RUNNING:
            current = current.model_copy(update={"status": RunStatus.RUNNING})
            self.store.save_run(current)
        session.run = current
        await self.checkpoint(
            current.id,
            "runtime_started",
            self.checkpoint_codec.encode(request, intent, request_frame, session),
        )

        initial_state = self.state_builder.build(
            request,
            request_frame,
            current,
            task=task,
            current_plan=session.current_plan,
            plan_completed_steps=session.completed_steps,
            plan_step_outputs=session.step_outputs,
            latest_observation=session.latest_observation,
            latest_failure=session.latest_failure,
            active_dataset_ids=sorted(session.dataset_ids),
            active_artifact_ids=sorted(session.artifact_ids),
            subagent_results=session.subagent_results,
            working_memory=session.working_memory,
        )

        async def refresh_state(_state):
            current_run = self.store.get_run(current.id) or session.run or current
            if current_run.status not in {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}:
                session.run = current_run
            if current_run.task_id:
                session.working_memory = self.store.get_working_memory(current_run.task_id) or session.working_memory
            return self.state_builder.build(
                request,
                request_frame,
                current_run,
                task=task,
                current_plan=session.current_plan,
                plan_completed_steps=session.completed_steps,
                plan_step_outputs=session.step_outputs,
                latest_observation=session.latest_observation,
                latest_failure=session.latest_failure,
                active_dataset_ids=sorted(session.dataset_ids),
                active_artifact_ids=sorted(session.artifact_ids),
                subagent_results=session.subagent_results,
                working_memory=session.working_memory,
            )

        async def decide(state):
            state_decision = self.decision_engine.decide_from_state(state)
            if state_decision is not None:
                await self.trace_decision(state_decision, state)
                return state_decision
            current_run = self.store.get_run(current.id) or session.run or current
            self.guard.check_turn(current_run)
            self.guard.check_execution_time(current_run)
            current_run = current_run.model_copy(update={"turn_count": current_run.turn_count + 1, "status": RunStatus.RUNNING})
            self.store.save_run(current_run)
            session.run = current_run
            if current_run.task_id:
                session.working_memory = self.store.get_working_memory(current_run.task_id) or session.working_memory
            if model_adapter is not None:
                decision = await self.model_provider.decide(
                    state,
                    session,
                    request=request,
                    run=current_run,
                    task=task,
                    intent=intent,
                    request_frame=request_frame,
                    model_adapter=model_adapter,
                    on_model_delta=on_model_delta,
                )
            else:
                decision = self.offline_provider.decide(
                    state,
                    session,
                    request=request,
                    task=task,
                    intent=intent,
                    request_frame=request_frame,
                )
            await self.trace_decision(decision, state)
            return decision

        async def dispatch(decision, state):
            return await self.dispatcher.dispatch(
                decision,
                state,
                session,
                request=request,
                run=session.run,
                task=task,
                intent=intent,
                request_frame=request_frame,
            )

        async def fast_path(state):
            if not session.fast_path_enabled or session.current_plan is None:
                return None
            if session.decision_provider == "offline" and session.current_plan.metadata.get("delegated_roles") and not session.subagent_results:
                session.fast_path_enabled = False
                return None
            step = self.plan_loop.next_executable_step(session.current_plan, session.completed_steps)
            if step is None:
                session.fast_path_enabled = False
                return None
            value = self.plan_fast_path(
                step,
                session=session,
                request=request,
                run=session.run,
                task=task,
                intent=intent,
                request_frame=request_frame,
            )
            return await value if inspect.isawaitable(value) else value

        return await self.runtime.run(
            initial_state,
            decide=decide,
            dispatch=dispatch,
            refresh_state=refresh_state,
            fast_path=fast_path,
            max_runtime_transitions=self.budget.max_runtime_transitions,
        )


__all__ = ["AgentRuntimeController"]
