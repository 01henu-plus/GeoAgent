"""State-driven Agent Runtime 的最小控制循环。"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.models import AgentDecision, AgentResultStatus, LoopDirective, Plan
from app.runtime.agent_state import AgentState


@dataclass(frozen=True, slots=True)
class RuntimeTransition:
    """一次 Action 执行后的结构化观察。"""

    terminal: bool = False
    status: AgentResultStatus | None = None
    final_response: str | None = None
    error: str | None = None
    observation: dict[str, Any] | None = None
    directive: LoopDirective = LoopDirective.CONTINUE
    latest_failure: dict[str, Any] | None = None
    clear_failure: bool = False
    findings: tuple[Any, ...] = ()
    dataset_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    subagent_results: tuple[Any, ...] = ()
    current_plan: Plan | None = None
    clear_plan: bool = False
    completed_steps: tuple[str, ...] = ()
    step_outputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuntimeOutcome:
    """AgentRuntime 的出口；不直接构造 AgentResult。"""

    terminal: bool
    status: AgentResultStatus | None = None
    final_response: str | None = None
    error: str | None = None
    findings: tuple[Any, ...] = ()
    dataset_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    latest_observation: dict[str, Any] | None = None
    directive: LoopDirective = LoopDirective.CONTINUE
    decision: AgentDecision | None = None
    state: AgentState | None = None
    iterations: int = 0


DecisionCallback = Callable[[AgentState], Awaitable[AgentDecision] | AgentDecision]
DispatchCallback = Callable[[AgentDecision, AgentState], Awaitable[RuntimeTransition] | RuntimeTransition]
RefreshCallback = Callable[[AgentState], Awaitable[AgentState] | AgentState]
FastPathCallback = Callable[[AgentState], Awaitable[RuntimeTransition | None] | RuntimeTransition | None]


class AgentRuntime:
    """统一 Observe -> Decide -> Execute -> Observe 的运行时控制流。"""

    def __init__(self, *, max_runtime_transitions: int = 100, max_iterations: int | None = None) -> None:
        # max_iterations 仅作为旧调用方兼容别名；语义统一为 Runtime transition safety limit。
        self.max_runtime_transitions = max(1, max_iterations if max_iterations is not None else max_runtime_transitions)

    async def run(
        self,
        initial_state: AgentState,
        *,
        decide: DecisionCallback,
        dispatch: DispatchCallback,
        refresh_state: RefreshCallback | None = None,
        fast_path: FastPathCallback | None = None,
        max_runtime_transitions: int | None = None,
        max_iterations: int | None = None,
    ) -> RuntimeOutcome:
        state = initial_state.model_copy(deep=True)
        limit = max(1, max_iterations or max_runtime_transitions or self.max_runtime_transitions)
        last_decision: AgentDecision | None = None
        last_transition = RuntimeTransition()
        for iteration in range(limit):
            if refresh_state is not None:
                state = await _maybe_await(refresh_state(state))
            transition: RuntimeTransition | None = None
            if fast_path is not None and state.current_plan is not None:
                transition = await _maybe_await(fast_path(state))
            if transition is None:
                last_decision = await _maybe_await(decide(state))
                transition = await _maybe_await(dispatch(last_decision, state))
            last_transition = transition
            state = self._apply_transition(state, transition)
            if transition.terminal:
                return RuntimeOutcome(
                    terminal=True,
                    status=transition.status,
                    final_response=transition.final_response,
                    error=transition.error,
                    findings=transition.findings,
                    dataset_ids=transition.dataset_ids,
                    artifact_ids=transition.artifact_ids,
                    latest_observation=transition.observation,
                    directive=transition.directive,
                    decision=last_decision,
                    state=state,
                    iterations=iteration + 1,
                )
        return RuntimeOutcome(
            terminal=True,
            status=AgentResultStatus.BLOCKED,
            error="AGENT_RUNTIME_BUDGET_EXCEEDED",
            findings=last_transition.findings,
            dataset_ids=last_transition.dataset_ids,
            artifact_ids=last_transition.artifact_ids,
            latest_observation=last_transition.observation,
            directive=LoopDirective.ABORT,
            decision=last_decision,
            state=state,
            iterations=limit,
        )

    @staticmethod
    def _apply_transition(state: AgentState, transition: RuntimeTransition) -> AgentState:
        updates: dict[str, Any] = {
            "latest_observation": transition.observation,
            "active_dataset_ids": _unique([*state.active_dataset_ids, *transition.dataset_ids]),
            "active_artifact_ids": _unique([*state.active_artifact_ids, *transition.artifact_ids]),
        }
        if transition.clear_failure:
            updates["latest_failure"] = None
        elif transition.latest_failure is not None:
            updates["latest_failure"] = transition.latest_failure
        if transition.subagent_results:
            updates["subagent_results"] = [dict(item) if isinstance(item, dict) else item for item in transition.subagent_results]
        if transition.clear_plan:
            updates["current_plan"] = None
        elif transition.current_plan is not None:
            updates["current_plan"] = transition.current_plan.model_copy(deep=True)
        if transition.completed_steps:
            updates["plan_completed_steps"] = _unique([*state.plan_completed_steps, *transition.completed_steps])
        if transition.step_outputs:
            updates["plan_step_outputs"] = {**state.plan_step_outputs, **transition.step_outputs}
        return state.model_copy(update=updates, deep=True)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


__all__ = ["AgentRuntime", "RuntimeOutcome", "RuntimeTransition"]
