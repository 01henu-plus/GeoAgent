"""State-driven Agent Runtime 使用的当前运行视图。

``AgentState`` 是由 Task/Run/WorkingMemory 派生出的决策视图，不是新的持久化
状态，也不承载协议历史、完整工具输出或模型 Context。每轮由
``AgentStateBuilder`` 重新读取 authoritative state，避免运行时长期持有过期快照。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.core.models import (
    AgentRequest,
    Plan,
    RequestFrame,
    Run,
    StrictModel,
    Task,
    WorkingMemory,
)
from app.state import StateStore


class AgentState(StrictModel):
    """当前 Run 的最小决策状态。"""

    request: AgentRequest
    request_frame: RequestFrame | None = None
    task_id: str | None = None
    run_id: str
    goal: str
    working_memory: WorkingMemory | None = None
    current_plan: Plan | None = None
    plan_completed_steps: list[str] = Field(default_factory=list)
    plan_step_outputs: dict[str, Any] = Field(default_factory=dict)
    active_dataset_ids: list[str] = Field(default_factory=list)
    active_artifact_ids: list[str] = Field(default_factory=list)
    latest_observation: dict[str, Any] | None = None
    latest_failure: dict[str, Any] | None = None
    subagent_results: list[Any] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    turn_count: int = 0
    tool_call_count: int = 0
    replan_count: int = 0


class AgentStateBuilder:
    """从当前持久化状态构建一个新 AgentState，不修改任何来源对象。"""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def build(
        self,
        request: AgentRequest,
        request_frame: RequestFrame | None,
        run: Run,
        *,
        task: Task | None = None,
        working_memory: WorkingMemory | None = None,
        current_plan: Plan | None = None,
        plan_completed_steps: list[str] | set[str] | None = None,
        plan_step_outputs: dict[str, Any] | None = None,
        latest_observation: dict[str, Any] | None = None,
        latest_failure: dict[str, Any] | None = None,
        active_dataset_ids: list[str] | None = None,
        active_artifact_ids: list[str] | None = None,
        subagent_results: list[Any] | None = None,
        unresolved_questions: list[str] | None = None,
    ) -> AgentState:
        current_run = self.store.get_run(run.id) or run
        memory = self.store.get_working_memory(current_run.task_id) if current_run.task_id else None
        memory = memory or working_memory
        memory_view = memory.model_copy(deep=True) if memory is not None else None
        task_view = task.model_copy(deep=True) if task is not None else None
        goal = (
            request_frame.goal.strip()
            if request_frame is not None and request_frame.goal.strip()
            else task_view.goal
            if task_view is not None
            else request.user_input
        )
        datasets = _unique(
            [
                *(memory_view.active_dataset_ids if memory_view else []),
                *(active_dataset_ids or []),
            ]
        )
        artifacts = _unique(
            [
                *(memory_view.active_artifact_ids if memory_view else []),
                *(active_artifact_ids or []),
            ]
        )
        unresolved = _unique_text(
            [
                *(memory_view.unresolved_questions if memory_view else []),
                *(unresolved_questions or []),
            ]
        )
        return AgentState(
            request=request.model_copy(deep=True),
            request_frame=request_frame.model_copy(deep=True) if request_frame is not None else None,
            task_id=current_run.task_id,
            run_id=current_run.id,
            goal=goal,
            working_memory=memory_view,
            current_plan=current_plan.model_copy(deep=True) if current_plan is not None else None,
            plan_completed_steps=sorted(set(plan_completed_steps or [])),
            plan_step_outputs=_deep_copy_dict(plan_step_outputs or {}),
            active_dataset_ids=datasets,
            active_artifact_ids=artifacts,
            latest_observation=_deep_copy_dict(latest_observation) if latest_observation is not None else None,
            latest_failure=_deep_copy_dict(latest_failure) if latest_failure is not None else None,
            subagent_results=[_deep_copy_value(item) for item in (subagent_results or [])],
            unresolved_questions=unresolved,
            turn_count=current_run.turn_count,
            tool_call_count=current_run.tool_call_count,
            replan_count=current_run.replan_count,
        )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _unique_text(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized.casefold() not in seen:
            result.append(normalized)
            seen.add(normalized.casefold())
    return result


def _deep_copy_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _deep_copy_value(item) for key, item in value.items()}


def _deep_copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _deep_copy_dict(value)
    if isinstance(value, list):
        return [_deep_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy_value(item) for item in value)
    return value


__all__ = ["AgentState", "AgentStateBuilder"]
