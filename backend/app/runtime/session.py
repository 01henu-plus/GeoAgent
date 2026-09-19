"""AgentRuntime 的可变会话状态与检查点恢复输入。

RuntimeSession 是一次运行期间的编排状态，不是 WorkingMemory，也不是给决策器的
完整上下文。WorkingMemory 仍然是 Task-scoped 的权威状态；AgentState 则是每轮
从权威状态和本会话派生出的有界决策视图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.models import Plan, Run, WorkingMemory
from app.models import ModelResponse


@dataclass(slots=True)
class RuntimeResumeState:
    """从 canonical checkpoint 解码出的运行时恢复输入。"""

    protocol_messages: list[dict[str, Any]] = field(default_factory=list)
    latest_observation: dict[str, Any] | None = None
    latest_failure: dict[str, Any] | None = None
    findings: list[Any] = field(default_factory=list)
    dataset_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    subagent_results: list[Any] = field(default_factory=list)
    completed_delegation_fingerprints: list[str] = field(default_factory=list)
    legacy_delegation_result: dict[str, Any] | None = None
    current_plan: Plan | None = None
    original_plan: Plan | None = None
    completed_steps: list[str] = field(default_factory=list)
    step_outputs: dict[str, Any] = field(default_factory=dict)
    previous_replan_reasons: list[str] = field(default_factory=list)
    runtime_mode: str | None = None


@dataclass(slots=True)
class AgentRuntimeSession:
    """一次 AgentRuntime 的 mutable orchestration state。"""

    protocol_messages: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Any] = field(default_factory=list)
    dataset_ids: set[str] = field(default_factory=set)
    artifact_ids: set[str] = field(default_factory=set)
    latest_observation: dict[str, Any] | None = None
    latest_failure: dict[str, Any] | None = None
    datasets: list[Any] = field(default_factory=list)
    current_plan: Plan | None = None
    original_plan: Plan | None = None
    completed_steps: set[str] = field(default_factory=set)
    step_outputs: dict[str, Any] = field(default_factory=dict)
    previous_replan_reasons: list[str] = field(default_factory=list)
    subagent_results: list[Any] = field(default_factory=list)
    completed_delegation_fingerprints: set[str] = field(default_factory=set)
    run: Run | None = None
    working_memory: WorkingMemory | None = None
    fast_path_enabled: bool = False
    decision_provider: str = "offline"
    last_response: ModelResponse | None = None

    @classmethod
    def from_resume(
        cls,
        *,
        run: Run,
        datasets: list[Any],
        plan: Plan | None,
        working_memory: WorkingMemory | None,
        resume: RuntimeResumeState | None = None,
        decision_provider: str = "offline",
    ) -> AgentRuntimeSession:
        resume = resume or RuntimeResumeState()
        current_plan = resume.current_plan or plan
        original_plan = resume.original_plan
        if original_plan is None and current_plan is not None:
            original_plan = current_plan.model_copy(deep=True)
        return cls(
            protocol_messages=[dict(item) for item in resume.protocol_messages],
            findings=list(resume.findings),
            dataset_ids=set(resume.dataset_ids),
            artifact_ids=set(resume.artifact_ids),
            latest_observation=_deep_copy_dict(resume.latest_observation) if resume.latest_observation else None,
            latest_failure=_deep_copy_dict(resume.latest_failure) if resume.latest_failure else None,
            datasets=list(datasets),
            current_plan=current_plan,
            original_plan=original_plan,
            completed_steps=set(resume.completed_steps),
            step_outputs=_deep_copy_dict(resume.step_outputs),
            previous_replan_reasons=list(resume.previous_replan_reasons),
            subagent_results=list(resume.subagent_results),
            completed_delegation_fingerprints=set(resume.completed_delegation_fingerprints),
            run=run,
            working_memory=working_memory.model_copy(deep=True) if working_memory is not None else None,
            fast_path_enabled=current_plan is not None,
            decision_provider=resume.runtime_mode or decision_provider,
        )

    @classmethod
    def from_checkpoint(
        cls,
        *,
        run: Run,
        datasets: list[Any],
        plan: Plan | None,
        working_memory: WorkingMemory | None,
        resume: RuntimeResumeState | None = None,
        decision_provider: str = "offline",
    ) -> AgentRuntimeSession:
        """兼容调用方使用更直观的恢复工厂名称。"""

        return cls.from_resume(
            run=run,
            datasets=datasets,
            plan=plan,
            working_memory=working_memory,
            resume=resume,
            decision_provider=decision_provider,
        )

def _deep_copy_dict(value: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            result[str(key)] = _deep_copy_dict(item)
        elif isinstance(item, list):
            result[str(key)] = [_deep_copy_value(child) for child in item]
        else:
            result[str(key)] = item
    return result


def _deep_copy_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _deep_copy_dict(value)
    if isinstance(value, list):
        return [_deep_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy_value(item) for item in value)
    return value


__all__ = ["AgentRuntimeSession", "RuntimeResumeState"]
