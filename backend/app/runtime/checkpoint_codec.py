"""Runtime checkpoint 的 canonical 编解码边界。"""

from __future__ import annotations

from typing import Any

from app.core.models import AgentRequest, Plan, RequestFrame
from app.runtime.protocol_history import extract_protocol_messages
from app.runtime.session import AgentRuntimeSession, RuntimeResumeState


class RuntimeCheckpointCodec:
    """集中维护 runtime state 的 canonical 字段和旧 checkpoint 兼容读取。"""

    CANONICAL_FIELDS = (
        "request",
        "request_frame",
        "current_plan",
        "original_plan",
        "completed_steps",
        "step_outputs",
        "protocol_messages",
        "latest_observation",
        "latest_failure",
        "findings",
        "dataset_ids",
        "artifact_ids",
        "subagent_results",
        "completed_delegation_fingerprints",
        "replan_count",
        "previous_replan_reasons",
        "runtime_mode",
    )

    @classmethod
    def decode(cls, state: dict[str, Any] | None) -> RuntimeResumeState:
        """读取 canonical state；旧 alias 只在这里解析。"""

        state = state or {}
        protocol_messages = state.get("protocol_messages")
        legacy_messages = state.get("messages")
        if not isinstance(protocol_messages, list):
            protocol_messages = None
        if not isinstance(legacy_messages, list):
            legacy_messages = None
        extracted = extract_protocol_messages(protocol_messages=protocol_messages, legacy_messages=legacy_messages)
        current_plan = _plan(state.get("current_plan")) or _plan(state.get("plan"))
        original_plan = _plan(state.get("original_plan"))
        return RuntimeResumeState(
            protocol_messages=extracted,
            latest_observation=state.get("latest_observation") if isinstance(state.get("latest_observation"), dict) else None,
            latest_failure=state.get("latest_failure") if isinstance(state.get("latest_failure"), dict) else None,
            findings=list(state.get("findings") or state.get("model_findings") or []),
            dataset_ids=[str(item) for item in (state.get("dataset_ids") or state.get("model_dataset_ids") or [])],
            artifact_ids=[str(item) for item in (state.get("artifact_ids") or state.get("model_artifact_ids") or [])],
            subagent_results=list(state.get("subagent_results") or []),
            completed_delegation_fingerprints=[str(item) for item in state.get("completed_delegation_fingerprints") or []],
            current_plan=current_plan,
            original_plan=original_plan,
            completed_steps=[str(item) for item in state.get("completed_steps") or []],
            step_outputs=dict(state.get("step_outputs") or {}),
            previous_replan_reasons=[str(item) for item in state.get("previous_replan_reasons") or []],
            runtime_mode=str(state["runtime_mode"]) if state.get("runtime_mode") else None,
        )

    @classmethod
    def encode(
        cls,
        request: AgentRequest,
        request_frame: RequestFrame | None,
        session: AgentRuntimeSession,
        *,
        replan_count: int | None = None,
    ) -> dict[str, Any]:
        """生成唯一 canonical runtime checkpoint payload。"""

        run = session.run
        plan = session.current_plan
        payload: dict[str, Any] = {
            "request": request.model_dump(mode="json"),
            "request_frame": request_frame.model_dump(mode="json") if request_frame else None,
            "current_plan": plan.model_dump(mode="json") if plan else None,
            "original_plan": session.original_plan.model_dump(mode="json") if session.original_plan else None,
            "completed_steps": sorted(session.completed_steps),
            "step_outputs": dict(session.step_outputs),
            "protocol_messages": list(session.protocol_messages),
            "latest_observation": session.latest_observation,
            "latest_failure": session.latest_failure,
            "findings": list(session.findings),
            "dataset_ids": sorted(session.dataset_ids),
            "artifact_ids": sorted(session.artifact_ids),
            "subagent_results": list(session.subagent_results),
            "completed_delegation_fingerprints": sorted(session.completed_delegation_fingerprints),
            "replan_count": replan_count if replan_count is not None else run.replan_count if run else 0,
            "previous_replan_reasons": list(session.previous_replan_reasons),
            "runtime_mode": session.decision_provider,
        }
        return payload


def _plan(value: Any) -> Plan | None:
    if not isinstance(value, dict):
        return None
    return Plan.model_validate(value)


__all__ = ["RuntimeCheckpointCodec"]
