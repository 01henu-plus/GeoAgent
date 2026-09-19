"""AgentDecision 到 RuntimeTransition 的动作边界。"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.models import AgentDecision, AgentResultStatus, DecisionType
from app.runtime.agent_runtime import RuntimeTransition
from app.runtime.session import AgentRuntimeSession

ActionHandler = Callable[..., Awaitable[RuntimeTransition] | RuntimeTransition]


class RuntimeActionDispatcher:
    """只分派一次动作，不负责 Run/Task finalization。"""

    def __init__(
        self,
        *,
        handlers: dict[DecisionType, ActionHandler] | None = None,
    ) -> None:
        self.handlers = dict(handlers or {})

    async def dispatch(
        self,
        decision: AgentDecision,
        state,
        session: AgentRuntimeSession,
        **context: Any,
    ) -> RuntimeTransition:
        if decision.type is DecisionType.FINAL:
            response = (decision.final_response or "").strip()
            if not response:
                return RuntimeTransition(terminal=True, error="EMPTY_MODEL_RESPONSE")
            for finding in decision.metadata.get("findings") or []:
                if finding not in session.findings:
                    session.findings.append(finding)
            session.dataset_ids.update(str(item) for item in decision.metadata.get("datasets") or [])
            session.artifact_ids.update(str(item) for item in decision.metadata.get("artifacts") or [])
            if decision.source == "model" or decision.metadata.get("model"):
                session.findings.append({"model": decision.metadata.get("model"), "content": response})
            return RuntimeTransition(
                terminal=True,
                status=_status_from_metadata(decision.metadata),
                final_response=response,
                error=decision.metadata.get("error"),
                findings=tuple(session.findings),
                dataset_ids=tuple(sorted(session.dataset_ids)),
                artifact_ids=tuple(sorted(session.artifact_ids)),
            )
        if decision.type is DecisionType.ASK_USER:
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.BLOCKED,
                final_response=decision.final_response or decision.reasoning_summary,
                error="WAITING_USER",
            )
        if decision.type is DecisionType.ABORT:
            if decision.metadata.get("empty_response"):
                return RuntimeTransition(terminal=True, error="EMPTY_MODEL_RESPONSE")
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.FAILED,
                error=decision.metadata.get("error_code") or decision.reasoning_summary,
                findings=tuple(session.findings),
                dataset_ids=tuple(sorted(session.dataset_ids)),
                artifact_ids=tuple(sorted(session.artifact_ids)),
            )

        handler = self.handlers.get(decision.type)
        if handler is None:
            return RuntimeTransition(
                terminal=True,
                status=AgentResultStatus.FAILED,
                error=f"不支持的运行时动作：{decision.type.value}",
            )
        value = handler(decision, state, session=session, **context)
        return await value if inspect.isawaitable(value) else value


def _status_from_metadata(metadata: dict[str, Any]) -> AgentResultStatus:
    value = metadata.get("status")
    try:
        return AgentResultStatus(value) if value else AgentResultStatus.SUCCESS
    except ValueError:
        return AgentResultStatus.SUCCESS


__all__ = ["RuntimeActionDispatcher"]
