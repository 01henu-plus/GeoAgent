"""无模型时的确定性 Decision Provider。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.core.models import (
    AgentDecision,
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    DecisionType,
    IntentResult,
    RequestFrame,
    Task,
)
from app.decision.decomposer import TaskDecomposer
from app.decision.intent import IntentResolver
from app.decision.router import AgentRouter
from app.knowledge import KnowledgeRetriever
from app.runtime.session import AgentRuntimeSession
from app.understanding.compat import LegacyIntentAdapter


class OfflineDecisionProvider:
    """复用旧 Router/Planner 提示，但只产生 AgentDecision。"""

    def __init__(
        self,
        *,
        legacy_intent_adapter: LegacyIntentAdapter,
        intent_resolver: IntentResolver,
        router: AgentRouter,
        decomposer: TaskDecomposer,
        knowledge: KnowledgeRetriever,
        next_executable_step: Callable[[Any, set[str]], Any],
        result_to_decision: Callable[[AgentResult, str], AgentDecision],
        diagnose_runs: Callable[[AgentRequest, Any], AgentResult],
        interpret_result: Callable[[AgentRequest, Any], AgentResult],
        conversation_reply: Callable[[str], str],
    ) -> None:
        self.legacy_intent_adapter = legacy_intent_adapter
        self.intent_resolver = intent_resolver
        self.router = router
        self.decomposer = decomposer
        self.knowledge = knowledge
        self.next_executable_step = next_executable_step
        self.result_to_decision = result_to_decision
        self.diagnose_runs = diagnose_runs
        self.interpret_result = interpret_result
        self.conversation_reply = conversation_reply

    def decide(
        self,
        state,
        session: AgentRuntimeSession,
        *,
        request: AgentRequest,
        task: Task | None,
        intent: IntentResult | None,
        request_frame: RequestFrame | None,
    ) -> AgentDecision:
        if session.legacy_delegation_result:
            return self.result_to_decision(
                AgentResult.model_validate(session.legacy_delegation_result),
                "offline_compat",
            )
        if intent is None:
            intent = self.legacy_intent_adapter.to_intent(request_frame, request, session.datasets)

        current_plan = session.current_plan
        if current_plan is not None:
            if current_plan.clarification:
                return AgentDecision(
                    type=DecisionType.ASK_USER,
                    reasoning_summary=current_plan.clarification,
                    final_response=current_plan.clarification,
                    source="offline",
                )
            if current_plan.metadata.get("delegated_roles") and not session.subagent_results:
                return self.router.route(
                    intent,
                    current_plan,
                    session.datasets,
                    subtasks=self.decomposer.decompose(request, session.datasets),
                ).model_copy(update={"source": "offline"})
            if current_plan.metadata.get("delegated_roles") and session.subagent_results:
                completed = sum(item.get("status") == AgentResultStatus.SUCCESS.value for item in session.subagent_results if isinstance(item, dict))
                total = len(session.subagent_results)
                result = AgentResult(
                    agent_id="main",
                    task_id=task.id if task else state.task_id,
                    status=AgentResultStatus.SUCCESS if completed == total else AgentResultStatus.PARTIAL,
                    summary=f"已并行完成 {completed}/{total} 个主题分析，并汇总结果。",
                    findings=list(session.findings),
                    datasets=sorted(session.dataset_ids),
                    artifacts=sorted(session.artifact_ids),
                    trace_id=state.run_id,
                )
                return self.result_to_decision(result, "offline")
            if self.next_executable_step(current_plan, session.completed_steps) is None:
                operation = str(current_plan.metadata.get("operation") or intent.entities.get("operation") or "")
                result = AgentResult(
                    agent_id="main",
                    task_id=task.id if task else state.task_id,
                    status=AgentResultStatus.SUCCESS,
                    summary=_plan_result_summary(operation, current_plan, list(session.findings), sorted(session.dataset_ids), sorted(session.artifact_ids)),
                    findings=list(session.findings),
                    datasets=sorted(session.dataset_ids),
                    artifacts=sorted(session.artifact_ids),
                    trace_id=state.run_id,
                )
                return self.result_to_decision(result, "offline")

        if intent.intent.value == "RUN_DIAGNOSIS":
            return self.result_to_decision(self.diagnose_runs(request, session.run), "offline")
        if intent.intent.value == "RESULT_INTERPRETATION":
            return self.result_to_decision(self.interpret_result(request, session.run), "offline")
        if intent.intent.value == "UNKNOWN":
            return AgentDecision(
                type=DecisionType.FINAL,
                reasoning_summary="当前回合不需要 GIS 执行。",
                final_response=self.conversation_reply(request.user_input),
                source="offline",
            )
        findings = self.knowledge.retrieve(request.user_input) if intent.intent.value == "KNOWLEDGE_QUERY" else []
        result = AgentResult(
            agent_id="main",
            task_id=task.id if task else state.task_id,
            status=AgentResultStatus.SUCCESS,
            summary="已整理 GIS 知识上下文。" if findings else "已理解请求，但当前计划没有需要执行的 GIS 操作。",
            findings=findings,
            trace_id=state.run_id,
            warnings=[] if findings else ["当前未配置大模型，离线模式只支持有限的 GIS 操作。"],
        )
        return self.result_to_decision(result, "offline")


def _plan_result_summary(operation: str, plan, findings: list[Any], dataset_ids: list[str], artifact_ids: list[str]) -> str:
    labels = {"slope": "坡度", "buffer": "缓冲区", "clip": "裁剪", "intersection": "相交", "spatial_join": "空间连接"}
    operation_label = labels.get(operation, "空间分析")
    resource_count = len(dataset_ids) + len(artifact_ids)
    return f"已完成{operation_label}，生成或确认 {resource_count} 个结果资源。" if resource_count else f"已完成{operation_label}。"


__all__ = ["OfflineDecisionProvider"]
