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
    InteractionMode,
    RequestFrame,
    Task,
)
from app.decision.decomposer import TaskDecomposer
from app.knowledge import KnowledgeRetriever
from app.runtime.session import AgentRuntimeSession


class OfflineDecisionProvider:
    """基于 AgentState/RequestFrame 只产生确定性的 AgentDecision。"""

    def __init__(
        self,
        *,
        decomposer: TaskDecomposer,
        knowledge: KnowledgeRetriever,
        next_executable_step: Callable[[Any, set[str]], Any],
        result_to_decision: Callable[[AgentResult, str], AgentDecision],
        diagnose_runs: Callable[[AgentRequest, Any], AgentResult],
        interpret_result: Callable[[AgentRequest, Any], AgentResult],
        conversation_reply: Callable[[str], str],
    ) -> None:
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
        request_frame: RequestFrame | None,
    ) -> AgentDecision:
        frame = request_frame or RequestFrame(mode=InteractionMode.NEW_TASK, goal=request.user_input)

        current_plan = session.current_plan
        if current_plan is not None:
            if _should_delegate(frame, session.datasets) and not session.subagent_results:
                return _delegation_decision(request, session.datasets, self.decomposer)
            if _should_delegate(frame, session.datasets) and session.subagent_results:
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
            if current_plan.clarification:
                return AgentDecision(
                    type=DecisionType.ASK_USER,
                    reasoning_summary=current_plan.clarification,
                    final_response=current_plan.clarification,
                    source="offline",
                )
            if self.next_executable_step(current_plan, session.completed_steps) is None:
                operation = str(current_plan.metadata.get("operation") or _operation_from_goal(frame.goal) or "")
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

        if frame.mode is InteractionMode.QUERY and (
            "run_diagnosis" in frame.capabilities
            or any(term in frame.goal.casefold() for term in ("运行状态", "运行记录", "失败", "错误", "trace", "日志"))
        ):
            return self.result_to_decision(self.diagnose_runs(request, session.run), "offline")
        if frame.mode is InteractionMode.QUERY and "knowledge_lookup" not in frame.capabilities:
            return self.result_to_decision(self.interpret_result(request, session.run), "offline")
        if frame.mode in {InteractionMode.CHAT, InteractionMode.CANCEL_TASK} or (
            not frame.needs_planning and not frame.needs_tool and "knowledge_lookup" not in frame.capabilities
        ):
            return AgentDecision(
                type=DecisionType.FINAL,
                reasoning_summary="当前回合不需要 GIS 执行。",
                final_response=self.conversation_reply(request.user_input),
                source="offline",
            )
        findings = self.knowledge.retrieve(request.user_input) if "knowledge_lookup" in frame.capabilities else []
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


def _delegation_decision(request: AgentRequest, datasets, decomposer: TaskDecomposer) -> AgentDecision:
    return AgentDecision(
        type=DecisionType.DELEGATE,
        reasoning_summary="任务包含多个相互独立的 GIS 主题，交给临时智能体并行处理。",
        subtasks=decomposer.decompose(request, datasets),
        source="offline",
    )


def _should_delegate(request_frame: RequestFrame, datasets) -> bool:
    text = request_frame.goal.casefold()
    roles = sum(
        any(term in text for term in terms)
        for terms in (("道路", "路网", "road"), ("人口", "population"), ("dem", "高程", "地形", "terrain"))
    )
    return roles >= 2 and len(datasets) >= 2 and bool(set(request_frame.capabilities) & {"raster_analysis", "vector_analysis"})


__all__ = ["OfflineDecisionProvider"]


def _operation_from_goal(goal: str) -> str | None:
    terms = (
        ("buffer", ("缓冲", "buffer")),
        ("clip", ("裁剪", "clip")),
        ("intersection", ("相交", "交集", "intersection")),
        ("spatial_join", ("空间连接", "spatial join")),
        ("zonal_statistics", ("分区统计", "zonal")),
        ("slope", ("坡度", "slope")),
        ("reproject", ("重投影", "坐标转换", "reproject")),
        ("distance", ("距离分析", "测距", "distance")),
    )
    lowered = goal.casefold()
    return next((name for name, names in terms if any(term.casefold() in lowered for term in names)), None)
