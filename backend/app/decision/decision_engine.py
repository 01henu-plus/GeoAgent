"""把当前状态或模型响应转换成下一步 AgentDecision。

本模块只做 Decision，不执行工具、计划、Memory 或 checkpoint。模型响应的
工具参数在这里被转换成结构化 ToolCall；真实执行仍由 AgentRuntime/上层能力负责。
"""

from __future__ import annotations

import json
from typing import Any

from app.core.models import (
    AgentDecision,
    DecisionType,
    RequestResolutionStatus,
    ToolCall,
)
from app.models import ModelResponse

CONTROL_CAPABILITY_NAMES = frozenset({"agent.plan", "agent.delegate", "agent.ask_user", "agent.replan"})

CONTROL_CAPABILITY_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "agent.plan",
            "description": "为存在明确多步依赖的 GIS 目标建立执行计划。不要为了规划而规划。",
            "parameters": {"type": "object", "properties": {"goal": {"type": "string"}}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent.delegate",
            "description": "把相对独立的多个主题交给临时子智能体并行处理。只在确有并行价值时使用。",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent.ask_user",
            "description": "当缺少必须由用户提供的信息时向用户提问，而不是猜测。",
            "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent.replan",
            "description": "当前计划因失败或新信息不再适用时请求重新规划。",
            "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "additionalProperties": False},
        },
    },
]


class DecisionEngine:
    """当前阶段的轻量 Decision adapter。"""

    def from_model_response(self, response: ModelResponse, *, source: str = "model") -> AgentDecision:
        if response.tool_calls:
            call_views = [_model_call_view(raw_call, index) for index, raw_call in enumerate(response.tool_calls)]
            control_calls = [item for item in call_views if item["name"] in CONTROL_CAPABILITY_NAMES]
            gis_calls = [item for item in call_views if item["name"] not in CONTROL_CAPABILITY_NAMES]
            if control_calls:
                if len(control_calls) != 1 or gis_calls:
                    return AgentDecision(
                        type=DecisionType.ABORT,
                        reasoning_summary="INVALID_AGENT_DECISION_BATCH：内部控制动作不能与其他控制动作或 GIS 工具混在同一批次。",
                        source=source,
                        metadata={
                            "error_code": "INVALID_AGENT_DECISION_BATCH",
                            "raw_tool_calls": response.tool_calls,
                            "model": response.model,
                        },
                    )
                return _control_decision(control_calls[0], response, source=source)
            tool_calls: list[ToolCall] = []
            invalid_calls: list[dict[str, str]] = []
            for item in call_views:
                call_id = item["id"]
                name = item["name"]
                arguments = item["arguments"]
                try:
                    parsed_arguments = _parse_arguments(arguments)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    parsed_arguments = {}
                    invalid_calls.append({"id": call_id, "name": name or "model.tool_call", "error": str(exc)})
                tool_calls.append(ToolCall(id=call_id, name=name or "model.tool_call", arguments=parsed_arguments))
            return AgentDecision(
                type=DecisionType.TOOL,
                reasoning_summary=f"模型决定调用 {len(tool_calls)} 个工具。",
                tool_calls=tool_calls,
                source=source,
                metadata={
                    "model": response.model,
                    "content": response.content or "",
                    "raw_tool_calls": response.tool_calls,
                    "invalid_tool_calls": invalid_calls,
                },
            )
        if response.content.strip():
            return AgentDecision(
                type=DecisionType.FINAL,
                reasoning_summary="模型返回了当前回合的最终回答。",
                final_response=response.content.strip(),
                source=source,
                metadata={"model": response.model},
            )
        return AgentDecision(
            type=DecisionType.ABORT,
            reasoning_summary="模型未返回可执行的工具调用或文本回答。",
            source=source,
            metadata={"empty_response": True, "model": response.model},
        )

    def decide(self, state: Any, response: ModelResponse | None = None, *, source: str = "runtime") -> AgentDecision | None:
        """执行不依赖模型的高置信状态门控。"""

        if response is not None:
            return self.from_model_response(response, source=source)
        return self.decide_from_state(state)

    def decide_from_state(self, state: Any) -> AgentDecision | None:
        """只返回明确的状态动作；没有确定性动作时返回 None 交给模型。"""

        frame = getattr(state, "request_frame", None)
        if frame is not None and frame.resolution_status is not RequestResolutionStatus.RESOLVED:
            return AgentDecision(
                type=DecisionType.ASK_USER,
                reasoning_summary="请求仍有未解析的必要信息。",
                final_response="请补充必要信息后再继续。",
                source="state_gate",
            )
        if getattr(state, "latest_failure", None):
            failure = state.latest_failure
            action = str(failure.get("directive") or failure.get("action", ""))
            if failure.get("deterministic", True) is False:
                action = ""
            if action == "ASK_USER":
                return AgentDecision(type=DecisionType.ASK_USER, reasoning_summary="上一步执行需要用户补充信息。", source="state_gate")
            if action == "ABORT":
                return AgentDecision(type=DecisionType.ABORT, reasoning_summary="上一步执行要求终止当前运行。", source="state_gate")
            if action == "REPLAN":
                return AgentDecision(type=DecisionType.REPLAN, reasoning_summary=failure.get("error") or "上一步执行策略不再适用，需要重新规划。", source="runtime_recovery")
        if getattr(state, "current_plan", None) is None and frame is not None and frame.needs_planning:
            return AgentDecision(
                type=DecisionType.PLAN,
                reasoning_summary="当前请求需要先建立一个可执行计划。",
                plan_goal=state.goal,
                source="state_gate",
            )
        return None


def _model_call_view(raw_call: dict[str, Any], index: int) -> dict[str, Any]:
    function = raw_call.get("function") or raw_call
    return {
        "id": str(raw_call.get("id") or f"model_call_{index}"),
        "name": str(function.get("name") or raw_call.get("name") or ""),
        "arguments": function.get("arguments", raw_call.get("arguments", {})),
        "raw": raw_call,
    }


def _control_decision(call: dict[str, Any], response: ModelResponse, *, source: str) -> AgentDecision:
    name = call["name"]
    try:
        arguments = _parse_arguments(call["arguments"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return AgentDecision(
            type=DecisionType.ABORT,
            reasoning_summary=f"INVALID_AGENT_DECISION_ARGUMENTS：{exc}",
            source=source,
            metadata={"error_code": "INVALID_AGENT_DECISION_ARGUMENTS", "raw_tool_calls": response.tool_calls, "model": response.model},
        )
    common_metadata = {"model": response.model, "control_capability": name, "raw_tool_calls": response.tool_calls, "arguments": arguments}
    if name == "agent.plan":
        return AgentDecision(type=DecisionType.PLAN, reasoning_summary="模型请求建立执行计划。", plan_goal=str(arguments.get("goal") or ""), source=source, metadata=common_metadata)
    if name == "agent.delegate":
        return AgentDecision(type=DecisionType.DELEGATE, reasoning_summary=str(arguments.get("reason") or "模型请求委派相对独立的主题。"), source=source, metadata=common_metadata)
    if name == "agent.ask_user":
        question = str(arguments.get("question") or "请补充完成当前任务所需的信息。")
        return AgentDecision(type=DecisionType.ASK_USER, reasoning_summary=question, final_response=question, source=source, metadata=common_metadata)
    return AgentDecision(type=DecisionType.REPLAN, reasoning_summary=str(arguments.get("reason") or "模型请求根据当前失败重新规划。"), source=source, metadata=common_metadata)


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value or "{}")
    if not isinstance(value, dict):
        raise TypeError("模型工具参数必须是 JSON 对象")
    return value


__all__ = ["CONTROL_CAPABILITY_DEFINITIONS", "CONTROL_CAPABILITY_NAMES", "DecisionEngine"]
