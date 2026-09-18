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


class DecisionEngine:
    """当前阶段的轻量 Decision adapter。"""

    def from_model_response(self, response: ModelResponse, *, source: str = "model") -> AgentDecision:
        if response.tool_calls:
            tool_calls: list[ToolCall] = []
            invalid_calls: list[dict[str, str]] = []
            for index, raw_call in enumerate(response.tool_calls):
                call_id = str(raw_call.get("id") or f"model_call_{index}")
                function = raw_call.get("function") or raw_call
                name = str(function.get("name") or raw_call.get("name") or "")
                arguments = function.get("arguments", raw_call.get("arguments", {}))
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

    def decide(self, state: Any, response: ModelResponse | None = None, *, source: str = "runtime") -> AgentDecision:
        """执行不依赖模型的高置信状态门控。"""

        if response is not None:
            return self.from_model_response(response, source=source)
        frame = getattr(state, "request_frame", None)
        if frame is not None and frame.resolution_status is not RequestResolutionStatus.RESOLVED:
            return AgentDecision(
                type=DecisionType.ASK_USER,
                reasoning_summary="请求仍有未解析的必要信息。",
                final_response="请补充必要信息后再继续。",
                source="state_gate",
            )
        if getattr(state, "latest_failure", None):
            action = str(state.latest_failure.get("action", ""))
            if action == "ASK_USER":
                return AgentDecision(type=DecisionType.ASK_USER, reasoning_summary="上一步执行需要用户补充信息。", source="state_gate")
            if action == "ABORT":
                return AgentDecision(type=DecisionType.ABORT, reasoning_summary="上一步执行要求终止当前运行。", source="state_gate")
        if getattr(state, "current_plan", None) is None and frame is not None and frame.needs_planning:
            return AgentDecision(
                type=DecisionType.PLAN,
                reasoning_summary="当前请求需要先建立一个可执行计划。",
                plan_goal=state.goal,
                source="state_gate",
            )
        return AgentDecision(
            type=DecisionType.FINAL,
            reasoning_summary="当前状态没有需要由确定性门控执行的下一步。",
            final_response=None,
            source="state_gate",
        )


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value or "{}")
    if not isinstance(value, dict):
        raise TypeError("模型工具参数必须是 JSON 对象")
    return value


__all__ = ["DecisionEngine"]
