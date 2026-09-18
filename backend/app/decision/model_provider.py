"""模型 Decision Provider。

Provider 只负责把当前 AgentState/RuntimeSession 转换为 AgentDecision，不执行工具、
规划、状态写入或运行结束处理。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.core.models import (
    AgentDecision,
    AgentRequest,
    IntentResult,
    RequestFrame,
    RequestResources,
    Run,
    Task,
)
from app.decision.decision_engine import DecisionEngine
from app.models import ModelAdapter, ModelRequest, ModelResponse
from app.runtime.session import AgentRuntimeSession


class ModelDecisionProvider:
    """将模型调用边界从 MainAgent 运行时中分离出来。"""

    def __init__(
        self,
        *,
        decision_engine: DecisionEngine,
        build_messages: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]],
        refresh_datasets: Callable[..., list[Any]],
        resolve_request_resources: Callable[[AgentRequest], RequestResources],
        max_tokens: int,
    ) -> None:
        self.decision_engine = decision_engine
        self.build_messages = build_messages
        self.refresh_datasets = refresh_datasets
        self.resolve_request_resources = resolve_request_resources
        self.max_tokens = max_tokens

    async def decide(
        self,
        state,
        session: AgentRuntimeSession,
        *,
        request: AgentRequest,
        run: Run,
        task: Task | None,
        intent: IntentResult | None,
        request_frame: RequestFrame | None,
        model_adapter: ModelAdapter,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentDecision:
        datasets = self.refresh_datasets(
            request,
            session.datasets,
            session.dataset_ids,
            session.working_memory,
            session.latest_observation,
        )
        session.datasets = datasets
        request_resources = self.resolve_request_resources(request)
        messages, bounded_protocol, tools = self.build_messages(
            request,
            run,
            task,
            datasets,
            intent,
            session.current_plan,
            request_frame,
            session.protocol_messages,
            working_memory=session.working_memory,
            request_resources=request_resources,
            current_observation=session.latest_observation,
            plan_progress={"completed_steps": sorted(session.completed_steps), "step_outputs": session.step_outputs},
            latest_failure=session.latest_failure,
        )
        session.protocol_messages = bounded_protocol
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        input_tokens = output_tokens = 0
        model_name: str | None = None
        async for chunk in model_adapter.stream(ModelRequest(messages=messages, tools=tools, max_tokens=self.max_tokens)):
            if chunk.content:
                content_parts.append(chunk.content)
                if on_model_delta is not None:
                    await on_model_delta(chunk.content)
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
            input_tokens = chunk.input_tokens or input_tokens
            output_tokens = chunk.output_tokens or output_tokens
            model_name = chunk.model or model_name
        response = ModelResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_name,
        )
        session.last_response = response
        return self.decision_engine.from_model_response(response, source="model")


__all__ = ["ModelDecisionProvider"]
