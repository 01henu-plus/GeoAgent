"""LLM Provider 的最小适配协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int = 2000


class ModelResponse(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None


class ModelStreamChunk(BaseModel):
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    done: bool = False


class ModelAdapter(ABC):
    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """执行一次模型请求。"""

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        """流式执行一次模型请求；未实现流式接口的适配器退化为单片段。"""
        response = await self.complete(request)
        yield ModelStreamChunk(
            content=response.content,
            tool_calls=response.tool_calls,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
            done=True,
        )

    async def close(self) -> None:
        return None
